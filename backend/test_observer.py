"""Offline tests for everything around the model call.

The model call itself needs credentials; the logic that guards it does not, and
that logic is where the bugs actually live. Run with: python -m backend.test_observer
"""

from __future__ import annotations

from pathlib import Path

from backend.observer import (
    MAX_DELTA_PER_EVENT,
    Observer,
    format_votes,
    infer_vote,
    _merge_contradictions,
    _settle,
    infer_elimination,
)
from backend.schema import (
    BeliefState,
    ClaimEntry,
    Contradiction,
    GameEvent,
    ObserverOutput,
    SuspicionEntry,
    Transcript,
)
from backend.scoring import (
    count_unexplained_reversals,
    final_verdict,
    grade,
    measure_drift,
    mentions,
    verdict,
)

PLAYERS = ["P1", "P2", "P3", "P4"]
UNIFORM = {p: 0.25 for p in PLAYERS}


def check(label: str, cond: bool) -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {label}")
    assert cond, label


def test_settle_normalizes() -> None:
    print("_settle")
    out = _settle(
        [SuspicionEntry(player=p, score=s) for p, s in zip(PLAYERS, [0.9, 0.9, 0.9, 0.9])],
        UNIFORM,
        PLAYERS,
        first_event=True,
    )
    check("sums to 1.0", abs(sum(out.values()) - 1.0) < 1e-9)

    # A player the model omitted must not be silently exonerated.
    out = _settle(
        [SuspicionEntry(player="P1", score=0.5)], {"P1": 0.1, "P2": 0.7, "P3": 0.1, "P4": 0.1},
        PLAYERS, first_event=False,
    )
    check("omitted player keeps its rank", max(out, key=lambda p: out[p]) == "P2")

    # Rate limit: P1 cannot leap from 0.1 to 0.95 in one event.
    out = _settle(
        [SuspicionEntry(player="P1", score=0.95)], {"P1": 0.1, "P2": 0.3, "P3": 0.3, "P4": 0.3},
        PLAYERS, first_event=False,
    )
    check("single-event swing is capped", out["P1"] < 0.5)


def test_cap_holds_after_normalization() -> None:
    """The regression: clamping before normalizing let a 0.76 player print 0.418,
    a 34-point move under a 30-point cap. Reproduces that exact shape."""
    print("_settle cap (post-normalization)")
    prior = {"P1": 0.04, "P2": 0.14, "P3": 0.15, "P4": 0.76}
    out = _settle(
        [
            SuspicionEntry(player="P1", score=0.05),
            SuspicionEntry(player="P2", score=0.10),
            SuspicionEntry(player="P3", score=0.60),
            SuspicionEntry(player="P4", score=0.10),  # model wants a huge drop
        ],
        prior,
        PLAYERS,
        first_event=False,
    )
    worst = max(abs(out[p] - prior[p]) for p in PLAYERS)
    check(f"no player moves more than the cap (worst {worst:.3f})", worst <= MAX_DELTA_PER_EVENT + 1e-6)
    check("still sums to 1.0", abs(sum(out.values()) - 1.0) < 1e-6)
    check("the intended direction survives", out["P4"] < prior["P4"] and out["P3"] > prior["P3"])


def test_eliminated_players_leave_the_distribution() -> None:
    print("elimination")
    dead_line = GameEvent(
        round=1, phase="night", speaker="MODERATOR",
        statement="Night falls. In the morning, P5 is found dead.",
    )
    check("death is read from the moderator", infer_elimination(dead_line) == "P5")
    check(
        "explicit field wins over the text",
        infer_elimination(
            GameEvent(round=2, phase="vote", speaker="MODERATOR",
                      statement="The village has decided.", eliminated="P2")
        ) == "P2",
    )
    check(
        "ordinary dialogue kills nobody",
        infer_elimination(
            GameEvent(round=1, phase="day", speaker="P1", statement="P4 has been loud today.")
        ) is None,
    )

    alive = ["P1", "P2", "P3"]
    out = _settle(
        [SuspicionEntry(player=p, score=0.3) for p in PLAYERS],  # model still scores dead P4
        UNIFORM, alive, first_event=True,
    )
    check("the dead are not in the distribution", "P4" not in out)
    check("the living still sum to 1.0", abs(sum(out.values()) - 1.0) < 1e-9)


def test_contradictions_are_append_only() -> None:
    print("_merge_contradictions")
    a = Contradiction(player="P4", earlier="said x", now="said y", round_noticed=1)
    b = Contradiction(player="P2", earlier="said q", now="said r", round_noticed=2)
    check("dropped entry is restored", len(_merge_contradictions([a], [b])) == 2)
    check("duplicate is not re-added", len(_merge_contradictions([a], [a, b])) == 2)


def test_reversal_counting() -> None:
    print("count_unexplained_reversals")
    events = [
        GameEvent(round=1, phase="day", speaker="P1", statement="hello"),
        GameEvent(round=1, phase="day", speaker="P1", statement="nothing about anyone"),
        GameEvent(round=1, phase="day", speaker="P1", statement="actually P4 seems fine to me"),
    ]
    history = [
        {"P4": 0.60, "P1": 0.40},
        {"P4": 0.20, "P1": 0.80},  # unexplained: nobody mentioned P4
        {"P4": 0.05, "P1": 0.95},  # explained: the line is about P4
    ]
    check("counts only the unexplained drop", count_unexplained_reversals(history, events) == 1)


def test_full_loop_with_stub_model() -> None:
    print("Observer.observe (stubbed model)")
    transcript = Transcript.model_validate_json(
        Path("data/fallback_transcript.json").read_text(encoding="utf-8")
    )

    class StubLLM:
        """A StructuredLLM that nudges P4 upward, ignoring the prompt entirely."""

        def __init__(self) -> None:
            self.calls = 0
            self.prompts: list[str] = []

        def structured(self, system: str, user: str, schema):
            self.calls += 1
            self.prompts.append(user)
            scores = {p: 0.15 for p in transcript.setup.players}
            scores["P4"] = 0.15 + min(0.6, 0.05 * self.calls)
            return ObserverOutput(
                suspicion=[SuspicionEntry(player=p, score=s) for p, s in scores.items()],
                claims_tracked=[ClaimEntry(player="P4", claim=f"tracked as of call {self.calls}")],
                # Real lines from the transcript -- the attribution check drops
                # quotes the accused player never actually said.
                contradictions_noticed=(
                    [
                        Contradiction(
                            player="P4",
                            earlier="I'm voting P2.",
                            now="I vote P1.",
                            round_noticed=2,
                        )
                    ]
                    if self.calls > 18
                    else []
                ),
                reasoning=f"stub reasoning {self.calls}",
            )

    stub = StubLLM()
    # skip_trivial and batch_opening both off: this test is about the loop's
    # bookkeeping, and it counts calls, so it wants the one-call-per-event
    # shape on purpose.
    observer = Observer(
        transcript.setup, llm=stub, skip_trivial=False, batch_opening=False
    )
    for event in transcript.events:
        observer.observe(event)

    check("one call per event", stub.calls == len(transcript.events))
    check("history is one snapshot per event", len(observer.history) == len(transcript.events))
    check(
        "prior state is fed back into every prompt after the first",
        all("Suspicion:" in p for p in stub.prompts[1:]),
    )
    check(
        "ground truth never reaches the model",
        not any("werewolf is" in p.lower() or "ground_truth" in p for p in stub.prompts),
    )
    check("contradiction survives to the end", len(observer.state.contradictions_noticed) == 1)

    score = grade(observer.state, observer.history, transcript.events, transcript.ground_truth)
    check("scorer reads the top suspect", score.predicted == "P4" and score.accuracy)
    check("consistency is in range", 0.0 <= score.consistency <= 1.0)
    check("history is exported for the chart", len(score.suspicion_history) == len(transcript.events))


def test_verdict_survives_the_wolf_being_voted_out() -> None:
    """Regression: the game ends by eliminating the werewolf, which removes them
    from the distribution. Reading the last state would mark a correct observer
    wrong, because its answer had just been voted out."""
    print("verdict")
    events = [
        GameEvent(round=2, phase="day", speaker="P3", statement="P4 lied about the vote."),
        GameEvent(round=2, phase="vote", speaker="P2", statement="I vote P4."),
        GameEvent(round=2, phase="vote", speaker="MODERATOR", statement="P4 is eliminated."),
    ]
    history = [
        {"P2": 0.2, "P3": 0.2, "P4": 0.6},
        {"P2": 0.2, "P3": 0.2, "P4": 0.6},
        {"P2": 0.5, "P3": 0.5},  # P4 gone -- last state cannot name them
    ]
    check("verdict ignores the resolving event", verdict(history, events) == "P4")

    score = grade(
        BeliefState(eliminated=["P4"], suspicion=history[-1]),
        history, events, {"P2": "villager", "P3": "villager", "P4": "werewolf"},
    )
    check("a correct observer is graded correct", score.accuracy and score.predicted == "P4")
    check("confidence comes from the same snapshot", abs(score.final_confidence - 0.6) < 1e-9)


def test_resume_continues_instead_of_restarting() -> None:
    """A run stopped by quota must pick up with its memory intact."""
    print("resume")
    transcript = Transcript.model_validate_json(
        Path("data/fallback_transcript.json").read_text(encoding="utf-8")
    )

    class CountingLLM:
        def __init__(self) -> None:
            self.calls = 0

        def structured(self, system: str, user: str, schema):
            self.calls += 1
            return ObserverOutput(
                suspicion=[SuspicionEntry(player=p, score=0.25) for p in transcript.setup.players],
                claims_tracked=[ClaimEntry(player="P4", claim="noted early")],
                # `earlier` is from before the restart and `now` from after, so
                # this only survives if restore() rebuilt the record of who said
                # what. Both are real P4 lines inside the first ten events.
                contradictions_noticed=[
                    Contradiction(
                        player="P4",
                        earlier="P5 wasn't going to be much help anyway",
                        now="I'm voting P2.",
                        round_noticed=1,
                    )
                ],
                reasoning="stub",
            )

    # batch_opening off: resume is about paying only for events after the
    # restart, and a batched opening would fold the first six into one call.
    first = Observer(transcript.setup, llm=CountingLLM(), skip_trivial=False, batch_opening=False)
    for event in transcript.events[:6]:
        first.observe(event)
    saved_state, saved_history = first.state, first.history

    second_llm = CountingLLM()
    second = Observer(transcript.setup, llm=second_llm, skip_trivial=False, batch_opening=False)
    second.restore(
        BeliefState.model_validate(saved_state.model_dump()),
        saved_history,
        first.states,
        transcript.events[:6],
    )
    for event in transcript.events[6:10]:
        second.observe(event)

    check("only the remaining events are paid for", second_llm.calls == 4)
    check("event numbering continues", second.state.event_index == 9)
    check("history is continuous", len(second.history) == 10)
    check("the claims ledger survived the restart", "P4" in second.state.claims_tracked)
    check(
        "caught contradictions survived the restart",
        len(second.state.contradictions_noticed) == 1,
    )
    check("the dead stayed dead across the restart", "P5" in second.state.eliminated)


def test_recording_round_trips_into_the_replay_shape() -> None:
    """A saved run must carry each turn's whole state, not just its numbers --
    otherwise `feeder.py --replay` drives the UI with blank reasoning panes."""
    print("recording shape")
    transcript = Transcript.model_validate_json(
        Path("data/fallback_transcript.json").read_text(encoding="utf-8")
    )

    class StubLLM:
        def structured(self, system: str, user: str, schema):
            return ObserverOutput(
                suspicion=[SuspicionEntry(player=p, score=0.25) for p in transcript.setup.players],
                claims_tracked=[ClaimEntry(player="P4", claim="dismissed the victim")],
                # Verbatim P4 lines; invented quotes are dropped by the
                # attribution check, which is the point of that check.
                contradictions_noticed=[
                    Contradiction(
                        player="P4",
                        earlier="P5 wasn't going to be much help anyway",
                        now="I'm voting P2.",
                        round_noticed=1,
                    )
                ],
                reasoning="P4 moved without explaining why.",
            )

    # batch_opening off: this pins the shape of a *per-event* recording, and
    # eight buffered events would be carried forward without a call at all.
    observer = Observer(transcript.setup, llm=StubLLM(), batch_opening=False)
    for event in transcript.events[:8]:
        observer.observe(event)

    check("a state is kept per turn", len(observer.states) == 8)

    import json as _json
    import tempfile

    from backend.run_observer import _write_run

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "run.json"
        _write_run(path, transcript, observer, None, 8)
        record = _json.loads(path.read_text(encoding="utf-8"))

    for key in ("setup", "ground_truth", "turns", "events_expected", "history"):
        check(f"recording carries '{key}'", key in record)
    check("one turn per event", len(record["turns"]) == 8)

    turn = record["turns"][0]
    check("turns pair event with state", "event" in turn and "state" in turn)
    check("reasoning survives to disk", turn["state"]["reasoning"] != "")
    check(
        "contradictions survive to disk",
        len(record["turns"][-1]["state"]["contradictions_noticed"]) == 1,
    )

    # The shape Lane A's feeder validates against.
    from backend.state import Turn as StateTurn

    check("turns validate as state.Turn", StateTurn.model_validate(turn).index == 0)


def test_drift_sees_what_consistency_misses() -> None:
    """The real run that motivated this: the observer led with P4, wandered to P3
    for seven events, then came back -- and scored a perfect 1.00 consistency."""
    print("drift")
    wobble = (
        [{"P4": 0.55, "P3": 0.20, "P2": 0.25}] * 3        # P4 leads
        + [{"P4": 0.10, "P3": 0.65, "P2": 0.25}] * 7      # P3 leads for seven
        + [{"P4": 0.80, "P3": 0.12, "P2": 0.08}] * 3      # P4 again
    )
    changes, off = measure_drift(wobble, "P4")
    check("counts both handovers", changes == 2)
    check("counts the seven events off verdict", off == 7)

    steady = [{"P4": 0.4 + i * 0.02, "P3": 0.3, "P2": 0.3} for i in range(10)]
    changes, off = measure_drift(steady, "P4")
    check("a steady read drifts not at all", changes == 0 and off == 0)
    check("empty history is survivable", measure_drift([], "P4") == (0, 0))
    check("a verdict that never led reports no wobble", measure_drift(steady, "P9") == (0, 0))


def test_transcript_validation() -> None:
    """Real datasets are malformed in all of these ways. Catch them at load,
    not twenty paid-for model calls into a run."""
    print("transcript validation")
    import copy

    base = Transcript.model_validate_json(
        Path("data/fallback_transcript.json").read_text(encoding="utf-8")
    ).model_dump()

    def rejects(label: str, mutate) -> None:
        broken = copy.deepcopy(base)
        mutate(broken)
        try:
            Transcript.model_validate(broken)
        except Exception:
            check(f"rejects {label}", True)
        else:
            check(f"rejects {label}", False)

    rejects("no werewolf", lambda d: d["ground_truth"].update({"P4": "villager"}))
    rejects("two werewolves", lambda d: d["ground_truth"].update({"P1": "werewolf"}))
    rejects("a player with no role", lambda d: d["ground_truth"].pop("P3"))
    rejects("a role for a non-player", lambda d: d["ground_truth"].update({"P9": "villager"}))
    rejects("duplicate players", lambda d: d["setup"].update({"players": ["P1", "P1", "P2"]}))
    rejects(
        "a speaker who isn't in the game",
        lambda d: d["events"].append(
            {"round": 3, "phase": "day", "speaker": "P9", "statement": "hello"}
        ),
    )
    rejects(
        "an event that reveals the answer",
        lambda d: d["events"].append(
            {"round": 3, "phase": "day", "speaker": "MODERATOR", "statement": "P4 was the werewolf."}
        ),
    )

    # ...without rejecting things that are fine.
    ok = copy.deepcopy(base)
    ok["events"].append(
        {"round": 3, "phase": "day", "speaker": "NARRATOR", "statement": "The village sleeps."}
    )
    Transcript.model_validate(ok)
    check("accepts a narrator who isn't a player", True)

    ok2 = copy.deepcopy(base)
    ok2["events"].append(
        {"round": 3, "phase": "day", "speaker": "P2", "statement": "I think P4 is the werewolf."}
    )
    Transcript.model_validate(ok2)
    check("accepts a player accusing someone (not a reveal)", True)


def test_only_the_narrator_can_eliminate() -> None:
    """Found on real Mafia chat: players say "ashton is dead" and "who is dead?"
    constantly. Read as eliminations, any player could remove a rival from the
    suspicion distribution -- including the person currently accusing them."""
    print("elimination is narrator-only")
    said_by_player = GameEvent(round=2, phase="day", speaker="Elliot", statement="ashton is dead")
    check("a player cannot kill anyone", infer_elimination(said_by_player) is None)
    check(
        "nor by mimicking the moderator",
        infer_elimination(
            GameEvent(round=2, phase="day", speaker="Eden",
                      statement="Noah was voted out. Their role was mafia")
        ) is None,
    )
    check(
        "the moderator still can",
        infer_elimination(
            GameEvent(round=2, phase="vote", speaker="Game-Manager",
                      statement="Noah was voted out. Their role was bystander")
        ) == "Noah",
    )
    check(
        "an explicit field wins regardless of speaker",
        infer_elimination(
            GameEvent(round=1, phase="day", speaker="Eden", statement="hm", eliminated="Kai")
        ) == "Kai",
    )


def test_mentions_needs_a_real_name_not_a_substring() -> None:
    """33 collisions across the real games: "Sage" inside "messages", "Lee"
    inside "feeling". Each one excused a suspicion drop as explained, quietly
    inflating consistency."""
    print("name mentions")
    for name, text in [("Sage", "look at the previos messages"), ("Lee", "this feeling"),
                       ("Ari", "im curious"), ("Adrian", "adriann sucks alot")]:
        check(f"{name!r} not matched inside {text.split()[-1]!r}", not mentions(name, text))
    for name, text in [("Rowan", "rowans reasoning"), ("Winter", "#vote_for_winter"),
                       ("Kai", "@kai you there"), ("Sage", "sages question"),
                       ("Kai", "Ronny, Kai, Ari")]:
        check(f"{name!r} matched in {text!r}", mentions(name, text))


def test_contradictions_must_be_the_accused_players_own_words() -> None:
    """Real failure on a Mafia game: the model reported Sidney contradicting
    themselves, quoting Sidney's line as `earlier` and Whitney's as `now`."""
    print("contradiction attribution")
    said = {
        "Sidney": ["whitney the hat is burning on the theif's head", "i think it is whitney"],
        "Whitney": ["I do not have a hat", "Let's vote out Rowan"],
    }
    misattributed = Contradiction(
        player="Sidney",
        earlier="whitney the hat is burning on the theif's head",
        now="I do not have a hat",  # Whitney said this
        round_noticed=1,
    )
    check("someone else's line is not a contradiction", not _merge_contradictions([], [misattributed], said))

    genuine = Contradiction(
        player="Whitney", earlier="Let's vote out Rowan", now="I do not have a hat", round_noticed=1
    )
    check("both halves theirs is kept", len(_merge_contradictions([], [genuine], said)) == 1)
    check(
        "with no record of who said what, nothing is filtered",
        len(_merge_contradictions([], [misattributed], None)) == 1,
    )


def test_resume_keeps_the_attribution_record() -> None:
    """Regression: `restore` used to clear `said`, so after a resume every quote
    from before the restart looked misattributed and was dropped."""
    print("resume keeps attribution")
    transcript = Transcript.model_validate_json(
        Path("data/fallback_transcript.json").read_text(encoding="utf-8")
    )
    quote = transcript.events[6].statement  # P4, round 1

    class Stub:
        def structured(self, system, user, schema):
            return ObserverOutput(
                suspicion=[SuspicionEntry(player=p, score=0.2) for p in transcript.setup.players],
                claims_tracked=[],
                contradictions_noticed=[
                    Contradiction(player="P4", earlier=quote, now="I vote P1.", round_noticed=1)
                ],
                reasoning="x",
            )

    blind = Observer(transcript.setup, llm=Stub())
    blind.restore(BeliefState(), [], [])
    for event in transcript.events[13:16]:
        blind.observe(event)
    check("without the events the old quote is unverifiable", not blind.state.contradictions_noticed)

    restored = Observer(transcript.setup, llm=Stub())
    restored.restore(BeliefState(), [], [], transcript.events[:13])
    for event in transcript.events[13:16]:
        restored.observe(event)
    check("with them the contradiction survives", len(restored.state.contradictions_noticed) == 1)


def test_vote_record() -> None:
    """The vote graph is the one measurably strong signal: across the 33 real
    games, under 4% of a liar's votes land on their own partner."""
    print("vote record")
    check(
        "the narrator's announcement is a vote",
        infer_vote(
            GameEvent(round=1, phase="vote", speaker="Game-Manager", statement="Kai voted for Sutton")
        ) == ("Kai", "Sutton"),
    )
    check(
        "a player typing the same thing is not",
        infer_vote(
            GameEvent(round=2, phase="day", speaker="Eden", statement="noah voted for bystander")
        ) is None,
    )

    votes = {1: {"A": "C", "B": "C", "C": "A"}, 2: {"A": "D", "B": "D"}}
    rendered = format_votes(votes, ["A", "B", "C", "D"])
    check("every round is shown", "round 1:" in rendered and "round 2:" in rendered)
    check("the never-voted pair is surfaced", "A/B" in rendered)
    check("a pair who did vote at each other is not", "A/C" not in rendered)
    check("nothing to show before any vote", format_votes({}, ["A"]) == "(nobody has voted yet)")

    # End to end: the record reaches the prompt, and survives a resume.
    transcript = Transcript.model_validate_json(
        Path("data/fallback_transcript.json").read_text(encoding="utf-8")
    )

    class Stub:
        def __init__(self) -> None:
            self.last = ""

        def structured(self, system, user, schema):
            self.last = user
            return ObserverOutput(
                suspicion=[SuspicionEntry(player=p, score=0.2) for p in transcript.setup.players],
                claims_tracked=[], contradictions_noticed=[], reasoning="x",
            )

    stub = Stub()
    observer = Observer(transcript.setup, llm=stub)
    for event in transcript.events[:14]:
        observer.observe(event)
    check("votes are tracked from the transcript", observer.votes.get(1, {}).get("P1") == "P4")
    check("the record reaches the prompt", "THE VOTE RECORD" in stub.last)

    resumed = Observer(transcript.setup, llm=Stub())
    resumed.restore(BeliefState(), [], [], transcript.events[:14])
    check("and is rebuilt on resume", resumed.votes == observer.votes)


def test_belief_state_defaults() -> None:
    print("BeliefState")
    check("empty state has no top suspect", BeliefState().top_suspect is None)


if __name__ == "__main__":
    for fn in [
        test_settle_normalizes,
        test_cap_holds_after_normalization,
        test_eliminated_players_leave_the_distribution,
        test_contradictions_are_append_only,
        test_reversal_counting,
        test_full_loop_with_stub_model,
        test_verdict_survives_the_wolf_being_voted_out,
        test_resume_continues_instead_of_restarting,
        test_recording_round_trips_into_the_replay_shape,
        test_drift_sees_what_consistency_misses,
        test_transcript_validation,
        test_only_the_narrator_can_eliminate,
        test_mentions_needs_a_real_name_not_a_substring,
        test_contradictions_must_be_the_accused_players_own_words,
        test_resume_keeps_the_attribution_record,
        test_vote_record,
        test_belief_state_defaults,
    ]:
        fn()
    print("\nall offline checks passed")
