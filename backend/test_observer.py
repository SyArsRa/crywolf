"""Offline tests for everything around the model call.

The model call itself needs credentials; the logic that guards it does not, and
that logic is where the bugs actually live. Run with: python -m backend.test_observer
"""

from __future__ import annotations

from pathlib import Path

from backend.observer import (
    MAX_DELTA_PER_EVENT,
    Observer,
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
from backend.scoring import count_unexplained_reversals, grade, verdict

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
                contradictions_noticed=(
                    [Contradiction(player="P4", earlier="would vote P2", now="voted P1", round_noticed=2)]
                    if self.calls > 18
                    else []
                ),
                reasoning=f"stub reasoning {self.calls}",
            )

    stub = StubLLM()
    observer = Observer(transcript.setup, llm=stub)
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
                contradictions_noticed=[
                    Contradiction(player="P4", earlier="said A", now="said B", round_noticed=1)
                ],
                reasoning="stub",
            )

    first = Observer(transcript.setup, llm=CountingLLM())
    for event in transcript.events[:6]:
        first.observe(event)
    saved_state, saved_history = first.state, first.history

    second_llm = CountingLLM()
    second = Observer(transcript.setup, llm=second_llm)
    second.restore(BeliefState.model_validate(saved_state.model_dump()), saved_history)
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
        test_belief_state_defaults,
    ]:
        fn()
    print("\nall offline checks passed")
