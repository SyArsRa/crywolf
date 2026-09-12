"""Offline tests for the computed evidence and the deliberation tier.

No API key and no model call. Run with: python -m backend.test_evidence

The properties worth pinning down here are the ones that would fail silently.
An evidence function that returns a confident answer when it has seen nothing,
or a deliberation call that fires on every event, would both still produce a
plausible-looking demo while costing money and being wrong.
"""

from __future__ import annotations

from pathlib import Path

from backend.evidence import (
    P_TARGET_IS_LIAR_GIVEN_INNOCENT,
    P_TARGET_IS_LIAR_GIVEN_LIAR,
    format_evidence,
    marginals,
)
from backend.observer import (
    MAX_DELTA_PER_EVENT,
    PRE_EVIDENCE_CEILING_MULTIPLE,
    PRE_EVIDENCE_MAX_DELTA,
    Observer,
    _settle,
)
from backend.schema import (
    ClaimEntry,
    Deliberation,
    GameEvent,
    GameSetup,
    ObserverOutput,
    SuspicionEntry,
    Transcript,
)

PLAYERS = ["A", "B", "C", "D", "E"]


def check(label: str, cond: bool) -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {label}")
    assert cond, label


def test_evidence_is_silent_until_a_role_is_announced() -> None:
    """The single most important property here.

    Returning a uniform distribution instead of `{}` would look identical on the
    chart and would quietly tell the observer it had evidence when the game has
    produced none. `format_evidence` has to say so in words, because the prompt
    is the only place that distinction can be acted on.
    """
    print("evidence is silent before any reveal")

    votes = {1: {"A": "B", "C": "D"}}
    check("no votes, no reveals -> nothing", marginals({}, {}, PLAYERS, 2) == {})
    check("votes but no reveals -> nothing", marginals(votes, {}, PLAYERS, 2) == {})
    check(
        "a reveal nobody voted for -> still nothing",
        marginals(votes, {"E": "bystander"}, PLAYERS, 2) == {},
    )

    blank = format_evidence({})
    check("the empty rendering says there is no evidence", "no structural evidence" in blank)
    check(
        "and warns against inventing a suspect",
        "do not manufacture" in blank.lower(),
    )


def test_voting_for_a_revealed_liar_clears_you() -> None:
    """The signal itself, in the direction the measurement says it runs.

    A vote aimed at someone later revealed to be a liar is something a liar
    almost never does, so the voter should end up *less* suspected than a player
    who did not.
    """
    print("the direction of the signal")

    check(
        "the two rates did not get swapped",
        P_TARGET_IS_LIAR_GIVEN_LIAR < P_TARGET_IS_LIAR_GIVEN_INNOCENT,
    )

    # A voted for D, who turns out to be a liar. B voted for C, who does not.
    scores = marginals(
        {1: {"A": "D", "B": "C"}},
        {"D": "mafia", "C": "bystander"},
        ["A", "B", "E"],
        1,
    )
    check("A is cleared by voting at the revealed liar", scores["A"] < scores["B"])
    check("and the silent player sits between them", scores["A"] < scores["E"])
    check(
        "the numbers sum to the liars remaining",
        abs(sum(scores.values()) - 1.0) < 1e-6,
    )


def test_evidence_sums_to_the_liar_count() -> None:
    """Same convention as `_settle`: each number is P(this player is lying), not
    a share of one pool. Two liars left means the bars total 2.0."""
    print("evidence uses the liars-remaining convention")

    scores = marginals(
        {1: {"A": "E", "B": "C"}, 2: {"C": "E", "D": "A"}},
        {"E": "mafia", "C": "bystander"},
        ["A", "B", "C", "D"],
        2,
    )
    check("sums to 2.0 with two liars left", abs(sum(scores.values()) - 2.0) < 1e-6)
    check("every living player is scored", set(scores) == {"A", "B", "C", "D"})
    check("nobody exceeds certainty", all(0.0 <= v <= 1.0 for v in scores.values()))


def test_the_pre_evidence_cap_actually_binds() -> None:
    """Before evidence exists the observer must not be able to commit.

    This is the fix for chasing whoever talks loudest in round 1, so it is worth
    a test that fails if someone raises the constant back to parity.
    """
    print("the evidence-free opening is throttled")

    check("the pre-evidence cap is the tighter one", PRE_EVIDENCE_MAX_DELTA < MAX_DELTA_PER_EVENT)

    prior = {p: 0.2 for p in PLAYERS}
    shouting = [SuspicionEntry(player="A", score=0.99)] + [
        SuspicionEntry(player=p, score=0.01) for p in PLAYERS[1:]
    ]
    throttled = _settle(shouting, prior, PLAYERS, False, 1.0, PRE_EVIDENCE_MAX_DELTA)
    loose = _settle(shouting, prior, PLAYERS, False, 1.0, MAX_DELTA_PER_EVENT)

    check("one loud line cannot run away with it", throttled["A"] <= 0.2 + PRE_EVIDENCE_MAX_DELTA + 1e-9)
    check("it moves less than it would after evidence", throttled["A"] < loose["A"])
    check("but it is throttled, not frozen", throttled["A"] > prior["A"])


def test_the_pre_evidence_ceiling_bounds_a_slow_commitment() -> None:
    """The per-event cap bounds the speed of a commitment, not its size.

    A live run made the point: throttled to 0.06 a line, the observer still
    walked a bystander from 0.25 to 0.60 across the twenty evidence-free events
    before the first reveal. Twenty small steps in one direction is still a
    commitment, and it is the size that reaches the verdict. So the opening also
    carries an absolute bound.
    """
    print("the opening ceiling bounds sustained pressure")

    roster = ["A", "B", "C", "D", "E", "F", "G", "H"]
    ceiling = PRE_EVIDENCE_CEILING_MULTIPLE * 2 / len(roster)
    shouting = [SuspicionEntry(player="A", score=0.99)] + [
        SuspicionEntry(player=p, score=0.01) for p in roster[1:]
    ]

    scores = {p: 0.25 for p in roster}
    for _ in range(30):
        scores = _settle(shouting, scores, roster, False, 2.0, PRE_EVIDENCE_MAX_DELTA, ceiling)

    check("thirty events of pressure cannot pass the ceiling", scores["A"] <= ceiling + 1e-9)
    check("nor can anyone else", all(v <= ceiling + 1e-9 for v in scores.values()))
    check("and the bars still total the liar count", abs(sum(scores.values()) - 2.0) < 1e-6)
    check("it is a bound, not a freeze", scores["A"] > 0.25)

    # Once evidence exists the bound lifts and the observer may commit properly.
    free = _settle(shouting, scores, roster, False, 2.0, MAX_DELTA_PER_EVENT, None)
    check("evidence lifts the ceiling", free["A"] > ceiling)


def test_deliberation_fires_only_on_a_role_reveal() -> None:
    """The whole economic argument for the second tier.

    If this fired per event it would be strictly worse than what it replaced --
    more expensive and no deeper. It must fire on role announcements only, and a
    deep call must not add a history entry, because the contract with Lane A and
    the scorer is one snapshot per event.
    """
    print("deliberation fires on reveals only")

    setup = GameSetup(
        players=["A", "B", "C"], deceiver_role="mafia", deceiver_count=1, deceiver_plural="mafia"
    )

    class TwoTierLLM:
        def __init__(self) -> None:
            self.shallow = 0
            self.deep = 0
            self.deep_prompts: list[str] = []

        def structured(self, system: str, user: str, schema):
            if schema is Deliberation:
                self.deep += 1
                self.deep_prompts.append(user)
                return Deliberation(
                    suspicion=[SuspicionEntry(player=p, score=0.5) for p in ["A", "B"]],
                    revised="I was wrong about A; they voted for the revealed mafia.",
                    reasoning="B now leads.",
                )
            self.shallow += 1
            return ObserverOutput(
                suspicion=[SuspicionEntry(player=p, score=0.33) for p in setup.players],
                claims_tracked=[ClaimEntry(player="A", claim="spoke")],
                contradictions_noticed=[],
                reasoning="per-event note",
            )

    events = [
        GameEvent(round=1, phase="day", speaker="A", statement="hello"),
        GameEvent(round=1, phase="day", speaker="B", statement="hi"),
        GameEvent(round=1, phase="vote", speaker="MODERATOR", statement="A voted for C"),
        GameEvent(
            round=1,
            phase="vote",
            speaker="MODERATOR",
            statement="C was voted out. Their role was bystander",
        ),
    ]

    llm = TwoTierLLM()
    observer = Observer(setup, llm=llm, deliberate=True)
    for event in events:
        observer.observe(event)

    check("one shallow call per event", llm.shallow == len(events))
    check("exactly one deep call, on the reveal", llm.deep == 1)
    check("the observer counted it", observer.deliberations == 1)
    check("history is still one entry per event", len(observer.history) == len(events))
    check("states are still one per event", len(observer.states) == len(events))
    check(
        "the deep prompt carries the round verbatim",
        "A voted for C" in llm.deep_prompts[0],
    )
    check(
        "the revision is shown to the user, not just logged",
        "wrong about A" in observer.state.reasoning,
    )

    quiet = TwoTierLLM()
    off = Observer(setup, llm=quiet, deliberate=False)
    for event in events:
        off.observe(event)
    check("--no-deliberate spends nothing extra", quiet.deep == 0)


def test_a_failed_deliberation_does_not_lose_the_run() -> None:
    """The deep call is an improvement on an already-valid state. Losing a demo
    to a timeout on an optional call would be a poor trade, so it is swallowed."""
    print("a failed deep call is survivable")

    setup = GameSetup(
        players=["A", "B", "C"], deceiver_role="mafia", deceiver_count=1, deceiver_plural="mafia"
    )

    class FlakyLLM:
        def structured(self, system: str, user: str, schema):
            if schema is Deliberation:
                raise RuntimeError("gateway timeout")
            return ObserverOutput(
                suspicion=[SuspicionEntry(player=p, score=0.33) for p in setup.players],
                claims_tracked=[],
                contradictions_noticed=[],
                reasoning="still here",
            )

    observer = Observer(setup, llm=FlakyLLM(), deliberate=True)
    observer.observe(
        GameEvent(
            round=1,
            phase="vote",
            speaker="MODERATOR",
            statement="C was voted out. Their role was bystander",
        )
    )
    check("the per-event state survived", observer.state.reasoning == "still here")
    check("nothing was counted as deliberated", observer.deliberations == 0)
    check("C still left the distribution", "C" not in observer.state.suspicion)


def test_a_wrong_shaped_deliberation_is_survivable_too() -> None:
    """A backend that answers the deep call with the wrong object must not kill
    the run. This is not hypothetical: a stub that ignored `schema` returned an
    ObserverOutput for a Deliberation and crashed a whole suite on `.revised`.
    """
    print("a wrong-shaped deep response is survivable")

    setup = GameSetup(
        players=["A", "B", "C"], deceiver_role="mafia", deceiver_count=1, deceiver_plural="mafia"
    )

    class SchemaBlindLLM:
        """Returns an ObserverOutput no matter what was asked for."""

        def structured(self, system: str, user: str, schema):
            return ObserverOutput(
                suspicion=[SuspicionEntry(player=p, score=0.33) for p in setup.players],
                claims_tracked=[],
                contradictions_noticed=[],
                reasoning="shallow only",
            )

    observer = Observer(setup, llm=SchemaBlindLLM(), deliberate=True)
    observer.observe(
        GameEvent(
            round=1,
            phase="vote",
            speaker="MODERATOR",
            statement="C was voted out. Their role was bystander",
        )
    )
    check("the run survived", observer.state.reasoning == "shallow only")
    check("the bad response was not counted", observer.deliberations == 0)
    check("the state is still coherent", abs(sum(observer.state.suspicion.values()) - 1.0) < 1e-6)


def test_evidence_reaches_the_prompt_on_a_real_game() -> None:
    """End to end on a real transcript, with a stub model: once a role has been
    announced, the per-event prompt must actually carry the computed block."""
    print("evidence reaches the prompt on real data")

    transcript = Transcript.model_validate_json(
        Path("data/mafia/llmafia-0002.json").read_text(encoding="utf-8")
    )

    class Recorder:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        def structured(self, system: str, user: str, schema):
            if schema is Deliberation:
                return Deliberation(
                    suspicion=[SuspicionEntry(player="Sidney", score=1.0)],
                    revised="",
                    reasoning="deep",
                )
            self.prompts.append(user)
            return ObserverOutput(
                suspicion=[
                    SuspicionEntry(player=p, score=0.25) for p in transcript.setup.players
                ],
                claims_tracked=[],
                contradictions_noticed=[],
                reasoning="shallow",
            )

    rec = Recorder()
    observer = Observer(transcript.setup, llm=rec, deliberate=True)
    for event in transcript.events:
        observer.observe(event)

    before = [p for p in rec.prompts if "no structural evidence" in p]
    after = [p for p in rec.prompts if "P(this player is mafia)" in p]
    check("the opening stretch is told it has nothing", len(before) > 0)
    check("later events carry computed numbers", len(after) > 0)
    check("every prompt has one or the other", len(before) + len(after) == len(rec.prompts))
    check(
        "ground truth never reaches the model",
        not any("ground_truth" in p for p in rec.prompts),
    )
    check("at least one reveal drove a deep call", observer.deliberations > 0)


def test_resume_rebuilds_the_announced_roles() -> None:
    """A resumed run must keep its evidence.

    `revealed` is what grades the votes. A restore that dropped it would compute
    no evidence at all and quietly fall back to reading tone -- and it would look
    completely normal on screen, because the bars would still move. Same class of
    silent regression as the `said` ledger, which is why it gets the same test.
    """
    print("resume keeps the evidence")

    transcript = Transcript.model_validate_json(
        Path("data/mafia/llmafia-0002.json").read_text(encoding="utf-8")
    )

    class Quiet:
        def structured(self, system: str, user: str, schema):
            if schema is Deliberation:
                return Deliberation(
                    suspicion=[SuspicionEntry(player="Sidney", score=1.0)],
                    revised="",
                    reasoning="deep",
                )
            return ObserverOutput(
                suspicion=[
                    SuspicionEntry(player=p, score=0.25) for p in transcript.setup.players
                ],
                claims_tracked=[],
                contradictions_noticed=[],
                reasoning="shallow",
            )

    first = Observer(transcript.setup, llm=Quiet(), deliberate=False)
    for event in transcript.events:
        first.observe(event)
    check("the straight run saw role announcements", len(first.revealed) > 0)
    check("and therefore computed evidence", first.evidence() != {})

    resumed = Observer(transcript.setup, llm=Quiet(), deliberate=False)
    resumed.restore(first.state, first.history, first.states, transcript.events)
    check("the announced roles came back", resumed.revealed == first.revealed)
    check("so the evidence survives the restart", resumed.evidence() == first.evidence())


if __name__ == "__main__":
    for fn in [
        test_evidence_is_silent_until_a_role_is_announced,
        test_voting_for_a_revealed_liar_clears_you,
        test_evidence_sums_to_the_liar_count,
        test_the_pre_evidence_cap_actually_binds,
        test_the_pre_evidence_ceiling_bounds_a_slow_commitment,
        test_deliberation_fires_only_on_a_role_reveal,
        test_a_failed_deliberation_does_not_lose_the_run,
        test_a_wrong_shaped_deliberation_is_survivable_too,
        test_evidence_reaches_the_prompt_on_a_real_game,
        test_resume_rebuilds_the_announced_roles,
    ]:
        fn()
    print("")
    print("all offline evidence checks passed")
