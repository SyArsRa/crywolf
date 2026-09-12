"""Offline tests for everything around the model call.

The model call itself needs credentials; the logic that guards it does not, and
that logic is where the bugs actually live. Run with: python -m backend.test_observer
"""

from __future__ import annotations

from pathlib import Path

from backend.observer import Observer, _merge_contradictions, _settle
from backend.schema import (
    BeliefState,
    ClaimEntry,
    Contradiction,
    GameEvent,
    ObserverOutput,
    SuspicionEntry,
    Transcript,
)
from backend.scoring import count_unexplained_reversals, grade

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


def test_belief_state_defaults() -> None:
    print("BeliefState")
    check("empty state has no top suspect", BeliefState().top_suspect is None)


if __name__ == "__main__":
    for fn in [
        test_settle_normalizes,
        test_contradictions_are_append_only,
        test_reversal_counting,
        test_full_loop_with_stub_model,
        test_belief_state_defaults,
    ]:
        fn()
    print("\nall offline checks passed")
