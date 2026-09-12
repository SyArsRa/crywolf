"""Grading the observer against ground truth.

Accuracy is the easy half. Consistency is the interesting half, and it needs a
definition that is actually computable rather than vibes.

A *self-contradiction* here is an *unexplained reversal*: the observer's
suspicion of some player falls sharply during an event in which that player was
neither the speaker nor mentioned by name. Nothing new was said about them, so
nothing justified the drop -- the observer simply lost the thread. Drops that
follow a line about that player are fair updating, not forgetting, and are not
counted.

Note this measures the observer contradicting *itself*. The count of
contradictions it caught among the *players* is reported separately -- that one
is a feature, not a fault.
"""

from __future__ import annotations

from typing import Dict, List

from backend.observer import infer_elimination
from backend.schema import BeliefState, GameEvent, Score

# A fall smaller than this is ordinary drift as probability mass shifts around.
REVERSAL_THRESHOLD = 0.15


def count_unexplained_reversals(
    history: List[Dict[str, float]],
    events: List[GameEvent],
    threshold: float = REVERSAL_THRESHOLD,
) -> int:
    """`history[i]` is the belief state produced by `events[i]`."""
    reversals = 0
    for i in range(1, min(len(history), len(events))):
        before, after, event = history[i - 1], history[i], events[i]
        statement = event.statement.lower()
        for player, prev in before.items():
            drop = prev - after.get(player, prev)
            if drop <= threshold:
                continue
            if player == event.speaker or player.lower() in statement:
                continue  # something was actually said about them
            reversals += 1
    return reversals


def verdict(history: List[Dict[str, float]], events: List[GameEvent]) -> str | None:
    """The observer's final call, taken before the game answers the question itself.

    Games end by eliminating someone, and an eliminated player leaves the
    distribution. Reading the top suspect from the very last state would
    therefore ask "who do you suspect?" after the suspect has already been
    removed -- an observer that named the werewolf correctly would score as
    wrong, because its answer was voted out a line earlier.

    So the verdict is the last snapshot produced by an event that killed nobody:
    the observer's last opinion formed from argument rather than from the
    village resolving it.
    """
    for i in range(min(len(history), len(events)) - 1, -1, -1):
        if infer_elimination(events[i]) is None and history[i]:
            return max(history[i], key=lambda p: history[i][p])
    return None


def measure_drift(history: List[Dict[str, float]], final_verdict: str | None) -> tuple[int, int]:
    """How much the observer's accusation wandered.

    `count_unexplained_reversals` catches single-event jerks and nothing else,
    which is how a run where the observer pointed at the wrong player for seven
    straight events still scored a perfect 1.00: each step down was small enough,
    or landed on an event that named the player, so none of it registered.

    Returns (lead_changes, events_off_verdict):

    * lead_changes -- how many times the top suspect changed hands at all.
    * events_off_verdict -- once the eventual verdict first took the lead, how
      many later events had someone else on top. This is the wobble: the stretch
      where the observer had the right answer, let go of it, and came back.

    Neither is a fault on its own. An observer that never moved would score zero
    on both and be useless; changing your mind on evidence is the job. They are
    here so the wobble is visible rather than hidden behind a perfect score.
    """
    leaders = [max(s, key=lambda p: s[p]) for s in history if s]
    if not leaders:
        return 0, 0

    lead_changes = sum(1 for a, b in zip(leaders, leaders[1:]) if a != b)

    if final_verdict is None or final_verdict not in leaders:
        return lead_changes, 0
    first_lead = leaders.index(final_verdict)
    events_off = sum(1 for leader in leaders[first_lead:] if leader != final_verdict)
    return lead_changes, events_off


def _confidence(
    history: List[Dict[str, float]], events: List[GameEvent], predicted: str | None
) -> float:
    """How sure the observer was, read from the same snapshot as the verdict."""
    if not predicted:
        return 0.0
    for i in range(min(len(history), len(events)) - 1, -1, -1):
        if infer_elimination(events[i]) is None and predicted in history[i]:
            return history[i][predicted]
    return 0.0


def grade(
    final: BeliefState,
    history: List[Dict[str, float]],
    events: List[GameEvent],
    ground_truth: Dict[str, str],
) -> Score:
    actual = next(p for p, role in ground_truth.items() if role.lower() == "werewolf")
    predicted = verdict(history, events) or final.top_suspect

    reversals = count_unexplained_reversals(history, events)
    lead_changes, events_off = measure_drift(history, predicted)
    # One reversal per event would be a total loss of the plot; scale against that.
    consistency = 1.0 - (reversals / len(history)) if history else 1.0

    return Score(
        accuracy=predicted == actual,
        predicted=predicted,
        actual=actual,
        final_confidence=_confidence(history, events, predicted),
        consistency=max(0.0, round(consistency, 3)),
        self_contradictions=reversals,
        lead_changes=lead_changes,
        events_off_verdict=events_off,
        contradictions_caught=len(final.contradictions_noticed),
        rounds_observed=final.round,
        suspicion_history=history,
    )
