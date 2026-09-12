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


def grade(
    final: BeliefState,
    history: List[Dict[str, float]],
    events: List[GameEvent],
    ground_truth: Dict[str, str],
) -> Score:
    actual = next(p for p, role in ground_truth.items() if role.lower() == "werewolf")
    predicted = final.top_suspect

    reversals = count_unexplained_reversals(history, events)
    # One reversal per event would be a total loss of the plot; scale against that.
    consistency = 1.0 - (reversals / len(history)) if history else 1.0

    return Score(
        accuracy=predicted == actual,
        predicted=predicted,
        actual=actual,
        final_confidence=final.suspicion.get(predicted, 0.0) if predicted else 0.0,
        consistency=max(0.0, round(consistency, 3)),
        self_contradictions=reversals,
        contradictions_caught=len(final.contradictions_noticed),
        rounds_observed=final.round,
        suspicion_history=history,
    )
