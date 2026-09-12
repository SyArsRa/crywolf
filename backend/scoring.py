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

import re
from typing import Dict, List, Optional

from backend.commit import Commitment
from backend.observer import infer_elimination
from backend.schema import DECEIVER_ROLES, BeliefState, GameEvent, Score

# A fall smaller than this is ordinary drift as probability mass shifts around.
REVERSAL_THRESHOLD = 0.15


def mentions(player: str, statement: str) -> bool:
    """Does this line actually name that player?

    A plain substring test is wrong and was quietly inflating consistency: with
    real names, "Sage" hides inside "messages", "Lee" inside "feeling", "Ari"
    inside "curious". Thirty-three such collisions across the 33 Mafia games,
    each one excusing a suspicion drop as "explained" when nobody had said a word
    about that player.

    Letters either side are what disqualify a match, rather than \\b, so
    possessives and chat punctuation still count as mentions: "rowans reasoning",
    "#vote_for_winter", "@kai".
    """
    return re.search(
        rf"(?<![a-z]){re.escape(player.lower())}(?:'s|’s|s)?(?![a-z])",
        statement.lower(),
    ) is not None


def count_unexplained_reversals(
    history: List[Dict[str, float]],
    events: List[GameEvent],
    threshold: float = REVERSAL_THRESHOLD,
) -> int:
    """`history[i]` is the belief state produced by `events[i]`."""
    reversals = 0
    for i in range(1, min(len(history), len(events))):
        before, after, event = history[i - 1], history[i], events[i]
        for player, prev in before.items():
            drop = prev - after.get(player, prev)
            if drop <= threshold:
                continue
            if player == event.speaker or mentions(player, event.statement):
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
    snapshot = _verdict_snapshot(history, events)
    return max(snapshot, key=lambda p: snapshot[p]) if snapshot else None


def _verdict_snapshot(
    history: List[Dict[str, float]], events: List[GameEvent]
) -> Dict[str, float]:
    """The last belief state produced by an event that eliminated nobody."""
    for i in range(min(len(history), len(events)) - 1, -1, -1):
        if infer_elimination(events[i]) is None and history[i]:
            return history[i]
    return {}


def final_verdict(history: List[Dict[str, float]]) -> str | None:
    """The observer's top suspect in its very last belief state.

    Reported alongside `verdict()` because the two games end differently and
    neither reading is right for both.

    A Werewolf game that ends by voting the wolf out leaves the observer
    answering about survivors only -- the player it spent the game accusing has
    just left the room -- so `verdict()` looks back to the last event that
    eliminated nobody. But a Mafia game where the mafia win ends with them still
    at the table, and the closing eliminations announce roles the observer
    legitimately uses: on game 0002 it moved to Ashton (mafia) at 53% on the
    final line, and `verdict()` threw that away for a snapshot two events older.

    Rather than pick whichever rule flatters the run, both are reported.
    """
    return max(history[-1], key=lambda p: history[-1][p]) if history and history[-1] else None


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
    """How sure the observer was, read from the same snapshot as the verdict.

    Reads `_verdict_snapshot` rather than re-walking the history. It used to have
    its own copy of that loop with one extra condition, which is the kind of
    duplication that agrees on every run you test and disagrees on the one you
    demo.
    """
    if not predicted:
        return 0.0
    return _verdict_snapshot(history, events).get(predicted, 0.0)


def precision_at_n(
    history: List[Dict[str, float]],
    events: List[GameEvent],
    actual: List[str],
) -> float:
    """Of the N players the observer suspects most, how many really are liars?

    With one werewolf, "did it name them" says everything. With three mafia among
    nine players it says very little -- pointing at one of three is a far easier
    shot than finding all three, and reporting it as a clean hit would flatter
    the observer. So the headline stays `accuracy` (is the top suspect a liar)
    and this is the honest companion: 1.0 means every one of its top N is guilty.

    Scored on the same snapshot as the verdict, over players still alive there --
    asking about the dead would be scoring information the game already gave away.
    """
    snapshot = _verdict_snapshot(history, events)
    if not snapshot or not actual:
        return 0.0
    still_in = [p for p in actual if p in snapshot]
    if not still_in:
        return 0.0
    ranked = sorted(snapshot, key=lambda p: snapshot[p], reverse=True)
    top = ranked[: len(still_in)]
    return sum(1 for p in top if p in still_in) / len(still_in)


def grade(
    final: BeliefState,
    history: List[Dict[str, float]],
    events: List[GameEvent],
    ground_truth: Dict[str, str],
    committed: Optional[Commitment] = None,
    events_available: int = 0,
) -> Score:
    actual = [p for p, role in ground_truth.items() if role.lower() in DECEIVER_ROLES]
    if not actual:
        raise ValueError("ground_truth names nobody on the lying team")
    # A run that stopped itself is graded on what it stopped to say. Falling back
    # to `verdict()` here would quietly re-derive an answer from the history and
    # could grade the observer on something it never actually claimed.
    if committed:
        predicted = committed.players[0]
    else:
        predicted = verdict(history, events) or final.top_suspect

    reversals = count_unexplained_reversals(history, events)
    lead_changes, events_off = measure_drift(history, predicted)
    # One reversal per event would be a total loss of the plot; scale against that.
    consistency = 1.0 - (reversals / len(history)) if history else 1.0

    # How often the observer has had a liar on top, over the whole run so far.
    # This is the number a live scoreboard wants: one verdict is one bit, but
    # every event is a graded prediction.
    leaders = [max(s, key=lambda p: s[p]) for s in history if s]
    running = sum(1 for leader in leaders if leader in actual) / len(leaders) if leaders else 0.0

    at_end = final_verdict(history)
    return Score(
        running_accuracy=round(running, 3),
        accuracy=predicted in actual,
        final_verdict=at_end,
        final_accuracy=at_end in actual if at_end else False,
        precision_at_n=round(precision_at_n(history, events, actual), 3),
        predicted=predicted,
        actual=actual,
        final_confidence=_confidence(history, events, predicted),
        consistency=max(0.0, round(consistency, 3)),
        self_contradictions=reversals,
        lead_changes=lead_changes,
        events_off_verdict=events_off,
        contradictions_caught=len(final.contradictions_noticed),
        rounds_observed=final.round,
        committed_at=committed.events_observed if committed else None,
        events_available=events_available or len(events),
        suspicion_history=history,
    )
