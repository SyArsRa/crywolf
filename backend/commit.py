"""When the observer has seen enough.

The run loop's only stop condition used to be "the transcript ran out", which
makes it a stream processor rather than an agent: nothing it believed could ever
change when it stopped. This is the missing half -- a goal the loop can actually
reach, so a run ends because the observer decided it was done.

Deliberately dependency-free. `scoring` reaches `observer` reaches `llm`, and the
replay path is supposed to stay clear of that chain (no API key, no model calls),
but replay still needs to evaluate the same rule against the same belief stream.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field

# Measured, not guessed: `python -m backend.commit` replays every recording in
# runs/ through this rule and reports where it would have stopped and whether it
# was right. Tuned for precision rather than for firing often, because the first
# setting tried (0.55 / 0.18 / 3) committed in 28 of 40 runs and was right in
# only 12 -- worse than letting the game finish, and on the shipped demo
# recording it committed to the wrong player at event 17 of 27 when the full run
# got it right.
#
# At these thresholds it fires on 2 of 18 distinct runs and is right on both,
# turning nothing right into something wrong. On one of them it is strictly
# better than finishing: run-20260912-125154 commits to P4 (correct) at 17/27,
# while reading to the end drifts to P2 and scores wrong.
#
# Firing on 11% of runs is the honest cost of that precision. A run that never
# reaches the bar simply reads to the end, which is the old behaviour.
MIN_CONFIDENCE = 0.75
MIN_MARGIN = 0.18
HOLD = 5


class Commitment(BaseModel):
    """The observer's decision to stop watching and name its suspects."""

    players: List[str] = Field(..., description="The call: the top `deceiver_count` suspects.")
    confidence: float = Field(..., description="Score of the leading suspect when it committed.")
    margin: float = Field(..., description="Gap between the last named suspect and the next one.")
    event_index: int = Field(..., description="0-based index of the event it committed on.")
    events_observed: int = Field(..., description="How many events it had seen. 1-based, for display.")


def _ranked(snapshot: Dict[str, float]) -> List[str]:
    return sorted(snapshot, key=lambda p: snapshot[p], reverse=True)


def _call(snapshot: Dict[str, float], n: int) -> tuple[List[str], float, float]:
    """The top `n` suspects, the leader's score, and the gap to the next player.

    Scores sum to the number of liars still in play, not to 1.0, so each one is
    already that player's own P(liar): 0.75 means the same thing whether there is
    one werewolf or three mafia. Confidence is read off the leader rather than
    the last name, which matches how `Score.accuracy` grades the call.
    """
    ranked = _ranked(snapshot)
    top = ranked[:n]
    # Nobody left to be wrong about: a call with no rival is a full margin.
    margin = snapshot[top[-1]] - snapshot[ranked[n]] if len(ranked) > n else snapshot[top[-1]]
    return top, snapshot[ranked[0]], margin


def commitment(
    history: List[Dict[str, float]],
    deceiver_count: int = 1,
    min_confidence: float = MIN_CONFIDENCE,
    min_margin: float = MIN_MARGIN,
    hold: int = HOLD,
) -> Optional[Commitment]:
    """Has the observer earned the right to stop, as of the latest snapshot?

    Three conditions, all of which must hold across the last `hold` events:

    * the same set of suspects is on top -- a call that changes every event is
      not a call,
    * the leader is at `min_confidence` or better,
    * and the last named suspect leads the first unnamed one by `min_margin`.

    The hold is what separates a conclusion from a spike. A single line can push
    one player to 70%; three consecutive events cannot, unless the observer
    actually believes it.
    """
    if hold < 1 or len(history) < hold:
        return None

    window = history[-hold:]
    if any(not snapshot for snapshot in window):
        return None

    n = max(1, deceiver_count)
    called: Optional[set] = None
    for snapshot in window:
        if len(snapshot) < n + 2:
            # Naming n of n+1 survivors is not a read, it is arithmetic -- and
            # confidence is nearly free once the field is that small, because the
            # scores sum to the number of liars still in play. Committing here
            # would buy a cheap "called it" at the moment the game was about to
            # answer the question itself.
            return None
        top, confidence, margin = _call(snapshot, n)
        if confidence < min_confidence or margin < min_margin:
            return None
        if called is None:
            called = set(top)
        elif set(top) != called:
            return None

    top, confidence, margin = _call(window[-1], n)
    return Commitment(
        players=top,
        confidence=round(confidence, 4),
        margin=round(margin, 4),
        event_index=len(history) - 1,
        events_observed=len(history),
    )


def _report() -> None:
    """Replay every recording in runs/ through the rule and print the result.

    How the thresholds above were chosen: the useful question is not "does it
    fire" but "does it fire early, and is it right when it does".
    """
    import glob
    import json

    rows = []
    for path in sorted(glob.glob("runs/*.json")) + sorted(glob.glob("data/demo_run.json")):
        blob = json.loads(open(path).read())
        history = blob.get("history") or []
        setup = blob.get("setup") or {}
        truth = blob.get("ground_truth") or {}
        liars = {p for p, r in truth.items() if r.lower() in {"werewolf", "wolf", "mafia", "scum"}}
        if not history or not liars:
            continue
        n = setup.get("deceiver_count", 1)
        total = blob.get("events_expected") or len(history)

        call = None
        for i in range(1, len(history) + 1):
            call = commitment(history[:i], deceiver_count=n)
            if call:
                break
        if call:
            hit = call.players[0] in liars
            rows.append((path, f"{call.events_observed}/{total}", f"{call.confidence:.0%}",
                         ",".join(call.players), "HIT" if hit else "miss"))
        else:
            rows.append((path, f"-/{total}", "-", "never committed", "-"))

    for path, when, conf, who, hit in rows:
        print(f"{path:46} {when:>9} {conf:>5}  {who:22} {hit}")
    fired = [r for r in rows if r[4] != "-"]
    hits = [r for r in fired if r[4] == "HIT"]
    print(f"\ncommitted in {len(fired)}/{len(rows)} runs; correct in {len(hits)}/{len(fired)}")


if __name__ == "__main__":
    _report()
