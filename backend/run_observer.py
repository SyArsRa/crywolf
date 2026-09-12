"""Replay a transcript straight through the observer, no server involved.

This is Lane B's test harness and the demo's safety net: if the API, the feeder,
or the websocket misbehaves on stage, this still produces the whole run.

    python -m backend.run_observer
    python -m backend.run_observer data/fallback_transcript.json --json out/run1.json

`--json` writes the full run (every belief state plus the final score) so a run
can be replayed into the UI later without spending another call.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from backend.observer import Observer
from backend.schema import BeliefState, Transcript
from backend.scoring import grade

BAR_WIDTH = 28


def _bars(state, roster: list[str], liars: set[str]) -> str:
    """One row per player, always in roster order.

    Sorting by score made rows swap places between events, so you couldn't
    follow a single player down the screen -- which is exactly what you want to
    watch. Fixed rows turn the output into a chart you can read over time.
    """
    # Real datasets have names like "Whitney" and "Kai", not P1..P5, so pad to
    # the widest or the bars stop lining up and the chart becomes unreadable.
    width = max((len(p) for p in roster), default=4)
    lines = []
    for player in roster:
        mark = " <" if player in liars else ""
        name = player.ljust(width)
        if player in state.eliminated:
            lines.append(f"    {name}  {'-' * BAR_WIDTH}   out{mark}")
            continue
        score = state.suspicion.get(player, 0.0)
        filled = round(score * BAR_WIDTH)
        lines.append(
            f"    {name}  {'#' * filled}{'.' * (BAR_WIDTH - filled)}  {score * 100:5.1f}%{mark}"
        )
    return "\n".join(lines)


def _write_run(path, transcript, observer, score, completed: int) -> None:
    """Dump a run to disk in the same shape `state.Run.to_record` writes.

    `score` is None for a partial run that crashed or was interrupted.

    `turns` is the key that matters: pairing each event with the *whole* state it
    produced -- reasoning and contradictions included, not just the suspicion
    numbers -- is what lets `feeder.py --replay` drive the UI from this file with
    no model calls. Without it a replay shows moving bars and blank reasoning,
    which is not a demo.
    """
    events = transcript.events[:completed]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "complete": score is not None,
                "live": True,
                "events_observed": completed,
                "events_expected": len(transcript.events),
                "error": None,
                "setup": transcript.setup.model_dump(),
                "ground_truth": transcript.ground_truth,
                "events": [e.model_dump() for e in events],
                "history": observer.history,
                "turns": [
                    {"index": i, "event": event.model_dump(), "state": state.model_dump()}
                    for i, (event, state) in enumerate(zip(events, observer.states))
                ],
                "final_state": observer.state.model_dump(),
                "score": score.model_dump() if score else None,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replay a Werewolf or Mafia transcript through the observer."
    )
    parser.add_argument("transcript", nargs="?", default="data/fallback_transcript.json")
    parser.add_argument("--json", dest="out", help="Write the full run to this path.")
    parser.add_argument(
        "--spoil",
        action="store_true",
        help="Mark the real liars in the printed bars. Never shown to the model.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue a partial run from --json instead of restarting. "
             "Free-tier quota runs out mid-game; this picks up where it stopped.",
    )
    parser.add_argument(
        "--no-deliberate",
        action="store_true",
        help="Skip the per-round deliberation call fired on each role reveal. "
             "Only for pricing a run against the per-event loop alone.",
    )
    args = parser.parse_args()

    transcript = Transcript.model_validate_json(Path(args.transcript).read_text(encoding="utf-8"))
    liars = set(transcript.deceivers())
    observer = Observer(transcript.setup, deliberate=not args.no_deliberate)
    start_at = 0

    if args.resume:
        if not args.out:
            print("--resume needs --json to say which run to continue.")
            return 2
        saved = Path(args.out)
        if not saved.exists():
            print(f"No run at {saved} to resume; starting fresh.")
        else:
            data = json.loads(saved.read_text(encoding="utf-8"))
            if data.get("complete"):
                print(f"{saved} is already a complete run. Delete it to start over.")
                return 0
            start_at = data["events_observed"]
            observer.restore(
                BeliefState.model_validate(data["final_state"]),
                data["history"],
                [BeliefState.model_validate(t["state"]) for t in data.get("turns", [])],
                transcript.events[:start_at],
            )
            print(f"Resuming from {saved} at event {start_at + 1} "
                  f"-- {start_at} events already observed, not re-spent.\n")

    total = len(transcript.events)
    pace = float(os.environ.get("CRYWOLF_MIN_INTERVAL", 0))
    print(f"Replaying {args.transcript} -- {total} events, "
          f"players {', '.join(transcript.setup.players)}")
    print(f"Model: {os.environ.get('CRYWOLF_MODEL', 'claude-haiku-4-5')}")
    if pace:
        print(f"Pacing at {pace:.0f}s/event -- about {total * pace / 60:.0f} minutes.")
    print()

    completed = start_at
    try:
        for i, event in enumerate(transcript.events[start_at:], start_at + 1):
            state = observer.observe(event)
            completed = i
            # Save after every event, not just at the end. Quota can stop us at
            # any point and each event costs a call we can't get back.
            if args.out:
                _write_run(Path(args.out), transcript, observer, None, completed)
            print(f"[{i:>2}/{total}] R{event.round} {event.phase} "
                  f"{event.speaker}: {event.statement}")
            remaining = observer.liars_remaining
            label = transcript.setup.deceiver_role
            print(f"    (each bar is P({label}); {remaining} still in the game)")
            print(_bars(state, transcript.setup.players, liars if args.spoil else set()))
            print(f"    -> {state.reasoning}")
            if state.contradictions_noticed:
                latest = state.contradictions_noticed[-1]
                print(f"    !! {latest.player} (R{latest.round_noticed}): "
                      f"\"{latest.earlier}\" vs \"{latest.now}\"")
            print()
    except (Exception, KeyboardInterrupt) as exc:
        print(f"\nStopped after {completed}/{total} events: "
              f"{type(exc).__name__}: {str(exc)[:200]}")
        if args.out and completed:
            # Already written after the last successful event; nothing is lost.
            print(f"Progress is in {args.out}. Continue with the same command plus --resume.")
        return 2

    score = grade(observer.state, observer.history, transcript.events, transcript.ground_truth)

    print("=" * 60)
    role = transcript.setup.deceiver_role
    print(f"top suspect        : {score.predicted}  ({score.final_confidence * 100:.1f}% confident)")
    print(f"actually {role:<10}: {', '.join(score.actual)}")
    print(f"accuracy           : {'HIT' if score.accuracy else 'MISS'}"
          f"   (is the top suspect one of them?)")
    if len(score.actual) > 1:
        print(f"precision@{len(score.actual)}        : {score.precision_at_n:.2f}"
              f"   (of its top {len(score.actual)} living suspects, how many really are)")
    print(f"consistency        : {score.consistency:.2f} "
          f"({score.self_contradictions} unexplained reversals)")
    print(f"drift              : {score.lead_changes} lead changes, "
          f"{score.events_off_verdict} events off the final verdict")
    print(f"contradictions caught among players: {score.contradictions_caught}")
    print(f"deep calls (one per role reveal)   : {observer.deliberations}")

    if args.out:
        _write_run(Path(args.out), transcript, observer, score, len(transcript.events))
        print(f"\nwrote {args.out}")

    return 0 if score.accuracy else 1


if __name__ == "__main__":
    sys.exit(main())
