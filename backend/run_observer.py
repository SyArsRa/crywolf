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
import sys
from pathlib import Path

from backend.observer import Observer
from backend.schema import Transcript
from backend.scoring import grade

BAR_WIDTH = 28


def _bars(suspicion: dict, wolf: str | None) -> str:
    lines = []
    for player, score in sorted(suspicion.items(), key=lambda kv: -kv[1]):
        filled = round(score * BAR_WIDTH)
        mark = " <" if player == wolf else ""
        lines.append(
            f"    {player}  {'#' * filled}{'.' * (BAR_WIDTH - filled)}  {score * 100:5.1f}%{mark}"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay a Werewolf transcript through the observer.")
    parser.add_argument("transcript", nargs="?", default="data/fallback_transcript.json")
    parser.add_argument("--json", dest="out", help="Write the full run to this path.")
    parser.add_argument(
        "--spoil",
        action="store_true",
        help="Mark the real werewolf in the printed bars. Never shown to the model.",
    )
    args = parser.parse_args()

    transcript = Transcript.model_validate_json(Path(args.transcript).read_text(encoding="utf-8"))
    wolf = transcript.werewolf()
    observer = Observer(transcript.setup)

    print(f"Replaying {args.transcript} -- {len(transcript.events)} events, "
          f"players {', '.join(transcript.setup.players)}\n")

    for i, event in enumerate(transcript.events, 1):
        state = observer.observe(event)
        print(f"[{i:>2}/{len(transcript.events)}] R{event.round} {event.phase} "
              f"{event.speaker}: {event.statement}")
        print(_bars(state.suspicion, wolf if args.spoil else None))
        print(f"    -> {state.reasoning}")
        if state.contradictions_noticed:
            latest = state.contradictions_noticed[-1]
            print(f"    !! {latest.player} (R{latest.round_noticed}): "
                  f"\"{latest.earlier}\" vs \"{latest.now}\"")
        print()

    score = grade(observer.state, observer.history, transcript.events, transcript.ground_truth)

    print("=" * 60)
    print(f"predicted werewolf : {score.predicted}  ({score.final_confidence * 100:.1f}% confident)")
    print(f"actual werewolf    : {score.actual}")
    print(f"accuracy           : {'HIT' if score.accuracy else 'MISS'}")
    print(f"consistency        : {score.consistency:.2f} "
          f"({score.self_contradictions} unexplained reversals)")
    print(f"contradictions caught among players: {score.contradictions_caught}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {
                    "events": [e.model_dump() for e in transcript.events],
                    "history": observer.history,
                    "final_state": observer.state.model_dump(),
                    "score": score.model_dump(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nwrote {out_path}")

    return 0 if score.accuracy else 1


if __name__ == "__main__":
    sys.exit(main())
