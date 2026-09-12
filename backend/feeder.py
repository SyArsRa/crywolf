"""Replays a game into the API, one line at a time, as if it were happening now.

Two modes:

    # live -- real observer calls, writes a recording
    python -m backend.feeder --transcript data/fallback_transcript.json --interval 3.0

    # replay -- a recording, no model calls, no API key
    python -m backend.feeder --replay runs/run-20260912-143000.json --interval 2.5

The feeder waits for each POST to come back before pacing the next one, so a
turn costs max(observer latency, interval). That means lines and belief updates
can never arrive out of order no matter how slow the model is -- an uneven pause
reads as thinking, whereas an update landing against the wrong line reads as
broken.

It prints per-turn and total timings, because tuning the run into the 90-second
demo slot is a measurement problem, not a guess.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import httpx

from backend.schema import BeliefState, GameEvent, GameSetup, Transcript
from backend.state import Turn

DEFAULT_API = "http://127.0.0.1:8000"
DEMO_BUDGET_SECONDS = 90.0

# The observer can take a while on a hard line; never time out before it does.
TIMEOUT = httpx.Timeout(180.0, connect=10.0)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_transcript(path: Path) -> Transcript:
    return Transcript.model_validate_json(path.read_text(encoding="utf-8"))


def load_recording(path: Path, fallback: Optional[Path]) -> Tuple[Transcript, List[Turn], int]:
    """Rebuild a transcript and its turns from a recorded run.

    Handles two shapes: the one `state.Run.to_record` writes, and the older
    `run_observer._write_run` one, which has no `setup` and no `turns` -- only
    per-event suspicion numbers. For the older shape the setup and ground truth
    come from the transcript file, and each turn is reconstructed with just its
    suspicion (the reasoning was never saved, so the UI shows blank reasoning on
    those replays).

    Also returns how many events the *original* game had, which is not
    len(events) when the recording is partial.
    """
    record = json.loads(path.read_text(encoding="utf-8"))
    events = [GameEvent.model_validate(e) for e in record.get("events", [])]
    expected = record.get("events_expected")

    if record.get("setup"):
        setup = GameSetup.model_validate(record["setup"])
        ground_truth = record.get("ground_truth") or {}
    else:
        if fallback is None or not fallback.exists():
            raise SystemExit(
                f"{path} has no 'setup' (it's an older run_observer file). "
                f"Pass --transcript pointing at the transcript it came from."
            )
        source = load_transcript(fallback)
        setup, ground_truth = source.setup, source.ground_truth
        # Older recordings don't store the original total; the transcript does.
        expected = expected or len(source.events)

    transcript = Transcript(setup=setup, events=events, ground_truth=ground_truth)

    if record.get("turns"):
        turns = [Turn.model_validate(t) for t in record["turns"]]
    else:
        history = record.get("history", [])
        turns = [
            Turn(
                index=i,
                event=event,
                state=BeliefState(
                    round=event.round,
                    phase=event.phase,
                    event_index=i,
                    suspicion=suspicion,
                ),
            )
            for i, (event, suspicion) in enumerate(zip(events, history))
        ]
        if turns:
            print(
                f"note: {path.name} predates full turn recording -- "
                f"replaying suspicion only, reasoning will be blank."
            )

    if not turns:
        raise SystemExit(f"{path} contains no turns to replay.")
    return transcript, turns, expected or len(turns)


# ---------------------------------------------------------------------------
# Feeding
# ---------------------------------------------------------------------------


def _pace(started: float, interval: float) -> float:
    """Sleep out the remainder of this turn's interval. Returns time slept."""
    remaining = interval - (time.perf_counter() - started)
    if remaining > 0:
        time.sleep(remaining)
        return remaining
    return 0.0


def _report(durations: List[float], total: float, expected: int) -> None:
    done = len(durations)
    print()
    print("=" * 62)
    print(f"fed {done}/{expected} events in {total:.1f}s")
    if durations:
        print(
            f"per turn: mean {sum(durations) / done:.2f}s  "
            f"min {min(durations):.2f}s  max {max(durations):.2f}s"
        )
    if total > DEMO_BUDGET_SECONDS:
        over = total - DEMO_BUDGET_SECONDS
        per_turn_needed = DEMO_BUDGET_SECONDS / expected if expected else 0
        print(
            f"OVER the {DEMO_BUDGET_SECONDS:.0f}s demo budget by {over:.1f}s -- "
            f"needs {per_turn_needed:.2f}s/turn. Lower --interval, or cut events."
        )
    else:
        print(f"inside the {DEMO_BUDGET_SECONDS:.0f}s demo budget, {DEMO_BUDGET_SECONDS - total:.1f}s to spare.")


def feed_live(client: httpx.Client, transcript: Transcript, interval: float) -> int:
    start = client.post("/game/start", json={"transcript": transcript.model_dump(), "live": True})
    if start.status_code != 200:
        print(f"couldn't start a live run: {start.status_code} {start.text[:400]}")
        print("\nIf that's a missing key, either export GEMINI_API_KEY, or replay a")
        print("recorded run instead:  python -m backend.feeder --replay <run file>")
        return 2
    total = len(transcript.events)
    print(f"live run: {total} events, players {', '.join(transcript.setup.players)}\n")

    durations: List[float] = []
    run_started = time.perf_counter()
    failed = False

    for i, event in enumerate(transcript.events, 1):
        turn_started = time.perf_counter()
        response = client.post("/event", json=event.model_dump())
        call = time.perf_counter() - turn_started

        if response.status_code != 200:
            print(f"\n[{i:>2}/{total}] FAILED after {call:.1f}s: {response.status_code} {response.text[:300]}")
            failed = True
            break

        state = response.json()
        top = max(state["suspicion"], key=state["suspicion"].get) if state["suspicion"] else "?"
        print(
            f"[{i:>2}/{total}] {call:5.2f}s  {event.speaker}: {event.statement[:48]}"
            f"\n           -> top suspect {top} "
            f"({state['suspicion'].get(top, 0) * 100:.0f}%)  {state['reasoning'][:70]}"
        )

        slept = _pace(turn_started, interval)
        durations.append(call + slept)

    elapsed = time.perf_counter() - run_started
    end = client.post("/game/end")
    _report(durations, elapsed, total)
    if end.status_code == 200:
        print(f"recording: {end.json().get('recording')}")
    return 1 if failed else 0


def feed_replay(
    client: httpx.Client,
    transcript: Transcript,
    turns: List[Turn],
    interval: float,
    expected: int,
) -> int:
    client.post(
        "/game/start",
        json={"transcript": transcript.model_dump(), "live": False, "events_expected": expected},
    ).raise_for_status()
    total = len(turns)
    partial = "" if total == expected else f" (a partial recording of a {expected}-event game)"
    print(f"replay: {total} recorded turns, no model calls{partial}\n")

    run_started = time.perf_counter()
    durations: List[float] = []

    for i, turn in enumerate(turns, 1):
        turn_started = time.perf_counter()
        client.post("/turn", json=turn.model_dump(mode="json")).raise_for_status()
        event = turn.event
        print(f"[{i:>2}/{total}] {event.speaker}: {event.statement[:60]}")
        _pace(turn_started, interval)
        durations.append(time.perf_counter() - turn_started)

    elapsed = time.perf_counter() - run_started
    client.post("/game/end")
    _report(durations, elapsed, total)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay a Werewolf game into the Cry Wolf API.")
    parser.add_argument(
        "--transcript",
        type=Path,
        default=Path("data/fallback_transcript.json"),
        help="Transcript to feed live. Also supplies setup/ground truth when replaying an older recording.",
    )
    parser.add_argument("--replay", type=Path, help="Replay this recorded run instead of making model calls.")
    parser.add_argument("--interval", type=float, default=3.0, help="Seconds per turn (a floor, not a cap).")
    parser.add_argument("--api", default=DEFAULT_API, help=f"API base URL (default {DEFAULT_API}).")
    args = parser.parse_args()

    with httpx.Client(base_url=args.api, timeout=TIMEOUT) as client:
        try:
            client.get("/health").raise_for_status()
        except httpx.HTTPError as exc:
            print(f"can't reach the API at {args.api}: {exc}")
            print("start it with:  uvicorn backend.main:app --reload")
            return 2

        try:
            if args.replay:
                transcript, turns, expected = load_recording(args.replay, args.transcript)
                return feed_replay(client, transcript, turns, args.interval, expected)
            return feed_live(client, load_transcript(args.transcript), args.interval)
        except KeyboardInterrupt:
            print("\ninterrupted -- closing the run so the recording is kept.")
            client.post("/game/end")
            return 130


if __name__ == "__main__":
    sys.exit(main())
