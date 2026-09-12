"""Cry Wolf ingestion API.

    POST /game/start   open a run (feeder -> api)
    POST /event        push one line of the game, get the new belief state back
    POST /turn         push a pre-computed turn (replay only, no model calls)
    POST /game/end     close the run
    GET  /score        accuracy + consistency against ground truth
    WS   /live         snapshot on connect, then one message per turn
    GET  /health       liveness + what run is open
    GET  /            the frontend (mounted last, so it never shadows a route)

The seam that matters is POST /event: the feeder on one side is entirely
replaceable -- a real game feed, speech-to-text, a second transcript source --
and nothing downstream of this endpoint would know the difference.

The scoring seam is `scoring.grade`, which reads only the belief states -- so
`GET /score` works for a replayed run exactly as it does for a live one.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.feeder import load_recording, load_transcript
from backend.hub import Hub
from backend.schema import GameEvent, Transcript
from backend.scoring import grade
from backend.state import Run, Turn

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("crywolf.api")

app = FastAPI(title="Cry Wolf", description="Observer ingestion API")
hub = Hub()

# One game at a time. A second /game/start replaces the first.
RUN: Optional[Run] = None

# The observer folds each event into a belief state it carries forward, so two
# events being processed at once would interleave inside it and corrupt the
# state -- and because each call takes seconds, the window is wide open. The
# feeder is strictly sequential so this cannot happen today; the lock is here so
# it stays impossible when a second client, a retry, or an impatient hand on
# curl shows up. Ordering matters as much as safety: turn N must reach the
# websocket before turn N+1.
INGEST = asyncio.Lock()

# Seconds per event while the game is still evidence-free. See `_warmup`.
WARMUP_INTERVAL = float(os.environ.get("CRYWOLF_WARMUP_INTERVAL", 0.35))


class StartRequest(BaseModel):
    transcript: Transcript
    live: bool = True
    # Only set when replaying a partial recording, where the transcript holds
    # fewer events than the original game had.
    events_expected: Optional[int] = None


def _require_run() -> Run:
    if RUN is None:
        raise HTTPException(status_code=409, detail="no run open -- POST /game/start first")
    return RUN


def _snapshot(run: Run) -> dict:
    return {
        "type": "snapshot",
        "setup": run.setup.model_dump(),
        "live": run.live,
        "events_expected": run.total_expected,
        "complete": run.complete,
        "committed": run.committed.model_dump(mode="json") if run.committed else None,
        "error": run.error,
        "turns": [t.model_dump(mode="json") for t in run.turns],
    }


@app.get("/health")
async def health() -> dict:
    return {
        "ok": True,
        "run_open": RUN is not None,
        "live": RUN.live if RUN else None,
        "turns": len(RUN.turns) if RUN else 0,
        "expected": RUN.total_expected if RUN else 0,
        "complete": RUN.complete if RUN else False,
        "clients": len(hub),
        "playing": _playing(),
    }


@app.post("/game/start")
async def game_start(req: StartRequest) -> dict:
    """Open a run. Constructing the Observer happens here, once per game, so its
    belief state carries across every subsequent /event."""
    global RUN
    try:
        RUN = Run(req.transcript, live=req.live, events_expected=req.events_expected)
    except Exception as exc:
        # Almost always a missing API key. Say so plainly rather than dumping a
        # 500 on whoever just cloned the repo.
        raise HTTPException(status_code=503, detail=f"{type(exc).__name__}: {exc}")
    log.info(
        "run started: %s, %d players, %d events expected -> %s",
        "live" if req.live else "replay",
        len(RUN.players),
        RUN.total_expected,
        RUN.path,
    )
    await hub.broadcast(_snapshot(RUN))
    return {
        "ok": True,
        "live": RUN.live,
        "players": RUN.players,
        "events_expected": RUN.total_expected,
        "recording": str(RUN.path),
    }


@app.post("/event")
async def ingest_event(event: GameEvent) -> dict:
    """The front door. One line of the game in, one belief state out."""
    run = _require_run()
    if run.observer is None:
        raise HTTPException(status_code=409, detail="this run is a replay -- POST /turn instead")

    async with INGEST:
        return await _observe_locked(run, event)


async def _observe_locked(run: Run, event: GameEvent) -> dict:
    try:
        # observe() is synchronous and does network I/O. Off the event loop it
        # goes, or the websocket stops updating for the duration of every call.
        turn = await run_in_threadpool(run.observe, event)
    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}"
        log.error("observer failed on event %d: %s", len(run.turns), detail)
        run.finish(error=detail)
        run.write()  # never lose the events already paid for
        await hub.broadcast({"type": "run_error", "detail": detail, "turns": len(run.turns)})
        raise HTTPException(status_code=502, detail=detail)

    run.write()
    await hub.broadcast({"type": "turn", **turn.model_dump(mode="json")})
    log.info(
        "[%2d/%2d] %s: %s",
        turn.index + 1,
        run.total_expected,
        event.speaker,
        event.statement[:60],
    )
    # `committed` rides on the response so a feeder driving this from outside
    # learns the loop is done and can stop sending. Without it the stop condition
    # would only work for games the server plays itself.
    return {**turn.state.model_dump(mode="json"), "committed": await _committed(run)}


@app.post("/turn")
async def ingest_turn(turn: Turn) -> dict:
    """Replay only: a turn that was already computed in an earlier run.

    Same websocket traffic as a live turn, so the UI cannot tell the difference
    -- which is the point.
    """
    run = _require_run()
    if run.observer is not None:
        raise HTTPException(status_code=409, detail="this run is live -- POST /event instead")

    async with INGEST:
        stored = run.add_turn(turn.event, turn.state)
        await hub.broadcast({"type": "turn", **stored.model_dump(mode="json")})
        committed = await _committed(run)
    return {"ok": True, "index": stored.index, "committed": committed}


@app.get("/score")
async def score() -> dict:
    """Grade the run against ground truth.

    Answerable mid-game as well as at the end -- the numbers just describe fewer
    events. `complete` says which you're looking at, so the UI can show a live
    "currently accusing X" and a final verdict with the same call.
    """
    run = _require_run()
    if not run.turns:
        raise HTTPException(status_code=409, detail="nothing observed yet")

    result = grade(
        run.final_state,
        run.history,
        run.events,
        run.transcript.ground_truth,
        committed=run.committed,
        events_available=run.total_expected,
    )
    # `actual` names the real liars. It stays in this payload because the scorer
    # and the tests need it, but nothing on screen reads it -- the live scoreboard
    # is driven by `running_accuracy` and `final_accuracy`, which are already
    # reduced to numbers.
    return {
        **result.model_dump(),
        "complete": run.complete,
        "events_observed": len(run.turns),
        "events_expected": run.total_expected,
    }


@app.post("/game/end")
async def game_end() -> dict:
    run = _require_run()
    run.finish()
    path = run.write()
    log.info("run ended: %d/%d turns, complete=%s", len(run.turns), run.total_expected, run.complete)
    await hub.broadcast(
        {
            "type": "game_end",
            "complete": run.complete,
            "turns": len(run.turns),
            "expected": run.total_expected,
        }
    )
    return {"ok": True, "complete": run.complete, "turns": len(run.turns), "recording": str(path)}


# ---------------------------------------------------------------------------
# Playing a game from the UI.
#
# The feeder is still the real seam -- an outside process posting events is what
# a live game feed would do. These endpoints are the convenience path: they run
# the same loop inside the server so a button in the browser can start a game
# with nobody at a terminal.
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
GAME_ROOTS = ("data", "runs")


class PlayRequest(BaseModel):
    game: str
    mode: Literal["live", "replay"] = "replay"
    interval: float = 2.5


def _resolve_game(relative: str) -> Path:
    """Only files under data/ or runs/. The path comes from a browser, so it is
    not allowed to wander off into the filesystem."""
    candidate = (REPO_ROOT / relative).resolve()
    roots = [(REPO_ROOT / r).resolve() for r in GAME_ROOTS]
    if not any(candidate.is_relative_to(root) for root in roots):
        raise HTTPException(status_code=400, detail="game must live under data/ or runs/")
    if not candidate.is_file():
        raise HTTPException(status_code=404, detail=f"no such game: {relative}")
    return candidate


def _describe(path: Path) -> Optional[dict]:
    """Summarise a file for the picker, or None if it isn't a game."""
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(blob, dict) or "setup" not in blob:
        return None

    setup = blob.get("setup") or {}
    recording = "turns" in blob or "history" in blob
    count = len(blob.get("turns") or blob.get("events") or [])
    return {
        "id": str(path.relative_to(REPO_ROOT)),
        "label": path.stem.replace("_", " "),
        "kind": "recording" if recording else "transcript",
        "players": len(setup.get("players") or []),
        "events": count,
        "deceiver_role": setup.get("deceiver_role", "werewolf"),
        "deceiver_count": setup.get("deceiver_count", 1),
        "complete": blob.get("complete") if recording else None,
    }


@app.get("/games")
async def games() -> dict:
    """Everything playable: transcripts to run live, recordings to replay."""
    found = []

    base = REPO_ROOT / "data"
    if base.is_dir():
        for path in sorted(base.rglob("*.json")):
            described = _describe(path)
            if described:
                found.append(described)

    # runs/ accumulates fast. Offer only the newest few real runs -- a replay of
    # a replay is a copy of its source and adds nothing to the list.
    runs = REPO_ROOT / "runs"
    if runs.is_dir():
        recent = sorted(
            (p for p in runs.glob("*.json") if not p.stem.endswith("-replay")),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )[:5]
        for path in recent:
            described = _describe(path)
            if described:
                found.append(described)

    # Recordings first: they need no key, so they're the safe default to offer.
    found.sort(key=lambda g: (g["kind"] != "recording", g["id"]))
    return {"games": found}


PLAYER: Optional[asyncio.Task] = None


def _playing() -> bool:
    return PLAYER is not None and not PLAYER.done()


def _warmup(interval: float) -> float:
    """How fast to play the evidence-free opening.

    Until a role is announced the game contains nothing that can be checked, and
    the observer is capped almost flat on purpose -- so this stretch is most of a
    minute of bars declining to move. It is played fast rather than skipped:
    every line still reaches the observer and still appears in the transcript,
    it just does not sit on screen at full speed waiting to be read.
    """
    return min(interval, WARMUP_INTERVAL)


async def _committed(run: Run) -> bool:
    """Stop condition: the observer has seen enough and says so.

    Announced on its own message rather than folded into `game_end`, so the UI
    can show the call landing at the moment it happens instead of after the run
    has already torn down.

    Idempotent, because both drivers reach it: the server's own play loop and
    the feeder posting events from outside. Only the transition broadcasts.
    """
    already = run.committed is not None
    call = run.consider_commitment()
    if not call:
        return False
    if already:
        return True
    log.info(
        "observer committed after %d/%d events: %s at %.0f%%",
        call.events_observed, run.total_expected,
        ", ".join(call.players), call.confidence * 100,
    )
    await hub.broadcast({"type": "committed", **call.model_dump(mode="json")})
    return True


async def _play_game(path: Path, mode: str, interval: float) -> None:
    """Run a whole game into the hub. Cancellable: stopping keeps the partial
    run on disk, exactly as a crashed live run does."""
    global RUN
    try:
        if mode == "replay":
            transcript, turns, expected = load_recording(path, REPO_ROOT / "data" / "fallback_transcript.json")
            RUN = Run(transcript, live=False, events_expected=expected)
            await hub.broadcast(_snapshot(RUN))
            for turn in turns:
                async with INGEST:
                    stored = RUN.add_turn(turn.event, turn.state)
                    await hub.broadcast({"type": "turn", **stored.model_dump(mode="json")})
                    if await _committed(RUN):
                        break
                await asyncio.sleep(_warmup(interval) if RUN.in_opening else interval)
        else:
            transcript = load_transcript(path)
            RUN = Run(transcript, live=True)
            await hub.broadcast(_snapshot(RUN))
            for event in transcript.events:
                started = time.perf_counter()
                async with INGEST:
                    await _observe_locked(RUN, event)
                    if await _committed(RUN):
                        break
                # The model sets the pace when it is slower than the interval.
                pace = _warmup(interval) if RUN.in_opening else interval
                await asyncio.sleep(max(0.0, pace - (time.perf_counter() - started)))

        RUN.finish()
        RUN.write()
        await hub.broadcast(
            {
                "type": "game_end",
                "complete": RUN.complete,
                "turns": len(RUN.turns),
                "expected": RUN.total_expected,
            }
        )
        log.info("play finished: %d/%d turns", len(RUN.turns), RUN.total_expected)

    except asyncio.CancelledError:
        if RUN is not None:
            RUN.finish(error="stopped")
            RUN.write()
            await hub.broadcast(
                {"type": "game_end", "complete": False, "turns": len(RUN.turns),
                 "expected": RUN.total_expected}
            )
        log.info("play stopped by request")
        raise
    except HTTPException:
        # _observe_locked already recorded the failure and told the clients.
        log.error("play aborted after an observer failure")
    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}"
        log.error("play failed: %s", detail)
        if RUN is not None:
            RUN.finish(error=detail)
            RUN.write()
        await hub.broadcast({"type": "run_error", "detail": detail,
                             "turns": len(RUN.turns) if RUN else 0})


@app.post("/play")
async def play(req: PlayRequest) -> dict:
    """Start a game from the UI. One at a time."""
    global PLAYER
    if _playing():
        raise HTTPException(status_code=409, detail="a game is already playing -- POST /stop first")

    path = _resolve_game(req.game)
    if req.mode == "live":
        # Fail now, with a readable message, rather than on the first event.
        try:
            from backend.llm import get_backend

            get_backend()
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"{type(exc).__name__}: {exc}")

    PLAYER = asyncio.create_task(_play_game(path, req.mode, max(0.0, req.interval)))
    log.info("play requested: %s (%s, %.1fs/turn)", req.game, req.mode, req.interval)
    return {"ok": True, "game": req.game, "mode": req.mode, "interval": req.interval}


@app.post("/stop")
async def stop() -> dict:
    global PLAYER
    if not _playing():
        return {"ok": True, "stopped": False}
    PLAYER.cancel()
    try:
        await PLAYER
    except asyncio.CancelledError:
        pass
    return {"ok": True, "stopped": True}


@app.websocket("/live")
async def live(ws: WebSocket) -> None:
    await hub.connect(ws)
    try:
        if RUN is not None:
            await hub.send(ws, _snapshot(RUN))
        else:
            await hub.send(ws, {"type": "snapshot", "turns": [], "setup": None})
        # Nothing is expected from the client; this just parks until it leaves.
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        hub.disconnect(ws)


# Mounted last and on purpose: a mount at "/" swallows anything not already
# matched above, so every API route has to be declared before this line. One
# process then serves both the API and the page, which keeps the websocket
# same-origin and sidesteps CORS entirely.
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
else:  # pragma: no cover - only before the first `npm run build`
    log.warning(
        "no built frontend at %s -- serving the API only. "
        "Build it with: cd frontend && npm install && npm run build",
        FRONTEND_DIR,
    )
