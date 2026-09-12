"""Cry Wolf ingestion API.

    POST /game/start   open a run (feeder -> api)
    POST /event        push one line of the game, get the new belief state back
    POST /turn         push a pre-computed turn (replay only, no model calls)
    POST /game/end     close the run
    GET  /score        accuracy + consistency against ground truth
    WS   /live         snapshot on connect, then one message per turn
    GET  /health       liveness + what run is open

The seam that matters is POST /event: the feeder on one side is entirely
replaceable -- a real game feed, speech-to-text, a second transcript source --
and nothing downstream of this endpoint would know the difference.

The scoring seam is `scoring.grade`, which reads only the belief states -- so
`GET /score` works for a replayed run exactly as it does for a live one.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

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
    return turn.state.model_dump(mode="json")


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
    return {"ok": True, "index": stored.index}


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

    result = grade(run.final_state, run.history, run.events, run.transcript.ground_truth)
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
