"""Websocket fan-out.

Every connected browser gets a snapshot of the run so far the moment it
connects, then one message per turn as they happen. The snapshot is what makes
a mid-demo refresh survivable -- reconnecting shows the full game to date
instead of empty panes.

Message envelope, always: {"type": ..., ...}

    snapshot    the whole run so far: setup, every turn, whether it ended
    turn        one {event, state} pair
    run_error   the observer failed; the run stopped here
    game_end    the last event has been processed
"""

from __future__ import annotations

import asyncio
import logging
from typing import Set

from fastapi import WebSocket

log = logging.getLogger("crywolf.hub")


class Hub:
    def __init__(self) -> None:
        self._connections: Set[WebSocket] = set()

    def __len__(self) -> int:
        return len(self._connections)

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._connections.add(ws)

    def disconnect(self, ws: WebSocket) -> None:
        self._connections.discard(ws)

    async def send(self, ws: WebSocket, message: dict) -> None:
        await ws.send_json(message)

    async def broadcast(self, message: dict) -> None:
        """Send to everyone, and quietly drop anyone who has gone away.

        A browser closed mid-run must never take the run down with it, so every
        send is allowed to fail independently.
        """
        if not self._connections:
            return
        targets = list(self._connections)
        results = await asyncio.gather(
            *(ws.send_json(message) for ws in targets), return_exceptions=True
        )
        for ws, result in zip(targets, results):
            if isinstance(result, Exception):
                log.debug("dropping dead websocket: %s", result)
                self.disconnect(ws)
