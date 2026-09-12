"""The run: everything the server knows about the game currently in progress.

One run at a time, held in memory, mirrored to disk after every turn. The disk
copy is the insurance policy -- if the network dies on stage, or a live run goes
badly, a recorded run can be replayed through the same websocket with no model
calls at all (see `feeder.py --replay`).

Ground truth lives here and goes nowhere near the observer or the websocket.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel

from backend.schema import BeliefState, GameEvent, GameSetup, Transcript

RUNS_DIR = Path("runs")


class Turn(BaseModel):
    """One event and the belief state it produced.

    This is the unit the websocket sends and the unit the suspicion-over-time
    chart plots -- keeping them paired is what lets the UI show a line and the
    reasoning behind it together.
    """

    index: int
    event: GameEvent
    state: BeliefState


class Run:
    """A single game in flight.

    `live=False` skips constructing an Observer entirely, which also means no
    API key is needed -- that's what makes replay runnable on a laptop with no
    credentials.
    """

    def __init__(
        self,
        transcript: Transcript,
        live: bool = True,
        runs_dir: Path = RUNS_DIR,
        events_expected: Optional[int] = None,
    ):
        self.transcript = transcript
        self.live = live
        # Replaying a partial recording, the transcript holds only the events
        # that were reached -- so the original total has to be carried over, or
        # a half-finished game reports itself complete.
        self._events_expected = events_expected
        self.turns: List[Turn] = []
        self.complete = False
        self.error: Optional[str] = None
        self.started_at = datetime.now(timezone.utc)

        self.observer = None
        if live:
            # Imported here so replay never touches llm.py, which raises without a key.
            from backend.observer import Observer

            self.observer = Observer(transcript.setup)

        stamp = self.started_at.strftime("%Y%m%d-%H%M%S")
        self.path = runs_dir / f"run-{stamp}{'' if live else '-replay'}.json"

    # -- views ------------------------------------------------------------

    @property
    def setup(self) -> GameSetup:
        return self.transcript.setup

    @property
    def players(self) -> List[str]:
        return self.transcript.setup.players

    @property
    def total_expected(self) -> int:
        return self._events_expected or len(self.transcript.events)

    @property
    def events(self) -> List[GameEvent]:
        return [t.event for t in self.turns]

    @property
    def history(self) -> List[Dict[str, float]]:
        """Per-event suspicion snapshots -- the shape `scoring.grade` wants."""
        return [dict(t.state.suspicion) for t in self.turns]

    @property
    def final_state(self) -> BeliefState:
        if self.turns:
            return self.turns[-1].state
        uniform = 1.0 / len(self.players) if self.players else 0.0
        return BeliefState(suspicion={p: uniform for p in self.players})

    # -- mutation ---------------------------------------------------------

    def observe(self, event: GameEvent) -> Turn:
        """Fold one event into the run. Blocking: does network I/O."""
        if self.observer is None:
            raise RuntimeError("this run is a replay; use add_turn() instead")
        state = self.observer.observe(event)
        return self.add_turn(event, state)

    def add_turn(self, event: GameEvent, state: BeliefState) -> Turn:
        turn = Turn(index=len(self.turns), event=event, state=state)
        self.turns.append(turn)
        return turn

    def finish(self, error: Optional[str] = None) -> None:
        self.error = error
        self.complete = error is None and len(self.turns) >= self.total_expected

    # -- disk -------------------------------------------------------------

    def to_record(self) -> dict:
        """The recording format.

        A superset of what `run_observer._write_run` writes: the `complete` /
        `events` / `history` / `final_state` / `score` keys are kept so anything
        expecting that shape still reads this, and `turns` is added because
        those keys alone lose each turn's reasoning and contradictions -- which
        is everything the UI needs to replay a run rather than just chart it.
        """
        return {
            "complete": self.complete,
            "live": self.live,
            "started_at": self.started_at.isoformat(),
            "events_observed": len(self.turns),
            "events_expected": self.total_expected,
            "error": self.error,
            "setup": self.setup.model_dump(),
            "ground_truth": self.transcript.ground_truth,
            "events": [t.event.model_dump() for t in self.turns],
            "history": self.history,
            "turns": [t.model_dump() for t in self.turns],
            "final_state": self.final_state.model_dump(),
            "score": None,  # GET /score is a separate item
        }

    def write(self) -> Path:
        """Mirror the run to disk. Called after every turn: quota and time are
        expensive enough that losing a half-finished run is not acceptable."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.to_record(), indent=2), encoding="utf-8")
        return self.path
