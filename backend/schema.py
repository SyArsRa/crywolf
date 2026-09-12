"""Shared data contract for Cry Wolf.

Lane A owns the transport (API, feeder, websocket); Lane B owns the observer.
This module is the seam between them -- both import from here, neither
redefines these shapes locally.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

Phase = Literal["night", "day", "vote"]


# --------------------------------------------------------------------------
# Inbound: what the feeder posts to POST /event
# --------------------------------------------------------------------------


class GameEvent(BaseModel):
    """One line of the game, as it arrives from the feed."""

    round: int
    phase: Phase
    speaker: str
    statement: str
    timestamp: Optional[str] = None


class GameSetup(BaseModel):
    """Everything the observer is allowed to know before the game starts.

    Deliberately excludes roles -- ground truth lives in `Transcript`, and the
    observer is never handed one.
    """

    players: List[str]
    premise: str = (
        "Standard Werewolf. Exactly one werewolf hides among the villagers. "
        "The werewolf knows who they are; nobody else does."
    )


class Transcript(BaseModel):
    """A full game on disk: setup, events, and the answer key."""

    setup: GameSetup
    events: List[GameEvent]
    ground_truth: Dict[str, str] = Field(
        ..., description='Player -> role, e.g. {"P2": "werewolf", "P1": "villager"}'
    )

    def werewolf(self) -> str:
        for player, role in self.ground_truth.items():
            if role.lower() == "werewolf":
                return player
        raise ValueError("transcript ground_truth contains no werewolf")


# --------------------------------------------------------------------------
# The model's per-event output.
#
# Strict JSON schema cannot express an open-keyed object (`{"P2": 0.62}`), so
# the model emits lists of records and we project them into dicts on the way
# out. Nothing downstream sees this wire shape.
# --------------------------------------------------------------------------


class SuspicionEntry(BaseModel):
    player: str
    score: float = Field(..., ge=0.0, le=1.0)


class ClaimEntry(BaseModel):
    player: str
    claim: str = Field(..., description="What this player has asserted or implied, cumulative.")


class Contradiction(BaseModel):
    player: str
    earlier: str = Field(..., description="What they said before.")
    now: str = Field(..., description="What they just said that conflicts with it.")
    round_noticed: int


class ObserverOutput(BaseModel):
    """Exactly what we ask the model for. Kept flat and small on purpose."""

    suspicion: List[SuspicionEntry]
    claims_tracked: List[ClaimEntry]
    contradictions_noticed: List[Contradiction]
    reasoning: str = Field(..., description="Two sentences at most, on what changed and why.")


# --------------------------------------------------------------------------
# Outbound: what the UI and the scorer consume
# --------------------------------------------------------------------------


class BeliefState(BaseModel):
    """The observer's running guess. Rewritten after every single event."""

    round: int = 0
    phase: Phase = "day"
    event_index: int = -1
    suspicion: Dict[str, float] = Field(default_factory=dict)
    claims_tracked: Dict[str, str] = Field(default_factory=dict)
    contradictions_noticed: List[Contradiction] = Field(default_factory=list)
    reasoning: str = ""

    @property
    def top_suspect(self) -> Optional[str]:
        if not self.suspicion:
            return None
        return max(self.suspicion, key=lambda p: self.suspicion[p])


class Score(BaseModel):
    """Final grade, served by GET /score."""

    accuracy: bool
    predicted: Optional[str]
    actual: str
    final_confidence: float
    consistency: float = Field(..., description="1.0 = never contradicted itself. See scoring.py.")
    self_contradictions: int
    contradictions_caught: int
    rounds_observed: int
    suspicion_history: List[Dict[str, float]] = Field(
        default_factory=list, description="Per-event snapshot, for the suspicion-over-time chart."
    )
