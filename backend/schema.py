"""Shared data contract for Cry Wolf.

Lane A owns the transport (API, feeder, websocket); Lane B owns the observer.
This module is the seam between them -- both import from here, neither
redefines these shapes locally.
"""

from __future__ import annotations

import re
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

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
    eliminated: Optional[str] = Field(
        None,
        description=(
            "Player this event kills or votes out, if any. Optional -- the observer "
            "falls back to reading the moderator's wording. Set it if you have it."
        ),
    )


class GameSetup(BaseModel):
    """Everything the observer is allowed to know before the game starts.

    Deliberately excludes who holds which role -- ground truth lives in
    `Transcript`, and the observer is never handed one. How *many* deceivers
    there are is different: in both Werewolf and Mafia every player is told that
    at the start, so withholding it would handicap the observer against the
    humans it is being compared to.
    """

    players: List[str]
    deceiver_role: str = Field(
        "werewolf",
        description='What the liars are called in this game: "werewolf", "mafia", ...',
    )
    deceiver_count: int = Field(
        1, ge=1, description="How many of them there are. Public knowledge in both games."
    )
    deceiver_plural: Optional[str] = Field(
        None,
        description=(
            'Plural of deceiver_role, when adding "s" is wrong. "mafia" is already '
            'both singular and plural; "werewolf" pluralises normally.'
        ),
    )

    def deceivers_phrase(self) -> str:
        """How to name the liars in a prompt, with the right number agreement."""
        if self.deceiver_count == 1:
            return f"1 {self.deceiver_role}"
        plural = self.deceiver_plural
        if not plural:
            role = self.deceiver_role
            # werewolf -> werewolves, not werewolfs.
            plural = f"{role[:-1]}ves" if role.endswith("f") else f"{role}s"
        return f"{self.deceiver_count} {plural}"
    premise: str = (
        "Standard Werewolf. Exactly one werewolf hides among the villagers. "
        "The werewolf knows who they are; nobody else does."
    )


# Speakers that aren't players. A narrator by any name.
NARRATORS = {"moderator", "narrator", "host", "system", "gm", "game", "game-manager"}

# Role names that mean "the ones who are lying", whichever game this is.
DECEIVER_ROLES = {"werewolf", "wolf", "werewolves", "mafia", "scum"}

# "P4 was the werewolf" / "the mafia was P4" / "P4 is the wolf".
#
# Deliberately NOT matched: "X was voted out. Their role was mafia." In both
# games an eliminated player's role is announced to everyone, so that is public
# information the human players had -- withholding it from the observer would be
# a different kind of cheating. Eliminated players leave the suspicion
# distribution anyway, so a revealed role can't inflate the score.
_ANSWER_LEAK = re.compile(
    r"(?:\b(\w+)\b\s+(?:was|is|were|are)\s+the\s+(?:were)?(?:wolf|mafia))"
    r"|(?:the\s+(?:were)?(?:wolf|mafia)\s+(?:was|is)\s+\b(\w+)\b)",
    re.I,
)


class Transcript(BaseModel):
    """A full game on disk: setup, events, and the answer key.

    Validated on load rather than trusted. A malformed transcript that gets
    through fails somewhere deep in a run instead -- after the model calls have
    been paid for -- and real datasets are malformed in all of these ways.
    """

    setup: GameSetup
    events: List[GameEvent]
    ground_truth: Dict[str, str] = Field(
        ..., description='Player -> role, e.g. {"P2": "werewolf", "P1": "villager"}'
    )

    @model_validator(mode="after")
    def _check(self) -> "Transcript":
        players = set(self.setup.players)
        if not players:
            raise ValueError("setup.players is empty")
        if len(players) != len(self.setup.players):
            raise ValueError("setup.players contains duplicates")

        deceivers = self.deceivers()
        if not deceivers:
            raise ValueError(
                f"ground_truth names no {self.setup.deceiver_role} "
                f"(looked for any of: {', '.join(sorted(DECEIVER_ROLES))})"
            )
        if len(deceivers) != self.setup.deceiver_count:
            raise ValueError(
                f"setup.deceiver_count is {self.setup.deceiver_count} but ground_truth "
                f"names {len(deceivers)} ({', '.join(deceivers)}) -- the observer is told "
                f"the count, so it has to be true"
            )

        unknown = set(self.ground_truth) - players
        if unknown:
            raise ValueError(f"ground_truth names non-players: {', '.join(sorted(unknown))}")
        missing = players - set(self.ground_truth)
        if missing:
            raise ValueError(f"ground_truth has no role for: {', '.join(sorted(missing))}")

        for i, event in enumerate(self.events):
            if event.speaker not in players and event.speaker.lower() not in NARRATORS:
                raise ValueError(
                    f"event {i} is spoken by {event.speaker!r}, who is neither a player "
                    f"nor a narrator ({', '.join(sorted(NARRATORS))})"
                )
            if event.eliminated and event.eliminated not in players:
                raise ValueError(f"event {i} eliminates {event.eliminated!r}, not a player")

            # The observer reads every statement, so a transcript that announces
            # the answer hands it the game and the accuracy number becomes a lie.
            #
            # Only the narrator can leak, though. "Morgan is the mafia" from a
            # player is an accusation -- it is the entire game, and half of them
            # are wrong. Checking every speaker rejected 6 of 33 real LLMafia
            # games for playing Mafia correctly.
            if event.speaker.lower() in NARRATORS:
                match = _ANSWER_LEAK.search(event.statement)
                if match:
                    named = match.group(1) or match.group(2)
                    if named in players:
                        raise ValueError(
                            f"event {i}: the narrator reveals a role to the observer "
                            f"({event.statement!r}). Keep reveals in ground_truth only."
                        )
        return self

    def deceivers(self) -> List[str]:
        """Every player on the lying team, in roster order."""
        return [
            p
            for p in self.setup.players
            if self.ground_truth.get(p, "").lower() in DECEIVER_ROLES
        ]

    def werewolf(self) -> str:
        """The single deceiver. Only meaningful in one-wolf games -- use
        `deceivers()` for anything that must handle Mafia too."""
        found = self.deceivers()
        if len(found) != 1:
            raise ValueError(
                f"this game has {self.setup.deceivers_phrase()}; use deceivers() instead"
            )
        return found[0]


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
    earlier: str = Field(
        ...,
        description=(
            "The player's own earlier words, quoted verbatim. Not a paraphrase, "
            "not your summary of them -- these are shown on screen as a quotation."
        ),
    )
    now: str = Field(
        ...,
        description=(
            "The words they just said that conflict with it, quoted verbatim from "
            "this line. Never your own gloss or conclusion about what they said."
        ),
    )
    round_noticed: int


class ObserverOutput(BaseModel):
    """Exactly what we ask the model for. Kept flat and small on purpose."""

    suspicion: List[SuspicionEntry]
    claims_tracked: List[ClaimEntry]
    contradictions_noticed: List[Contradiction]
    reasoning: str = Field(
        ...,
        description=(
            "What THIS line changed and why, in at most two sentences and under 200 "
            "characters. It is displayed in a narrow panel beside the transcript, so "
            "anything longer is cut off mid-word. Do not recap the game so far."
        ),
    )


class Deliberation(BaseModel):
    """The deeper, less frequent call: a whole round reconsidered at once.

    Fired only when an elimination announces a role, because that is the only
    moment new structural evidence enters the game -- every vote cast that round
    has just been graded. A 70-event game triggers about three of these, against
    70 shallow per-event calls, so this is the cheap half of the loop as well as
    the good half.

    Unlike `ObserverOutput` this is explicitly allowed to throw out an earlier
    read wholesale. `revised` is what makes that visible: an observer that
    quietly stops suspecting someone is the failure mode the per-event loop kept
    falling into, so here it has to say out loud what it abandoned and why.
    """

    suspicion: List[SuspicionEntry]
    revised: str = Field(
        ...,
        description=(
            "What you changed your mind about and what specifically changed it, or "
            "the empty string if this round genuinely changed nothing. If you have "
            "dropped a suspicion you held, name the player and say what resolved it."
        ),
    )
    reasoning: str = Field(
        ...,
        description=(
            "Your read of the game after this round, in at most two sentences and "
            "under 200 characters. Shown in the same narrow panel as the per-event "
            "note, so anything longer is cut off mid-word."
        ),
    )


# --------------------------------------------------------------------------
# Outbound: what the UI and the scorer consume
# --------------------------------------------------------------------------


class BeliefState(BaseModel):
    """The observer's running guess. Rewritten after every single event."""

    round: int = 0
    phase: Phase = "day"
    event_index: int = -1
    eliminated: List[str] = Field(
        default_factory=list, description="Dead players, in the order they died."
    )
    suspicion: Dict[str, float] = Field(
        default_factory=dict, description="Living players only, summing to 1.0."
    )
    claims_tracked: Dict[str, str] = Field(default_factory=dict)
    contradictions_noticed: List[Contradiction] = Field(default_factory=list)
    reasoning: str = ""
    deliberated: bool = Field(
        False,
        description=(
            "This state was rewritten by the deep pass, not just the per-event one. "
            "Set when a role reveal let the observer re-read a whole round against "
            "graded evidence -- the moment it is allowed to change its mind wholesale."
        ),
    )

    @property
    def top_suspect(self) -> Optional[str]:
        if not self.suspicion:
            return None
        return max(self.suspicion, key=lambda p: self.suspicion[p])


class Score(BaseModel):
    """Final grade, served by GET /score."""

    accuracy: bool = Field(
        ..., description="Is the observer's top suspect actually on the lying team?"
    )
    predicted: Optional[str] = Field(
        None, description="Top suspect before the game resolved itself. See scoring.verdict."
    )
    final_verdict: Optional[str] = Field(
        None, description="Top suspect in the very last belief state."
    )
    final_accuracy: bool = Field(
        False,
        description=(
            "Was final_verdict a liar? Differs from accuracy when the closing "
            "events changed the observer's mind -- report both, not the kinder one."
        ),
    )
    actual: List[str] = Field(..., description="Everyone who really was lying.")
    precision_at_n: float = Field(
        0.0,
        description=(
            "Of the observer's top N living suspects, the fraction who really are "
            "liars, where N is how many are still in the game. The honest companion "
            "to accuracy once there is more than one."
        ),
    )
    final_confidence: float
    consistency: float = Field(
        ..., description="1.0 = never contradicted itself. Per-event only -- see lead_changes."
    )
    lead_changes: int = Field(
        0, description="Times the top suspect changed hands over the whole game."
    )
    events_off_verdict: int = Field(
        0,
        description=(
            "After the final verdict first took the lead, events where someone else "
            "was on top. The wobble consistency alone does not see."
        ),
    )
    self_contradictions: int
    contradictions_caught: int
    running_accuracy: float = Field(
        0.0,
        description=(
            "Across every event so far, the fraction whose top suspect really was "
            "a liar. Derived here rather than in the client so a live scoreboard "
            "never has to be handed `actual` to compute it."
        ),
    )
    rounds_observed: int
    committed_at: Optional[int] = Field(
        None,
        description=(
            "Events observed when the loop stopped itself, or None if it read to "
            "the end. The point of the stop condition: how much of the game it "
            "did not need to see."
        ),
    )
    events_available: int = Field(
        0, description="Events the game held, whether or not they were read."
    )
    suspicion_history: List[Dict[str, float]] = Field(
        default_factory=list, description="Per-event snapshot, for the suspicion-over-time chart."
    )
