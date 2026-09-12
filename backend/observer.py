"""The observer: one Claude call per event, carrying its whole belief forward.

This is the part of Cry Wolf the build is actually about. Everything else moves
bytes around; this is the thing that has to remember round 1 when it is looking
at round 4.

Design notes worth knowing before you change anything here:

* The prior belief state goes back into the prompt in full on every call. The
  model gets no conversation history -- the belief state IS the memory, which is
  what makes it inspectable. If it forgets something, you can see the forgetting.
* Output is constrained by a strict JSON schema, so we never parse prose.
* Which model answers is `llm.py`'s problem, not this file's. Nothing below
  mentions a provider.
* The model's raw numbers are advisory. `_settle` normalizes them and rate-limits
  how far any one player can move in a single event, because an unchecked model
  will happily swing a player from 0.1 to 0.9 on one sarcastic remark and the
  suspicion chart stops meaning anything.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from backend.llm import StructuredLLM, get_backend
from backend.schema import (
    BeliefState,
    Contradiction,
    GameEvent,
    GameSetup,
    ObserverOutput,
)

# No single line of dialogue is worth more than this much suspicion. Tuned by
# watching replays: below ~0.2 the observer looks catatonic, above ~0.4 the
# chart turns into noise.
MAX_DELTA_PER_EVENT = 0.30

SYSTEM = """\
You are an observer of a live Werewolf game. You do not play. You do not speak \
to the players. You watch, and you keep one running judgement: for each living \
player, how likely is it that they are the werewolf?

You see the game one line at a time, in order, exactly as it happens. Before \
each line you are given your own belief state from a moment ago -- your \
suspicion numbers, the claims you have been tracking, and the contradictions \
you have already caught. That state is your only memory. Whatever you do not \
carry forward in it is gone.

How to reason:

- Weigh behaviour over assertion. Everyone says they are a villager; that tells \
you nothing. What tells you something is who a player defends, who they steer \
attention toward, when they change position, and what they conveniently avoid.
- A werewolf knows who the werewolf is. Watch for someone arguing from \
knowledge they should not have, or being oddly incurious about a player they \
would normally suspect.
- Hold your earlier reads. If you flagged a player in round 1, either that \
reason still stands or something specific has resolved it -- and if something \
resolved it, say so in your reasoning. Quietly abandoning a suspicion is the \
single worst thing you can do here.
- Move in proportion to evidence. One ambiguous remark nudges. A genuine \
contradiction shoves. Nothing in a single line justifies a total reversal.
- Suspicion across all living players should sum to roughly 1.0 -- these are \
shares of one werewolf, not independent verdicts.

`claims_tracked` is your ledger: carry every player's entry forward and extend \
it, do not overwrite the history with only the latest thing they said.

`contradictions_noticed` is cumulative too. Add a new entry only when a player \
says something that genuinely conflicts with something they said earlier -- not \
merely something you disagree with. Never drop an entry you have already made.

`reasoning` is at most two sentences, about what this line changed and why."""


def _fmt_state(state: BeliefState, players: List[str]) -> str:
    if state.event_index < 0:
        return "(nothing yet -- this is the first line of the game)"

    lines = ["Suspicion:"]
    for p in players:
        lines.append(f"  {p}: {state.suspicion.get(p, 0.0):.2f}")

    lines.append("")
    lines.append("Claims you are tracking:")
    if state.claims_tracked:
        for p, claim in state.claims_tracked.items():
            lines.append(f"  {p}: {claim}")
    else:
        lines.append("  (none yet)")

    lines.append("")
    lines.append("Contradictions you have already caught:")
    if state.contradictions_noticed:
        for c in state.contradictions_noticed:
            lines.append(
                f"  R{c.round_noticed} {c.player}: said \"{c.earlier}\", then \"{c.now}\""
            )
    else:
        lines.append("  (none yet)")

    lines.append("")
    lines.append(f"Your last note: {state.reasoning}")
    return "\n".join(lines)


def render(event: GameEvent, state: BeliefState, setup: GameSetup) -> str:
    return f"""GAME
{setup.premise}
Players: {", ".join(setup.players)}

YOUR BELIEF STATE, BEFORE THIS LINE
{_fmt_state(state, setup.players)}

THE NEXT LINE
Round {event.round}, {event.phase} -- {event.speaker}: "{event.statement}"

Update your belief state to account for this line. Carry forward everything that still holds."""


def _settle(
    raw: List,
    prior: Dict[str, float],
    players: List[str],
    first_event: bool,
) -> Dict[str, float]:
    """Turn the model's numbers into a distribution we can plot.

    Fills in players the model forgot, rate-limits per-event movement, then
    normalizes to sum 1.0.
    """
    uniform = 1.0 / len(players)
    proposed = {e.player: max(0.0, min(1.0, e.score)) for e in raw if e.player in players}

    settled: Dict[str, float] = {}
    for p in players:
        before = prior.get(p, uniform)
        # A player the model omitted keeps its previous value rather than
        # silently falling to zero -- omission is not exoneration.
        after = proposed.get(p, before)
        if not first_event:
            after = max(before - MAX_DELTA_PER_EVENT, min(before + MAX_DELTA_PER_EVENT, after))
        settled[p] = after

    total = sum(settled.values())
    if total <= 0:
        return {p: uniform for p in players}
    return {p: v / total for p, v in settled.items()}


def _merge_contradictions(
    prior: List[Contradiction], fresh: List[Contradiction]
) -> List[Contradiction]:
    """Append-only. The model is told never to drop one; this enforces it."""
    seen = {(c.player, c.earlier, c.now) for c in prior}
    merged = list(prior)
    for c in fresh:
        if (c.player, c.earlier, c.now) not in seen:
            seen.add((c.player, c.earlier, c.now))
            merged.append(c)
    return merged


class Observer:
    """Stateful across one game. Construct once, feed events in order."""

    def __init__(self, setup: GameSetup, llm: Optional[StructuredLLM] = None):
        self.setup = setup
        self.llm = llm or get_backend()
        uniform = 1.0 / len(setup.players)
        self.state = BeliefState(suspicion={p: uniform for p in setup.players})
        self.history: List[Dict[str, float]] = []

    def observe(self, event: GameEvent) -> BeliefState:
        """Fold one event into the belief state and return the new one."""
        prior = self.state
        first = prior.event_index < 0

        out: ObserverOutput = self.llm.structured(
            system=SYSTEM,
            user=render(event, prior, self.setup),
            schema=ObserverOutput,
        )

        claims = dict(prior.claims_tracked)
        for entry in out.claims_tracked:
            claims[entry.player] = entry.claim

        self.state = BeliefState(
            round=event.round,
            phase=event.phase,
            event_index=prior.event_index + 1,
            suspicion=_settle(out.suspicion, prior.suspicion, self.setup.players, first),
            claims_tracked=claims,
            contradictions_noticed=_merge_contradictions(
                prior.contradictions_noticed, out.contradictions_noticed
            ),
            reasoning=out.reasoning,
        )
        self.history.append(dict(self.state.suspicion))
        return self.state
