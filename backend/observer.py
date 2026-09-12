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

import re
from typing import Dict, List, Optional

from backend.llm import StructuredLLM, get_backend
from backend.schema import (
    DECEIVER_ROLES,
    NARRATORS,
    BeliefState,
    Contradiction,
    GameEvent,
    GameSetup,
    ObserverOutput,
)

# No single line of dialogue is worth more than this much suspicion. Tuned by
# watching replays: below ~0.2 the observer looks catatonic, above ~0.4 the
# chart turns into noise.
#
# Symmetric on purpose, and we considered changing that. In one run the observer
# had the werewolf at 55%, then spent seven events down at 10% pointing at
# someone else, then came back to 80% when the lie surfaced. An asymmetric cap
# (suspicion easy to gain, hard to shed) would have kept the chart tidier.
#
# We decided against it. That swing was not forgetting -- the claims ledger held
# throughout, which is exactly why it recovered the instant hard evidence
# arrived. It had considered a theory that P3 was shielding the werewolf, and
# dropped it when the evidence said otherwise. Rigging the arithmetic so the
# observer cannot change its mind would buy a prettier demo at the cost of the
# reasoning being real. The wobble is reported instead, by
# `scoring.measure_drift`.
MAX_DELTA_PER_EVENT = 0.30

# Who has died. Lane A can set `GameEvent.eliminated` explicitly; failing that we
# read the moderator's announcement, which is the only place deaths are stated.
_ROLE_REVEAL = re.compile(r"their role was (?P<role>\w+)", re.I)

_DEATH = re.compile(
    r"\b(?P<who>[A-Z]\w*)\b\s+(?:is|was|has been)\s+(?:found\s+)?(?:dead|killed|eliminated|lynched|voted out)",
    re.I,
)


# "Kai voted for Sutton" -- the narrator reporting a cast vote.
_VOTE_REPORTED = re.compile(r"^(?P<voter>\S+) voted for (?P<target>[A-Za-z]\w*)", re.I)

# "I vote P4." -- a player declaring their own, during the voting phase.
_VOTE_DECLARED = re.compile(r"^i(?:'m| am)? vot(?:e|ing)(?: for)? (?P<target>[A-Za-z]\w*)", re.I)


def infer_vote(event: GameEvent) -> Optional[tuple]:
    """(voter, target) for a publicly cast vote, or None.

    Two shapes, because the two transcripts record votes differently: LLMafia has
    the narrator announce each one, while the hand-written game has players say
    "I vote P4." in the voting phase. Both are public declarations and both
    belong in the record -- reading only the first left the vote table empty for
    the entire Werewolf transcript.

    A player's line counts only during the voting phase. "Let's vote out Rowan"
    in mid-argument is a suggestion, not a vote, and treating it as one would
    poison the strongest signal we have.
    """
    if event.speaker.lower() in NARRATORS:
        match = _VOTE_REPORTED.match(event.statement.strip())
        return (match.group("voter"), match.group("target")) if match else None
    if event.phase != "vote":
        return None
    match = _VOTE_DECLARED.match(event.statement.strip())
    return (event.speaker, match.group("target")) if match else None


def format_votes(votes: Dict[int, Dict[str, str]], alive: List[str]) -> str:
    """The vote record, laid out so the pair signal is visible at a glance.

    Measured across the 33 real Mafia games in this dataset: of 160 day-votes
    cast by a mafia member, only 6 (3.8%) were aimed at their own partner.
    Random voting would put that near 15-20%. So "these two have had every
    opportunity to vote for each other and never have" is the strongest single
    signal available -- and it is a fact about the whole game, not one line, so
    the model should not have to reconstruct it from memory every turn.
    """
    if not votes:
        return "(nobody has voted yet)"

    lines = []
    for rnd in sorted(votes):
        cast = ", ".join(f"{v} -> {t}" for v, t in votes[rnd].items())
        lines.append(f"  round {rnd}: {cast}")

    # Who has never voted for whom, among players still in the game.
    voted_for: Dict[str, set] = {}
    for round_votes in votes.values():
        for voter, target in round_votes.items():
            voted_for.setdefault(voter, set()).add(target)

    never = []
    for a in alive:
        for b in alive:
            if a >= b or a not in voted_for or b not in voted_for:
                continue
            if b not in voted_for[a] and a not in voted_for[b]:
                never.append(f"{a}/{b}")
    if never:
        lines.append("")
        lines.append(
            "  Both voted, never at each other: " + ", ".join(never)
        )
    return "\n".join(lines)


def infer_elimination(event: GameEvent) -> Optional[str]:
    """The player this event announces the death of, if any.

    Only the narrator can eliminate anyone. Players say "ashton is dead" and
    "who is dead?" all the time -- in the real Mafia games they say it a lot --
    and reading those as eliminations let any player remove a rival from the
    suspicion distribution just by claiming they were gone. Seven such lines
    across the 33 games, one of which would have "killed" the speaker's own
    accuser mid-argument.
    """
    if event.eliminated:
        return event.eliminated
    if event.speaker.lower() not in NARRATORS:
        return None
    match = _DEATH.search(event.statement)
    return match.group("who") if match else None

SYSTEM = """\
You are an observer of a live social deduction game -- Werewolf, Mafia, or a \
variant of one. You do not play. You do not speak to the players. You watch, and \
you keep one running judgement: for each living player, how likely is it that \
they are on the lying team?

Each turn you are told what the liars are called in this game and how many there \
are. Both are public knowledge every player at the table has. Where there is more \
than one, they know each other and are working together -- so watch for players \
who defend each other, never vote for each other, or take turns steering \
suspicion onto the same outsider.

You see the game one line at a time, in order, exactly as it happens. Before \
each line you are given your own belief state from a moment ago -- your \
suspicion numbers, the claims you have been tracking, and the contradictions \
you have already caught. That state is your only memory. Whatever you do not \
carry forward in it is gone.

How to reason. These are ordered: the first is worth more than all the rest \
together, and the last is a trap.

1. THE VOTE RECORD IS YOUR BEST EVIDENCE. It is given to you exact, every turn, \
so you never have to recall it. Teammates almost never vote for each other -- \
measured across real games of this kind, under 4% of a liar's votes land on \
their own partner, against 15-20% if they voted at random. So two players who \
have both been voting, for rounds, and never once at each other, are the single \
strongest pair you can find. Weigh who votes for whom far above anything anyone \
says.

2. Watch what a vote costs the voter. Voting for a player nobody else suspects, \
or switching targets the moment a partner comes under pressure, or piling onto \
whoever is already losing -- these move suspicion. A vote is an action; talk is \
free.

3. The liars know who each other are. Watch for someone arguing from knowledge \
they should not have, or being oddly incurious about a player they would \
otherwise suspect.

4. Time matters. Liars tend to say less early and more once the game has shape. \
Someone who led the first round loudly and then went quiet is more often \
innocent than guilty.

5. DO NOT treat aggression, confidence or baseless accusation as guilt. This is \
the trap, and it is the most common way to get this wrong. Frightened villagers \
accuse people constantly, loudly, and on no evidence -- in real games they do it \
just as often as the liars do, and pointing at whoever is shouting is how an \
observer ends up accusing three innocents in a row. Being wrong is not the same \
as lying. Only raise suspicion on an accusation if the accuser gains something \
specific from that person being gone.
- Hold your earlier reads. If you flagged a player in round 1, either that \
reason still stands or something specific has resolved it -- and if something \
resolved it, say so in your reasoning. Quietly abandoning a suspicion is the \
single worst thing you can do here.
- Move in proportion to evidence. One ambiguous remark nudges. A genuine \
contradiction shoves. Nothing in a single line justifies a total reversal.
- Suspicion across living players should sum to roughly 1.0. It is one pool of \
suspicion shared out, not independent verdicts: for one player to rise, others \
must fall. Where there are several liars, expect the pool to be split between \
them rather than piled on one.

`claims_tracked` is your ledger: carry every player's entry forward and extend \
it, do not overwrite the history with only the latest thing they said.

`contradictions_noticed` is cumulative too. Add a new entry only when a player \
says something that genuinely conflicts with something they said earlier -- not \
merely something you disagree with. Never drop an entry you have already made. \
Quote both sides in the player's own words: `earlier` and `now` are shown on \
screen as quotations, so a paraphrase there puts words in a player's mouth.

Two hard rules:

- Never say or assume a player is dead unless they appear in the dead list you \
are given. Players who are still speaking are alive.
- `reasoning` is at most two sentences and under 200 characters. It sits in a \
narrow panel beside the transcript. Say what changed and why; leave out the \
recap of what everyone did."""


def _fmt_state(state: BeliefState, alive: List[str]) -> str:
    if state.event_index < 0:
        return "(nothing yet -- this is the first line of the game)"

    lines = ["Suspicion:"]
    for p in alive:
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


def render(
    event: GameEvent,
    state: BeliefState,
    setup: GameSetup,
    alive: Optional[List[str]] = None,
    votes: Optional[Dict[int, Dict[str, str]]] = None,
) -> str:
    alive = alive if alive is not None else list(setup.players)
    dead = [p for p in setup.players if p not in alive]
    roster = ", ".join(alive)
    if dead:
        roster += f"   (out of the game, cannot be suspected: {', '.join(dead)})"

    count = setup.deceiver_count
    who = setup.deceivers_phrase()
    if count > 1:
        who += ", who know each other and are working together,"

    return f"""GAME
{setup.premise}
There {'is' if count == 1 else 'are'} {who} among these players.
Players still alive: {roster}

THE VOTE RECORD SO FAR (public, and exact -- you do not have to remember it)
{format_votes(votes or {}, alive)}

YOUR BELIEF STATE, BEFORE THIS LINE
{_fmt_state(state, alive)}

THE NEXT LINE
Round {event.round}, {event.phase} -- {event.speaker}: "{event.statement}"

Update your belief state to account for this line. Carry forward everything that still holds."""


def _normalize(values: Dict[str, float], target: float = 1.0) -> Dict[str, float]:
    """Scale so the values sum to `target`.

    `target` is the number of liars still in the game, not 1.0, and that
    distinction turned out to matter enormously. With two mafia and a total of
    1.0, the two guilty players are forced to compete for one pool of suspicion:
    raising one mathematically requires lowering the other, so the observer can
    never say "it's these two" however strongly it believes it. It picks one and
    suppresses its partner. That is a property of the arithmetic, not of the
    reasoning.

    Summing to the number of remaining liars makes each value an honest
    P(this player is lying): two players at 0.5 each means "one of these two,
    probably, and I can't split them", which is exactly the thing we want it to
    be able to say.
    """
    total = sum(values.values())
    if total <= 0:
        share = target / len(values) if values else 0.0
        return {p: share for p in values}
    return {p: v * target / total for p, v in values.items()}


def _settle(
    raw: List,
    prior: Dict[str, float],
    alive: List[str],
    first_event: bool,
    remaining_liars: float = 1.0,
) -> Dict[str, float]:
    """Turn the model's numbers into a distribution we can plot.

    Three jobs: fill in players the model forgot, hold per-event movement to
    MAX_DELTA_PER_EVENT, and make the living players sum to 1.0.

    The order matters and used to be wrong. Clamping before normalizing does not
    bound anything, because dividing by the total moves every value again -- an
    early run clamped a player to 0.46 and printed 0.418, a 34-point drop under a
    30-point cap. So: normalize first, then clamp, then hand the leftover
    probability to players who still have room, and repeat until it settles.
    """
    if not alive:
        return {}
    target = max(0.0, min(float(remaining_liars), float(len(alive))))
    uniform = target / len(alive)

    proposed = {e.player: max(0.0, min(1.0, e.score)) for e in raw if e.player in alive}
    # A player the model omitted keeps its previous value rather than silently
    # falling to zero -- omission is not exoneration.
    current = _normalize({p: proposed.get(p, prior.get(p, uniform)) for p in alive}, target)

    if first_event:
        return current

    floor = {p: max(0.0, prior.get(p, uniform) - MAX_DELTA_PER_EVENT) for p in alive}
    ceiling = {p: min(1.0, prior.get(p, uniform) + MAX_DELTA_PER_EVENT) for p in alive}

    for _ in range(20):
        clamped = {p: max(floor[p], min(ceiling[p], v)) for p, v in current.items()}
        residual = target - sum(clamped.values())
        if abs(residual) < 1e-9:
            return clamped

        # Push the leftover onto whoever is not already pinned at a bound.
        room = {
            p: (ceiling[p] - clamped[p]) if residual > 0 else (clamped[p] - floor[p])
            for p in alive
        }
        available = sum(room.values())
        if available < 1e-9:
            # Every player is pinned. The bars have to total 100%, so the sum
            # wins and the cap gives way. Only reachable if the model returns
            # something wild on a very small roster.
            return _normalize(clamped, target)
        current = {p: clamped[p] + residual * (room[p] / available) for p in alive}

    return _normalize(current, target)


def _normalize_quote(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", text.lower()).strip()


def _said_by(quote: str, statements: List[str]) -> bool:
    """Did this player actually say something like this?

    Loose on purpose -- the model paraphrases lightly, trims, or merges two
    sentences -- but it must be recognisably one of *their* lines, not someone
    else's.
    """
    needle = _normalize_quote(quote)
    if not needle:
        return False
    for said in statements:
        hay = _normalize_quote(said)
        if needle in hay or hay in needle:
            return True
        # A short prefix match covers trimmed quotes without matching everything.
        if len(needle) >= 20 and needle[:20] in hay:
            return True
    return False


def _merge_contradictions(
    prior: List[Contradiction],
    fresh: List[Contradiction],
    said: Optional[Dict[str, List[str]]] = None,
) -> List[Contradiction]:
    """Append-only, and only for quotes the accused player actually said.

    Append-only because the model is told never to drop an entry and we enforce
    it rather than trust it.

    The attribution check exists because of a real failure: on a Mafia game the
    model reported Sidney contradicting themselves, quoting Sidney's line as
    `earlier` and *Whitney's* line as `now`. Two speakers, filed as one player
    caught lying. That goes straight onto the screen as a quotation, so a
    contradiction whose halves can't both be traced to the accused is dropped.
    """
    seen = {(c.player, c.earlier, c.now) for c in prior}
    merged = list(prior)
    for c in fresh:
        key = (c.player, c.earlier, c.now)
        if key in seen:
            continue
        if said is not None:
            statements = said.get(c.player, [])
            if not (_said_by(c.earlier, statements) and _said_by(c.now, statements)):
                continue  # misattributed -- someone else said one of these
        seen.add(key)
        merged.append(c)
    return merged


class Observer:
    """Stateful across one game. Construct once, feed events in order."""

    def __init__(self, setup: GameSetup, llm: Optional[StructuredLLM] = None):
        self.setup = setup
        self.llm = llm or get_backend()
        # Before anyone speaks, every player is equally likely, and the
        # numbers sum to the number of liars -- not to 1.
        uniform = setup.deceiver_count / len(setup.players)
        self.state = BeliefState(suspicion={p: uniform for p in setup.players})
        self.history: List[Dict[str, float]] = []
        # Every state, not just its numbers. `history` is enough to plot a chart;
        # replaying a run into the UI needs the reasoning and contradictions too.
        self.states: List[BeliefState] = []
        # Everything each player has actually said, so a claimed
        # contradiction can be checked against their own words.
        self.said: Dict[str, List[str]] = {}
        # round -> {voter: target}, from the narrator's announcements only.
        self.votes: Dict[int, Dict[str, str]] = {}
        # Roles announced publicly when a player is eliminated. Public
        # knowledge -- every player at the table hears it.
        self.revealed: Dict[str, str] = {}

    def _liars_left(self, eliminated: List[str]) -> int:
        """Liars still in the game, given this list of the dead.

        Takes the eliminated list explicitly because the event that reveals a
        liar must count that reveal immediately -- reading it off the state
        would use the previous turn's dead list and be one event stale.

        Floors at 1: if every liar has been caught the game is already over
        and the numbers mean nothing, but a target of 0 would blank the chart.
        """
        caught = sum(
            1
            for player, role in self.revealed.items()
            if role.lower() in DECEIVER_ROLES and player in eliminated
        )
        return max(1, self.setup.deceiver_count - caught)

    @property
    def liars_remaining(self) -> int:
        """How many of the lying team are still in the game.

        Derived only from what the table knows: the announced total, minus
        any eliminated player whose revealed role was a liar. Never from
        ground truth.
        """
        return self._liars_left(self.state.eliminated)

    @property
    def alive(self) -> List[str]:
        """Living players, in the roster's original order -- so the UI can keep
        rows in a fixed position instead of resorting them every event."""
        return [p for p in self.setup.players if p not in self.state.eliminated]

    def restore(
        self,
        state: BeliefState,
        history: List[Dict[str, float]],
        states: Optional[List[BeliefState]] = None,
        observed: Optional[List[GameEvent]] = None,
    ) -> None:
        """Pick up where a previous run stopped. See `run_observer.py --resume`.

        `observed` is the events already processed. Without them the record of
        who said what starts empty, and the contradiction attribution check then
        rejects every quote from before the restart as misattributed -- a resumed
        run would quietly stop catching lies told in round 1.
        """
        self.state = state
        self.history = list(history)
        self.states = list(states) if states else []
        self.said = {}
        self.votes = {}
        for event in observed or []:
            if event.speaker in self.setup.players:
                self.said.setdefault(event.speaker, []).append(event.statement)
            vote = infer_vote(event)
            if vote and vote[0] in self.setup.players and vote[1] in self.setup.players:
                self.votes.setdefault(event.round, {})[vote[0]] = vote[1]

    def observe(self, event: GameEvent) -> BeliefState:
        """Fold one event into the belief state and return the new one."""
        if event.speaker in self.setup.players:
            self.said.setdefault(event.speaker, []).append(event.statement)
        vote = infer_vote(event)
        if vote and vote[0] in self.setup.players and vote[1] in self.setup.players:
            self.votes.setdefault(event.round, {})[vote[0]] = vote[1]
        gone = infer_elimination(event)
        reveal = _ROLE_REVEAL.search(event.statement)
        if gone in self.setup.players and reveal:
            self.revealed[gone] = reveal.group('role')
        prior = self.state
        first = prior.event_index < 0

        out: ObserverOutput = self.llm.structured(
            system=SYSTEM,
            user=render(event, prior, self.setup, self.alive, self.votes),
            schema=ObserverOutput,
        )

        claims = dict(prior.claims_tracked)
        for entry in out.claims_tracked:
            claims[entry.player] = entry.claim

        # A death announced by this event takes effect immediately: the victim
        # leaves the bars on the same line that kills them.
        eliminated = list(prior.eliminated)
        victim = infer_elimination(event)
        if victim in self.setup.players and victim not in eliminated:
            eliminated.append(victim)
        alive_after = [p for p in self.setup.players if p not in eliminated]

        self.state = BeliefState(
            round=event.round,
            phase=event.phase,
            event_index=prior.event_index + 1,
            eliminated=eliminated,
            suspicion=_settle(
                out.suspicion, prior.suspicion, alive_after, first, self._liars_left(eliminated)
            ),
            claims_tracked=claims,
            contradictions_noticed=_merge_contradictions(
                prior.contradictions_noticed, out.contradictions_noticed, self.said
            ),
            reasoning=out.reasoning,
        )
        self.history.append(dict(self.state.suspicion))
        self.states.append(self.state)
        return self.state
