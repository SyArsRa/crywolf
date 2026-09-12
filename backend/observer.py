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

from backend.evidence import format_evidence, marginals
from backend.llm import StructuredLLM, get_backend
from backend.schema import (
    DECEIVER_ROLES,
    NARRATORS,
    BeliefState,
    Contradiction,
    GameEvent,
    Deliberation,
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

# The cap that applies before any structural evidence exists -- that is, before
# the first eliminated player's role has been announced.
#
# This is the fix for the observer's worst habit. The first reveal lands a median
# 43% of the way through a transcript; until then the game contains no evidence
# at all, only chat, much of it "Hi" and "hello all". The old loop spent that
# entire stretch at full swing, committed hard to whoever talked loudest, and
# then had to spend the rest of the game climbing back out -- and the symmetric
# cap that lets it climb out is also what let it climb in.
#
# So movement is throttled, not frozen. A genuine contradiction still registers;
# a confident tone no longer does. An observer that says "I do not know yet" for
# the first 40% and then moves hard on real evidence beats one that has been
# wrong since line three and is defending it.
#
# Note this is emphatically NOT the asymmetric cap rejected in the note above:
# it is symmetric, and it is keyed to whether evidence exists rather than to
# which direction the number is moving.
PRE_EVIDENCE_MAX_DELTA = 0.06

# And an absolute bound, as a multiple of each player's starting share, for as
# long as no role has been announced.
#
# The per-event cap alone turned out not to be enough, which a live run showed
# plainly: throttled to 0.06 a line, the observer still walked a bystander from
# 0.25 to 0.60 over the twenty evidence-free events before the first reveal. A
# per-event limit bounds the speed of a commitment, not its size, and it is the
# size that ends up on screen and in the verdict.
#
# So during the opening no player may exceed twice their opening share -- 0.50
# where eight players hold two liars between them. Expressed as a multiple
# rather than a constant because the opening share depends on the roster: 0.25
# each in that game, 0.20 in a five-player one-wolf game.
#
# It still leaves room to lean: twice the field is a clear accusation, and a
# genuine round-one contradiction reads as one. What it forbids is arriving at
# near-certainty on an evening of chat that contains no evidence.
PRE_EVIDENCE_CEILING_MULTIPLE = 2.0

# When the opening stops being buffered and starts being read line by line.
#
# Just the opening, once (see `Observer._flush_opening`); from there every line
# gets its own call. An elimination, or evidence arriving, closes it early.
#
# Counted in lines worth reading rather than in events, because an event is a bad
# unit here: `needs_model` already drops narrator vote lines and anything under
# four words, and a typical opening is greetings. Measured over the 33 games, a
# flat five-event buffer usually held exactly one substantive line -- it ended the
# opening before there was anything in it to read, and saved 0.9% of calls doing
# so. Five substantive lines lands at a median event 11 (range 6-19).
#
# The event cap is the backstop, not the trigger. It was once the trigger, at 40,
# and that broke the live run: a buffered event carries its prior forward
# unchanged, so nothing moved for a minute and the run looked hung.
OPENING_MIN_LINES = 5
OPENING_MAX_EVENTS = 20

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
    """The vote record, laid out in full. It is the raw material the computed
    evidence in `evidence.py` is derived from, so the model can check that
    derivation against the votes themselves.

    The "never voted at each other" line below is reported but DEMOTED, and the
    prompt now says so. It reads as the strongest signal here -- 3.8% of a
    liar's votes land on their partner against 15-20% at random -- and that
    framing was wrong. Scoring candidate pairs by it alone lands at 0.03 top-1
    accuracy, *below* the 0.37 chance rate, and adding it to the reveal signal
    drags that signal from 0.67 down to 0.48.

    The reason is worth keeping so nobody reinstates it: almost no pair ever
    votes internally, so the test barely separates one candidate pair from
    another, and what it actually rewards is pairs who cast few votes. It is a
    quiet-player detector wearing a coalition-detector's clothes. It stays on
    screen as context; it is not evidence.
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
        lines.append(
            "  (Context only, NOT evidence. There are usually a dozen or more such"
            " pairs and the true pair is only sometimes among them, so it barely"
            " narrows anything. Do not build a theory on this line.)"
        )
    return "\n".join(lines)


# A player line this short carries no reasoning to read: "Hi", "why", "yes",
# "I agree". Measured across the 33 games, these are 37% of all events.
TRIVIAL_WORD_COUNT = 3

_WORD = re.compile(r"[a-z0-9']+")


def needs_model(event: GameEvent) -> bool:
    """Is this line worth a model call?

    Measured over the 33 games, 55% of events are not:

      * 18% are the narrator announcing a vote. `infer_vote` already parses
        those exactly, and they reach the model anyway inside the vote table and
        the computed evidence on the next line that does get a call. Paying a
        model to read "Kai voted for Sutton" buys a restatement of something we
        parsed perfectly.
      * 37% are player lines of three words or fewer.

    What is deliberately NOT skipped: any other narrator line. Deaths, phase
    changes and role announcements all change the shape of the game, and a role
    announcement is what triggers the deliberation call.

    Skipping is not the same as ignoring. A skipped event still updates the vote
    record, the claims ledger, the dead list and the evidence, and still emits
    its own belief state -- one history entry per event, exactly as before, so
    nothing downstream can tell the difference except the bill.
    """
    if event.speaker.lower() in NARRATORS:
        return infer_vote(event) is None
    return len(_WORD.findall(event.statement.lower())) > TRIVIAL_WORD_COUNT


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

1. THE COMPUTED EVIDENCE IS YOUR BEST EVIDENCE. Each turn you are handed a \
block headed COMPUTED EVIDENCE holding a probability per player, worked out \
from the vote record by arithmetic rather than by anyone's judgement. Here is \
what it is: whenever a player is voted out their role is announced to the whole \
table, and that tells you retroactively whether each vote cast that round was \
aimed at a liar or at an innocent. Liars know who their teammates are and steer \
away from them; innocents are guessing, and hit a liar far more often. Measured \
over real games of this kind, a vote cast by a liar lands on a player later \
revealed to be a liar 2% of the time, against 26% for a vote cast by an \
innocent. Start from those numbers and move off them only for a specific reason \
you can name.

Two things it cannot do, which is where you earn your keep. It is silent until \
the first role is announced, usually about halfway through the game -- when it \
says it has nothing, the game genuinely holds no evidence yet, and you should \
stay close to even rather than invent a suspect. And it is right roughly half \
the time, not always, so a player it likes who is caught in a flat \
contradiction is still caught.

2. DO NOT reason from "these two have never voted for each other". You will see \
such pairs listed; they are context, not evidence. There are usually more than \
a dozen of them and the true pair is only sometimes among them, so the line \
narrows almost nothing -- and an observer that leans on it measurably does \
worse than one that ignores it.

3. Watch what a vote costs the voter. Voting for a player nobody else suspects, \
or switching targets the moment a partner comes under pressure, or piling onto \
whoever is already losing -- these move suspicion. A vote is an action; talk is \
free.

4. The liars know who each other are. Watch for someone arguing from knowledge \
they should not have, or being oddly incurious about a player they would \
otherwise suspect.

5. Time matters. Liars tend to say less early and more once the game has shape. \
Someone who led the first round loudly and then went quiet is more often \
innocent than guilty.

6. DO NOT treat aggression, confidence or baseless accusation as guilt. This is \
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
    evidence: Optional[Dict[str, float]] = None,
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

COMPUTED EVIDENCE (arithmetic over the graded votes, not an opinion)
{format_evidence(evidence or {}, setup.deceiver_role)}

THE VOTE RECORD SO FAR (public, and exact -- you do not have to remember it)
{format_votes(votes or {}, alive)}

YOUR BELIEF STATE, BEFORE THIS LINE
{_fmt_state(state, alive)}

THE NEXT LINE
Round {event.round}, {event.phase} -- {event.speaker}: "{event.statement}"

Update your belief state to account for this line. Carry forward everything that still holds."""


def render_opening(
    events: List[GameEvent],
    setup: GameSetup,
    alive: List[str],
    votes: Optional[Dict[int, Dict[str, str]]] = None,
    evidence: Optional[Dict[str, float]] = None,
) -> str:
    """The whole evidence-free opening, as one prompt.

    The per-line renderer shows one statement against a running belief. This
    shows the opening entire, because that is the only way it is worth reading:
    nothing in it can be checked against anything, so the only signal available
    is the shape of the conversation -- who steered, who answered, who never
    committed to anything -- and that is invisible one line at a time.
    """
    count = setup.deceiver_count
    who = setup.deceivers_phrase()
    if count > 1:
        who += ", who know each other and are working together,"

    lines = "\n".join(
        f"  [{i}] {e.speaker} (R{e.round} {e.phase}): {e.statement}"
        for i, e in enumerate(events, 1)
    )

    return f"""GAME
{setup.premise}
There {'is' if count == 1 else 'are'} {who} among these players.
Players still alive: {', '.join(alive)}

COMPUTED EVIDENCE (arithmetic over the graded votes, not an opinion)
{format_evidence(evidence or {}, setup.deceiver_role)}

THE VOTE RECORD SO FAR (public, and exact -- you do not have to remember it)
{format_votes(votes or {}, alive)}

THE OPENING, IN FULL
Every line of the game so far, in order. This is the first time you are reading
any of it, and you are reading all of it at once rather than one line at a time.

{lines}

Form your first belief state from the whole of this, not from its last line.

You have no graded evidence yet -- nobody's role has been checked against
anything -- so what you have is conversational shape: who pushed a name and who
followed, who answered a direct question and who deflected it, who committed to
a position they could later be held to. Record what each player has claimed, in
`claims_tracked`, because that ledger is what a later contradiction gets checked
against. Flag a contradiction only if someone has already contradicted themselves
within these lines, quoting both halves verbatim.

Stay close to even. An opening of pure conversation is not grounds for
confidence, and a number you cannot justify now is one you will spend the rest of
the game defending. Lean where someone has genuinely earned it; do not
manufacture a suspect because the field looks flat."""


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
    cap: float = MAX_DELTA_PER_EVENT,
    ceiling: Optional[float] = None,
) -> Dict[str, float]:
    """Turn the model's numbers into a distribution we can plot.

    Three jobs: fill in players the model forgot, hold per-event movement to
    `cap`, and make the living players sum to `remaining_liars`.

    `cap` is MAX_DELTA_PER_EVENT once the game has produced structural evidence
    and the much tighter PRE_EVIDENCE_MAX_DELTA before then -- see the comment on
    that constant for why the evidence-free opening needs a different cap.

    `ceiling`, when given, is an absolute bound no player may pass, applied on
    top of the per-event cap. The opening uses it because a per-event limit
    bounds how fast a commitment forms and not how large it gets, and twenty
    small steps in one direction is still a commitment.

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

    if first_event and ceiling is None:
        return current

    hard = 1.0 if ceiling is None else max(uniform, min(1.0, ceiling))
    if first_event:
        # Nothing to move away from yet, so only the absolute bound applies.
        # It still goes through the loop below rather than being clamped and
        # renormalized in one shot: dividing by the total moves every value
        # again, so a one-shot clamp does not actually bound anything. That is
        # the same trap documented above for the per-event cap.
        floor = {p: 0.0 for p in alive}
        ceiling = {p: hard for p in alive}
    else:
        floor = {p: max(0.0, prior.get(p, uniform) - cap) for p in alive}
        ceiling = {p: min(hard, prior.get(p, uniform) + cap) for p in alive}

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


DELIBERATION_SYSTEM = """You are the same observer, but this is not a line-by-line update. A round has just ended and a player's role has been announced, which is the only moment in this game when genuinely new evidence arrives: every vote cast this round has just been graded. You are being given the whole round at once, verbatim, instead of one line at a time.

Take the opportunity the per-line pass does not have. Read the round as a whole and ask what the graded votes mean about each living player.

You are explicitly permitted -- expected, when warranted -- to revise wholesale. If you have been suspecting someone for several rounds and this round's evidence does not support it, say so plainly in `revised` and move the number. "I was wrong about Whitney; she voted for the player just revealed as mafia, which a mafia member almost never does" is exactly the kind of sentence this call exists to produce. The one thing you must not do is drop a suspicion silently: if a number falls a long way, `revised` has to say why.

The COMPUTED EVIDENCE block is arithmetic over the graded votes and is your strongest input. Start from it. Depart from it only where you can name the specific thing that outweighs it.

Do not treat volume, confidence or aggression as guilt. Frightened innocents accuse constantly and at random; measured over real games they do it as often as the liars do.

Return a suspicion number for every living player. They should sum to roughly the number of liars still in the game, so each one reads as P(this player is lying) -- two players at 0.5 means "one of these two, and I cannot split them", which is a legitimate and useful thing to say."""


def render_deliberation(
    round_events: List[GameEvent],
    state: BeliefState,
    setup: GameSetup,
    alive: List[str],
    votes: Dict[int, Dict[str, str]],
    evidence: Dict[str, float],
    liars_remaining: int,
) -> str:
    """The whole round verbatim, plus everything computed about it."""
    transcript = "\n".join(
        f"  R{e.round} {e.phase} {e.speaker}: {e.statement}" for e in round_events
    ) or "  (no lines recorded for this round)"

    return f"""GAME
{setup.premise}
There are {setup.deceivers_phrase()} among these players.
Players still alive: {', '.join(alive)}
Still on the lying team, by public reveals alone: {liars_remaining}

COMPUTED EVIDENCE (arithmetic over the graded votes, not an opinion)
{format_evidence(evidence, setup.deceiver_role)}

THE VOTE RECORD SO FAR (public, and exact)
{format_votes(votes, alive)}

THE ROUND THAT JUST ENDED, IN FULL
{transcript}

YOUR BELIEF STATE GOING INTO THIS
{_fmt_state(state, alive)}

A role has just been announced, so this round's votes are now graded. Reconsider
the whole game in that light. Revise as far as the evidence takes you, and say in
`revised` what you changed and why."""


class Observer:
    """Stateful across one game. Construct once, feed events in order."""

    def __init__(
        self,
        setup: GameSetup,
        llm: Optional[StructuredLLM] = None,
        deliberate: bool = True,
        skip_trivial: bool = True,
        batch_opening: bool = True,
    ):
        self.setup = setup
        self.llm = llm or get_backend()
        # The second tier. On by default; `run_observer --no-deliberate` turns it
        # off so a run can be priced against the per-event loop alone.
        self.deliberate = deliberate
        # Skip the model on lines that carry nothing to read. See `needs_model`.
        # `--all-events` turns it off to compare like for like.
        self.skip_trivial = skip_trivial
        # Model calls actually made, against len(history), so a run can report
        # what it cost rather than what it would have cost.
        self.calls = 0
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
        # Every line of the round in progress, kept verbatim so the deliberation
        # call can read the round whole instead of through its own summary.
        self.round_log: List[GameEvent] = []
        # Deep calls made, for pricing a run against the per-event loop.
        self.deliberations = 0

        # The opening, read in one call instead of fourteen. See `_flush_opening`.
        self.batch_opening = batch_opening
        self.opening: List[GameEvent] = []
        self.opening_read = not batch_opening
        self.openings = 0

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

    def evidence(self, eliminated: Optional[List[str]] = None) -> Dict[str, float]:
        """P(liar) per living player from the vote record alone. `{}` if the game
        has not graded a single vote yet -- which is not the same as "all clear",
        and `format_evidence` says so in as many words."""
        dead = self.state.eliminated if eliminated is None else eliminated
        alive = [p for p in self.setup.players if p not in dead]
        return marginals(self.votes, self.revealed, alive, self._liars_left(dead))

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

        The announced roles are rebuilt here for the same reason and it matters
        just as much: they are what grades the votes, so a resumed run that
        dropped them would compute no evidence at all and fall back to reading
        tone -- the exact failure the evidence exists to prevent, and invisible
        on screen because the bars would still move.
        """
        self.state = state
        self.history = list(history)
        self.states = list(states) if states else []
        # The restored state is already somebody's opinion of the opening, so the
        # opening is over however few events were observed. Without this a resumed
        # run re-enters the buffer, stops paying per line, and -- if nobody dies
        # before the transcript ends -- reads the rest of the game in one call it
        # never gets around to making.
        self.opening = []
        self.opening_read = True
        self.said = {}
        self.votes = {}
        self.revealed = {}
        self.round_log = []
        for event in observed or []:
            if event.speaker in self.setup.players:
                self.said.setdefault(event.speaker, []).append(event.statement)
            vote = infer_vote(event)
            if vote and vote[0] in self.setup.players and vote[1] in self.setup.players:
                self.votes.setdefault(event.round, {})[vote[0]] = vote[1]
            gone = infer_elimination(event)
            reveal = _ROLE_REVEAL.search(event.statement)
            if gone in self.setup.players and reveal:
                self.revealed[gone] = reveal.group("role")
            if self.round_log and self.round_log[-1].round != event.round:
                self.round_log = []
            self.round_log.append(event)

    def observe(self, event: GameEvent) -> BeliefState:
        """Fold one event into the belief state and return the new one."""
        if event.speaker in self.setup.players:
            self.said.setdefault(event.speaker, []).append(event.statement)
        vote = infer_vote(event)
        if vote and vote[0] in self.setup.players and vote[1] in self.setup.players:
            self.votes.setdefault(event.round, {})[vote[0]] = vote[1]
        gone = infer_elimination(event)
        reveal = _ROLE_REVEAL.search(event.statement)
        # A reveal is the only moment new structural evidence enters the game,
        # and it must be recorded before the evidence is computed for this event
        # -- reading it a line later would grade this round's votes a turn late.
        announced_role = bool(gone in self.setup.players and reveal)
        if announced_role:
            self.revealed[gone] = reveal.group('role')

        # A round boundary closes the buffer the deliberation call reads from.
        if self.round_log and self.round_log[-1].round != event.round:
            self.round_log = []
        self.round_log.append(event)

        prior = self.state
        first = prior.event_index < 0
        evidence = self.evidence()

        # The opening is buffered and read in one call rather than one per line.
        # Until the game eliminates somebody there is no structural evidence in
        # it at all, and `PRE_EVIDENCE_MAX_DELTA` caps belief movement at six
        # points across the whole stretch -- so a per-line call there buys a
        # capped nudge at full price. Read whole it is strictly more informative
        # (the model sees the round's shape, not one line of it) and costs one
        # call instead of a median of fourteen.
        if not self.opening_read:
            self.opening.append(event)
            # One batch, at the start, and then never again: the opening is read
            # whole because its lines only mean anything against each other, and
            # from there the game is read line by line. Chunking the whole
            # evidence-free stretch this way saves more calls, but it costs the
            # per-event granularity everything downstream is built on -- the
            # chart, the deltas, the live grade all go still between chunks.
            if evidence or self._opening_ends(event):
                self._flush_opening(prior, announced_role, evidence)
                self.opening = []
                self.opening_read = True
            else:
                self._carry_forward(event, prior, announced_role)
                # Say so on screen. A buffered event carries its prior forward
                # unchanged, so the bars do not move and the panel would
                # otherwise sit empty -- which reads as a hung run rather than as
                # an observer that has not spoken yet.
                self.state = self.state.model_copy(
                    update={
                        "reasoning": (
                            f"Reading the opening — {len(self.opening)} "
                            f"line{'' if len(self.opening) == 1 else 's'} so far, held for a "
                            f"single pass. No evidence has been graded yet."
                        )
                    }
                )
                if self.states:
                    self.states[-1] = self.state
            return self.state

        # A skipped line still moves the game forward -- the vote record, the
        # dead list and the evidence above have all been updated already. What
        # it does not do is pay a model to have no opinion about "Hi".
        if self.skip_trivial and not needs_model(event):
            self._carry_forward(event, prior, announced_role)
            return self.state

        self.calls += 1
        out: ObserverOutput = self.llm.structured(
            system=SYSTEM,
            user=render(event, prior, self.setup, self.alive, self.votes, evidence),
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
                out.suspicion,
                prior.suspicion,
                alive_after,
                first,
                self._liars_left(eliminated),
                # Before the game has graded a single vote there is nothing to
                # be confident about, so the per-line pass is held on a short
                # leash. See PRE_EVIDENCE_MAX_DELTA.
                MAX_DELTA_PER_EVENT if evidence else PRE_EVIDENCE_MAX_DELTA,
                None
                if evidence
                else PRE_EVIDENCE_CEILING_MULTIPLE
                * self._liars_left(eliminated)
                / max(1, len(alive_after)),
            ),
            claims_tracked=claims,
            contradictions_noticed=_merge_contradictions(
                prior.contradictions_noticed, out.contradictions_noticed, self.said
            ),
            reasoning=out.reasoning,
        )

        # The deep pass, on the one event that justifies it.
        if self.deliberate and announced_role:
            self._deliberate(alive_after, eliminated)

        self.history.append(dict(self.state.suspicion))
        self.states.append(self.state)
        return self.state

    def flush(self) -> None:
        """Read a still-buffered opening because the game has ended.

        Normally the buffer closes on an elimination, or on OPENING_MAX_EVENTS.
        A game that does neither would otherwise end with the opening unread and
        the observer holding no opinion at all -- every buffered event carried
        forward, no call ever made, uniform suspicion on screen. No game in
        `data/` reaches that today, but `adapters/werewolf_among_us.py` builds
        games where nobody is ever eliminated, so it is one dataset away.
        """
        if self.opening_read or not self.opening:
            return
        self.opening_read = True
        alive = self.alive
        try:
            out: ObserverOutput = self.llm.structured(
                system=SYSTEM,
                user=render_opening(self.opening, self.setup, alive, self.votes, self.evidence()),
                schema=ObserverOutput,
            )
            self.calls += 1
            self.openings += 1
        except Exception:
            return

        claims = dict(self.state.claims_tracked)
        for entry in out.claims_tracked:
            claims[entry.player] = entry.claim
        settled = _settle(
            out.suspicion,
            self.state.suspicion,
            alive,
            first_event=True,
            remaining_liars=self._liars_left(self.state.eliminated),
        )
        # Rewritten in place, the way `_deliberate` does it: the last event
        # already emitted a carried-forward state, and appending a second one
        # would break the one-entry-per-event contract `history` owes the scorer
        # and the UI. This is the same moment in the game, read properly.
        self.state = self.state.model_copy(
            update={
                "suspicion": settled,
                "claims_tracked": claims,
                "contradictions_noticed": _merge_contradictions(
                    self.state.contradictions_noticed, out.contradictions_noticed, self.said
                ),
                "reasoning": out.reasoning,
            }
        )
        if self.history:
            self.history[-1] = dict(settled)
        if self.states:
            self.states[-1] = self.state

    def _opening_ends(self, event: GameEvent) -> bool:
        """Has the buffer got enough in it to be worth a call?

        Three ways to be done, in order of what they mean:

        * somebody was eliminated -- the game just said something checkable, and
          whatever is buffered should be read against it;
        * `OPENING_MIN_LINES` players have said something substantive -- there is
          now material to read, which is the whole point of holding it back;
        * `OPENING_MAX_EVENTS` events have gone by regardless -- a backstop, both
          so the prompt cannot grow without bound and so a quiet table cannot
          leave the observer with no opinion for very long.
        """
        if infer_elimination(event) in self.setup.players:
            return True
        if len(self.opening) >= OPENING_MAX_EVENTS:
            return True
        lines = sum(
            1
            for e in self.opening
            if e.speaker in self.setup.players and needs_model(e)
        )
        return lines >= OPENING_MIN_LINES

    def _flush_opening(
        self, prior: BeliefState, announced_role: bool, evidence: Dict[str, float]
    ) -> None:
        """Read the whole buffered opening in a single call.

        Emits one belief state, for the event that closed the buffer. The earlier
        buffered events already emitted their own carried-forward states as they
        arrived, so `history` still holds exactly one entry per event and the
        replay in the UI is unchanged in shape.

        A failure here is swallowed the same way `_deliberate`'s is: the carried
        forward state is already valid, and an opening nobody could read is not
        worth losing the rest of the game over.
        """
        event = self.opening[-1]
        alive_before = self.alive

        try:
            out: ObserverOutput = self.llm.structured(
                system=SYSTEM,
                user=render_opening(self.opening, self.setup, alive_before, self.votes, evidence),
                schema=ObserverOutput,
            )
            self.calls += 1
            self.openings += 1
        except Exception:
            self._carry_forward(event, prior, announced_role)
            return

        claims = dict(prior.claims_tracked)
        for entry in out.claims_tracked:
            claims[entry.player] = entry.claim

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
            # `first_event=True`: this is the observer's first opinion of the
            # game, so there is no prior movement for a per-event cap to bound.
            # The opening ceiling still applies -- a single call that has read
            # only chat must not come out of it certain.
            suspicion=_settle(
                out.suspicion,
                prior.suspicion,
                alive_after,
                first_event=True,
                remaining_liars=self._liars_left(eliminated),
                # Exactly the rule the per-line pass uses, and usually a no-op
                # here: the buffer is flushed on an elimination, and that event's
                # reveal was recorded before `evidence` was computed, so by now
                # the first votes have been graded. It bites only when the
                # backstop fires -- a game still talking after OPENING_MAX_EVENTS
                # with nobody dead, where certainty really would be unearned.
                # `first_event` waives the per-event movement cap alone, which
                # has nothing to bound on a first opinion.
                ceiling=(
                    None
                    if evidence
                    else PRE_EVIDENCE_CEILING_MULTIPLE
                    * self._liars_left(eliminated)
                    / max(1, len(alive_after))
                ),
            ),
            claims_tracked=claims,
            contradictions_noticed=_merge_contradictions(
                prior.contradictions_noticed, out.contradictions_noticed, self.said
            ),
            reasoning=out.reasoning,
        )

        if self.deliberate and announced_role:
            self._deliberate(alive_after, eliminated)

        self.history.append(dict(self.state.suspicion))
        self.states.append(self.state)

    def _carry_forward(self, event: GameEvent, prior: BeliefState, announced_role: bool) -> None:
        """Advance the state over an event we did not pay to read.

        Suspicion is unchanged, which is the honest answer: nothing was said.
        Everything else -- the round, the phase, the event index, the dead --
        moves exactly as it would have, so `history` and `states` stay one entry
        per event and Lane A's replay is byte-identical in shape.

        A role announcement is never skipped, but this stays correct if one ever
        were: the deep call still fires.
        """
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
            # Renormalised only because a death may have left the distribution;
            # the living players keep their relative standing.
            suspicion=_normalize(
                {p: prior.suspicion.get(p, 0.0) for p in alive_after},
                self._liars_left(eliminated),
            ),
            claims_tracked=dict(prior.claims_tracked),
            contradictions_noticed=list(prior.contradictions_noticed),
            reasoning=prior.reasoning,
        )
        if self.deliberate and announced_role:
            self._deliberate(alive_after, eliminated)
        self.history.append(dict(self.state.suspicion))
        self.states.append(self.state)

    def _deliberate(self, alive: List[str], eliminated: List[str]) -> None:
        """Reconsider the whole round now that its votes have been graded.

        Overwrites `self.state.suspicion` in place rather than appending: the
        contract with Lane A and the scorer is one history entry per event, and
        a deep call is not an event. It is the same moment in the game, read
        better.

        `MAX_DELTA_PER_EVENT` deliberately does not apply. The cap exists to stop
        a single throwaway line swinging the chart; this call has just read an
        entire round against graded evidence, which is precisely the situation
        the observer should be allowed to change its mind wholesale in. Capping
        it here would reintroduce the stickiness the two-tier design is meant to
        cure.

        A failure here is swallowed, and the whole body is inside the try for
        that reason -- not just the network call. A backend that returns the
        wrong shape is as much a failure as a timeout, and it used to crash the
        run from outside the guard. The deep call is an improvement on top of a
        belief state that is already valid; losing a game to an optional call is
        a poor trade however the call went wrong.
        """
        if not alive:
            return
        try:
            out = self.llm.structured(
                system=DELIBERATION_SYSTEM,
                user=render_deliberation(
                    self.round_log,
                    self.state,
                    self.setup,
                    alive,
                    self.votes,
                    self.evidence(eliminated),
                    self._liars_left(eliminated),
                ),
                schema=Deliberation,
            )
            if not isinstance(out, Deliberation):
                return
            settled = _settle(
                out.suspicion,
                self.state.suspicion,
                alive,
                first_event=True,  # no cap: see the docstring
                remaining_liars=self._liars_left(eliminated),
            )
            # `revised` is kept in the reasoning the UI shows, because an
            # observer that abandons a read without saying so is the exact
            # failure this tier was added to fix -- it belongs on screen, not
            # only in a log.
            note = out.reasoning
            if out.revised.strip():
                note = f"{out.revised.strip()} {out.reasoning}".strip()
        except Exception:
            return

        self.deliberations += 1
        self.state = self.state.model_copy(
            update={"suspicion": settled, "reasoning": note[:400], "deliberated": True}
        )
