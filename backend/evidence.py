"""Structural evidence computed in code, not remembered by a model.

The observer's one reliable signal is arithmetic, so it is done here rather than
asked for in a prompt.

WHAT THE SIGNAL IS

When a day-vote eliminates someone, the moderator announces their role to the
whole table. That announcement retroactively labels every vote cast that round:
we now know whether each voter was aiming at a liar or at an innocent. Liars
know who their teammates are, so they steer away from them; villagers are
guessing, and guess right a fair share of the time.

Measured across the 33 LLMafia games in `data/mafia/`, over every vote whose
target was eventually revealed:

    P(your target is revealed a liar | you are a liar)     = 0.024   (n=124)
    P(your target is revealed a liar | you are innocent)   = 0.265   (n=266)

An eleven-fold likelihood ratio, and the single most discriminating thing in the
dataset by a wide margin.

WHAT IT IS WORTH

Scoring every candidate liar-set by this likelihood alone, with no model call at
all -- leave-one-game-out, replayed causally, final reveal withheld:

    top-1 accuracy   0.52      (chance 0.37)
    precision@N      0.58      (chance 0.37)

Not an oracle. A real edge, from about thirty lines of arithmetic, available for
free on every event.

WHAT IS DELIBERATELY NOT HERE

"Teammates never vote for each other" is not used, and must not be added back.
It measures 3.8% against 15-20% at random and looks compelling, but scoring sets
by it lands at 0.03 top-1 -- far *below* chance -- and adding it to the signal
above drags 0.67 down to 0.48. The reason is worth keeping: almost no pair ever
votes internally, so the term barely separates one pair from another, and what
it actually ends up maximising is candidate sets that cast few votes. It is a
quiet-player detector wearing a coalition-detector's clothes.

TIMING

The first reveal lands a median 43% of the way through a transcript. Before it
there is no structural evidence in the game at all -- only chat. `marginals`
returns nothing in that stretch, and the observer is expected to stay near
uniform rather than invent a suspect. See `PRE_EVIDENCE_MAX_DELTA` in
`observer.py`.
"""

from __future__ import annotations

import itertools
import math
from typing import Dict, List, Optional

from backend.schema import DECEIVER_ROLES

# P(the player you voted for is revealed a liar | you are one) and the same
# given you are innocent. Pooled over all 33 games; see the module docstring.
# Laplace-smoothed, which is why neither is exactly zero.
P_TARGET_IS_LIAR_GIVEN_LIAR = 0.0238
P_TARGET_IS_LIAR_GIVEN_INNOCENT = 0.2649

# Enumerating candidate sets is a binomial, so it is bounded rather than
# trusted: 12 players choose 3 is 220, which is nothing, but a malformed setup
# claiming 6 liars among 20 would be 38,760 sets on every single event.
MAX_CANDIDATE_SETS = 20_000


def _log_likelihood(
    candidate: frozenset,
    votes: List[tuple],
    revealed: Dict[str, str],
) -> float:
    """How well this candidate liar-set explains the votes we have labels for.

    Only votes whose target has since been revealed carry information: those are
    the ones the game has scored for us.
    """
    total = 0.0
    for voter, target in votes:
        role = revealed.get(target)
        if role is None:
            continue
        p = (
            P_TARGET_IS_LIAR_GIVEN_LIAR
            if voter in candidate
            else P_TARGET_IS_LIAR_GIVEN_INNOCENT
        )
        hit = role.lower() in DECEIVER_ROLES
        total += math.log(p if hit else 1.0 - p)
    return total


def marginals(
    votes: Dict[int, Dict[str, str]],
    revealed: Dict[str, str],
    alive: List[str],
    liars_remaining: int,
) -> Dict[str, float]:
    """P(each living player is a liar), from the vote record alone.

    Returns `{}` when there is nothing to go on -- no reveals yet, or no votes
    aimed at anyone whose role is known. An empty result means "the game has not
    said anything yet", and the caller must not read it as "everyone is clear".

    The numbers sum to `liars_remaining`, matching the convention in
    `observer._settle`: each one is an honest P(this player is lying), not a
    share of a single pool.
    """
    if not alive or liars_remaining <= 0:
        return {}

    flat = [
        (voter, target)
        for round_votes in votes.values()
        for voter, target in round_votes.items()
    ]
    labelled = [(v, t) for v, t in flat if t in revealed]
    if not labelled:
        return {}

    size = min(liars_remaining, len(alive))
    if math.comb(len(alive), size) > MAX_CANDIDATE_SETS:
        return {}

    scores = {
        frozenset(c): _log_likelihood(frozenset(c), labelled, revealed)
        for c in itertools.combinations(alive, size)
    }
    if not scores:
        return {}

    # Softmax in log space -- these are sums of many logs and exponentiating
    # them directly underflows to zero on a long game.
    top = max(scores.values())
    weights = {c: math.exp(s - top) for c, s in scores.items()}
    total = sum(weights.values())
    if total <= 0:
        return {}

    out = {p: 0.0 for p in alive}
    for candidate, weight in weights.items():
        share = weight / total
        for player in candidate:
            out[player] += share
    return out


def format_evidence(scores: Dict[str, float], role: str = "liar") -> str:
    """The marginals, rendered for the prompt.

    Stated as a prior the model is asked to weigh, not as an answer. It is right
    about half the time, which is much better than the model's reading of round
    one but nowhere near certain, and a prompt that oversells it would just swap
    one overconfident wrong answer for another.
    """
    if not scores:
        return (
            "(nothing yet -- no eliminated player's role has been announced, so no\n"
            "  vote has been graded. There is no structural evidence in the game at\n"
            "  this point. Do not manufacture a suspect from tone alone.)"
        )

    lines = [
        f"  From the vote record alone, P(this player is {role}):",
    ]
    for player in sorted(scores, key=lambda p: scores[p], reverse=True):
        lines.append(f"    {player}: {scores[player]:.2f}")
    lines.append("")
    lines.append(
        "  This is computed, not guessed: every vote aimed at a player whose role\n"
        "  was later announced is now graded evidence about the voter. It is right\n"
        "  about half the time -- far better than reading tone, far short of proof.\n"
        "  Weigh it heavily, but a specific contradiction can still outweigh it."
    )
    return "\n".join(lines)
