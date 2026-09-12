"""Run the observer over many games and report the aggregate, not the anecdote.

This exists because the project tuned against one game (llmafia-0002) three
times in a row and drew a conclusion from it that the full dataset contradicts.
One game is a sample of one: the difference between 0.37 and 0.52 accuracy is
invisible at that scale, and both look like success or failure depending on
which game you happened to pick.

    # the arithmetic baseline, no API key and no calls at all
    python -m backend.batch --evidence-only

    # the real thing, over the six cheapest games
    python -m backend.batch --limit 6

    # hold games out: tune on the cheapest 20, report on the rest
    python -m backend.batch --evidence-only --holdout 20

`--evidence-only` is the one to reach for first. It scores `evidence.py` alone
with no model in the loop, runs over all 33 games in under a second, and is the
floor any change to the prompt has to clear to be worth paying for.

COST. A full run is one call per event plus one per role reveal: about 73 calls
on a median game, 2,900 across all 33. Start with `--limit`, and pick games by
cost rather than by result -- choosing the games where it already works is how
you get a number that does not survive the demo.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional

from backend.evidence import marginals
from backend.observer import Observer, infer_elimination, infer_vote, _ROLE_REVEAL
from backend.schema import DECEIVER_ROLES, Transcript


def load(pattern: str, limit: Optional[int], holdout: int) -> List[Path]:
    """Games to score, cheapest first.

    Sorted by event count so `--limit N` means "the N cheapest" rather than "the
    first N alphabetically", which is what makes a partial run affordable without
    making it a biased sample of easy games.
    """
    paths = sorted(Path(p) for p in glob.glob(pattern))
    paths.sort(key=lambda p: len(json.loads(p.read_text(encoding="utf-8"))["events"]))
    paths = paths[holdout:]
    return paths[:limit] if limit else paths


def evidence_only(transcript: Transcript) -> Dict[str, float]:
    """Replay the bookkeeping with no model, and score the arithmetic alone.

    Stops at the last role reveal rather than running to the end. The final
    reveal in a game the village wins is very nearly the answer, and scoring
    after it would measure the transcript rather than the observer -- the same
    trap `scoring.verdict` exists to avoid.
    """
    players = transcript.setup.players
    votes: Dict[int, Dict[str, str]] = {}
    revealed: Dict[str, str] = {}
    dead: List[str] = []

    reveals = [
        i
        for i, e in enumerate(transcript.events)
        if infer_elimination(e) in players and _ROLE_REVEAL.search(e.statement)
    ]
    stop = reveals[-1] if reveals else len(transcript.events)

    for event in transcript.events[:stop]:
        vote = infer_vote(event)
        if vote and vote[0] in players and vote[1] in players:
            votes.setdefault(event.round, {})[vote[0]] = vote[1]
        gone = infer_elimination(event)
        if gone in players:
            if gone not in dead:
                dead.append(gone)
            match = _ROLE_REVEAL.search(event.statement)
            if match:
                revealed[gone] = match.group("role")

    alive = [p for p in players if p not in dead]
    caught = sum(
        1 for p, r in revealed.items() if r.lower() in DECEIVER_ROLES and p in dead
    )
    left = max(1, transcript.setup.deceiver_count - caught)
    return marginals(votes, revealed, alive, left)


def _rank(scores: Dict[str, float], liars: List[str]):
    """(top suspect is a liar, precision@N) over the players still being scored."""
    live = [p for p in liars if p in scores]
    if not scores or not live:
        return None, None
    ranked = sorted(scores, key=lambda p: scores[p], reverse=True)
    top_n = ranked[: len(live)]
    return ranked[0] in live, sum(1 for p in top_n if p in live) / len(live)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Score the observer over many games and report the aggregate."
    )
    parser.add_argument("--games", default="data/mafia/llmafia-*.json")
    parser.add_argument("--limit", type=int, help="Score only the N cheapest games. Use it.")
    parser.add_argument(
        "--holdout",
        type=int,
        default=0,
        help="Skip the N cheapest games, so you can tune on those and report on the rest.",
    )
    parser.add_argument(
        "--evidence-only",
        action="store_true",
        help="Score evidence.py alone. No model, no API key, no cost.",
    )
    parser.add_argument("--no-deliberate", action="store_true", help="Per-event calls only.")
    parser.add_argument(
        "--no-batch-opening",
        action="store_true",
        help="Read the opening one line at a time instead of in a single call.",
    )
    parser.add_argument("--json", dest="out", help="Write per-game rows here.")
    args = parser.parse_args()

    paths = load(args.games, args.limit, args.holdout)
    if not paths:
        print(f"No games matched {args.games!r}.")
        return 2

    mode = "evidence only (no model calls)" if args.evidence_only else "full observer"
    print(f"{len(paths)} games, {mode}")
    print("")

    rows = []
    hits: List[bool] = []
    precisions: List[float] = []
    chances: List[float] = []
    calls = 0

    for path in paths:
        transcript = Transcript.model_validate_json(path.read_text(encoding="utf-8"))
        liars = transcript.deceivers()
        try:
            if args.evidence_only:
                scores = evidence_only(transcript)
                observer = None
            else:
                observer = Observer(
                    transcript.setup,
                    deliberate=not args.no_deliberate,
                    batch_opening=not args.no_batch_opening,
                )
                for event in transcript.events:
                    observer.observe(event)
                # A game that never eliminated anyone leaves the opening unread.
                observer.flush()
                scores = observer.state.suspicion
        except (Exception, KeyboardInterrupt) as exc:
            print(f"  {path.name}: stopped -- {type(exc).__name__}: {str(exc)[:120]}")
            break

        hit, precision = _rank(scores, liars)
        if hit is None:
            print(f"  {path.name}: no scorable state (no evidence, or every liar already out)")
            continue

        alive_liars = [p for p in liars if p in scores]
        chance = len(alive_liars) / len(scores) if scores else 0.0
        hits.append(hit)
        precisions.append(precision)
        chances.append(chance)
        if observer is not None:
            calls += len(transcript.events) + observer.deliberations

        top = max(scores, key=lambda p: scores[p])
        rows.append(
            {
                "game": path.name,
                "events": len(transcript.events),
                "top_suspect": top,
                "hit": hit,
                "precision_at_n": round(precision, 3),
                "chance": round(chance, 3),
                "actual": liars,
            }
        )
        print(
            f"  {path.name:<22} {len(transcript.events):>4} events  "
            f"top={top:<9} {'HIT ' if hit else 'miss'}  p@N={precision:.2f}  "
            f"(chance {chance:.2f})"
        )

    if not hits:
        print("")
        print("Nothing scored.")
        return 2

    n = len(hits)
    base = sum(chances) / n
    # A run of n coin flips at the chance rate has this much slop in it. Printed
    # because "0.52 against 0.37" over six games is not a result, and the number
    # that says so belongs on screen beside it rather than in someone's head.
    err = math.sqrt(max(base * (1.0 - base), 1e-9) / n)

    print("")
    print("=" * 66)
    print(f"games scored        : {n}")
    print(f"top-1 accuracy      : {sum(hits) / n:.3f}   (chance {base:.3f})")
    print(f"mean precision@N    : {sum(precisions) / n:.3f}")
    print(f"1 s.e. at chance    : +/-{err:.3f}   ({base + 2 * err:.3f} would be 2 s.e. up)")
    if calls:
        print(f"model calls spent   : {calls}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print("")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
