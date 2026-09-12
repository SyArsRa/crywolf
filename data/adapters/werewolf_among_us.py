"""Werewolf Among Us (ACL 2023 Findings) -> Cry Wolf `Transcript`.

    # look before converting -- reports roles, player counts, what is usable
    python -m data.adapters.werewolf_among_us probe raw/test.json

    # convert the games that fit our format
    python -m data.adapters.werewolf_among_us convert raw/test.json --out data/wau/

Source schema, taken from the authors' own loader (`baselines/read_data.py` in
SALT-NLP/PersuationGames), not guessed:

    {
      "Game_ID": ..., "video_name" | "EG_ID": ...,
      "playerNames": ["Alice", "Bob", ...],
      "startRoles":  ["Werewolf", "Seer", ...],   # parallel to playerNames
      "votingOutcome": [...],
      "Dialogue": [{"utterance": str, "speaker": str,
                    "annotation": [str], "Rec_Id": ...}, ...]
    }

THE FORMAT MISMATCH, up front, because it drives everything below:

The dataset is *One Night Ultimate Werewolf*, not the multi-round Werewolf our
observer was built for. The differences that matter:

1. One night, one day, one vote. There are no rounds, no night kills, and nobody
   is eliminated mid-game -- so every utterance maps to round 1, phase "day",
   and no event ever carries an elimination.
2. Usually TWO werewolves, not one. Our `Transcript` validator requires exactly
   one, and the observer's prompt tells it exactly one hides among villagers.
   Games with a different count are skipped by default rather than silently
   mangled -- `probe` tells you how many that costs.
3. Roles can swap during the night (Robber, Troublemaker), so `startRoles` is
   who they *started* as, not who they were when voting happened. We treat
   startRoles as ground truth and record that choice in the output, because the
   dataset's own vote-outcome files are the only place final roles live and they
   are a separate download.

None of this makes the data unusable; it makes it a different game with the same
shape. Read `probe` output before trusting a conversion.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Iterator, List, Optional

from backend.schema import GameEvent, GameSetup, Transcript

# Speakers the authors' loader drops -- not players.
NON_PLAYERS = {"automated", "game audio", "siri", "alexa", "unknown", "none", ""}

WEREWOLF_ROLES = {"werewolf", "wolf", "werewolves"}

ONUW_PREMISE = (
    "One Night Ultimate Werewolf. Roles are dealt in secret and the game lasts a "
    "single night and a single day: there are no eliminations along the way. The "
    "players talk once, then everyone votes at the same time. Exactly one "
    "werewolf hides among them and knows who they are; nobody else does."
)


def _why(exc: Exception) -> str:
    """Pydantic's multi-line validation dump is unreadable in a tally. Pull out
    the sentence the validator actually raised."""
    text = " ".join(str(exc).splitlines())
    if "Value error, " in text:
        text = text.split("Value error, ", 1)[1]
    text = text.split(" [type=", 1)[0]
    return " ".join(text.split())[:70]


def _game_id(game: dict) -> str:
    for key in ("video_name", "EG_ID", "Game_ID"):
        if game.get(key):
            base = str(game[key])
            suffix = game.get("Game_ID")
            if key != "Game_ID" and suffix is not None:
                return _slug(f"{base}-g{suffix}")
            return _slug(base)
    return "unknown"


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-").lower()


def _players_and_roles(game: dict) -> tuple[List[str], dict]:
    names = [str(n).strip() for n in game.get("playerNames", [])]
    roles = [str(r).strip() for r in game.get("startRoles", [])]
    ground_truth = {
        name: ("werewolf" if role.lower() in WEREWOLF_ROLES else role.lower())
        for name, role in zip(names, roles)
    }
    return names, ground_truth


def _dialogue(game: dict, players: List[str]) -> Iterator[GameEvent]:
    known = {p.lower(): p for p in players}
    for line in game.get("Dialogue", []):
        speaker = str(line.get("speaker", "")).strip()
        utterance = str(line.get("utterance", "")).strip()
        if not utterance or speaker.lower() in NON_PLAYERS:
            continue
        # Names in Dialogue don't always match playerNames' casing.
        canonical = known.get(speaker.lower())
        if canonical is None:
            continue  # a voice that isn't one of the seated players
        yield GameEvent(round=1, phase="day", speaker=canonical, statement=utterance)


def convert_game(game: dict) -> Transcript:
    """One raw game -> a validated Transcript. Raises if it doesn't fit."""
    players, ground_truth = _players_and_roles(game)
    if not players:
        raise ValueError("no playerNames")
    if len(ground_truth) != len(players):
        raise ValueError("startRoles does not line up with playerNames")

    events = list(_dialogue(game, players))
    if not events:
        raise ValueError("no usable dialogue")

    # Transcript's own validator enforces the one-werewolf rule and the rest.
    return Transcript(
        setup=GameSetup(players=players, premise=ONUW_PREMISE),
        events=events,
        ground_truth=ground_truth,
    )


def load_games(path: Path) -> List[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        # Some splits are keyed by game id rather than being a bare list.
        return list(data.values())
    return data


# ---------------------------------------------------------------------------
# probe -- look before you convert
# ---------------------------------------------------------------------------


def probe(path: Path) -> int:
    games = load_games(path)
    print(f"{path}: {len(games)} games\n")

    wolf_counts: Counter = Counter()
    roles: Counter = Counter()
    usable, reasons = 0, Counter()
    lengths, player_counts = [], Counter()

    for game in games:
        players, ground_truth = _players_and_roles(game)
        player_counts[len(players)] += 1
        for role in ground_truth.values():
            roles[role] += 1
        wolves = sum(1 for r in ground_truth.values() if r == "werewolf")
        wolf_counts[wolves] += 1
        try:
            transcript = convert_game(game)
        except Exception as exc:
            reasons[_why(exc)] += 1
            continue
        usable += 1
        lengths.append(len(transcript.events))

    print("werewolves per game:")
    for count, n in sorted(wolf_counts.items()):
        print(f"  {count} wolves: {n:>3} games{'   <- what we can use' if count == 1 else ''}")

    print("\nplayers per game:")
    for count, n in sorted(player_counts.items()):
        print(f"  {count} players: {n:>3} games")

    print("\nroles seen:")
    for role, n in roles.most_common():
        print(f"  {role:<16} {n}")

    print(f"\nusable as-is: {usable}/{len(games)} games")
    if lengths:
        lengths.sort()
        print(
            f"  utterances per usable game: min {lengths[0]}, "
            f"median {lengths[len(lengths) // 2]}, max {lengths[-1]}"
        )
        print(f"  a 27-event run costs ~27 calls; the median game here is {lengths[len(lengths) // 2]}")
    if reasons:
        print("\nskipped because:")
        for reason, n in reasons.most_common():
            print(f"  {n:>3}x  {reason}")
    return 0


# ---------------------------------------------------------------------------
# convert
# ---------------------------------------------------------------------------


def convert(path: Path, out_dir: Path, limit: Optional[int], max_events: Optional[int]) -> int:
    games = load_games(path)
    out_dir.mkdir(parents=True, exist_ok=True)

    written, skipped = 0, Counter()
    for game in games:
        try:
            transcript = convert_game(game)
        except Exception as exc:
            skipped[_why(exc)] += 1
            continue
        if max_events and len(transcript.events) > max_events:
            skipped[f"longer than {max_events} events"] += 1
            continue

        target = out_dir / f"{_game_id(game)}.json"
        payload = transcript.model_dump()
        payload["_source"] = {
            "dataset": "Werewolf Among Us (ACL 2023 Findings)",
            "file": path.name,
            "game_id": _game_id(game),
            "ground_truth_note": (
                "Roles are startRoles -- who each player was dealt. In One Night "
                "Ultimate Werewolf, Robber and Troublemaker can swap roles during "
                "the night, so these may differ from who was a werewolf at the vote."
            ),
        }
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        written += 1
        if limit and written >= limit:
            break

    print(f"wrote {written} transcripts to {out_dir}")
    if skipped:
        print("skipped:")
        for reason, n in skipped.most_common():
            print(f"  {n:>3}x  {reason}")
    if not written:
        print("\nNothing converted. Run `probe` on the same file to see why.")
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("probe", help="Report what's in the file without converting.")
    p.add_argument("path", type=Path)

    c = sub.add_parser("convert", help="Write Cry Wolf transcripts.")
    c.add_argument("path", type=Path)
    c.add_argument("--out", type=Path, default=Path("data/wau"))
    c.add_argument("--limit", type=int, help="Stop after this many transcripts.")
    c.add_argument(
        "--max-events",
        type=int,
        help="Skip games longer than this. Every event is one model call.",
    )
    args = parser.parse_args()

    if not args.path.exists():
        print(f"{args.path} not found. See data/adapters/README.md for where to get it.")
        return 2

    if args.command == "probe":
        return probe(args.path)
    return convert(args.path, args.out, args.limit, args.max_events)


if __name__ == "__main__":
    sys.exit(main())
