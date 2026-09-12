"""LLMafia -> Cry Wolf `Transcript`.

    python -m data.adapters.llmafia probe data/raw/games
    python -m data.adapters.llmafia convert data/raw/games --out data/mafia/

Source: https://huggingface.co/datasets/niveck/LLMafia (MIT). 33 online Mafia
games, 7-12 human players each plus one LLM agent, day/night rounds with real
eliminations. Each game is a directory:

    all_messages.txt        "[HH:MM:SS] Speaker: text", every phase, chronological
    public_daytime_chat.txt daytime only
    public_nighttime_chat.txt  MAFIA ONLY -- see below
    config.json             players, is_mafia, is_llm
    mafia_names.txt         the answer key
    player_names.txt        roster

THE ONE THING THAT MATTERS HERE:

Nighttime is mafia-only chat. Checked across the dataset: in 26 of 26 games with
any night chat, *every single nighttime speaker is mafia*. Feeding night messages
to the observer would not be a subtle leak, it would be handing over the answer
and then congratulating ourselves on the accuracy. The night votes leak the same
way -- only mafia vote at night.

So this adapter keeps exactly what a seat at the table would have seen:

  kept     daytime player chat
           daytime vote announcements ("Brook voted for Remi")
           elimination results, including the revealed role
  dropped  every nighttime player message
           every nighttime vote (those are the mafia choosing a victim)
           lobby chatter before the first Daytime

Elimination results keep "Their role was bystander/mafia" on purpose: in this
game the role of anyone voted out is announced to everyone, so it is public
knowledge the humans had. It cannot inflate our score either, because eliminated
players leave the suspicion distribution entirely.

`real_names.txt` exists in the raw data and is never read. The released names are
already anonymised pseudonyms and that is all that should travel any further.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import List, Optional

from backend.schema import GameEvent, GameSetup, Transcript

MESSAGE = re.compile(r"^\[(?P<time>[0-9:]+)\]\s+(?P<speaker>[^:]+):\s*(?P<text>.*)$")
MANAGER = "Game-Manager"

DAY_STARTS = re.compile(r"now it's daytime", re.I)
NIGHT_STARTS = re.compile(r"now it's nighttime", re.I)
DAY_VOTE_STARTS = re.compile(r"daytime has ended.*time to vote", re.I)
NIGHT_VOTE_STARTS = re.compile(r"nighttime has ended.*time to vote", re.I)
VOTE_CAST = re.compile(r"^(?P<voter>\S+) voted for (?P<target>\S+)$")
VOTED_OUT = re.compile(r"^(?P<who>\S+) was voted out\.?\s*(?:Their role was (?P<role>\w+))?", re.I)

PREMISE = (
    "Online Mafia, played in a group chat. Each day everyone talks, then votes "
    "someone out; the role of whoever is voted out is announced to all. Each "
    "night the mafia quietly choose a bystander to eliminate -- you do not see "
    "that conversation, only who is gone in the morning. The mafia know each "
    "other. Nobody else knows anything."
)


class _Phase:
    """Where we are in the day/night cycle while walking the log.

    `public` is the only thing the adapter really decides: whether what we are
    reading now was visible to every player, or only to the mafia.
    """

    def __init__(self) -> None:
        self.round = 0
        self.phase = "day"
        self.public = False  # nothing before the first Daytime counts

    def manager_said(self, text: str) -> None:
        if DAY_STARTS.search(text):
            self.round += 1
            self.phase, self.public = "day", True
        elif NIGHT_STARTS.search(text):
            self.phase, self.public = "night", False
        elif DAY_VOTE_STARTS.search(text):
            self.phase, self.public = "vote", True
        elif NIGHT_VOTE_STARTS.search(text):
            # The mafia picking a victim. Individual votes stay hidden.
            self.phase, self.public = "night", False


def _mafia_channel(directory: Path) -> set:
    """(timestamp, speaker, text) for every message on the mafia-only channel.

    The phase machine below infers visibility from the manager's announcements,
    and that is *almost* right -- but the dataset files are the real authority on
    who could see what, and they disagree in one important way. Messages land in
    the nighttime channel by audience, not by clock: in game 0006 a mafia player
    posts at 13:54:17 and Nighttime does not begin until 13:56:39, yet that
    message went only to the mafia. The asynchronous LLM agent makes this common.

    Treating such a message as public would show the observer something no
    villager saw. So whatever the clock says, anything in this set is withheld.
    """
    path = directory / "public_nighttime_chat.txt"
    if not path.exists():
        return set()
    out = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("[") or "] " not in line or ": " not in line:
            continue
        time, _, rest = line[1:].partition("] ")
        speaker, _, text = rest.partition(": ")
        out.add((time.strip(), speaker.strip(), text.strip()))
    return out


def _parse_game(directory: Path) -> tuple[List[str], dict, List[GameEvent], dict]:
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    players = [p["name"] for p in config["players"]]
    ground_truth = {
        p["name"]: ("mafia" if p.get("is_mafia") else "bystander") for p in config["players"]
    }
    llm_players = [p["name"] for p in config["players"] if p.get("is_llm")]

    log = (directory / "all_messages.txt").read_text(encoding="utf-8")
    mafia_only = _mafia_channel(directory)
    state, events = _Phase(), []
    dropped = Counter()

    for line in log.splitlines():
        match = MESSAGE.match(line.strip())
        if not match:
            continue
        speaker = match.group("speaker").strip()
        text = match.group("text").strip()
        if not text:
            continue

        if speaker == MANAGER:
            state.manager_said(text)

            out = VOTED_OUT.match(text)
            if out and out.group("who") in players:
                # Public in both phases: everyone learns who is gone, and what
                # they were. Phase records whether the village or the mafia did it.
                events.append(
                    GameEvent(
                        round=max(state.round, 1),
                        phase=state.phase,
                        speaker="MODERATOR",
                        statement=text,
                        eliminated=out.group("who"),
                        timestamp=match.group("time"),
                    )
                )
                continue

            vote = VOTE_CAST.match(text)
            if vote:
                if not state.public:
                    dropped["night vote (mafia choosing a victim)"] += 1
                    continue
                events.append(
                    GameEvent(
                        round=max(state.round, 1),
                        phase="vote",
                        speaker="MODERATOR",
                        statement=text,
                        timestamp=match.group("time"),
                    )
                )
            continue

        if speaker not in players:
            dropped[f"non-player speaker {speaker!r}"] += 1
            continue
        if (match.group("time"), speaker, text) in mafia_only:
            # On the mafia channel regardless of what the clock said.
            dropped["mafia-only channel message"] += 1
            continue
        if not state.public:
            dropped["nighttime message"] += 1
            continue

        events.append(
            GameEvent(
                round=max(state.round, 1),
                phase=state.phase if state.phase != "vote" else "vote",
                speaker=speaker,
                statement=text,
                timestamp=match.group("time"),
            )
        )

    meta = {
        "llm_players": llm_players,
        "dropped": dict(dropped),
        "rounds": state.round,
    }
    return players, ground_truth, events, meta


def convert_game(directory: Path) -> tuple[Transcript, dict]:
    players, ground_truth, events, meta = _parse_game(directory)
    if not players:
        raise ValueError("no players in config.json")
    if not events:
        raise ValueError("no public events (all chat was nighttime?)")

    mafia = [p for p, role in ground_truth.items() if role == "mafia"]
    transcript = Transcript(
        setup=GameSetup(
            players=players,
            deceiver_role="mafia",
            deceiver_plural="mafia",
            deceiver_count=len(mafia),
            premise=PREMISE,
        ),
        events=events,
        ground_truth=ground_truth,
    )
    return transcript, meta


def _games(root: Path) -> List[Path]:
    return sorted(d for d in root.iterdir() if d.is_dir() and (d / "config.json").exists())


def probe(root: Path) -> int:
    games = _games(root)
    print(f"{root}: {len(games)} games\n")

    mafia_counts, player_counts, rounds = Counter(), Counter(), Counter()
    lengths, failures, dropped_total = [], Counter(), Counter()

    for directory in games:
        try:
            transcript, meta = convert_game(directory)
        except Exception as exc:
            failures[" ".join(str(exc).splitlines())[:70]] += 1
            continue
        mafia_counts[transcript.setup.deceiver_count] += 1
        player_counts[len(transcript.setup.players)] += 1
        rounds[meta["rounds"]] += 1
        lengths.append(len(transcript.events))
        for reason, n in meta["dropped"].items():
            dropped_total[reason] += n

    print("mafia per game:")
    for count, n in sorted(mafia_counts.items()):
        print(f"  {count}: {n:>3} games")
    print("\nplayers per game:")
    for count, n in sorted(player_counts.items()):
        print(f"  {count:>2}: {n:>3} games")
    print("\nday/night rounds per game:")
    for count, n in sorted(rounds.items()):
        print(f"  {count:>2}: {n:>3} games")

    if lengths:
        lengths.sort()
        median = lengths[len(lengths) // 2]
        print(f"\nusable events per game (one model call each):")
        print(f"  min {lengths[0]}, median {median}, max {lengths[-1]}")
        print(f"  under 40 events: {sum(1 for n in lengths if n <= 40)} games")

    print("\nwithheld from the observer:")
    for reason, n in dropped_total.most_common():
        print(f"  {n:>5}  {reason}")

    if failures:
        print("\nunusable:")
        for reason, n in failures.most_common():
            print(f"  {n:>3}x  {reason}")
    return 0


def convert(root: Path, out_dir: Path, max_events: Optional[int], limit: Optional[int]) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    written, skipped = 0, Counter()

    for directory in _games(root):
        try:
            transcript, meta = convert_game(directory)
        except Exception as exc:
            skipped[" ".join(str(exc).splitlines())[:70]] += 1
            continue
        if max_events and len(transcript.events) > max_events:
            skipped[f"longer than {max_events} events"] += 1
            continue

        payload = transcript.model_dump()
        payload["_source"] = {
            "dataset": "LLMafia (niveck/LLMafia, MIT)",
            "game": directory.name,
            "llm_players": meta["llm_players"],
            "withheld": meta["dropped"],
            "note": (
                "Nighttime chat and night votes are mafia-only and are excluded. "
                "Everything here was visible to every player at the table."
            ),
        }
        (out_dir / f"llmafia-{directory.name}.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        written += 1
        if limit and written >= limit:
            break

    print(f"wrote {written} transcripts to {out_dir}")
    for reason, n in skipped.most_common():
        print(f"  skipped {n:>3}x  {reason}")
    return 0 if written else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("probe", help="Report what's there without converting.")
    p.add_argument("root", type=Path, nargs="?", default=Path("data/raw/games"))

    c = sub.add_parser("convert", help="Write Cry Wolf transcripts.")
    c.add_argument("root", type=Path, nargs="?", default=Path("data/raw/games"))
    c.add_argument("--out", type=Path, default=Path("data/mafia"))
    c.add_argument("--max-events", type=int, help="Skip longer games. Each event is a call.")
    c.add_argument("--limit", type=int)
    args = parser.parse_args()

    if not args.root.exists():
        print(f"{args.root} not found -- see data/adapters/README.md for the download.")
        return 2
    if args.command == "probe":
        return probe(args.root)
    return convert(args.root, args.out, args.max_events, args.limit)


if __name__ == "__main__":
    sys.exit(main())
