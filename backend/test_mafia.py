"""Tests for multi-deceiver games and the LLMafia adapter.

The adapter tests run against the real download if it's present and skip
otherwise, so this suite is still useful on a fresh clone.

    python -m backend.test_mafia
"""

from __future__ import annotations

import json
from pathlib import Path

from backend.schema import BeliefState, GameEvent, GameSetup, Transcript
from backend.scoring import grade, precision_at_n

RAW = Path("data/raw/games")
CONVERTED = Path("data/mafia")


def check(label: str, cond: bool) -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {label}")
    assert cond, label


def skip(label: str, why: str) -> None:
    print(f"  SKIP  {label} ({why})")


def _mafia_transcript(events=None) -> Transcript:
    return Transcript(
        setup=GameSetup(
            players=["A", "B", "C", "D"], deceiver_role="mafia", deceiver_count=2
        ),
        events=events
        or [GameEvent(round=1, phase="day", speaker="A", statement="morning")],
        ground_truth={"A": "mafia", "B": "bystander", "C": "mafia", "D": "bystander"},
    )


def test_multiple_deceivers_are_allowed() -> None:
    print("multi-deceiver schema")
    t = _mafia_transcript()
    check("two mafia validate", t.deceivers() == ["A", "C"])
    check("werewolf() refuses to answer for a 2-mafia game", _raises(t.werewolf))

    try:
        Transcript(
            setup=GameSetup(players=["A", "B"], deceiver_role="mafia", deceiver_count=1),
            events=[GameEvent(round=1, phase="day", speaker="A", statement="hi")],
            ground_truth={"A": "mafia", "B": "mafia"},
        )
    except Exception:
        check("a count that disagrees with ground_truth is rejected", True)
    else:
        check("a count that disagrees with ground_truth is rejected", False)


def _raises(fn) -> bool:
    try:
        fn()
    except Exception:
        return True
    return False


def test_precision_at_n() -> None:
    print("precision@N")
    events = [GameEvent(round=1, phase="day", speaker="A", statement="talk")] * 3
    actual = ["A", "C"]

    perfect = [{"A": 0.4, "C": 0.4, "B": 0.1, "D": 0.1}] * 3
    check("both mafia on top scores 1.0", precision_at_n(perfect, events, actual) == 1.0)

    half = [{"A": 0.4, "B": 0.35, "C": 0.15, "D": 0.1}] * 3
    check("one of two scores 0.5", precision_at_n(half, events, actual) == 0.5)

    wrong = [{"B": 0.5, "D": 0.3, "A": 0.1, "C": 0.1}] * 3
    check("neither scores 0.0", precision_at_n(wrong, events, actual) == 0.0)

    score = grade(
        BeliefState(suspicion=half[-1]),
        half,
        events,
        {"A": "mafia", "B": "bystander", "C": "mafia", "D": "bystander"},
    )
    check("accuracy is true when the top suspect is any mafia", score.accuracy)
    check("actual lists every mafia", sorted(score.actual) == ["A", "C"])
    check("precision tells the fuller story", score.precision_at_n == 0.5)


def test_two_suspects_can_both_be_high() -> None:
    """The bug behind "it only ever picks one suspect": suspicion was normalised
    to sum to 1.0, so with two mafia the guilty pair had to compete for one pool
    -- raising one mathematically lowered the other."""
    print("suspicion sums to the number of liars")
    from backend.observer import Observer, _settle
    from backend.schema import ObserverOutput, SuspicionEntry

    alive = ["A", "B", "C", "D"]
    out = _settle(
        [SuspicionEntry(player=p, score=s) for p, s in zip(alive, [0.9, 0.85, 0.2, 0.1])],
        {p: 0.5 for p in alive}, alive, first_event=False, remaining_liars=2,
    )
    check("two suspects stay high together", out["A"] > 0.6 and out["B"] > 0.6)
    check("they sum to the liar count, not 1.0", abs(sum(out.values()) - 2.0) < 1e-6)

    single = _settle(
        [SuspicionEntry(player="A", score=0.9)],
        {p: 0.25 for p in alive}, alive, first_event=True, remaining_liars=1,
    )
    check("one-liar games are unchanged", abs(sum(single.values()) - 1.0) < 1e-6)

    # And the observer derives the count from public reveals, never ground truth.
    transcript = _mafia_transcript(
        events=[
            GameEvent(round=1, phase="day", speaker="A", statement="morning"),
            GameEvent(round=1, phase="vote", speaker="MODERATOR",
                      statement="C was voted out. Their role was mafia", eliminated="C"),
        ]
    )

    class Stub:
        def structured(self, system, user, schema):
            return ObserverOutput(
                suspicion=[SuspicionEntry(player=p, score=0.5) for p in transcript.setup.players],
                claims_tracked=[], contradictions_noticed=[], reasoning="x",
            )

    observer = Observer(transcript.setup, llm=Stub())
    check("starts expecting both liars", observer.liars_remaining == 2)
    for event in transcript.events:
        observer.observe(event)
    check("a revealed mafia reduces the count", observer.liars_remaining == 1)
    check(
        "and the living suspicion re-targets that",
        abs(sum(observer.state.suspicion.values()) - 1.0) < 1e-6,
    )


def test_adapter_withholds_the_mafia_only_channel() -> None:
    """The one that matters. Nighttime chat is mafia-only in this dataset -- every
    nighttime speaker is mafia in all 26 games that have any. If a single one of
    those messages reaches the observer, the accuracy number is worthless."""
    print("adapter: night channel withheld")
    if not RAW.exists():
        skip("needs the raw download", f"{RAW} missing")
        return

    from data.adapters.llmafia import _games, _parse_game

    def night_messages(path: Path) -> set[tuple[str, str, str]]:
        """(timestamp, speaker, text) triples from the mafia-only channel.

        Matching on text alone gives false positives -- "hi" and "?" get said in
        both phases by different people. The timestamp makes it exact.
        """
        out = set()
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.startswith("[") or "] " not in line or ": " not in line:
                continue
            time, _, rest = line[1:].partition("] ")
            speaker, _, text = rest.partition(": ")
            out.add((time.strip(), speaker.strip(), text.strip()))
        return out

    leaked, checked = [], 0
    for directory in _games(RAW):
        night_file = directory / "public_nighttime_chat.txt"
        if not night_file.exists():
            continue
        checked += 1
        forbidden = night_messages(night_file)
        _, _, events, _ = _parse_game(directory)
        for event in events:
            if event.speaker == "MODERATOR":
                continue  # manager announcements are public by definition
            if (event.timestamp, event.speaker, event.statement) in forbidden:
                leaked.append((directory.name, event.speaker, event.statement[:45]))

    for row in leaked[:5]:
        print(f"        LEAKED: {row}")
    check(
        f"not one mafia-only message survives conversion ({checked} games)",
        not leaked,
    )

    # And no player-spoken event may sit in a night phase.
    for directory in _games(RAW)[:5]:
        _, _, events, _ = _parse_game(directory)
        offenders = [
            e for e in events if e.phase == "night" and e.speaker != "MODERATOR"
        ]
        check(f"{directory.name}: no player speaks during night", not offenders)


def test_converted_games_are_sane() -> None:
    print("converted transcripts")
    if not CONVERTED.exists() or not any(CONVERTED.glob("*.json")):
        skip("needs `python -m data.adapters.llmafia convert`", f"{CONVERTED} empty")
        return

    files = sorted(CONVERTED.glob("*.json"))
    check("all 33 games converted", len(files) == 33)

    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        transcript = Transcript.model_validate(
            {k: v for k, v in data.items() if not k.startswith("_")}
        )
        assert transcript.setup.deceiver_count == len(transcript.deceivers()), path.name
        assert transcript.events, path.name

        # The setup is the only thing handed to the observer up front. It may say
        # "there are 2 mafia" -- every player knows that -- but must never say
        # *which* players. So no mafia's name may appear anywhere in it beyond the
        # plain roster.
        setup = transcript.setup.model_dump()
        roster = setup.pop("players")
        blurb = json.dumps(setup)
        for name in transcript.deceivers():
            assert name in roster, path.name
            assert name not in blurb, f"{path.name} names {name} in the setup blurb"

    check("every converted game re-validates", True)
    check("no converted setup names a mafia player", True)

    sample = json.loads(files[0].read_text(encoding="utf-8"))
    check("source provenance is recorded", sample["_source"]["dataset"].startswith("LLMafia"))
    check("withheld counts are recorded", "withheld" in sample["_source"])
    check(
        "rounds actually advance (multi-round, unlike One Night)",
        max(e["round"] for e in sample["events"]) >= 2,
    )
    check(
        "eliminations are marked",
        any(e.get("eliminated") for e in sample["events"]),
    )


if __name__ == "__main__":
    for fn in [
        test_multiple_deceivers_are_allowed,
        test_precision_at_n,
        test_two_suspects_can_both_be_high,
        test_adapter_withholds_the_mafia_only_channel,
        test_converted_games_are_sane,
    ]:
        fn()
    print("\nall mafia checks passed")
