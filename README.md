# Cry Wolf

An observer that watches a game of Werewolf and tracks, in real time, who it
thinks is lying.

It reads one line of the game at a time and rewrites its belief after each one:
a suspicion score for every living player, a ledger of what each has claimed,
the contradictions it has caught, and a sentence explaining why it moved. All of
that is on screen while the game runs, and all of it is graded against ground
truth when the game ends.

The game is a proxy. The real question is whether a model's stated reasoning is
the reasoning that actually produced its answer — and the only way to ask that
is to make the belief explicit, watch it change, and score it.

---

## Quickstart

Requirements: Python 3.11+, Node 18+, and a bash shell (macOS or Linux; Git Bash
or WSL on Windows).

```bash
npm run setup
```

That creates the Python virtualenv, installs both halves, and copies
`.env.example` to `.env`.

Then pick one:

```bash
npm run replay   # a recorded game, no API key, no cost, deterministic
npm run demo     # a live run -- real model calls, needs a key in .env
```

Either way the UI opens at <http://127.0.0.1:8000>. Ctrl-C stops everything.

**Start with `npm run replay`.** It plays a real recorded run — genuine model
output, saved to disk — through the whole stack at 2.5 seconds a turn. Nothing
is mocked, nothing is billed, and there is no network dependency.

There is also a third way in, which costs nothing and takes under a second:

```bash
python -m backend.batch --evidence-only
```

That scores the structural signal across all 33 games with no model in the loop
at all. More on why that number matters below.

---

## What you are looking at

The page is laid out as a case file.

- **The table** — every player, alive or dead, with the current suspicion on
  each. Rows hold their position rather than re-sorting, so you can follow one
  player down the screen across the whole game.
- **The case file** — what each player has claimed, and every contradiction the
  observer has caught, stamped with the round it noticed.
- **The belief board** — suspicion over time. This is the chart that tells you
  whether the observer is reasoning or thrashing.
- **The transcript and the evidence panel** — the game as it arrives, and the
  computed vote evidence the observer is working from.

Three moments are worth watching for.

**The opening is deliberately quiet.** The first role reveal lands a median 43%
of the way through a transcript. Before it, the game contains no structural
evidence whatsoever — only chat, much of it "hi" and "hello all". The bars stay
bunched through that stretch on purpose. An observer that says *I do not know
yet* for the first 40% beats one that has been wrong since line three and is
defending it.

**A role reveal is where the work happens.** When a voted-out player's role is
announced, the observer makes a second, deeper call: it re-reads the entire
round verbatim against the newly graded vote record and may revise wholesale.
That beat takes roughly twice as long as a normal event. It is thinking, not
hanging. There are about three of these in a median game against seventy cheap
ones.

**It is allowed to say it was wrong.** When the deep pass abandons a read, it
says so in the reasoning line, on screen. An agent that quietly drops a theory
and pretends it never held one is the exact failure that pass exists to catch,
so the retraction belongs in the interface rather than in a log.

---

## How it works

Three components, connected by two deliberately thin seams.

```
  transcript ──▶ feeder ──▶ POST /event ──▶ observer ──▶ WS /live ──▶ UI
                                              │
                                          evidence.py
```

**The feeder** (`backend/feeder.py`) pushes one game event at a time to the API.
Anything that can produce lines of a game can sit on that side — a recorded
transcript, a live table, speech-to-text — and nothing downstream would notice
the difference.

**The API** (`backend/main.py`, FastAPI) holds one run at a time. Each event is
folded into the observer's belief, and the new state goes straight out over the
websocket.

**The observer** (`backend/observer.py`) is the part the project is actually
about. It gets no conversation history. Its entire memory is the belief state,
which is fed back into the prompt in full on every call. That is the central
design choice: it forces every carried thought to be written down somewhere you
can read it, so when the observer forgets something, you can watch it forget.

**The evidence model** (`backend/evidence.py`) computes the one signal that is
worth computing rather than asking a model to remember. Thirty lines of
arithmetic, described next.

### The signal

When a day vote eliminates someone, their role is announced to the whole table.
That announcement retroactively grades every vote cast that round: we now know
whether each voter was aiming at a liar or at an innocent. Liars know their
teammates and steer away from them; villagers are guessing.

Measured across the 33 games in `data/mafia/`:

| | |
|---|---|
| P(your target is revealed a liar \| **you are a liar**) | 0.024 (n=124) |
| P(your target is revealed a liar \| **you are innocent**) | 0.265 (n=266) |

An elevenfold likelihood ratio, and the most discriminating thing in the dataset
by a wide margin. Scoring candidate liar-sets by that likelihood alone, with no
model call anywhere:

```
33 games, top-1 0.485 against a 0.373 chance rate, precision@N 0.475
```

That is the floor. Any prompt change has to beat it to be worth paying for.

### What is deliberately not a signal

Two things that feel obviously true to humans measure at or below chance, and
both were built, measured, and deleted:

- **"Teammates never vote for each other."** It looks compelling — about 3.8%
  against 15–20% at random — but scoring sets by it lands at 0.03 top-1, *far
  below* chance, and adding it to the vote signal drags 0.67 down to 0.48.
  Almost no pair ever votes internally, so the term barely separates one pair
  from another; what it actually rewards is candidate sets that cast few votes.
  It is a quiet-player detector wearing a coalition-detector's clothes.
- **Aggression, confidence, accusation volume.** Rhetorical heat is not
  evidence. Scoring it is how the early observer talked itself into committing
  to whoever spoke loudest before any evidence existed.

Both are documented where someone would go to re-add them. Please read those
comments before doing so.

---

## Measuring a change

Never on one game. This project drew a wrong conclusion from `llmafia-0002`
three times running.

```bash
python -m backend.batch --evidence-only              # free, all 33, under a second
python -m backend.batch --limit 6                    # the real loop, cheapest games
python -m backend.batch --evidence-only --holdout 20 # tune on 20, report on 13
```

`--limit` takes the *cheapest* games rather than the first N, so a partial run is
not a biased sample of easy ones. The output prints the chance rate and one
standard error beside every score, because "0.52 against 0.37" over six games is
not a result, and the number that says so belongs next to it.

Four things get graded, and the last two are the interesting ones:

| | |
|---|---|
| **accuracy** | is the top suspect actually lying |
| **precision@N** | of the top N living suspects, how many really are, where N is how many liars remain |
| **consistency** | did it reverse a read without saying why |
| **drift** | how many times the lead changed, and how many events it spent away from its final answer |

Accuracy alone will happily reward an observer that arrives at the right answer
by thrashing. Consistency and drift are what catch that.

---

## Results, honestly

The full loop beats chance. It is not close to solved.

- Evidence alone, all 33 games: **top-1 0.485** against a **0.373** chance rate.
- The full loop has been scored on a six-game subset, not all 33 — that is an
  API-cost limit, not a code limit.
- Our headline single run (`out/run_final.log`) ends with the right player at
  72% in the middle of the game and a **miss** on the final verdict, at
  precision@2 = 0.50.

The scoreboard reports `accuracy` and `final_accuracy` separately, and prints
both rather than the kinder one.

---

## Commands

| | |
|---|---|
| `npm run setup` | venv, dependencies, `.env` |
| `npm run replay` | recorded game into the UI — no key, no cost |
| `npm run demo` | live run into the UI — needs a key |
| `npm run dev` | both halves with hot reload, auto-feeding a recording |
| `npm run test` | offline test suite, no key needed |
| `python -m backend.run_observer --spoil` | replay through the observer with no server at all |
| `python -m backend.batch --evidence-only` | score the arithmetic across all 33 games |

`npm run dev` serves the UI on <http://localhost:5173> with hot module reload and
proxies the API through, so interface edits appear instantly with no rebuild.
`npm run demo` builds the UI and serves everything from one port instead.

---

## Configuration

Copy `.env.example` to `.env`. It is gitignored; never commit a real key. A real
exported environment variable overrides the file, and the server reads it at
startup — change `.env`, restart uvicorn.

| Variable | Default | |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | required for live runs only |
| `CRYWOLF_MODEL` | `claude-haiku-4-5` | ladder: haiku → `claude-sonnet-5` → `claude-opus-5` |
| `CRYWOLF_EFFORT` | `medium` | silently omitted for models that reject it |
| `CRYWOLF_MIN_INTERVAL` | `0` | seconds between calls, if rate limits bite |

Start on Haiku. One replay is roughly 27 calls and tuning means many replays.
Move up when the reads look shallow, not before.

---

## Layout

```
backend/
  observer.py    the loop: one call per event, belief carried forward
  evidence.py    the vote arithmetic -- read this one first
  scoring.py     accuracy, precision@N, consistency, drift
  llm.py         the only file that knows which model is answering
  main.py        FastAPI: ingestion, scoring, websocket, static UI
  feeder.py      pushes a transcript in, live or replayed
  batch.py       scores many games and reports the aggregate
  schema.py      the shared data contract
  README.md      the design document -- every tuning knob and what breaks
frontend/        React + Vite, no UI framework, hand-written CSS
data/
  mafia/         33 normalized LLMafia games
  adapters/      format adapters for the two public corpora
scripts/         dev.sh and demo.sh
```

`backend/README.md` is the real design document. It carries the measured
likelihoods, every tuning constant with the reasoning behind its value, and the
list of things that were tried and removed.

---

## Limitations

- **One game at a time.** A second `POST /game/start` replaces the first. No
  auth, no multi-user, no database — runs are JSON files on disk.
- **Local only.** No hosted deployment. The API serves the built UI from one
  port, so deploying it is possible, but it has not been done.
- **The live path replays recorded transcripts.** The `POST /event` seam is
  built for a live source; nothing is wired to one.
- **Bash required.** `scripts/*.sh` use `lsof`, `open`, and `seq`. Native
  Windows needs WSL or Git Bash.
- **Ports 8000 (and 5173 in dev) must be free.** The scripts refuse to start
  rather than fight for a port.
- **A live run takes a couple of minutes** — 2–5 seconds per event, roughly
  twice that on a role reveal, and it depends on your account's rate limits.
  `CRYWOLF_MIN_INTERVAL` exists for when they bite.
- **Desktop layout.** Websockets required. Not tested on mobile.

## Data and credit

Games come from the public **LLMafia** corpus of LLM agents playing Mafia,
normalized through the adapters in `data/adapters/`. A second public format,
Werewolf-Among-Us, is also supported. We wrote the adapters; we did not generate
the games.

Ground truth lives in `Transcript.ground_truth`, is never rendered into a
prompt, and there is a test that fails if it ever is.
