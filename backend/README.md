# The observer

This is the design document for the part of Cry Wolf that does the thinking.
The rest of the system moves bytes around; this is the thing that has to
remember round 1 while it is looking at round 4.

If you are only going to read one source file, read `evidence.py`. It is short,
and it is the part that carries the result.

---

## Using it

The observer is a plain object. Construct one per game, then hand it events.

```python
from backend.observer import Observer
from backend.scoring import grade
from backend.schema import Transcript

transcript = Transcript.model_validate_json(open("data/fallback_transcript.json").read())
observer = Observer(transcript.setup)          # once per game

state = observer.observe(event)                # once per POST /event -- blocking, ~2-5s
broadcast(state.model_dump())                  # straight onto the websocket

score = grade(observer.state, observer.history,
              transcript.events, transcript.ground_truth)   # GET /score
```

`observe()` is synchronous and does network I/O. In FastAPI, call it from a
worker thread (`run_in_threadpool`) or use the async client. Blocking the event
loop stops the websocket updating mid-game.

One timing note for anyone building a progress indicator: on the event where a
voted-out player's role is announced, `observe()` makes a second, deeper call and
takes roughly twice as long. That is about three events in a median game. It
returns the same `BeliefState` and still appends exactly one history entry, so
nothing downstream changes — but that is the beat where a spinner will sit
longest. `Observer(setup, deliberate=False)` turns it off.

---

## Shapes

`BeliefState` is what the UI renders, and it is defined in `schema.py`:

| field | |
|---|---|
| `round`, `phase`, `event_index` | where in the game this belief was formed |
| `eliminated` | the dead, in the order they died |
| `suspicion` | `{player: float}`, **living players only** |
| `claims_tracked` | `{player: str}` — what each has asserted |
| `contradictions_noticed` | `[{player, earlier, now, round_noticed}]` |
| `reasoning` | one paragraph, written for a human to read |
| `deliberated` | this state came from the deep pass, not the per-event one |
| `top_suspect` | property, not a field |

Two things about the shapes that are worth knowing before you touch them.

`contradictions_noticed` holds objects rather than strings, so the interface can
render "P4 said X, then Y" without parsing prose back out of a sentence.

The model emits `suspicion` as a list of `{player, score}` pairs, because a
strict JSON schema cannot express an object with open-ended keys. It is
converted to the dict above before anything downstream sees it. That is a wire
shape and nothing more.

### What the numbers mean

`suspicion` holds **living players only**, and the values sum to the number of
liars still in the game — not to 1.0. Each one is an honest *P(this player is
lying)*, not a share of a single pool. In a game with two liars left, two
players sitting at 0.7 and 0.6 is a coherent belief rather than a bug.

(The docstring on the field in `schema.py` still says 1.0. It is stale;
`observer._settle` and `evidence.marginals` are the truth. Worth fixing.)

`BeliefState.eliminated` lists the dead. Set `GameEvent.eliminated` if you know
who died; otherwise the observer reads it out of the moderator's wording
("P5 is found dead", "P4 is eliminated").

Scoring deliberately reads the verdict from the last event that killed nobody.
Otherwise the final vote removes the werewolf from the distribution, and a
correct observer grades as wrong. See `scoring.verdict()`.

### Ground truth

Ground truth lives in `Transcript.ground_truth` and is never rendered into a
prompt. There is a test asserting that. Keep reveals out of `events` too — the
fallback transcript's last line says "P4 is eliminated", not "P4 was the
werewolf".

---

## How the observer reasons

Two tiers and one piece of arithmetic.

### The arithmetic: computed evidence

When a player is voted out, their role is announced, which retroactively grades
every vote cast that round: we now know whether each voter was aiming at a liar
or at an innocent. Liars steer away from their own team; innocents guess.

Measured over the 33 games in `data/mafia/`, a vote cast by a liar lands on a
player later revealed as a liar **2.4%** of the time, against **26.5%** for an
innocent. An elevenfold likelihood ratio, and the most discriminating thing in
the dataset by a distance. `marginals()` scores every candidate liar-set by that
likelihood and returns per-player probabilities.

Scored on its own, with no model in the loop at all:

```bash
python -m backend.batch --evidence-only
# 33 games, top-1 0.485 against a 0.373 chance rate, precision@N 0.475
```

That is the floor. Any prompt change has to beat it to be worth paying for.

### Tier 1: one call per event

One cheap call per line of the game, so the bars keep moving live. It receives
the computed evidence block rather than being asked to reconstruct the vote
record from memory.

The prior belief goes back into the prompt in full every time. The model gets no
conversation history — the belief state *is* the memory, which is what makes it
inspectable. If it forgets something, you can see the forgetting.

### Tier 2: one call per role reveal

A deeper call, fired only when an elimination announces a role, because that is
the only moment new evidence enters the game. It sees the whole round verbatim,
the full vote table, the computed evidence, and its own prior reasoning — and it
may revise wholesale, including "I was wrong about Whitney". About three of these
in a median game against seventy shallow ones.

Its `revised` field is surfaced in `reasoning` on purpose. An observer that
abandons a read without saying so is the exact failure this tier exists to fix,
so the retraction belongs on screen rather than in a log.

`--no-deliberate` turns it off, to price a run against the per-event loop alone.

### The evidence-free opening

The first reveal lands a median 43% of the way through a transcript. Before it,
the game contains no structural evidence at all — only chat. The old loop spent
that entire stretch committing hard to whoever talked loudest, and then had to
spend the rest of the game climbing back out.

Movement is now throttled to `PRE_EVIDENCE_MAX_DELTA` until the first role is
announced, with an absolute ceiling on top of it. Throttled, not frozen: a real
contradiction still registers, a confident tone no longer does.

### Two things that are deliberately not signals

Both were built, measured, and removed. Read the comments in `evidence.py` and
`observer.format_votes` before reinstating either.

- **"Teammates never vote for each other."** 0.03 top-1 on its own — below
  chance — and it drags the reveal signal from 0.67 down to 0.48 when added to
  it. Almost no pair ever votes internally, so the term barely separates one pair
  from another; what it actually maximises is candidate sets that cast few votes.
  It is a quiet-player detector wearing a coalition-detector's clothes.
- **Aggression, confidence, accusation volume.** Rhetorical heat is not evidence,
  and scoring it is how the observer used to talk itself into a suspect before
  any evidence existed.

---

## Tuning knobs

Each of these has a longer comment at its definition explaining how the value was
arrived at. The short version:

| constant | where | default | |
|---|---|---|---|
| `MAX_DELTA_PER_EVENT` | `observer.py` | 0.30 | per-event suspicion cap once evidence exists. Lower it if the chart looks jumpy; raise it if the observer looks asleep. |
| `PRE_EVIDENCE_MAX_DELTA` | `observer.py` | 0.06 | the same cap before the first role reveal. Raising it to parity reintroduces round-one overcommitment, and there is a test that fails if you do. |
| `PRE_EVIDENCE_CEILING_MULTIPLE` | `observer.py` | 2.0 | during the opening, no player may exceed this multiple of their starting share. A per-event cap bounds the *speed* of a commitment, not its size, and it is the size that ends up in the verdict. |
| `P_TARGET_IS_LIAR_GIVEN_LIAR` / `..._GIVEN_INNOCENT` | `evidence.py` | 0.0238 / 0.2649 | the measured likelihoods. Refit them if the dataset changes; do not hand-tune them. |

The caps are symmetric on purpose, and we considered changing that. In one run
the observer had the werewolf at 55%, spent seven events down at 10% pointing
somewhere else, then came back to 80% when the lie surfaced. An asymmetric cap —
suspicion easy to gain, hard to shed — would have kept that chart tidier.

We decided against it. That swing was not forgetting: the claims ledger held
throughout, which is exactly why the observer recovered the instant hard evidence
arrived. It had considered a theory and dropped it when the evidence said
otherwise. Rigging the arithmetic so the observer cannot change its mind buys a
prettier demo at the cost of the reasoning being real. The wobble is reported
instead, by `scoring.measure_drift`.

---

## Measuring a change

Never on one game. This project drew a wrong conclusion from `llmafia-0002`
three times running.

```bash
python -m backend.batch --evidence-only                # free, all 33, under a second
python -m backend.batch --limit 6                      # the real loop, cheapest games
python -m backend.batch --evidence-only --holdout 20   # tune on 20, report on 13
```

`--limit` takes the *cheapest* games rather than the first N, so a partial run is
not a biased sample of easy ones. The output prints the chance rate and one
standard error beside the score, because "0.52 against 0.37" over six games is
not a result, and the number that says so belongs next to it.

---

## Running it without the server

```bash
python -m backend.run_observer --spoil --json out/run1.json
python -m backend.test_observer   # offline, no API key needed
python -m backend.test_evidence   # offline, the evidence and the deep call
```

This is also the demo's safety net: if the API, the feeder, or the websocket
misbehaves, this still produces the whole run.

`--json` saves after **every** event, so a run stopped by quota keeps everything
it paid for. Continue it with the same command plus `--resume`:

```bash
python -m backend.run_observer --spoil --json out/run1.json --resume
```

A finished run is marked `"complete": true` and will not be resumed over. The
saved file doubles as the stage fallback — the UI can replay it without spending
a call.

---

## Which model runs

`llm.py` is the only file that knows. `observer.py` asks for a validated object
and does not care how it was produced.

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...
```

| Variable | Default | |
|---|---|---|
| `CRYWOLF_MODEL` | `claude-haiku-4-5` | ladder: haiku → `claude-sonnet-5` → `claude-opus-5` |
| `CRYWOLF_EFFORT` | `medium` | ignored on models that do not accept it |
| `CRYWOLF_MIN_INTERVAL` | `0` | seconds between calls, if rate limits bite |

Start on Haiku: one replay is about 27 calls, and tuning means many replays. Move
up when the reads look shallow, not before.

Three things the code handles so you do not have to:

- **`effort` is rejected outright by Haiku 4.5 and Sonnet 4.5.** Sending it is a
  400 on every call, not a warning. `llm.py` omits it for those models.
- **The system prompt is identical every event, so it is cached** — 26 cache
  reads per run instead of 26 re-sends.
- **Rate limits and 5xx retry with backoff**, honoring the server's stated delay.

To swap providers, write a class with one `structured(system, user, schema)`
method that returns the validated model. Nothing outside `llm.py` changes.
