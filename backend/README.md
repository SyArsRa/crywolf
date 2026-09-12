# Lane B — the observer

## What Lane A needs to call

```python
from backend.observer import Observer
from backend.scoring import grade
from backend.schema import Transcript, GameEvent

transcript = Transcript.model_validate_json(open("data/fallback_transcript.json").read())
observer = Observer(transcript.setup)          # once per game

state = observer.observe(event)                # once per POST /event -- blocking, ~2-5s
broadcast(state.model_dump())                  # straight onto the websocket

score = grade(observer.state, observer.history,
              transcript.events, transcript.ground_truth)   # GET /score
```

`observe()` is synchronous and does network I/O. In FastAPI, call it from a
worker (`run_in_threadpool`) or use the async client — don't block the event loop
or the websocket stops updating mid-game.

One timing note: on the event where the moderator announces an eliminated
player's role, `observe()` makes a second, deeper call and takes roughly twice as
long. That is about three events in a median game. It returns the same
`BeliefState` and still appends exactly one history entry, so nothing downstream
changes -- but if there is a progress spinner, that is the beat where it will sit
longest. `Observer(setup, deliberate=False)` turns it off.

## Shapes

`BeliefState` (what the UI renders) is in `schema.py`:
`round`, `phase`, `event_index`, `suspicion: {player: float}` summing to 1.0,
`claims_tracked: {player: str}`, `contradictions_noticed: [{player, earlier, now, round_noticed}]`,
`reasoning`, and a `top_suspect` property.

Two departures from the plan artifact, both deliberate:

- `contradictions_noticed` holds objects, not strings — the UI can render
  "P4 said X, then Y" without parsing prose.
- The model emits `suspicion` as a list of `{player, score}` because a strict
  JSON schema can't express an open-keyed object. It's converted to the dict
  above before anything downstream sees it. Wire shape only.

## Ground truth

Lives in `Transcript.ground_truth` and is never rendered into a prompt — there's
a test asserting it. Keep any reveal out of `events` too (the fallback
transcript's last line says "P4 is eliminated", not "P4 was the werewolf").

## Running it without the server

```bash
python -m backend.run_observer --spoil --json out/run1.json
python -m backend.test_observer   # offline, no API key needed
python -m backend.test_evidence   # offline, the evidence and the deep call
```

`--json` saves after **every** event, so a run stopped by quota keeps everything
it paid for. Continue it with the same command plus `--resume`:

```bash
python -m backend.run_observer --spoil --json out/run1.json --resume
```

A finished run is marked `"complete": true` and won't be resumed over. The saved
file is also the stage fallback — the UI can replay it without spending a call.

## Deaths

`BeliefState.eliminated` lists the dead; `suspicion` holds **living players only**
and sums to 1.0 across them. Set `GameEvent.eliminated` if you know who died;
otherwise the observer reads the moderator's wording ("P5 is found dead",
"P4 is eliminated").

Scoring reads the verdict from the last event that killed nobody — otherwise the
final vote removes the werewolf from the distribution and a correct observer
grades as wrong. See `scoring.verdict()`.

## Which model runs

`llm.py` is the only file that knows. `observer.py` asks for a validated object
and doesn't care how it was produced.

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...
```

| Variable | Default | |
|---|---|---|
| `CRYWOLF_MODEL` | `claude-haiku-4-5` | ladder: haiku → `claude-sonnet-5` → `claude-opus-5` |
| `CRYWOLF_EFFORT` | `medium` | ignored on models that don't accept it |
| `CRYWOLF_MIN_INTERVAL` | `0` | seconds between calls, if rate limits bite |

Start on Haiku: one replay is ~27 calls and tuning means many replays. Move up
when reads look shallow, not before.

Two model-specific things the code handles so you don't have to:

- **`effort` is rejected by Haiku 4.5 and Sonnet 4.5** — sending it is a 400 on
  every call, not a warning. `llm.py` omits it for those models.
- The system prompt is identical every event, so it's cached: 26 cache reads per
  run instead of 26 re-sends.

Rate limits and 5xx retry with backoff, honoring the server's stated delay.

To swap providers later, write a class with one `structured(system, user, schema)`
method returning the validated model. Nothing outside `llm.py` changes.

## How the observer reasons now

Two tiers and one piece of arithmetic. Read `evidence.py` first -- it is short,
and it is the part that does the work.

**Computed evidence (`evidence.py`).** When a player is voted out their role is
announced, which retroactively grades every vote cast that round: we now know
whether each voter was aiming at a liar or at an innocent. Liars steer away from
their own team; innocents guess. Measured over the 33 games, a vote cast by a
liar lands on a player later revealed as a liar 2.4% of the time against 26.5%
for an innocent -- an elevenfold likelihood ratio, and the most discriminating
thing in the dataset by a distance. `marginals()` scores every candidate
liar-set by that likelihood and returns per-player probabilities.

Scored on its own, with no model in the loop at all:

```bash
python -m backend.batch --evidence-only
# 33 games, top-1 0.485 against a 0.373 chance rate, precision@N 0.475
```

That is the floor. Any prompt change has to beat it to be worth paying for.

**Tier 1, per event.** Unchanged in shape -- one cheap call per line, so the bars
still move live for the demo. It now receives the computed block instead of being
asked to reconstruct the vote record from memory.

**Tier 2, per role reveal.** A deeper call fired only when an elimination
announces a role, because that is the only moment new evidence enters the game.
It sees the whole round verbatim, the full vote table, the computed evidence and
its own prior reasoning, and may revise wholesale -- including "I was wrong about
Whitney". About three of these on a median game against 70 shallow ones. Its
`revised` field is surfaced in `reasoning` on purpose: an observer that abandons
a read without saying so is the failure this tier exists to fix, so it belongs on
screen rather than in a log. `--no-deliberate` turns it off, to price a run
against the per-event loop alone.

**The evidence-free opening.** The first reveal lands a median 43% of the way
through a transcript. Before it the game contains no structural evidence at all,
only chat, and the old loop spent that stretch committing hard to whoever talked
loudest. Movement is now throttled to `PRE_EVIDENCE_MAX_DELTA` until the first
role is announced. Throttled, not frozen: a real contradiction still registers, a
confident tone no longer does.

Two things are deliberately *not* signals, both because they measure worse than
chance. Read the comments in `evidence.py` and `observer.format_votes` before
reinstating either:

- "teammates never vote for each other" -- 0.03 top-1 on its own, and it drags
  the reveal signal from 0.67 down to 0.48 when added to it
- aggression, confidence, accusation volume

## Measuring a change

Never on one game. The project drew a wrong conclusion from `llmafia-0002` three
times running.

```bash
python -m backend.batch --evidence-only                # free, all 33, under a second
python -m backend.batch --limit 6                      # the real loop, cheapest games
python -m backend.batch --evidence-only --holdout 20   # tune on 20, report on 13
```

`--limit` takes the *cheapest* games rather than the first N, so a partial run is
not a biased sample of easy ones. The output prints the chance rate and one
standard error beside the score, because "0.52 against 0.37" over six games is
not a result, and the number that says so belongs next to it.

## Tuning knobs

- `MAX_DELTA_PER_EVENT` in `observer.py` (0.30) -- the per-event suspicion cap
  once evidence exists. Lower it if the chart looks jumpy; raise it if the
  observer looks asleep.
- `PRE_EVIDENCE_MAX_DELTA` (0.06) -- the same cap before the first role reveal.
  Raising it to parity reintroduces the round-1 overcommitment, and there is a
  test that fails if you do.
- `P_TARGET_IS_LIAR_GIVEN_LIAR` / `..._GIVEN_INNOCENT` in `evidence.py` -- the
  measured likelihoods. Refit them if the dataset changes; do not hand-tune them.
