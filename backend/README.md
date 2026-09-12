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

## Tuning knobs

- `MAX_DELTA_PER_EVENT` in `observer.py` (0.30) — the per-event suspicion cap.
  Lower it if the chart looks jumpy; raise it if the observer looks asleep.
