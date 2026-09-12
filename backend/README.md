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

`--json` saves a full run so the UI can replay it — that's the stage fallback if
a live run misbehaves.

## Which model runs

`llm.py` is the only file that knows. `observer.py` asks for a validated object
and doesn't care who produced it.

```bash
pip install -r requirements.txt
export GEMINI_API_KEY=...          # free key: https://aistudio.google.com/apikey
```

Gemini is the default. `gemini-2.5-flash` rather than pro because one replay is
27 calls and the free tier's pro quota won't survive a tuning session.

| Variable | Default | |
|---|---|---|
| `CRYWOLF_PROVIDER` | `gemini` | or `anthropic`, if a key ever appears |
| `CRYWOLF_MODEL` | per provider | e.g. `gemini-2.5-pro` |
| `CRYWOLF_EFFORT` | `medium` | Anthropic only |

Rate limits and transient 5xx are retried with backoff inside `llm.py` — free-tier
Gemini will throttle a 27-call replay, and that's expected, not a failure.

To add a third provider, write a class with one `structured(system, user, schema)`
method returning the validated model. Nothing else changes.

## Tuning knobs

- `MAX_DELTA_PER_EVENT` in `observer.py` (0.30) — the per-event suspicion cap.
  Lower it if the chart looks jumpy; raise it if the observer looks asleep.
