#!/usr/bin/env bash
#
# Run both halves at once, each with live reload, and auto-feed a recorded game
# so the page is never sitting empty:
#
#   backend   uvicorn on :8000, --reload (watches Python)
#   frontend  vite on :5173, hot module reload (watches src/)
#   autofeed  replays a recording into the API a couple of seconds after boot
#
# Open http://localhost:5173 -- vite proxies the API and the websocket through
# to the backend, so the page behaves identically but UI edits appear instantly
# with no rebuild.
#
# Knobs:
#   AUTOFEED=0              don't feed anything, just run the servers
#   AUTOFEED_LOOP=1         replay on repeat, so the UI always has motion
#   AUTOFEED_FILE=path      replay this recording instead of the newest one
#   AUTOFEED_INTERVAL=2.0   seconds per turn
#   API_PORT / UI_PORT      change ports
#
# Ctrl-C stops everything.
#
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
API_PORT="${API_PORT:-8000}"
UI_PORT="${UI_PORT:-5173}"
API_URL="http://127.0.0.1:${API_PORT}"

AUTOFEED="${AUTOFEED:-1}"
AUTOFEED_LOOP="${AUTOFEED_LOOP:-0}"
AUTOFEED_INTERVAL="${AUTOFEED_INTERVAL:-2.0}"

if [ ! -x "$PY" ]; then
  echo "No Python venv found. Run:  npm run setup"
  exit 1
fi
if [ ! -d frontend/node_modules ]; then
  echo "Frontend dependencies missing. Run:  npm run setup"
  exit 1
fi

for port in "$API_PORT" "$UI_PORT"; do
  if lsof -ti:"$port" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "Port $port is already in use."
    echo "Stop it with:  lsof -ti:$port | xargs kill -9"
    exit 1
  fi
done

# Prefer a real recorded run; fall back to the sample that ships with the repo.
# Replay-of-a-replay files are skipped -- they're just copies of their source.
pick_recording() {
  if [ -n "${AUTOFEED_FILE:-}" ]; then
    echo "$AUTOFEED_FILE"
    return
  fi
  local newest
  newest=$(ls -t runs/*.json 2>/dev/null | grep -v -- '-replay\.json$' | head -1 || true)
  if [ -n "$newest" ]; then
    echo "$newest"
  elif [ -f data/demo_run.json ]; then
    echo "data/demo_run.json"
  fi
}

pids=()
cleanup() {
  for pid in "${pids[@]:-}"; do
    [ -n "${pid:-}" ] || continue
    kill -- "-$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

set -m

echo "==> backend  $API_URL"
"$PY" -m uvicorn backend.main:app --reload --host 127.0.0.1 --port "$API_PORT" &
pids+=($!)

# localhost, not 127.0.0.1: vite binds to ::1 only, so the IPv4 spelling is
# refused even though the port is open.
echo "==> frontend http://localhost:$UI_PORT   <- open this one"
CRYWOLF_API="$API_URL" UI_PORT="$UI_PORT" \
  npm --prefix frontend run dev -- --port "$UI_PORT" &
pids+=($!)

set +m

if [ "$AUTOFEED" = "1" ]; then
  RECORDING="$(pick_recording)"
  if [ -z "$RECORDING" ] || [ ! -f "$RECORDING" ]; then
    echo "==> autofeed: no recording found, skipping (the page will say 'waiting')"
  else
    echo "==> autofeed: $RECORDING  (AUTOFEED=0 to disable)"
    (
      # Wait for the API rather than guessing -- uvicorn's first boot is slower
      # than it looks, and posting early just 502s.
      for _ in $(seq 1 80); do
        curl -fsS "$API_URL/health" >/dev/null 2>&1 && break
        sleep 0.25
      done
      # A beat for the browser's websocket to attach, so it sees turn one. A
      # late-joining page still backfills, so this is polish, not correctness.
      sleep 2
      while true; do
        "$PY" -m backend.feeder --api "$API_URL" \
          --replay "$RECORDING" --interval "$AUTOFEED_INTERVAL" >/dev/null 2>&1 || true
        [ "$AUTOFEED_LOOP" = "1" ] || break
        sleep 3
      done
    ) &
    pids+=($!)
  fi
fi

echo
echo "Feed a different game any time, in another terminal:"
echo "  npm run feed            # live run, needs ANTHROPIC_API_KEY"
echo "  npm run feed:replay     # the sample recording"
echo

# If either server dies, stop the other rather than leaving half a stack up.
wait -n 2>/dev/null || wait
