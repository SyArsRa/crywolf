#!/usr/bin/env bash
#
# One command: build the UI, start the API, wait for it to answer, open the
# page, then play a game into it. Ctrl-C stops everything.
#
# Any arguments are passed straight to the feeder, so:
#   npm run demo                     -- live run of the fallback transcript
#   npm run demo -- --replay out/run1.json --interval 2.5
#
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
PORT="${PORT:-8000}"
URL="http://127.0.0.1:${PORT}"

if [ ! -x "$PY" ]; then
  echo "No Python venv found. Run:  npm run setup"
  exit 1
fi
if [ ! -d frontend/node_modules ]; then
  echo "Frontend dependencies missing. Run:  npm run setup"
  exit 1
fi

# Default to a live run; anything passed through overrides it.
if [ "$#" -eq 0 ]; then
  set -- --transcript data/fallback_transcript.json --interval 3.0
fi

# A live run needs a key. Warn early rather than failing on /game/start.
case " $* " in
  *" --replay "*) ;;
  *)
    if [ ! -f .env ] && [ -z "${ANTHROPIC_API_KEY:-}" ]; then
      echo "warning: no .env and no ANTHROPIC_API_KEY -- a live run will fail."
      echo "         cp .env.example .env  (or use --replay for a no-key run)"
      echo
    fi
    ;;
esac

if lsof -ti:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "Port $PORT is already in use."
  echo "Stop it with:  lsof -ti:$PORT | xargs kill -9        (or run with PORT=8001)"
  exit 1
fi

echo "==> building the UI"
npm --prefix frontend run build

echo "==> starting the API on $PORT"
"$PY" -m uvicorn backend.main:app --host 127.0.0.1 --port "$PORT" &
SERVER=$!
trap 'kill "$SERVER" 2>/dev/null || true' EXIT INT TERM

printf "==> waiting for the API"
ready=""
for _ in $(seq 1 60); do
  if curl -fsS "$URL/health" >/dev/null 2>&1; then ready=1; break; fi
  printf "."
  sleep 0.25
done
echo
if [ -z "$ready" ]; then
  echo "the API never came up -- check the output above"
  exit 1
fi

echo "==> UI at $URL"
command -v open >/dev/null 2>&1 && open "$URL" || true
# Give the browser a moment to connect, so it sees the game from turn one.
sleep 1.5

echo "==> feeding the game"
"$PY" -m backend.feeder "$@" || echo "(the feeder stopped early -- the recording is still on disk)"

echo
echo "Game over. The server is still up at $URL so you can look at the final state."
echo "Ctrl-C to stop it."
wait "$SERVER"
