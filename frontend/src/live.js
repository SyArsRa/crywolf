/**
 * The websocket, and every piece of state derived from it.
 *
 * The server is the only source of truth: this reduces the message stream into
 * a run, and the components render it. Message shapes are documented at the top
 * of backend/hub.py.
 *
 *   snapshot   roster + every turn so far   -> applied silently, no animation
 *   turn       {index, event, state}        -> animated
 *   run_error  the observer failed
 *   game_end   last event processed
 */

import { useEffect, useReducer, useRef } from "react";

/** Kept in step with NARRATORS in backend/schema.py -- a speaker that isn't a
 *  player narrates from a banner instead of getting a seat at the table. */
export const NARRATORS = new Set([
  "moderator", "narrator", "host", "system", "gm", "game", "game-manager", "mod",
]);

export const isModerator = (speaker) =>
  NARRATORS.has(String(speaker ?? "").toLowerCase());

/** Seat colours: the inks a case file would actually be written in --
 *  fountain blue, crimson, lamp black, oxblood, sepia, iron gall, forest,
 *  violet copying-pencil. Mid-dark rather than pastel, because these sit on
 *  manila and have to read as a 1.5px chart line as well as a name.
 *
 *  The previous set was a generic categorical palette (indigo/violet/teal);
 *  it separated the series fine but looked like a charting library's default
 *  on a board that is meant to look hand-kept. */
export const PALETTE = [
  "#27476b", "#8c2119", "#1d1a17", "#5c3a1e",
  "#3f5d3a", "#6b3560", "#7a5c1f", "#2f5d63",
];

const EMPTY = {
  connected: false,
  setup: null,
  players: [],
  expected: 0,
  turns: [],
  history: [],
  killed: [],
  eliminated: [],
  complete: false,
  /** The observer's own decision to stop watching, once it makes one. */
  committed: null,
  error: null,
  /** true when the most recent change came from a snapshot, so the UI can skip
   *  the theatre and land straight on the current state */
  silent: true,
};

/** Fold one turn into the run. */
function applyTurn(run, turn) {
  const deadBefore = run.eliminated.length;
  const eliminated = run.eliminated.slice();

  for (const player of turn.state?.eliminated ?? []) {
    if (!eliminated.includes(player)) eliminated.push(player);
  }
  // `event.eliminated` is an optional hint from the feed; the observer's own
  // list is authoritative. Take whichever arrives.
  if (turn.event?.eliminated && !eliminated.includes(turn.event.eliminated)) {
    eliminated.push(turn.event.eliminated);
  }

  return {
    ...run,
    turns: [...run.turns, turn],
    history: [...run.history, { ...(turn.state?.suspicion ?? {}) }],
    killed: [...run.killed, eliminated.length > deadBefore],
    eliminated,
  };
}

function reducer(run, message) {
  switch (message.type) {
    case "__open":
      return { ...run, connected: true };
    case "__closed":
      return { ...run, connected: false };

    case "snapshot": {
      // A fresh run, or this tab joining one already in progress.
      let next = {
        ...EMPTY,
        connected: true,
        setup: message.setup ?? null,
        players: message.setup?.players ?? [],
        expected: message.events_expected ?? 0,
        complete: Boolean(message.complete),
        committed: message.committed ?? null,
        error: message.error ?? null,
        silent: true,
      };
      for (const turn of message.turns ?? []) next = applyTurn(next, turn);
      return next;
    }

    case "turn":
      return { ...applyTurn(run, message), silent: false };

    // The loop stopped itself: it has seen enough and is naming its suspects.
    case "committed":
      return { ...run, committed: message, silent: false };

    case "game_end":
      return { ...run, complete: Boolean(message.complete), silent: false };

    case "run_error":
      return { ...run, error: message.detail ?? "the observer failed" };

    default:
      return run;
  }
}

export function useLiveRun() {
  const [run, dispatch] = useReducer(reducer, EMPTY);
  const retry = useRef(null);

  useEffect(() => {
    let socket;
    let stopped = false;

    const connect = () => {
      if (stopped) return;
      const proto = location.protocol === "https:" ? "wss:" : "ws:";
      socket = new WebSocket(`${proto}//${location.host}/live`);

      socket.onopen = () => dispatch({ type: "__open" });
      socket.onmessage = (e) => {
        try {
          dispatch(JSON.parse(e.data));
        } catch (err) {
          console.error("unparsable message", err, e.data);
        }
      };
      socket.onclose = () => {
        dispatch({ type: "__closed" });
        // The server restarting mid-demo must not need a page refresh.
        clearTimeout(retry.current);
        retry.current = setTimeout(connect, 1500);
      };
      socket.onerror = () => socket.close();
    };

    connect();
    return () => {
      stopped = true;
      clearTimeout(retry.current);
      socket?.close();
    };
  }, []);

  return run;
}

/* ------------------------------------------------------------- selectors */

export const lastTurn = (run) => run.turns[run.turns.length - 1] ?? null;

export function seatColors(players) {
  return Object.fromEntries(players.map((p, i) => [p, PALETTE[i % PALETTE.length]]));
}

/** Suspicion from the turn before last, for the per-bar deltas. */
export const previousSuspicion = (run) => run.history[run.history.length - 2] ?? {};

/**
 * The observer's final call.
 *
 * Mirrors scoring.verdict(): the last snapshot produced by an event that killed
 * nobody. A game ends by eliminating someone and an eliminated player drops out
 * of the distribution, so reading the very last state would ask "who do you
 * suspect?" after the suspect had already been voted out -- an observer that
 * named the werewolf correctly would look wrong.
 */
export function verdict(run) {
  // Mafia has more than one liar, and the setup says how many -- so the call is
  // the top `deceiver_count` suspects, not always a single name.
  const count = Math.max(1, run.setup?.deceiver_count ?? 1);

  for (let i = run.history.length - 1; i >= 0; i--) {
    if (run.killed[i]) continue;
    const snapshot = run.history[i];
    const names = Object.keys(snapshot ?? {});
    if (!names.length) continue;
    const ranked = names.sort((a, b) => snapshot[b] - snapshot[a]).slice(0, count);
    return {
      players: ranked,
      confidence: snapshot[ranked[0]],
      role: run.setup?.deceiver_role ?? "werewolf",
    };
  }
  return null;
}
