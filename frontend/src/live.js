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

/** Initials for the avatar. "P4" stays "P4"; "Bailey" becomes "Ba" -- a real
 *  name doesn't fit in a 56px circle, and the full name sits under it anyway. */
export function initials(name) {
  const text = String(name ?? "");
  if (text.length <= 2) return text;
  const parts = text.split(/[\s_-]+/).filter(Boolean);
  if (parts.length > 1) return (parts[0][0] + parts[1][0]).toUpperCase();
  return text.slice(0, 2);
}

/** Seat colours. Mid-dark rather than pastel: these sit on a light background
 *  and have to carry white text inside an avatar and read as a 1.5px chart
 *  line. */
export const PALETTE = [
  "#d2691e", "#2f6fd0", "#8b5cf6", "#1f9d63",
  "#c08a00", "#d4426e", "#0f9b8e", "#7c3aed",
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
        error: message.error ?? null,
        silent: true,
      };
      for (const turn of message.turns ?? []) next = applyTurn(next, turn);
      return next;
    }

    case "turn":
      return { ...applyTurn(run, message), silent: false };

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
