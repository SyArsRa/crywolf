import { lastTurn } from "../live.js";
import Controls from "./Controls.jsx";

const STATE_LABEL = {
  idle: "Idle",
  playing: "Playing",
  vote: "Vote",
  verdict: "Verdict",
  failed: "Failed",
};

/** One word for where the game is right now. `playing` only covers games the
 *  server is driving itself -- one fed in from the feeder still has turns
 *  arriving, so the transcript filling up counts too. */
function currentState(run, playing) {
  if (run.error) return "failed";
  if (run.complete) return "verdict";
  const phase = lastTurn(run)?.event?.phase;
  if (phase === "vote") return "vote";
  if (playing || run.turns.length > 0) return "playing";
  return "idle";
}

export default function Header({ run, playing, onChanged }) {
  const event = lastTurn(run)?.event;
  const state = currentState(run, playing);
  const alive = run.players.length - run.eliminated.length;

  return (
    <header className="hud">
      <div className="hud-plaque">
        <span className="hud-mark">🐺</span>
        <span className="hud-brand">Cry Wolf</span>
      </div>

      <span className={`token token-${state}`}>{STATE_LABEL[state]}</span>

      {run.turns.length > 0 && (
        <div className="hud-stats">
          {event && (
            <span className="stat">
              R{event.round} · {event.phase}
            </span>
          )}
          <span className="stat">
            {run.turns.length}/{run.expected || 0}
          </span>
          <span className="stat">
            {alive}/{run.players.length} alive
          </span>
        </div>
      )}

      <div className="spacer" />

      <Controls playing={playing} onChanged={onChanged} />

      <div className={`conn ${run.connected ? "live" : "down"}`}>
        <span className="dot" />
        {!run.connected && <span>reconnecting…</span>}
      </div>
    </header>
  );
}
