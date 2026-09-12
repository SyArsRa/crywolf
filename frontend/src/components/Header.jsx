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


/** A counter that reads as a dossier gauge rather than a fraction: a big
 *  numeral, the total it is out of, and a bar that fills as the run proceeds.
 *  "24/27" told you nothing at a glance from across a room; this does. */
function Gauge({ label, value, total, fill, tone }) {
  return (
    <div className={`gauge${tone ? ` gauge-${tone}` : ""}`}>
      <span className="gauge-label">{label}</span>
      <span className="gauge-read">
        <span className="gauge-value">{value}</span>
        <span className="gauge-total">of {total}</span>
      </span>
      <span className="gauge-track">
        <span className="gauge-fill" style={{ width: `${Math.min(1, Math.max(0, fill)) * 100}%` }} />
      </span>
    </div>
  );
}

export default function Header({ run, playing, onChanged }) {
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
          <Gauge
            label="Testimony"
            value={run.turns.length}
            total={run.expected || 0}
            fill={run.expected ? run.turns.length / run.expected : 0}
          />
          <Gauge
            label="Still alive"
            value={alive}
            total={run.players.length}
            fill={run.players.length ? alive / run.players.length : 0}
            tone="alive"
          />
        </div>
      )}

      <div className="spacer" />

      <Controls playing={playing} onChanged={onChanged} />

      {!run.connected && <div className="conn down">reconnecting…</div>}
    </header>
  );
}
