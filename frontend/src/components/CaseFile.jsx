import { lastTurn } from "../live.js";

/** Leading suspect + the observer's latest note, side by side like a dossier
 *  opened on the table next to the game. */
export default function CaseFile({ run, colors }) {
  const state = lastTurn(run)?.state;
  const suspicion = state?.suspicion ?? {};
  const reasoning = state?.reasoning;
  const entries = Object.entries(suspicion).sort((a, b) => b[1] - a[1]);
  const [leader, leaderScore] = entries[0] ?? [null, 0];
  const runnerUp = entries[1]?.[1] ?? 0;
  const gap = Math.round((leaderScore - runnerUp) * 100);

  return (
    <section className="case-row">
      <div className="card suspect-card" style={{ "--seat-color": leader ? colors[leader] : undefined }}>
        <h2>Most suspected right now</h2>
        {leader ? (
          <>
            <div className="suspect-name">{leader}</div>
            <div className="suspect-pct">
              {(leaderScore * 100).toFixed(0)}%
              {gap > 0 && <span className="suspect-gap">+{gap} over {entries[1][0]}</span>}
            </div>
            <div className="suspect-caveat">still a guess, not proof</div>
          </>
        ) : (
          <p className="empty-hint">Waiting for the first read.</p>
        )}
      </div>

      <div className="card reading-card">
        <div className="reading-head">
          <h2>Why it thinks that</h2>
          {state?.deliberated && (
            <span className="deep-badge">it just re-read the whole round</span>
          )}
        </div>
        <p
          className={`reasoning${state?.deliberated ? " deep" : ""}`}
          key={run.turns.length}
        >
          {reasoning || "Nothing written yet."}
        </p>
      </div>
    </section>
  );
}
