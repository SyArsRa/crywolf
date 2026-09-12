import { lastTurn, previousSuspicion } from "../live.js";

function SuspicionBars({ run, colors }) {
  const state = lastTurn(run)?.state;
  const suspicion = state?.suspicion ?? {};
  const previous = previousSuspicion(run);
  const entries = Object.entries(suspicion).sort((a, b) => b[1] - a[1]);
  // Mafia has two liars, so two bars are "the call", not one.
  const topN = Math.max(1, run.setup?.deceiver_count ?? 1);

  if (!entries.length) {
    return <p className="empty-hint">Waiting for the first belief state.</p>;
  }

  return (
    <div>
      {entries.map(([player, score], rank) => {
        const before = previous[player];
        const diff = before === undefined ? 0 : score - before;
        const dead = run.eliminated.includes(player);
        return (
          <div
            key={player}
            className={`bar${rank < topN ? " top" : ""}${dead ? " dead" : ""}`}
            style={{ "--seat-color": colors[player] }}
          >
            <div className="bar-head">
              <span className="who">{player}</span>
              {Math.abs(diff) >= 0.005 && (
                <span className={`delta ${diff > 0 ? "up" : "down"}`}>
                  {diff > 0 ? "▲" : "▼"}
                  {Math.abs(diff * 100).toFixed(0)}
                </span>
              )}
              <span className="pct">{(score * 100).toFixed(0)}%</span>
            </div>
            <div className="track">
              <div className="fill" style={{ width: `${(score * 100).toFixed(1)}%` }} />
            </div>
          </div>
        );
      })}
    </div>
  );
}

function Contradictions({ run }) {
  // The observer carries these forward cumulatively, so the latest state holds
  // the complete list -- no need to accumulate them here.
  const found = lastTurn(run)?.state?.contradictions_noticed ?? [];
  if (!found.length) {
    return (
      <ul className="contradictions">
        <li className="empty">none yet</li>
      </ul>
    );
  }
  return (
    <ul className="contradictions">
      {found
        .slice()
        .reverse()
        .map((c, i) => (
          <li key={`${c.player}-${c.earlier}-${i}`}>
            <span className="who">{c.player}</span> · round {c.round_noticed}
            <span className="quote">
              said “{c.earlier}”
              <br />
              then “{c.now}”
            </span>
          </li>
        ))}
    </ul>
  );
}

/** Suspicion over time. A dead player leaves the distribution, so their line
 *  simply stops -- which is information, not a gap to paper over. */
function Chart({ run, colors }) {
  const W = 300;
  const H = 130;
  const pad = 4;
  const n = run.history.length;
  if (n < 2) return <svg className="chart" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" />;

  const xAt = (i) => (i / (n - 1)) * W;
  const yAt = (v) => pad + (1 - v) * (H - pad * 2);

  const last = run.history[n - 1] ?? {};
  const names = Object.keys(last);
  const leader = names.length
    ? names.reduce((best, p) => (last[p] > last[best] ? p : best), names[0])
    : null;

  return (
    <svg className="chart" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none">
      {[0, 0.25, 0.5, 0.75, 1].map((frac) => (
        <line
          key={frac}
          className="grid"
          x1={0}
          x2={W}
          y1={pad + (H - pad * 2) * frac}
          y2={pad + (H - pad * 2) * frac}
        />
      ))}
      {run.players.map((player) => {
        const points = [];
        run.history.forEach((snapshot, i) => {
          const v = snapshot?.[player];
          if (typeof v === "number") points.push(`${xAt(i).toFixed(1)},${yAt(v).toFixed(1)}`);
        });
        if (points.length < 2) return null;
        return (
          <path
            key={player}
            d={`M${points.join(" L")}`}
            stroke={colors[player] ?? "#888"}
            strokeWidth={player === leader ? 2.6 : 1.4}
            opacity={player === leader ? 1 : 0.65}
          />
        );
      })}
    </svg>
  );
}

export default function Panel({ run, colors }) {
  const reasoning = lastTurn(run)?.state?.reasoning;

  return (
    <aside className="col panel-col">
      {run.error && <p className="banner">Run stopped: {run.error}</p>}

      <h2>Suspicion</h2>
      <SuspicionBars run={run} colors={colors} />

      <h2>Observer's note</h2>
      <p className="reasoning" key={run.turns.length}>
        {reasoning || "—"}
      </p>

      <h2>Contradictions caught</h2>
      <Contradictions run={run} />

      <h2>Suspicion over time</h2>
      <Chart run={run} colors={colors} />
    </aside>
  );
}
