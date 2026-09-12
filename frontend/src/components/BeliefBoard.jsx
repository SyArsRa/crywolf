import { lastTurn, previousSuspicion } from "../live.js";

/** A tiny per-player line of P(mafia) over the game so far. A dead player's
 *  history simply stops -- that's information, not a gap to paper over. */
function Sparkline({ run, player, color }) {
  const W = 200;
  const H = 30;
  const pad = 2;
  const points = [];
  run.history.forEach((snapshot, i) => {
    const v = snapshot?.[player];
    if (typeof v !== "number") return;
    const x = (i / Math.max(1, run.history.length - 1)) * W;
    const y = pad + (1 - v) * (H - pad * 2);
    points.push(`${x.toFixed(1)},${y.toFixed(1)}`);
  });
  if (points.length < 2) {
    return <svg className="spark" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" />;
  }
  return (
    <svg className="spark" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none">
      <path d={`M${points.join(" L")}`} stroke={color ?? "#888"} />
    </svg>
  );
}

/** The phases the game has passed through, as proportional segments -- the
 *  same horizontal axis every sparkline above is drawn against. */
function phaseSegments(run) {
  const segments = [];
  run.turns.forEach((turn) => {
    const { round, phase } = turn.event;
    const last = segments[segments.length - 1];
    if (last && last.round === round && last.phase === phase) last.count += 1;
    else segments.push({ round, phase, count: 1 });
  });
  return segments;
}

export default function BeliefBoard({ run, colors }) {
  const state = lastTurn(run)?.state;
  const suspicion = state?.suspicion ?? {};
  const previous = previousSuspicion(run);
  const topN = Math.max(1, run.setup?.deceiver_count ?? 1);

  const living = Object.entries(suspicion).sort((a, b) => b[1] - a[1]);
  const dead = run.eliminated.filter((p) => run.players.includes(p));
  const segments = phaseSegments(run);

  return (
    <section className={`belief-board card${state?.deliberated ? " lurching" : ""}`}>
      <div className="belief-head">
        <h2>Belief over time</h2>
        <span className="belief-sub">
          {state?.deliberated
            ? "rewritten wholesale — the deep pass re-read the round"
            : "P(mafia) per player, rewritten after every message"}
        </span>
      </div>

      {!living.length && !dead.length && (
        <p className="empty-hint">Waiting for the first belief state.</p>
      )}

      <div className="belief-rows">
        {living.map(([player, score], rank) => {
          const before = previous[player];
          const diff = before === undefined ? 0 : score - before;
          return (
            <div key={player} className={`belief-row${rank < topN ? " top" : ""}`} style={{ "--seat-color": colors[player] }}>
              <span className="who">{player}</span>
              <Sparkline run={run} player={player} color={colors[player]} />
              <span className="pct">{(score * 100).toFixed(0)}%</span>
              {Math.abs(diff) >= 0.005 && (
                <span className={`delta ${diff > 0 ? "up" : "down"}`}>
                  {diff > 0 ? "▲" : "▼"}
                  {Math.abs(diff * 100).toFixed(0)}
                </span>
              )}
            </div>
          );
        })}
        {dead.map((player) => (
          <div key={player} className="belief-row dead">
            <span className="who">{player}</span>
            <span className="out">out</span>
          </div>
        ))}
      </div>

      {segments.length > 0 && (
        <div className="phase-axis">
          <span />
          <div className="phase-track">
            {segments.map((seg, i) => (
              <span
                key={`${seg.round}-${seg.phase}-${i}`}
                className={`phase-seg${i === segments.length - 1 ? " current" : ""}`}
                style={{ flexGrow: seg.count }}
              >
                R{seg.round} {seg.phase}
              </span>
            ))}
          </div>
        </div>
      )}
    </section>
  );
}
