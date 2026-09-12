import { lastTurn, previousSuspicion } from "../live.js";


/** "Night 1" rather than "R1 night" -- the round label is read by people who
 *  have never seen this screen before, and an R-prefix is an abbreviation they
 *  would have to be taught. */
function phaseLabel(round, phase) {
  const name = phase ? phase[0].toUpperCase() + phase.slice(1) : "";
  return `${name} ${round}`;
}

const W = 1000;
const H = 260;
const PAD_L = 34;
const PAD_R = 86;
const PAD_Y = 16;

const x = (i, n) => PAD_L + (i / Math.max(1, n - 1)) * (W - PAD_L - PAD_R);
const y = (v) => PAD_Y + (1 - v) * (H - PAD_Y * 2);

/** One player's run of suspicion, as a path plus wherever it ends.
 *
 *  A dead player's history simply stops -- that's information, not a gap to
 *  paper over -- so the line ends where they left the game and the label sits
 *  at that point rather than being dragged to the right edge. */
function series(run, player) {
  const pts = [];
  run.history.forEach((snap, i) => {
    const v = snap?.[player];
    if (typeof v === "number") pts.push([x(i, run.history.length), y(v), v]);
  });
  return pts;
}

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
  const n = run.history.length;

  // Everyone who ever had a number, leader last so their line is drawn on top.
  const tracked = run.players.filter((p) => run.history.some((s) => typeof s?.[p] === "number"));
  const leader = living[0]?.[0];
  const ordered = [...tracked].sort((a, b) => (a === leader ? 1 : b === leader ? -1 : 0));

  // Phase bands, in chart coordinates, so night reads as night behind the ink.
  let cursor = 0;
  const bands = segments.map((seg) => {
    const from = x(cursor, n);
    cursor += seg.count;
    return { ...seg, from, to: x(cursor - 1, n), key: `${seg.round}-${seg.phase}-${cursor}` };
  });

  return (
    <section className={`belief-board card${state?.deliberated ? " lurching" : ""}`}>
      <div className="belief-head">
        <h2>How suspicion changed</h2>
        <span className="belief-sub">
          {state?.deliberated
            ? "Rewritten from scratch — it just took a second, closer look"
            : "Chance each player is lying, updated after every message"}
        </span>
      </div>

      {n < 2 ? (
        <p className="empty-hint">Waiting for the first belief state.</p>
      ) : (
        <div className="chart-wrap">
          <svg className="belief-chart" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none">
            {/* Phase bands first: night, day and vote tint the paper behind. */}
            {bands.map((b) => (
              <rect
                key={b.key}
                className={`band band-${b.phase}`}
                x={b.from}
                y={PAD_Y}
                width={Math.max(0, b.to - b.from)}
                height={H - PAD_Y * 2}
              />
            ))}

            {/* Ruled lines, the way a ledger page is ruled. */}
            {[0, 0.25, 0.5, 0.75, 1].map((v) => (
              <g key={v}>
                <line className={`rule${v === 0.5 ? " rule-mid" : ""}`} x1={PAD_L} x2={W - PAD_R} y1={y(v)} y2={y(v)} />
                <text className="rule-label" x={PAD_L - 8} y={y(v) + 3} textAnchor="end">
                  {v * 100}
                </text>
              </g>
            ))}

            {ordered.map((player) => {
              const pts = series(run, player);
              if (pts.length < 2) return null;
              const isLeader = player === leader;
              const isDead = dead.includes(player);
              const [ex, ey, ev] = pts[pts.length - 1];
              return (
                <g key={player} className={`line${isLeader ? " leader" : ""}${isDead ? " gone" : ""}`}>
                  <path d={`M${pts.map((p) => `${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(" L")}`} stroke={colors[player]} />
                  <circle cx={ex} cy={ey} r={isLeader ? 4 : 3} fill={colors[player]} />
                  <text className="end-label" x={ex + 9} y={ey + 4} fill={colors[player]}>
                    {player} {isDead ? "out" : `${Math.round(ev * 100)}%`}
                  </text>
                </g>
              );
            })}
          </svg>
        </div>
      )}

      {/* The ledger underneath: the current number, and which way it just moved. */}
      <div className="belief-rows">
        {living.map(([player, score], rank) => {
          const before = previous[player];
          const diff = before === undefined ? 0 : score - before;
          return (
            <div
              key={player}
              className={`belief-row${rank < topN ? " top" : ""}`}
              style={{ "--seat-color": colors[player] }}
            >
              <span className="swatch" style={{ background: colors[player] }} />
              <span className="who">{player}</span>
              <span className="bar">
                <span className="bar-fill" style={{ width: `${score * 100}%`, background: colors[player] }} />
              </span>
              <span className="pct">{(score * 100).toFixed(0)}%</span>
              {Math.abs(diff) >= 0.005 ? (
                <span className={`delta ${diff > 0 ? "up" : "down"}`}>
                  {diff > 0 ? "▲" : "▼"}
                  {Math.abs(diff * 100).toFixed(0)}
                </span>
              ) : (
                <span className="delta" />
              )}
            </div>
          );
        })}
        {dead.map((player) => (
          <div key={player} className="belief-row dead">
            <span className="swatch" style={{ background: colors[player] }} />
            <span className="who">{player}</span>
            <span className="bar" />
            <span className="out">out</span>
            <span className="delta" />
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
                {phaseLabel(seg.round, seg.phase)}
              </span>
            ))}
          </div>
        </div>
      )}
    </section>
  );
}
