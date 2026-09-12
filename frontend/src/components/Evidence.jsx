import { lastTurn } from "../live.js";

export default function Evidence({ run }) {
  const state = lastTurn(run)?.state;
  const contradictions = state?.contradictions_noticed ?? [];
  const claims = state?.claims_tracked ?? {};

  return (
    <section className="card evidence-col">
      <div className="evidence-head">
        <h2>Evidence board</h2>
        <div className="evidence-counts">
          <span className="count-pin">{contradictions.length} times someone changed their story</span>
          <span className="count-pin">{Object.keys(claims).length} statements remembered</span>
        </div>
      </div>

      {run.error && <p className="banner">Run stopped: {run.error}</p>}

      <ul className="contradictions">
        {!contradictions.length && <li className="empty">nobody has changed their story yet</li>}
        {contradictions
          .slice()
          .reverse()
          .map((c, i) => (
            <li key={`${c.player}-${c.earlier}-${i}`}>
              <span className="who">{c.player}</span> · round {c.round_noticed}
              <span className="quote">
                <span className="quote-label">earlier</span> “{c.earlier}”
                <br />
                <span className="quote-label">now</span> “{c.now}”
              </span>
            </li>
          ))}
      </ul>
    </section>
  );
}
