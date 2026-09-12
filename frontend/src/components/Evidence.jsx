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
          <span className="count-pin">{contradictions.length} contradictions</span>
          <span className="count-pin">{Object.keys(claims).length} claims tracked</span>
        </div>
      </div>

      {run.error && <p className="banner">Run stopped: {run.error}</p>}

      <ul className="contradictions">
        {!contradictions.length && <li className="empty">none caught yet</li>}
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
