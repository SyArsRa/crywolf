import { useEffect, useState } from "react";

/**
 * The grade, live, while the game is still running.
 *
 * This is the feedback half of the loop made visible: the observer is being
 * scored against ground truth after every event, not once at the end. Nothing
 * here is fed back to the observer -- it never sees this, which is the whole
 * point of grading it from outside.
 *
 * Only reduced numbers are read. `actual` is in the payload for the scorer's
 * benefit, and deliberately goes unused here: putting the real liars into the
 * DOM would spoil the room on screen.
 */
export default function Scoreboard({ run }) {
  const [score, setScore] = useState(null);

  // Re-grade whenever a turn lands, so the number moves with the game.
  useEffect(() => {
    if (!run.turns.length) return setScore(null);
    let cancelled = false;
    fetch("/score")
      .then((r) => (r.ok ? r.json() : null))
      .then((data) => !cancelled && data && setScore(data))
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [run.turns.length]);

  if (!score) return null;

  const running = Math.round((score.running_accuracy ?? 0) * 100);
  const chance = run.players.length
    ? Math.round((100 * (run.setup?.deceiver_count ?? 1)) / Math.max(1, run.players.length - run.eliminated.length))
    : 0;

  return (
    <div className="scoreboard">
      <div className="score-main">
        <span className="score-label">Graded live</span>
        <span className={`score-now ${score.final_accuracy ? "hit" : "miss"}`}>
          {score.final_accuracy ? "on a liar" : "off"}
        </span>
      </div>

      <div className="score-stat">
        <span className="score-value">{running}%</span>
        <span className="score-sub">
          of {score.events_observed} events with a liar on top · chance {chance}%
        </span>
      </div>

      <div className="score-meters">
        <Meter label="consistency" value={score.consistency ?? 0} />
        <Meter label="precision@N" value={score.precision_at_n ?? 0} />
      </div>

      <div className="score-foot">
        {score.contradictions_caught} contradictions · {score.lead_changes} lead changes
        {score.committed_at ? ` · called it at ${score.committed_at}` : ""}
      </div>
    </div>
  );
}

function Meter({ label, value }) {
  return (
    <div className="meter">
      <div className="meter-head">
        <span>{label}</span>
        <span>{value.toFixed(2)}</span>
      </div>
      <div className="meter-track">
        <div className="meter-fill" style={{ width: `${Math.min(100, value * 100)}%` }} />
      </div>
    </div>
  );
}
