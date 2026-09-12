import { useEffect, useState } from "react";

/**
 * The observer's own report card, graded live.
 *
 * `GET /score` answers mid-game as readily as at the end -- it just describes
 * fewer events -- so this polls it rather than waiting for the verdict. That is
 * the whole point of the panel: the loop's feedback signal is visible while the
 * loop is still running, not only after it stops.
 *
 * What is deliberately NOT shown mid-game: whether the observer is currently
 * right. `/score` carries `actual`, and rendering a hit/miss while the game is
 * still being played would put the answer on screen before the observer reaches
 * it -- the run is the argument, and spoiling it costs more than the extra
 * number is worth. The numbers below grade *how* it is thinking, which needs no
 * reveal. The verdict lands when `complete` does.
 */
export default function Scorecard({ run }) {
  const [score, setScore] = useState(null);

  const turns = run.turns.length;
  // The last turn lands before the server marks the run finished, so grading on
  // the turn count alone leaves the card permanently one beat short of the
  // verdict. Completion is its own trigger -- it is the beat the stamp is for.
  const done = run.complete;

  useEffect(() => {
    let cancelled = false;
    // 409 until the first event lands. Not an error -- just nothing to grade.
    if (!turns) {
      setScore(null);
      return;
    }
    fetch("/score")
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => !cancelled && d && setScore(d))
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [turns, done]);

  const pct = (v) => `${Math.round((v ?? 0) * 100)}%`;
  const finished = Boolean(score?.complete) || run.complete;

  // `accuracy`/`predicted`, not `final_accuracy`/`final_verdict`.
  //
  // The final verdict reads the very last snapshot, which on a game that ends
  // by eliminating the wolf is taken *after* the wolf has been removed from the
  // distribution -- so it reports whoever leads among the survivors and grades
  // the run a miss even when the observer had the wolf at 80% one event
  // earlier. That is a scoring artefact, not a wrong read, and stamping MISSED
  // on a run that called it correctly would be the one genuinely misleading
  // thing on the board. `accuracy` is the call the observer made while the
  // wolf was still in play, which is the question being asked.
  const right = score?.accuracy;
  const called = score?.predicted;

  return (
    <section className="card scorecard">
      <div className="score-head">
        <h2>Report card</h2>
        <span className="score-sub">
          {score
            ? `graded against withheld truth · ${score.events_observed}/${score.events_expected}`
            : "graded against withheld truth"}
        </span>
      </div>

      {!score ? (
        <p className="empty-hint">Nothing graded yet.</p>
      ) : (
        <>
          <dl className="score-grid">
            <div className="score-cell">
              <dt>Precision@N</dt>
              <dd>{pct(score.precision_at_n)}</dd>
              <span className="score-note">right names in its top {score.actual?.length ?? 1}</span>
            </div>
            <div className="score-cell">
              <dt>Consistency</dt>
              <dd>{pct(score.consistency)}</dd>
              <span className="score-note">how steadily it held the read</span>
            </div>
            <div className="score-cell">
              <dt>Confidence</dt>
              <dd>{pct(score.final_confidence)}</dd>
              <span className="score-note">weight on its leading suspect</span>
            </div>
            <div className="score-cell">
              <dt>Mind changed</dt>
              <dd>{score.lead_changes ?? 0}×</dd>
              <span className="score-note">times the lead suspect moved</span>
            </div>
          </dl>

          {finished ? (
            <div className={`score-stamp ${right ? "hit" : "miss"}`}>
              {right ? "caught the wolf" : "missed the wolf"}
              <span className="score-actual">
                called {called} · answer {(score.actual ?? []).join(" · ")}
              </span>
            </div>
          ) : (
            <p className="score-sealed">verdict sealed until the run ends</p>
          )}
        </>
      )}
    </section>
  );
}
