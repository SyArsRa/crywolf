import { useEffect, useRef } from "react";
import { isModerator } from "../live.js";


/** "Night 1" rather than "R1 night" -- the round label is read by people who
 *  have never seen this screen before, and an R-prefix is an abbreviation they
 *  would have to be taught. */
function phaseLabel(round, phase) {
  const name = phase ? phase[0].toUpperCase() + phase.slice(1) : "";
  return `${name} ${round}`;
}

export default function Transcript({ run, colors }) {
  const scroller = useRef(null);

  // Follow the game as it arrives.
  useEffect(() => {
    const node = scroller.current;
    if (node) node.scrollTop = node.scrollHeight;
  }, [run.turns.length]);

  return (
    <section className="card transcript-col" ref={scroller}>
      <div className="transcript-head">
        <h2>Transcript</h2>
        <span className="transcript-count">
          {run.turns.length} / {run.expected || 0}
        </span>
      </div>
      {!run.turns.length && (
        <p className="empty-hint">Nothing yet — pick a game and press Start.</p>
      )}
      <ol className="log">
        {run.turns.map((turn) => {
          const { event } = turn;
          const mod = isModerator(event.speaker);
          return (
            <li
              key={turn.index}
              className={mod ? "mod" : undefined}
              style={{ "--seat-color": colors[event.speaker] ?? "var(--text)" }}
            >
              <span className="who">{event.speaker}</span>
              <span className="meta">
                {phaseLabel(event.round, event.phase)}
              </span>
              <span className="said">{event.statement}</span>
            </li>
          );
        })}
      </ol>
    </section>
  );
}
