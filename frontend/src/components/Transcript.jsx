import { useEffect, useRef } from "react";
import { isModerator } from "../live.js";

export default function Transcript({ run, colors }) {
  const scroller = useRef(null);

  // Follow the game as it arrives.
  useEffect(() => {
    const node = scroller.current;
    if (node) node.scrollTop = node.scrollHeight;
  }, [run.turns.length]);

  return (
    <section className="col transcript-col" ref={scroller}>
      <h2>Transcript</h2>
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
                R{event.round} {event.phase}
              </span>
              <span className="said">{event.statement}</span>
            </li>
          );
        })}
      </ol>
    </section>
  );
}
