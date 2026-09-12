import { useEffect, useMemo, useState } from "react";
import { isModerator, lastTurn, verdict } from "../live.js";
import Portrait from "./Portrait.jsx";

const BUBBLE_MS = 7000;
const SWEEP_MS = 1600;
const RETHINK_MS = 3200;

/** Seats are placed round a circle, starting at the top and going clockwise. */
function seatLayout(players) {
  const n = players.length || 1;
  return players.map((player, i) => {
    const angle = ((-90 + (360 / n) * i) * Math.PI) / 180;
    const radius = 41; // percent of the (square) stage
    const x = 50 + radius * Math.cos(angle);
    const y = 50 + radius * Math.sin(angle);
    // Bubbles always sit above the speaker, like a comic panel -- the stage
    // carries extra top padding so the topmost seat's bubble still fits.
    //
    // Horizontally they have to lean inward: a bubble centred above a seat on
    // the right of the circle runs off the edge of the stage and gets clipped.
    // The tail still points at the seat.
    const align = x > 62 ? "right" : x < 38 ? "left" : "center";
    // Speech points away from the table, never across it. Seats below the
    // centre line used to put their bubble "above", which with the larger
    // cards lands it on top of the round label in the middle of the table --
    // the two things you most need to read at once.
    const side = y > 55 ? "below" : "above";
    return { player, index: i, x, y, side, align };
  });
}

function Seat({ seat, color, dead, speaking, suspicion, bubble }) {
  return (
    <div
      className={`seat${dead ? " dead" : ""}${speaking ? " speaking" : ""}`}
      style={{ left: `${seat.x}%`, top: `${seat.y}%`, "--seat-color": color }}
    >
      <div
        className="avatar"
        style={{ "--suspicion": Math.min(1, (suspicion ?? 0) * 2.2).toFixed(2) }}
      >
        <Portrait index={seat.index} />
      </div>
      <div className="name">{seat.player}</div>
      {bubble && <div className={`bubble ${seat.side} align-${seat.align}`}>{bubble}</div>}
    </div>
  );
}

export default function Table({ run, colors }) {
  const turn = lastTurn(run);
  const event = turn?.event ?? null;
  const state = turn?.state ?? null;
  const phase = event?.phase ?? null;
  const round = event?.round ?? null;

  const seats = useMemo(() => seatLayout(run.players), [run.players]);

  /* The speech bubble lives for a few seconds, then goes. Keyed on the turn
     index so a new line always replaces the previous one rather than stacking. */
  const [speech, setSpeech] = useState(null);
  useEffect(() => {
    if (!event || isModerator(event.speaker)) return setSpeech(null);
    setSpeech({ speaker: event.speaker, text: event.statement });
    const timer = setTimeout(() => setSpeech(null), BUBBLE_MS);
    return () => clearTimeout(timer);
  }, [turn?.index]);

  /* Phase changes sweep a label across the stage -- but never while catching up
     on a snapshot, or a page refresh replays every transition at once. */
  const [sweep, setSweep] = useState(null);
  useEffect(() => {
    if (!phase || run.silent) return;
    setSweep({ key: `${round}-${phase}`, label: phase === "vote" ? "the vote" : `${phase} ${round}` });
    const timer = setTimeout(() => setSweep(null), SWEEP_MS);
    return () => clearTimeout(timer);
  }, [round, phase]);

  /* The deep pass is the loop's self-correction step, and it is otherwise
     invisible: a role reveal lets the observer re-read the whole round against
     graded evidence and rewrite its belief wholesale. That deserves the same
     theatre as a phase change, because it is the more interesting event. */
  const [rethink, setRethink] = useState(null);
  useEffect(() => {
    if (!state?.deliberated || run.silent) return;
    setRethink(turn.index);
    const timer = setTimeout(() => setRethink(null), RETHINK_MS);
    return () => clearTimeout(timer);
  }, [turn?.index, state?.deliberated]);

  const stars = useMemo(
    () =>
      Array.from({ length: 70 }, (_, i) => ({
        id: i,
        left: `${Math.random() * 100}%`,
        top: `${Math.random() * 62}%`,
        dur: `${(2.5 + Math.random() * 4).toFixed(2)}s`,
        delay: `${(Math.random() * 4).toFixed(2)}s`,
      })),
    []
  );

  const call = run.complete || run.error ? verdict(run) : null;
  const narration = event && isModerator(event.speaker) ? event.statement : null;

  const alive = run.players.length - run.eliminated.length;

  return (
    <section
      className={`stage${rethink !== null ? " rethinking" : ""}`}
      data-phase={phase ?? undefined}
      data-seats={run.players.length || undefined}
    >
      <div className="room-label">
        <h2>The room</h2>
        <span className="room-sub">
          {alive} of {run.players.length || 0} in play
        </span>
      </div>

      <div className="sky">
        <div className="stars">
          {stars.map((s) => (
            <div
              key={s.id}
              className="star"
              style={{ left: s.left, top: s.top, "--dur": s.dur, "--delay": s.delay }}
            />
          ))}
        </div>
        <div className="celestial" />
      </div>

      <div className="table-stage">
        <div className="table">
          <div>
            <div className="round-label">
              {round ? `Round ${round} · ${phase}` : "waiting"}
            </div>
            {call && (
              <div className="verdict">
                {run.complete ? "Verdict" : "Stopped early · leaning"}:{" "}
                <strong>{call.players.join(", ")}</strong>
                {call.players.length === 1 && ` at ${(call.confidence * 100).toFixed(0)}%`}
                <span className="verdict-role"> — the {call.role}</span>
              </div>
            )}
          </div>
        </div>

        {seats.map((seat) => (
          <Seat
            key={seat.player}
            seat={seat}
            color={colors[seat.player]}
            dead={run.eliminated.includes(seat.player)}
            speaking={speech?.speaker === seat.player}
            suspicion={state?.suspicion?.[seat.player]}
            bubble={speech?.speaker === seat.player ? speech.text : null}
          />
        ))}
      </div>

      <div className={`narrator${narration ? " show" : ""}`}>{narration}</div>

      {sweep && (
        <div className="sweep run" key={sweep.key}>
          <span>{sweep.label}</span>
        </div>
      )}

      {rethink !== null && (
        <div className="rethink-beat" key={rethink}>
          <span className="rethink-kicker">a role was revealed</span>
          <span className="rethink-title">re-reading the round</span>
          <span className="rethink-sub">deep pass · belief rewritten against graded evidence</span>
        </div>
      )}

      {run.committed && (
        <div className="commit-beat">
          <span className="commit-kicker">the observer stopped watching</span>
          <span className="commit-title">{run.committed.players.join(" · ")}</span>
          <span className="commit-sub">
            called it at message {run.committed.events_observed} of {run.expected || "?"} ·{" "}
            {(run.committed.confidence * 100).toFixed(0)}% confident
          </span>
        </div>
      )}
    </section>
  );
}
