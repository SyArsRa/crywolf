import { useCallback, useEffect, useMemo, useState } from "react";
import { lastTurn, seatColors, useLiveRun } from "./live.js";
import Table from "./components/Table.jsx";
import Panel from "./components/Panel.jsx";
import Transcript from "./components/Transcript.jsx";
import Controls from "./components/Controls.jsx";

export default function App() {
  const run = useLiveRun();
  const colors = useMemo(() => seatColors(run.players), [run.players]);

  // Whether a game is in flight isn't on the websocket -- the socket carries the
  // game, not the server's state -- so ask /health. Cheap, and it also notices a
  // game someone started from a terminal.
  const [playing, setPlaying] = useState(false);
  const refresh = useCallback(() => {
    fetch("/health")
      .then((r) => r.json())
      .then((d) => setPlaying(Boolean(d.playing)))
      .catch(() => {});
  }, []);

  useEffect(() => {
    refresh();
    const timer = setInterval(refresh, 1500);
    return () => clearInterval(timer);
  }, [refresh]);

  const event = lastTurn(run)?.event;
  const phase = event?.phase;

  return (
    <>
      <header className="hud">
        <div className="brand">
          Cry Wolf <span>observer</span>
        </div>
        <div className={`chip${phase ? ` phase-${phase}` : ""}`}>
          {event ? `round ${event.round} · ${phase}` : "no run"}
        </div>
        <div className="chip">
          {run.turns.length} / {run.expected || 0}
        </div>
        <div className="spacer" />
        <Controls run={run} playing={playing} onChanged={refresh} />
        <div className={`conn ${run.connected ? "live" : "down"}`}>
          <span className="dot" />
          <span>{run.connected ? "live" : "reconnecting…"}</span>
        </div>
      </header>

      <main>
        <Transcript run={run} colors={colors} />
        <Table run={run} colors={colors} />
        <Panel run={run} colors={colors} />
      </main>
    </>
  );
}
