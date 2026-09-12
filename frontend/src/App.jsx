import { useCallback, useEffect, useMemo, useState } from "react";
import { seatColors, useLiveRun } from "./live.js";
import Header from "./components/Header.jsx";
import Table from "./components/Table.jsx";
import CaseFile from "./components/CaseFile.jsx";
import BeliefBoard from "./components/BeliefBoard.jsx";
import Transcript from "./components/Transcript.jsx";
import Evidence from "./components/Evidence.jsx";
import Scorecard from "./components/Scorecard.jsx";

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

  return (
    <div className="board-app">
      <Header run={run} playing={playing} onChanged={refresh} />

      <main className="board">
        <Table run={run} colors={colors} />
        <CaseFile run={run} colors={colors} />
        <div className="chart-row">
          <BeliefBoard run={run} colors={colors} />
          <Scorecard run={run} />
        </div>
        <div className="ledger-row">
          <Transcript run={run} colors={colors} />
          <Evidence run={run} />
        </div>
      </main>
    </div>
  );
}
