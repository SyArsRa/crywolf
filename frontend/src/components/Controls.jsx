import { useEffect, useState } from "react";

/**
 * Start a game without a terminal.
 *
 * The feeder is still the real ingestion path -- this just asks the server to
 * run the same loop itself, so the demo needs one window instead of two.
 */
export default function Controls({ run, playing, onChanged }) {
  const [games, setGames] = useState([]);
  const [game, setGame] = useState("");
  const [mode, setMode] = useState("replay");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [startedAt, setStartedAt] = useState(0);

  useEffect(() => {
    fetch("/games")
      .then((r) => r.json())
      .then((data) => {
        const list = data.games ?? [];
        setGames(list);
        // Default to the shipped demo recording: it needs no API key and is a
        // complete game, so the first click always shows something good.
        const preferred =
          list.find((g) => g.id === "data/demo_run.json") ??
          list.find((g) => g.kind === "recording") ??
          list[0];
        if (preferred) {
          setGame(preferred.id);
          setMode(preferred.kind === "recording" ? "replay" : "live");
        }
      })
      .catch((e) => setError(String(e)));
  }, []);

  const selected = games.find((g) => g.id === game);

  // A recording can only be replayed; a transcript can only be run live.
  useEffect(() => {
    if (selected) setMode(selected.kind === "recording" ? "replay" : "live");
  }, [game]);

  async function start() {
    setBusy(true);
    setError(null);
    setStartedAt(Date.now());
    try {
      const response = await fetch("/play", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          game,
          mode,
          interval: mode === "live" ? 1.0 : 2.0,
        }),
      });
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        throw new Error(body.detail ?? `HTTP ${response.status}`);
      }
      onChanged?.();
    } catch (e) {
      setError(String(e.message ?? e));
    } finally {
      setBusy(false);
    }
  }

  async function stop() {
    // The Start and Stop buttons occupy the same spot, and React will happily
    // reuse the DOM node between them -- so a stray event on that node can land
    // on Stop a moment after Start was pressed and kill the game instantly.
    // Distinct keys (below) prevent the reuse; this refuses the click outright.
    if (Date.now() - startedAt < 1200) return;
    setBusy(true);
    try {
      await fetch("/stop", { method: "POST" });
      onChanged?.();
    } finally {
      setBusy(false);
    }
  }

  if (playing) {
    return (
      <div className="controls playing">
        <span className="controls-status">
          playing · {run.turns.length} / {run.expected || "?"}
        </span>
        <button key="stop" className="btn ghost" onClick={stop} disabled={busy}>
          Stop
        </button>
      </div>
    );
  }

  return (
    <div className="controls">
      <select value={game} onChange={(e) => setGame(e.target.value)} disabled={busy}>
        {games.map((g) => (
          <option key={g.id} value={g.id}>
            {g.label} · {g.players}p · {g.events} events
            {g.kind === "recording" ? " · recorded" : ` · ${g.deceiver_role}`}
          </option>
        ))}
      </select>
      <button key="start" className="btn" onClick={start} disabled={busy || !game}>
        {busy ? "Starting…" : mode === "live" ? "Start live game" : "Start game"}
      </button>
      {error && <span className="controls-error">{error}</span>}
    </div>
  );
}
