import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Two ways to run the UI:
//
//   npm run dev     -- vite with hot reload, proxying the API and the websocket
//                      through to uvicorn
//   npm run build   -- emits dist/, which FastAPI serves itself, so the demo is
//                      a single process with no node involved
//
// The proxy target follows CRYWOLF_API so `API_PORT=... npm run dev` actually
// works; hardcoding it means a non-default port proxies to the wrong server and
// the page sits there connected to nothing.
const API = process.env.CRYWOLF_API || "http://127.0.0.1:8000";
const WS = API.replace(/^http/, "ws");

export default defineConfig({
  plugins: [react()],
  build: { outDir: "dist", emptyOutDir: true },
  server: {
    port: Number(process.env.UI_PORT) || 5173,
    // Every backend route has to be listed. A path that isn't here is served by
    // vite itself, which 404s -- and only in dev, so it looks like the feature
    // is broken while working fine on :8000.
    proxy: {
      "/live": { target: WS, ws: true },
      "/health": API,
      "/games": API,
      "/game": API,
      "/event": API,
      "/turn": API,
      "/play": API,
      "/stop": API,
      "/score": API,
    },
  },
});
