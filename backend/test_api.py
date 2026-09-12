"""API tests that need no credentials.

Everything here runs a *replay* run (`live=False`), which skips the Observer
entirely -- so the endpoints, the websocket envelope and the scoring seam are all
exercised without spending a single model call.

    python -m backend.test_api
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

import backend.main as api
from backend.schema import BeliefState, GameEvent, Transcript

TRANSCRIPT = Path("data/fallback_transcript.json")
RECORDED = Path("out/run1.json")


def check(label: str, cond: bool) -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {label}")
    assert cond, label


def _turns_from_recording(transcript: Transcript):
    """Rebuild {event, state} pairs from a completed run_observer recording."""
    record = json.loads(RECORDED.read_text(encoding="utf-8"))
    events = [GameEvent.model_validate(e) for e in record["events"]]
    history = record["history"]
    final = record["final_state"]
    turns = []
    for i, (event, suspicion) in enumerate(zip(events, history)):
        # Only the last state was saved in full, so the earlier ones carry just
        # the suspicion numbers -- enough to drive scoring and the chart.
        state = BeliefState(
            round=event.round,
            phase=event.phase,
            event_index=i,
            eliminated=final["eliminated"] if i == len(events) - 1 else [],
            suspicion=suspicion,
        )
        turns.append({"index": i, "event": event.model_dump(), "state": state.model_dump()})
    return turns


def test_replay_run_scores() -> None:
    print("replay run + /score")
    transcript = Transcript.model_validate_json(TRANSCRIPT.read_text(encoding="utf-8"))
    turns = _turns_from_recording(transcript)

    with TestClient(api.app) as client:
        check("health responds with no run open", client.get("/health").json()["ok"] is True)
        check("scoring without a run is refused", client.get("/score").status_code == 409)

        start = client.post(
            "/game/start",
            json={"transcript": transcript.model_dump(), "live": False},
        )
        check("replay run starts without an API key", start.status_code == 200)
        check("scoring an empty run is refused", client.get("/score").status_code == 409)

        for turn in turns:
            assert client.post("/turn", json=turn).status_code == 200, turn["index"]

        mid = client.get("/score").json()
        check("scores mid-run", mid["events_observed"] == len(turns))

        client.post("/game/end")
        final = client.get("/score").json()

        check("the run is complete", final["complete"] is True)
        check(f"names the werewolf ({final['predicted']})", final["predicted"] == "P4")
        check("accuracy is a real hit", final["accuracy"] is True)
        check("confidence is reported", 0.0 < final["final_confidence"] <= 1.0)
        check("consistency is in range", 0.0 <= final["consistency"] <= 1.0)
        check(
            "chart history is included",
            len(final["suspicion_history"]) == len(turns),
        )
        check("ground truth is not leaked to the client", "ground_truth" not in final)


def test_live_endpoints_reject_replay_traffic() -> None:
    print("run-mode guards")
    transcript = Transcript.model_validate_json(TRANSCRIPT.read_text(encoding="utf-8"))
    with TestClient(api.app) as client:
        client.post("/game/start", json={"transcript": transcript.model_dump(), "live": False})
        event = transcript.events[0].model_dump()
        check(
            "POST /event is refused on a replay run",
            client.post("/event", json=event).status_code == 409,
        )


def test_websocket_snapshot() -> None:
    print("websocket")
    transcript = Transcript.model_validate_json(TRANSCRIPT.read_text(encoding="utf-8"))
    turns = _turns_from_recording(transcript)[:3]

    with TestClient(api.app) as client:
        client.post("/game/start", json={"transcript": transcript.model_dump(), "live": False})
        for turn in turns:
            client.post("/turn", json=turn)

        with client.websocket_connect("/live") as ws:
            snapshot = ws.receive_json()
            check("connects with a snapshot", snapshot["type"] == "snapshot")
            check("snapshot carries the turns so far", len(snapshot["turns"]) == 3)
            check(
                "snapshot does not carry ground truth",
                "ground_truth" not in json.dumps(snapshot),
            )


if __name__ == "__main__":
    for fn in [
        test_replay_run_scores,
        test_live_endpoints_reject_replay_traffic,
        test_websocket_snapshot,
    ]:
        fn()
    print("\nall API checks passed (no API key used)")
