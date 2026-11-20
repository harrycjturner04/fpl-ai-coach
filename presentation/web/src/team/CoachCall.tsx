import type { SnapshotResponse } from "../types";
import { chipName } from "./model";

export function CoachCall({ call, onSee }: { call: SnapshotResponse["call"]; onSee: () => void }) {
  return (
    <section className="call" aria-labelledby="call-title">
      <h2 id="call-title">{call.title}</h2>
      <p>{call.reason}</p>
      <button type="button" onClick={onSee}>
        {call.chip ? `See the ${chipName(call.chip)} team` : "See the coach's plan"}
      </button>
    </section>
  );
}
