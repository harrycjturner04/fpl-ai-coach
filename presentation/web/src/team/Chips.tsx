import type { Snapshot } from "../types";
import { chipName, chipsSoon } from "./model";

/** Chip scan gains are planning objective over five weeks, so they are never called points. */
export function Chips({ snap }: { snap: Snapshot }) {
  const list = chipsSoon(snap);
  return (
    <section className="list" aria-labelledby="chips-title">
      <h2 id="chips-title">Chips worth playing soon</h2>
      {list.length ? (
        <ul>
          {list.map((c) => (
            <li key={c.chip}>
              <span>{chipName(c.chip)} in GW {c.gameweek}, <span className="gain num">+{c.gain.toFixed(1)}</span> over no chip</span>
            </li>
          ))}
        </ul>
      ) : (
        <p className="empty">No chip beats holding it in the next 5 gameweeks.</p>
      )}
    </section>
  );
}
