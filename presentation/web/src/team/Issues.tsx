import type { Snapshot } from "../types";
import { issues } from "./model";

export function Issues({ snap }: { snap: Snapshot }) {
  const list = issues(snap);
  return (
    <section className="list" aria-labelledby="issues-title">
      <h2 id="issues-title">Injuries and doubts</h2>
      {list.length ? (
        <ul>
          {list.map(({ player, status, label, news, inSquad }) => (
            <li key={player.id}>
              <div>
                <span className="who">{player.name}, {player.team}</span>
                {news && <span className="sub">{news}</span>}
                {!inSquad && <span className="sub">Not in your squad</span>}
              </div>
              {label && <span className={`tag ${status}`}>{label}</span>}
            </li>
          ))}
        </ul>
      ) : (
        <p className="empty">No injury or suspension news for your players.</p>
      )}
    </section>
  );
}
