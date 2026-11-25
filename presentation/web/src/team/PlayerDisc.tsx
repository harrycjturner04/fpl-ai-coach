import { useEffect, useState } from "react";
import { statusOf, type Slot } from "./model";

const R = 21;
const C = 2 * Math.PI * R;

export function PlayerDisc({ slot, onPick }: { slot: Slot; onPick?: (id: number) => void }) {
  const { player, flag, points, p60, captain, vice, isNew } = slot;
  const { status, label, chance } = statusOf(player, flag);
  const fill = p60 ?? 0;
  // Start empty and fill on the next frame so the ring animates on mount; later changes animate via the CSS transition.
  const [shown, setShown] = useState(0);
  useEffect(() => {
    const id = requestAnimationFrame(() => setShown(fill));
    return () => cancelAnimationFrame(id);
  }, [fill]);

  const pts = points.toFixed(1);
  const aria = [
    player.name,
    `${pts} predicted points`,
    p60 === null ? "no minutes estimate" : `${Math.round(p60 * 100)}% chance of 60+ minutes`,
    label && `${chance ?? 0}% chance of playing`,
    isNew && "new",
    captain ? "captain" : vice ? "vice captain" : null,
  ]
    .filter(Boolean)
    .join(", ");

  const Tag = onPick ? "button" : "div";
  const props = onPick ? { type: "button" as const, onClick: () => onPick(player.id) } : { role: "img" };
  return (
    <Tag className={`slot ${status}${onPick ? " pick" : ""}`} aria-label={aria} {...props}>
      <div className="disc" aria-hidden="true">
        <svg viewBox="0 0 48 48">
          <circle className="track" cx="24" cy="24" r={R} />
          <circle className="ring" cx="24" cy="24" r={R} strokeDasharray={C} strokeDashoffset={C * (1 - shown)} />
        </svg>
        <span className="pts num">{pts}</span>
        {(captain || vice) && <span className="arm">{captain ? "C" : "V"}</span>}
        {isNew && <span className="new">New</span>}
      </div>
      <span className="chip" aria-hidden="true">
        <span className="name">{player.name}</span>
        {label && <span className="label">{label}</span>}
      </span>
    </Tag>
  );
}
