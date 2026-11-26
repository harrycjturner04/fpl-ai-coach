import { useEffect, useRef, useState } from "react";

const TICK_MS = 300;

/** `target`, counted up or down to on change over a short tick; instant under reduced motion. */
function useTick(target: number) {
  const [shown, setShown] = useState(target);
  const from = useRef(target);
  useEffect(() => {
    const start = from.current;
    const reduce = !window.matchMedia || window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (start === target || reduce) { from.current = target; setShown(target); return; }
    const t0 = performance.now();
    let raf = requestAnimationFrame(function step(t) {
      const k = Math.min(1, (t - t0) / TICK_MS);
      from.current = start + (target - start) * (1 - (1 - k) ** 3);
      setShown(from.current);
      if (k < 1) raf = requestAnimationFrame(step);
    });
    return () => cancelAnimationFrame(raf);
  }, [target]);
  return shown;
}

/** This week's predicted points for the team shown, and how far it is from your current team (none when there is no
 * current team to compare with, or the shown team is it). */
export function Headline({ points, vs }: { points: number; vs?: number | null }) {
  const shown = useTick(points).toFixed(1);
  const d = vs == null ? null : Math.round((points - vs) * 10) / 10;
  const tone = d === null || d === 0 ? "" : d > 0 ? "gain" : "loss";
  return (
    <div className="headline" role="group" aria-label="Predicted points">
      <p className="figures num"><span className="points">{shown}</span></p>
      {d !== null && <p className={`delta num ${tone}`}>{d === 0 ? "same as your team" : `${d > 0 ? "+" : "−"}${Math.abs(d).toFixed(1)} vs your team`}</p>}
      <p className="caption">Predicted points this week for the team shown</p>
    </div>
  );
}
