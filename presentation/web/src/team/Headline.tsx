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

/** This week's predicted points now and with the plan (the coach's, or your swaps); the plan figure is coloured by
 * gain or loss. */
export function Headline({ now, plan, by = "the coach's plan" }: { now: number | null; plan: number; by?: string }) {
  const [a, b] = [now?.toFixed(1), plan.toFixed(1)];
  const shown = useTick(plan).toFixed(1);
  const tone = a === undefined || a === b ? "" : Number(b) > Number(a) ? "gain" : "loss";
  return (
    <div className="headline" role="group" aria-label="Predicted points">
      <p className="figures num">
        {a !== undefined && (
          <>
            <span className="now">{a}</span>
            <span className="arrow" aria-hidden="true">→</span>
            <span className="sr-only">to</span>
          </>
        )}
        <span className={`plan ${tone}`}>{shown}</span>
      </p>
      <p className="caption">
        {a !== undefined ? `Predicted points for your team now, and with ${by}` : `Predicted points with ${by}`}
      </p>
    </div>
  );
}
