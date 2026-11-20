/** This week's predicted points now and with the plan; the plan figure is coloured by gain or loss. */
export function Headline({ now, plan }: { now: number | null; plan: number }) {
  const [a, b] = [now?.toFixed(1), plan.toFixed(1)];
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
        <span className={`plan ${tone}`}>{b}</span>
      </p>
      <p className="caption">
        {a !== undefined ? "Predicted points for your team now, and with the coach's plan" : "Predicted points with the coach's plan"}
      </p>
    </div>
  );
}
