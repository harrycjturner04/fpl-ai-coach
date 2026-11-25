import { useEffect, useState } from "react";
import { ApiError, getLatestRun, getRun, startRun } from "../api";
import type { Run } from "../types";

const POLL_MS = 3000;
const STALE_HOURS = 30;
const active = (r: Run | null) => r?.status === "queued" || r?.status === "running";

export function RunNow({ entry, onDone, onSettings }: { entry: number; onDone: () => void; onSettings: () => void }) {
  const [run, setRun] = useState<Run | null>(null);
  const [err, setErr] = useState<ApiError>();
  const [busy, setBusy] = useState(false);
  const [retry, setRetry] = useState(0);

  useEffect(() => {
    let live = true;
    getLatestRun(entry).then((r) => live && active(r) && setRun(r)).catch(() => {});
    return () => { live = false; };
  }, [entry]);

  useEffect(() => {
    if (!run || !active(run)) return;
    let live = true;
    const t = setTimeout(() => {
      getRun(run.id).then((r) => {
        if (!live) return;
        setRun(r);
        if (r.status === "done") onDone();
      }, () => live && setRetry((n) => n + 1));  // network error: keep polling
    }, POLL_MS);
    return () => { live = false; clearTimeout(t); };
  }, [run, retry, onDone]);

  const start = () => {
    setBusy(true);
    setErr(undefined);
    startRun(entry)
      .then(setRun)
      .catch((e) => setErr(e instanceof ApiError ? e : new ApiError(0, String(e))))
      .finally(() => setBusy(false));
  };

  const running = active(run);
  const done = new Set(run?.steps.map((s) => s.name));
  const current = run?.status === "running" ? run.steps_expected.find((n) => !done.has(n)) : undefined;
  const blocked = run?.status === "blocked";
  const failure = err ? err.detail : run?.status === "failed" || run?.status === "blocked" ? run.error : null;
  const age = run?.data_age_hours;

  return (
    <section className="run-now" aria-label="Run the coach">
      <button onClick={start} disabled={busy || running}>
        {failure && !(err?.status === 409) && !blocked ? "Try again" : "Run the coach now"}
      </button>
      {failure && (
        <p className="run-error" role="alert">
          {failure}
          {(blocked || (err?.status === 409 && (err.reason === "cap" || err.reason === "ai_off"))) && (
            <> <button className="link" onClick={onSettings}>Settings</button></>
          )}
        </p>
      )}
      {run?.status === "queued" && <p role="status">Waiting for the coach to start...</p>}
      {run?.status === "running" && (
        <ol className="steps" aria-live="polite">
          {run.steps_expected.map((n) => {
            const s = run.steps.find((x) => x.name === n);
            return (
              <li key={n} className={s ? "done" : n === current ? "now" : ""}>
                <span aria-hidden="true">{s ? "✓" : n === current ? "●" : "○"}</span> {n}
                <span className="state">{s ? ` ${s.secs.toFixed(1)} s` : n === current ? " in progress" : " pending"}</span>
              </li>
            );
          })}
        </ol>
      )}
      {(running || run?.status === "done") && age != null && (
        <p className="age">
          <span>{`Using data pulled ${Math.round(age)} h ago.`}</span>
          {age > STALE_HOURS && <span> Switch on your PC to refresh it.</span>}
        </p>
      )}
    </section>
  );
}
