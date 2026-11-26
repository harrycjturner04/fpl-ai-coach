import { useEffect, useRef, useState } from "react";
import { ApiError, adoptPlan } from "../api";
import { chipName } from "../team/model";
import { swappable } from "./SwapTab";
import { nextGws, preview, type Swap } from "./swap";
import type { Chip, PlanCheck, Snapshot, UserPlan } from "../types";

const KEY = "planCheckSeconds";
const DEFAULT_SECONDS = 30;
const DEADLINE_TEXT = "This gameweek's deadline has passed. Swaps for the next gameweek open after the coach's next run.";
const f1 = (n: number) => n.toFixed(1);
const signed = (n: number) => `${n >= 0 ? "+" : "−"}${Math.abs(n).toFixed(1)}`;

export type CheckState =
  | { kind: "checking"; started: number; estimate: number }
  | { kind: "error"; error: ApiError }
  | { kind: "done"; result: PlanCheck };

/** The last measured check time, or 30 s before any check has run. */
function estimate() {
  try {
    const n = Number(localStorage.getItem(KEY));
    return n > 0 ? n : DEFAULT_SECONDS;
  } catch { return DEFAULT_SECONDS; }
}

/** One check at a time, tagged with the key (run, swaps, chip) it was made for so a caller shows it only while that
 * key is current. Swap checks and re-checks share it: a newer check replaces an older one's answer, as does `clear`. */
export function useCheck() {
  const [state, setState] = useState<{ key: string; s: CheckState } | null>(null);
  const seq = useRef(0);
  const start = (key: string, run: () => Promise<PlanCheck>) => {
    const n = ++seq.current;
    setState({ key, s: { kind: "checking", started: Date.now(), estimate: estimate() } });
    run().then((result) => {
      try { localStorage.setItem(KEY, String(result.seconds)); } catch { /* estimate stays at its last value */ }
      if (n === seq.current) setState({ key, s: { kind: "done", result } });
    }, (e) => {
      if (n === seq.current) setState({ key, s: { kind: "error", error: e instanceof ApiError ? e : new ApiError(0, String(e)) } });
    });
  };
  const clear = () => { seq.current++; setState(null); };
  return { state, start, clear };
}

interface Props {
  snap: Snapshot;
  swaps: Swap[];
  chip: Chip | null;
  state: CheckState;
  /** Re-checking an adopted plan on a newer run (changes the 422 wording). */
  recheck?: boolean;
  onRetry: () => void;
  onKeep: () => void;
  onAdopted: (plan: UserPlan) => void;
}

export function CheckPanel({ snap, swaps, chip, state, recheck, onRetry, onKeep, onAdopted }: Props) {
  return (
    <section className="check-panel" aria-label="Check against the coach's plan">
      {state.kind === "checking" && <Progress started={state.started} estimate={state.estimate} />}
      {state.kind === "error" && <Failure error={state.error} recheck={recheck} onRetry={onRetry} />}
      {state.kind === "done" && <Verdict snap={snap} swaps={swaps} chip={chip} result={state.result} onKeep={onKeep} onAdopted={onAdopted} />}
    </section>
  );
}

function Progress({ started, estimate }: { started: number; estimate: number }) {
  const [now, setNow] = useState(Date.now);
  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), 250);
    return () => clearInterval(t);
  }, []);
  const elapsed = (now - started) / 1000;
  return (
    <>
      <p role="status">{elapsed < estimate ? `Checking against the coach's plan, about ${Math.round(estimate)} s` : "Still solving, nearly there"}</p>
      <progress max={estimate} value={Math.min(elapsed, estimate)} aria-label="Check progress" />
    </>
  );
}

function Failure({ error, recheck, onRetry }: { error: ApiError; recheck?: boolean; onRetry: () => void }) {
  const [title, detail, retry] =
    error.status === 409 && error.reason !== "busy" ? ["Deadline passed", DEADLINE_TEXT, false]
    : error.status === 409 ? ["Already checking", `${error.detail} Wait for it to finish, then try again.`, true]
    : error.status === 422 ? [recheck ? "This plan may no longer be legal on the latest run" : "Can't check these swaps",
      `${error.detail} ${recheck ? "Make new swaps in Try a swap." : "Change your swaps and check again."}`, false]
    : error.status === 429 ? ["Too many checks", "Too many checks this hour. Try again later.", false]
    : error.status === 0 ? ["Couldn't reach the coach", error.detail, true]
    : ["The check failed", `${error.detail} Try again in a moment.`, true];
  return (
    <div className="check-error" role="alert">
      <h3>{title}</h3>
      <p>{detail}</p>
      {retry && <button type="button" onClick={onRetry}>Try again</button>}
    </div>
  );
}

function Verdict({ snap, swaps, chip, result, onKeep, onAdopted }: Omit<Props, "state" | "recheck" | "onRetry"> & { result: PlanCheck }) {
  const [adopting, setAdopting] = useState(false);
  const [error, setError] = useState<ApiError | string>();
  const sending = useRef(false);
  const passed = typeof error === "object" && error.status === 409 && error.reason !== "busy";
  const coachChip = snap.plan.weeks[0].chip, weeks = snap.gameweeks.length;
  const likeForLike = (coachChip === "wildcard" || coachChip === "freehit") && result.chip !== coachChip;
  let five: number | null = null;
  try {
    if (swappable(snap)) {
      const [base, now] = [preview(snap, [], null).fiveWeek, preview(snap, swaps, chip).fiveWeek];
      if (base !== null && now !== null) five = now - base;
    }
  } catch { /* swaps no longer fit this run's squad: no quick estimate */ }

  const adopt = () => {
    if (sending.current) return;
    sending.current = true;
    setAdopting(true);
    setError(undefined);
    adoptPlan(result.id).then(onAdopted, (e) => {
      setError(e instanceof ApiError ? e : String(e));
      sending.current = false;
      setAdopting(false);
    });
  };

  return (
    <div className="verdict">
      <h3>The optimiser's verdict</h3>
      <p className="num">{`This week: ${f1(result.points)} vs coach ${f1(result.coach.points)}`}</p>
      <p>{`Optimiser's planning value: ${f1(result.objective)} vs coach ${f1(result.coach.objective)}${result.coach.exact ? "" : " (coach's plan not exact)"}`}</p>
      {five !== null && <p>{`Quick estimate: ${signed(five)} over ${nextGws(snap)} (no further transfers)`}</p>}
      <p className="explain">{`The planning value is the optimiser's score for ${weeks} week${weeks === 1 ? "" : "s"} with later transfers planned. The quick estimate keeps your squad fixed.`}</p>
      {likeForLike && <p className="like">{`The coach plays a ${chipName(coachChip!)} this week; turn on ${chipName(coachChip!)} to compare like for like.`}</p>}
      {error && (passed
        ? <div className="check-error" role="alert"><h3>Deadline passed</h3><p>{DEADLINE_TEXT}</p></div>
        : <p className="check-error" role="alert">{`Couldn't adopt this plan. ${typeof error === "string" ? error : error.detail}`}</p>)}
      <div className="verdict-actions">
        {!passed && <button type="button" className="adopt" disabled={adopting} onClick={adopt}>Adopt this plan</button>}
        <button type="button" onClick={onKeep}>Keep trying</button>
      </div>
    </div>
  );
}
