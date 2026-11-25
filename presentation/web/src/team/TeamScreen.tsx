import { useEffect, useMemo, useRef, useState, type KeyboardEvent } from "react";
import { getUserPlan, postPlanCheck } from "../api";
import { CheckPanel, useCheck } from "../swap/CheckPanel";
import { SwapTab, swappable } from "../swap/SwapTab";
import { preview, rescore, squadAfter, type Swap } from "../swap/swap";
import type { Chip, Snapshot, Snapshot2, SnapshotResponse, UserPlan, Week } from "../types";
import { Chips } from "./Chips";
import { CoachCall } from "./CoachCall";
import { Headline } from "./Headline";
import { Issues } from "./Issues";
import { chipName, deadlineLabel, moves, runLabel, transfersSummary } from "./model";
import { Pitch } from "./Pitch";
import { RunNow } from "./RunNow";
import "./team.css";
import "../app.css";

const fits = (snap: Snapshot2, swaps: Swap[]) => {
  try { squadAfter(snap, swaps); return true; } catch { return false; }
};

/** The owner's week with any player this run no longer has left out, so an older plan still renders. */
const known = (snap: Snapshot, w: Week): Week => {
  const has = new Set(snap.players.map((p) => p.id));
  const keep = (ids: number[]) => ids.filter((id) => has.has(id));
  return { ...w, starting: keep(w.starting), bench: keep(w.bench), transfers_in: keep(w.transfers_in), transfers_out: keep(w.transfers_out) };
};

type Tab = { id: "team" | "plan" | "swap"; label: string; week?: Week };

export function TeamScreen({ data, now = new Date(), onSettings, onReload }: { data: SnapshotResponse; now?: Date; onSettings: () => void; onReload: () => void }) {
  const plan = data.plan.weeks[0];
  const tabs: Tab[] = [
    ...(data.current ? [{ id: "team", label: "Your team", week: data.current } as const] : []),
    { id: "plan", label: "Coach's plan", week: plan },
    { id: "swap", label: "Try a swap" },
  ];
  const [tabId, setTabId] = useState(tabs[0].id);
  const refs = useRef<(HTMLButtonElement | null)[]>([]);
  const tab = tabs.find((t) => t.id === tabId) ?? tabs[0];
  // Kept here, not in the tab, so the headline follows the swaps and they survive a tab switch. Tied to the run that
  // produced them: a new snapshot starts empty, and a set that no longer fits the squad is dropped rather than thrown.
  const empty = { run: data.generated_at, swaps: [], chip: null };
  const [stored, setStored] = useState<{ run: string; swaps: Swap[]; chip: Chip | null }>(empty);
  const stale = stored.run !== data.generated_at || (swappable(data) && !fits(data, stored.swaps));
  if (stale) setStored(empty);
  const { swaps, chip } = stale || !swappable(data) ? empty : stored;
  const setSwap = (s: { swaps: Swap[]; chip: Chip | null }) => setStored({ run: data.generated_at, ...s });
  // Your team's figure, plus the preview's change, so the "now" number never moves between tabs.
  const mine = useMemo(() => {
    if (tab.id !== "swap" || !data.current || !swappable(data)) return null;
    const [base, now] = [preview(data, [], null), preview(data, swaps, chip)];
    return { now: data.current.points, plan: data.current.points + (now.weekPoints ?? base.weekPoints!) - base.weekPoints! };
  }, [tab.id, data, swaps, chip]);

  // The owner's adopted plan for this gameweek, and the check verdicts. A verdict is shown only while its key (run,
  // swaps, chip; or run and plan for a re-check) is current, so a changed set or a new run never shows a stale one.
  const [userPlan, setUserPlan] = useState<UserPlan | null>(null);
  const [whose, setWhose] = useState<"coach" | "yours">("coach");
  useEffect(() => {
    let live = true;
    getUserPlan(data.entry).then((p) => live && setUserPlan(p), () => {});
    return () => { live = false; };
  }, [data.entry, data.generated_at]);
  const check = useCheck();
  const snapName = data.report.replace(/\.md$/, ".json");
  const swapKey = JSON.stringify([data.generated_at, swaps, chip]);
  const recheckKey = JSON.stringify([data.generated_at, "recheck", userPlan?.id]);
  const runCheck = () => check.start(swapKey, () => postPlanCheck(data.entry, snapName, swaps, chip));
  const runRecheck = (p: UserPlan) => check.start(recheckKey, () => postPlanCheck(data.entry, snapName, p.swaps, p.chip));
  const adopted = (p: UserPlan) => {
    check.clear();
    setUserPlan(p);
    setWhose("yours");
    setTabId("plan");
  };
  const yours = tab.id === "plan" && userPlan && whose === "yours" ? userPlan : null;

  const select = (i: number) => {
    setTabId(tabs[i].id);
    refs.current[i]?.focus();
  };
  const onKey = (e: KeyboardEvent, i: number) => {
    const next = { ArrowRight: i + 1, ArrowLeft: i - 1, Home: 0, End: tabs.length - 1 }[e.key];
    if (next === undefined) return;
    e.preventDefault();
    select((next + tabs.length) % tabs.length);
  };

  return (
    <main className="team">
      <header className="team-head">
        <div className="head-row">
          <h1>{data.team_name}</h1>
          <p>{deadlineLabel(data.gameweek, data.deadline, now)}</p>
          <button className="gear" aria-label="Settings" onClick={onSettings}>⚙︎</button>
        </div>
        {data.status.stale
          ? <p className="run stale" role="status">{data.status.reason}</p>
          : <p className="run">{runLabel(data.generated_at)}</p>}
        <RunNow entry={data.entry} onDone={onReload} onSettings={onSettings} />
      </header>
      <div className="team-grid">
        {mine ? <Headline now={mine.now} plan={mine.plan} by="your swaps" /> : <Headline now={data.current?.points ?? null} plan={plan.points} />}
        <section className="team-pitch" aria-label="Pitch">
          <div className="tabs" role="tablist" aria-label="Which team to show">
            {tabs.map((t, i) => (
              <button
                key={t.id}
                ref={(el) => { refs.current[i] = el; }}
                id={`tab-${t.id}`}
                role="tab"
                aria-selected={t.id === tab.id}
                aria-controls="team-panel"
                tabIndex={t.id === tab.id ? 0 : -1}
                onClick={() => setTabId(t.id)}
                onKeyDown={(e) => onKey(e, i)}
              >
                {t.label}
              </button>
            ))}
          </div>
          <div id="team-panel" role="tabpanel" aria-labelledby={`tab-${tab.id}`}>
            {tab.id === "plan" && userPlan && (
              <div className="whose" role="group" aria-label="Whose plan">
                <button type="button" aria-pressed={whose === "coach"} onClick={() => setWhose("coach")}>Coach's</button>
                <button type="button" aria-pressed={whose === "yours"} onClick={() => setWhose("yours")}>Yours</button>
              </div>
            )}
            {yours ? (
              <>
                <p className="yours-label">Your plan, not the coach's</p>
                {yours.stale && (
                  <div className="stale-plan">
                    <p className="stale-title">Based on an older run</p>
                    {swappable(data) && (
                      <p>{`Re-scored on the latest predictions: ${(rescore(data, yours.result.weeks.slice(0, 1))[0]?.points ?? 0).toFixed(1)} this week vs the coach's new plan ${plan.points.toFixed(1)}`}</p>
                    )}
                    <button type="button" disabled={check.state?.key === recheckKey && check.state.s.kind === "checking"} onClick={() => runRecheck(yours)}>
                      Re-check on latest run
                    </button>
                  </div>
                )}
                {check.state?.key === recheckKey && (
                  <CheckPanel snap={data} swaps={yours.swaps} chip={yours.chip} state={check.state.s} recheck
                    onRetry={() => runRecheck(yours)} onKeep={check.clear} onAdopted={adopted} />
                )}
                <Pitch snap={data} week={known(data, yours.result.weeks[0])} />
                <PlanMoves snap={data} week={known(data, yours.result.weeks[0])} />
              </>
            ) : tab.week
              ? <Pitch snap={data} week={tab.week} />
              : <SwapTab snap={data} swaps={swaps} chip={chip} onChange={(swaps, chip) => setSwap({ swaps, chip })} onCheck={runCheck} />}
            {tab.id === "swap" && check.state?.key === swapKey && (
              <CheckPanel snap={data} swaps={swaps} chip={chip} state={check.state.s}
                onRetry={runCheck} onKeep={check.clear} onAdopted={adopted} />
            )}
            {tab.id === "plan" && !yours && <PlanMoves snap={data} week={plan} />}
          </div>
        </section>
        <CoachCall call={data.call} onSee={() => select(tabs.findIndex((t) => t.id === "plan"))} />
        <Issues snap={data} />
        <Chips snap={data} />
      </div>
    </main>
  );
}

function PlanMoves({ snap, week }: { snap: Snapshot; week: Week }) {
  const pairs = moves(snap, week);
  return (
    <div className="moves">
      <p className="moves-head">
        {transfersSummary(week, snap.state.hit_cost)}
        {week.chip && <span>{chipName(week.chip)} this week</span>}
      </p>
      {pairs.length > 0 && (
        <ul>
          {pairs.map(([out, inn]) => <li key={out.id}>{`${out.name} → ${inn.name}`}</li>)}
        </ul>
      )}
    </div>
  );
}
