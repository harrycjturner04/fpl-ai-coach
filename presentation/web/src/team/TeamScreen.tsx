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
  // The swap tab's figure: your team's, plus the preview's change, so it follows each swap.
  const swapPoints = useMemo(() => {
    if (tab.id !== "swap" || !swappable(data)) return null;
    const [base, now] = [preview(data, [], null), preview(data, swaps, chip)];
    const change = (now.weekPoints ?? base.weekPoints!) - base.weekPoints!;
    return data.current ? data.current.points + change : now.weekPoints ?? base.weekPoints!;
  }, [tab.id, data, swaps, chip]);

  // The owner's adopted plan for this gameweek, and the check verdicts. A verdict is shown only while its key (run,
  // swaps, chip; or run and plan for a re-check) is current, so a changed set or a new run never shows a stale one.
  const [userPlan, setUserPlan] = useState<UserPlan | null>(null);
  const [planFailed, setPlanFailed] = useState(false);
  const [planTry, setPlanTry] = useState(0);
  const [whose, setWhose] = useState<"coach" | "yours">("coach");
  useEffect(() => {
    let live = true;
    setUserPlan(null);
    setPlanFailed(false);
    getUserPlan(data.entry).then((p) => live && setUserPlan(p), () => live && setPlanFailed(true));
    return () => { live = false; };
  }, [data.entry, data.generated_at, planTry]);
  const check = useCheck();
  const snapName = data.report.replace(/\.md$/, ".json");
  const swapKey = JSON.stringify([data.generated_at, swaps, chip]);
  const recheckKey = JSON.stringify([data.generated_at, "recheck", userPlan?.id]);
  const checking = (key: string) => check.state?.key === key && check.state.s.kind === "checking";
  const runCheck = () => check.start(swapKey, () => postPlanCheck(data.entry, snapName, swaps, chip));
  const runRecheck = (p: UserPlan) => check.start(recheckKey, () => postPlanCheck(data.entry, snapName, p.swaps, p.chip));
  const adopted = (p: UserPlan) => {
    check.clear();
    setUserPlan(p);
    setWhose("yours");
    setTabId("plan");
  };
  const yours = tab.id === "plan" && userPlan && whose === "yours" ? userPlan : null;
  // The headline is this week's points for the team on show, against your current team. An adopted plan's stored
  // points (net of hits, bench on Bench Boost: the same basis as plan.points) come from its own run; when it is stale
  // they are re-scored on this run, and with no re-score there is no honest difference to show.
  const rescored = yours?.stale ? rescoredNet(data, yours.result.weeks[0]) : null;
  const shownPoints = swapPoints ?? (tab.id === "team" ? data.current?.points ?? plan.points : yours ? rescored ?? yours.result.weeks[0].points : plan.points);
  const vs = tab.id === "team" || (yours?.stale && rescored === null) ? null : data.current?.points;

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
        <Headline points={shownPoints} vs={vs} />
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
            {tab.id === "plan" && planFailed && (
              <p className="plan-note" role="alert">
                Couldn't load your plan. Try again shortly.{" "}
                <button type="button" onClick={() => setPlanTry((n) => n + 1)}>Try again</button>
              </p>
            )}
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
                    <Rescored net={rescored} coach={plan.points} />
                    <button type="button" disabled={checking(recheckKey)} onClick={() => runRecheck(yours)}>
                      Re-check on latest run
                    </button>
                  </div>
                )}
                {check.state?.key === recheckKey && (
                  <CheckPanel snap={data} swaps={yours.swaps} chip={yours.chip} state={check.state.s} recheck
                    onRetry={() => runRecheck(yours)} onKeep={check.clear} onAdopted={adopted} />
                )}
                {data.gameweeks.includes(yours.result.weeks[0].gameweek) && <Pitch snap={data} week={known(data, yours.result.weeks[0])} />}
                <PlanMoves snap={data} week={known(data, yours.result.weeks[0])} />
              </>
            ) : tab.week
              ? <Pitch snap={data} week={tab.week} />
              : <SwapTab snap={data} swaps={swaps} chip={chip} onChange={(swaps, chip) => setSwap({ swaps, chip })} onCheck={runCheck} checking={checking(swapKey)} />}
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

/** The owner's week-1 squad re-scored on this run's predictions, net of its hits like the coach's figure; null when
 * this run has no swap rules, does not cover that gameweek or the squad no longer makes an XI. */
function rescoredNet(snap: Snapshot, week: Week): number | null {
  try {
    const row = swappable(snap) ? rescore(snap, [week])[0] : undefined;
    return row && swappable(snap) ? row.points - week.hits * snap.rules.hit_cost : null;
  } catch { return null; }
}

/** The headline shows the re-scored figure; this line only sets it beside the coach's new plan. */
function Rescored({ net, coach }: { net: number | null; coach: number }) {
  return net === null ? null : <p>{`Your plan re-scored on the latest predictions (shown above). The coach's new plan: ${coach.toFixed(1)} this week`}</p>;
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
