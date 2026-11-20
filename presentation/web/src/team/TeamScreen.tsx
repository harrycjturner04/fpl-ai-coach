import { useRef, useState, type KeyboardEvent } from "react";
import type { Snapshot, SnapshotResponse, Week } from "../types";
import { Chips } from "./Chips";
import { CoachCall } from "./CoachCall";
import { Headline } from "./Headline";
import { Issues } from "./Issues";
import { chipName, deadlineLabel, transfersSummary } from "./model";
import { Pitch } from "./Pitch";
import "./team.css";

type Tab = { id: "team" | "plan"; label: string; week: Week };

export function TeamScreen({ data, now = new Date() }: { data: SnapshotResponse; now?: Date }) {
  const plan = data.plan.weeks[0];
  const tabs: Tab[] = [
    ...(data.current ? [{ id: "team", label: "Your team", week: data.current } as const] : []),
    { id: "plan", label: "Coach's plan", week: plan },
  ];
  const [tabId, setTabId] = useState(tabs[0].id);
  const refs = useRef<(HTMLButtonElement | null)[]>([]);
  const tab = tabs.find((t) => t.id === tabId)!;

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
        <h1>{data.team_name}</h1>
        <p>{deadlineLabel(data.gameweek, data.deadline, now)}</p>
      </header>
      {data.status.stale && <p className="notice" role="status">{data.status.reason}</p>}
      <div className="team-grid">
        <Headline now={data.current?.projected_points ?? null} plan={plan.projected_points} />
        <section className="team-pitch" aria-label="Pitch">
          <div className="tabs" role="tablist" aria-label="Which team to show">
            {tabs.map((t, i) => (
              <button
                key={t.id}
                ref={(el) => { refs.current[i] = el; }}
                id={`tab-${t.id}`}
                role="tab"
                aria-selected={t.id === tabId}
                aria-controls="team-panel"
                tabIndex={t.id === tabId ? 0 : -1}
                onClick={() => setTabId(t.id)}
                onKeyDown={(e) => onKey(e, i)}
              >
                {t.label}
              </button>
            ))}
          </div>
          <div id="team-panel" role="tabpanel" aria-labelledby={`tab-${tab.id}`}>
            <Pitch snap={data} week={tab.week} />
            {tab.id === "plan" && <PlanMoves snap={data} week={plan} />}
          </div>
        </section>
        <CoachCall call={data.call} onSee={() => select(tabs.length - 1)} />
        <Issues snap={data} />
        <Chips snap={data} />
      </div>
    </main>
  );
}

function PlanMoves({ snap, week }: { snap: Snapshot; week: Week }) {
  const name = (id: number) => snap.players.find((p) => p.id === id)?.name ?? `Player ${id}`;
  return (
    <div className="moves">
      <p className="moves-head">
        {transfersSummary(week)}
        {week.chip && <span>{chipName(week.chip)} this week</span>}
      </p>
      {week.transfers_in.length > 0 && (
        <ul>
          {week.transfers_out.map((out, i) => <li key={out}>{`${name(out)} → ${name(week.transfers_in[i])}`}</li>)}
        </ul>
      )}
    </div>
  );
}
