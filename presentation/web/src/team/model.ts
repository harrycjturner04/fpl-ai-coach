import type { Flag, Player, Snapshot, Week } from "../types";

export type Status = "fit" | "doubt" | "out";

export type Slot = { player: Player; flag?: Flag; points: number; p60: number | null; captain: boolean; vice: boolean; isNew: boolean };
export type StatusInfo = { status: Status; label: string | null; chance: number | null };

const POSITIONS = ["GKP", "DEF", "MID", "FWD"] as const;

const OUT: StatusInfo = { status: "out", label: "✕ 0%", chance: 0 };
const doubt = (chance: number): StatusInfo => ({ status: "doubt", label: `! ${chance}%`, chance });
const severity = (s: StatusInfo) => (s.status === "fit" ? Infinity : s.chance ?? 0);

/** The worse of FPL's status and the coach's flag (FPL i/s/u/n or out/suspended -> out, doubt -> its start probability; fit and rotation risk change nothing). */
export function statusOf(p: Player, flag?: Flag): StatusInfo {
  const fpl: StatusInfo =
    p.chance === 0 || ["i", "s", "u", "n"].includes(p.status) ? OUT
    : p.chance !== null && p.chance >= 25 && p.chance <= 75 ? doubt(p.chance)
    : { status: "fit", label: null, chance: p.chance };
  const coach = flag?.status === "out" || flag?.status === "suspended" ? OUT
    : flag?.status === "doubt" ? doubt(Math.round(flag.start_prob_next_gw * 100))
    : null;
  return coach && severity(coach) < severity(fpl) ? coach : fpl;
}

export const flagsById = (snap: Snapshot) => new Map(snap.flags.map((f) => [f.player_id, f]));

/** Starters grouped GKP, DEF, MID, FWD (render reversed for attack at the top), and the bench in order. */
export function lineup(snap: Snapshot, week: Week): { rows: Slot[][]; bench: Slot[] } {
  const byId = new Map(snap.players.map((p) => [p.id, p]));
  const flags = flagsById(snap);
  const gw = snap.gameweeks.indexOf(week.gameweek);
  const slot = (id: number): Slot => {
    const player = byId.get(id)!;
    return {
      player,
      flag: flags.get(id),
      points: player.points[gw],
      p60: player.p60,
      captain: id === week.captain,
      vice: id === week.vice_captain,
      isNew: week.transfers_in.includes(id),
    };
  };
  const starters = week.starting.map(slot);
  return {
    rows: POSITIONS.map((pos) => starters.filter((s) => s.player.position === pos)),
    bench: week.bench.map(slot),
  };
}

export const CHIPS: Record<string, string> = { wildcard: "Wildcard", freehit: "Free Hit", bboost: "Bench Boost", "3xc": "Triple Captain" };
export const chipName = (chip: string) => CHIPS[chip] ?? chip;

/** "Gameweek 11 deadline Sat 11:00 (38h)" in the viewer's time zone. */
export function deadlineLabel(gw: number, deadlineIso: string, now: Date, locale = "en-GB"): string {
  const deadline = new Date(deadlineIso);
  const fmt = (o: Intl.DateTimeFormatOptions) => deadline.toLocaleString(locale, o);
  const hours = (deadline.getTime() - now.getTime()) / 3_600_000;
  const left = hours <= 0 ? "passed" : hours < 1 ? "under 1h" : `${Math.floor(hours)}h`;
  return `Gameweek ${gw} deadline ${fmt({ weekday: "short" })} ${fmt({ hour: "2-digit", minute: "2-digit", hourCycle: "h23" })} (${left})`;
}

/** "Coach run Tue 10 Nov, 12:00" in the viewer's time zone. */
export function runLabel(generatedIso: string, locale = "en-GB"): string {
  const t = new Date(generatedIso);
  const date = t.toLocaleString(locale, { weekday: "short", day: "numeric", month: "short" });
  return `Coach run ${date}, ${t.toLocaleString(locale, { hour: "2-digit", minute: "2-digit", hourCycle: "h23" })}`;
}

export type Issue = { player: Player; status: Status; label: string; chance: number | null; news: string; inSquad: boolean };

/** Squad players (now or in plan week 1) who are doubtful, out or in the news, plus anyone the coach flagged; squad first, then by chance. */
export function issues(snap: Snapshot): Issue[] {
  const squad = new Set([...(snap.current?.squad ?? []), ...(snap.plan.weeks[0]?.squad ?? [])]);
  const flags = flagsById(snap);
  return snap.players
    .filter((p) => flags.has(p.id) || (squad.has(p.id) && (statusOf(p, flags.get(p.id)).status !== "fit" || !!p.news)))
    .map((p) => {
      const { status, label, chance } = statusOf(p, flags.get(p.id));
      return { player: p, status, label: label ?? "", chance, news: p.news || flags.get(p.id)?.summary || "", inSquad: squad.has(p.id) };
    })
    .sort((a, b) => Number(b.inSquad) - Number(a.inSquad) || chanceOf(a) - chanceOf(b));
}
const chanceOf = (i: Issue) => i.chance ?? 100;

/** Best positive gain over no chip per chip, best first. */
export function chipsSoon(snap: Snapshot): { chip: string; gameweek: number; gain: number }[] {
  const best = new Map<string, { chip: string; gameweek: number; gain: number }>();
  for (const r of snap.chip_scan ?? []) {
    if (r.gain_vs_none > 0 && r.gain_vs_none > (best.get(r.chip)?.gain ?? 0)) best.set(r.chip, { chip: r.chip, gameweek: r.gameweek, gain: r.gain_vs_none });
  }
  return [...best.values()].sort((a, b) => b.gain - a.gain);
}

export function transfersSummary(week: Week, hitCost: number): string {
  const n = week.transfers_in.length;
  if (!n) return "No transfers";
  return `${n} transfer${n === 1 ? "" : "s"}` + (week.hits ? `, −${hitCost * week.hits} hit` : "");
}

/** Outgoing and incoming players paired by position (the squad's composition is fixed, so counts match). */
export function moves(snap: Snapshot, week: Week): [Player, Player][] {
  const byId = new Map(snap.players.map((p) => [p.id, p]));
  const ins = week.transfers_in.map((id) => byId.get(id)!);
  return week.transfers_out.map((id) => {
    const out = byId.get(id)!;
    const i = ins.findIndex((p) => p.position === out.position);
    return [out, ins.splice(i, 1)[0]];
  });
}
