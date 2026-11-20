import type { Player, Snapshot, Week } from "../types";

export type Status = "fit" | "doubt" | "out";

export type Slot = { player: Player; points: number; p60: number | null; captain: boolean; vice: boolean; isNew: boolean };

const POSITIONS = ["GKP", "DEF", "MID", "FWD"] as const;

export function statusOf(p: Player): { status: Status; label: string | null } {
  if (p.chance === 0 || ["i", "s", "u"].includes(p.status)) return { status: "out", label: "✕ 0%" };
  if (p.chance !== null && p.chance >= 25 && p.chance <= 75) return { status: "doubt", label: `! ${p.chance}%` };
  return { status: "fit", label: null };
}

/** Starters grouped GKP, DEF, MID, FWD (render reversed for attack at the top), and the bench in order. */
export function lineup(snap: Snapshot, week: Week): { rows: Slot[][]; bench: Slot[] } {
  const byId = new Map(snap.players.map((p) => [p.id, p]));
  const gw = snap.gameweeks.indexOf(week.gameweek);
  const slot = (id: number): Slot => {
    const player = byId.get(id)!;
    return {
      player,
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

export type Issue = { player: Player; status: Status; label: string; news: string; inSquad: boolean };

/** Squad players (now or in plan week 1) who are doubtful, out or in the news, plus anyone the coach flagged; squad first, then by chance. */
export function issues(snap: Snapshot): Issue[] {
  const squad = new Set([...(snap.current?.squad ?? []), ...(snap.plan.weeks[0]?.squad ?? [])]);
  const flags = new Map(snap.flags.map((f) => [f.player_id, f]));
  return snap.players
    .filter((p) => flags.has(p.id) || (squad.has(p.id) && (statusOf(p).status !== "fit" || !!p.news)))
    .map((p) => {
      const { status, label } = statusOf(p);
      return { player: p, status, label: label ?? "", news: p.news || flags.get(p.id)?.summary || "", inSquad: squad.has(p.id) };
    })
    .sort((a, b) => Number(b.inSquad) - Number(a.inSquad) || chanceOf(a) - chanceOf(b));
}
const chanceOf = (i: Issue) => (i.status === "out" ? 0 : i.player.chance ?? 100);

/** Best positive gain over no chip per chip, best first. */
export function chipsSoon(snap: Snapshot): { chip: string; gameweek: number; gain: number }[] {
  const best = new Map<string, { chip: string; gameweek: number; gain: number }>();
  for (const r of snap.chip_scan ?? []) {
    if (r.gain_vs_none > 0 && r.gain_vs_none > (best.get(r.chip)?.gain ?? 0)) best.set(r.chip, { chip: r.chip, gameweek: r.gameweek, gain: r.gain_vs_none });
  }
  return [...best.values()].sort((a, b) => b.gain - a.gain);
}

export function transfersSummary(week: Week): string {
  const n = week.transfers_in.length;
  if (!n) return "No transfers";
  return `${n} transfer${n === 1 ? "" : "s"}` + (week.hits ? `, −${4 * week.hits} hit` : "");
}
