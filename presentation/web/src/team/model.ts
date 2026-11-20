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
