import { describe, expect, it } from "vitest";
import fixture from "../test/snapshot.json";
import type { Flag, Player, Snapshot } from "../types";
import { chipsSoon, deadlineLabel, issues, lineup, moves, runLabel, statusOf, transfersSummary } from "./model";

const snap = () => structuredClone(fixture) as Snapshot;
const player = (over: Partial<Player>): Player => ({ ...snap().players[0], chance: null, status: "a", news: "", ...over });

describe("lineup", () => {
  it("puts 11 starters in a legal formation and 4 on the bench", () => {
    const s = snap();
    const { rows, bench } = lineup(s, s.current!);
    const [gk, def, mid, fwd] = rows.map((r) => r.length);
    expect(gk + def + mid + fwd).toBe(11);
    expect(gk).toBe(1);
    expect(def).toBeGreaterThanOrEqual(3);
    expect(def).toBeLessThanOrEqual(5);
    expect(mid).toBeGreaterThanOrEqual(2);
    expect(mid).toBeLessThanOrEqual(5);
    expect(fwd).toBeGreaterThanOrEqual(1);
    expect(fwd).toBeLessThanOrEqual(3);
    expect(rows.map((r) => r.map((x) => x.player.position))).toEqual([["GKP"], Array(def).fill("DEF"), Array(mid).fill("MID"), Array(fwd).fill("FWD")]);
    expect(bench).toHaveLength(4);
  });

  it("marks the captain and vice", () => {
    const s = snap();
    const slots = lineup(s, s.current!).rows.flat();
    expect(slots.filter((x) => x.captain).map((x) => x.player.id)).toEqual([s.current!.captain]);
    expect(slots.filter((x) => x.vice).map((x) => x.player.id)).toEqual([s.current!.vice_captain]);
  });

  it("marks transfers in as new", () => {
    const s = snap();
    const week = s.plan.weeks[0];
    week.transfers_in = [week.starting[0], week.bench[0]];
    const { rows, bench } = lineup(s, week);
    const all = [...rows.flat(), ...bench];
    expect(all.filter((x) => x.isNew).map((x) => x.player.id).sort()).toEqual([...week.transfers_in].sort());
  });

  it("takes points for the week's gameweek", () => {
    const s = snap();
    const p1 = (w: typeof s.current) => [...lineup(s, w!).rows.flat(), ...lineup(s, w!).bench].find((x) => x.player.id === 1)!;
    expect(p1(s.current).points).toBe(0);
    expect(p1(s.plan.weeks[1]).points).toBe(4);
  });

  it("keeps a null p60 and zero points", () => {
    const s = snap();
    const p = s.players.find((x) => x.id === s.current!.starting[0])!;
    p.p60 = null;
    p.points = [0, 0];
    const slot = lineup(s, s.current!).rows.flat().find((x) => x.player.id === p.id)!;
    expect(slot.p60).toBeNull();
    expect(slot.points).toBe(0);
  });
});

describe("statusOf", () => {
  it.each([
    [{ chance: 0 }, "out", "✕ 0%"],
    [{ chance: 75 }, "doubt", "! 75%"],
    [{ chance: 25 }, "doubt", "! 25%"],
    [{ chance: 100 }, "fit", null],
    [{ chance: null }, "fit", null],
    [{ status: "s", chance: null }, "out", "✕ 0%"],
    [{ status: "i", chance: null }, "out", "✕ 0%"],
    [{ status: "u", chance: null }, "out", "✕ 0%"],
    [{ status: "n", chance: null }, "out", "✕ 0%"],
  ] as const)("%o -> %s", (over, status, label) => {
    expect(statusOf(player(over))).toMatchObject({ status, label });
  });

  const flag = (over: Partial<Flag>): Flag => ({ ...snap().flags[0], ...over });
  it.each([
    [{ chance: null }, { status: "out" }, "out", "✕ 0%"],
    [{ chance: null }, { status: "suspended" }, "out", "✕ 0%"],
    [{ chance: null }, { status: "doubt", start_prob_next_gw: 0.5 }, "doubt", "! 50%"],
    [{ chance: 75 }, { status: "doubt", start_prob_next_gw: 0.5 }, "doubt", "! 50%"],
    [{ chance: 25 }, { status: "doubt", start_prob_next_gw: 0.5 }, "doubt", "! 25%"],
    [{ chance: 0 }, { status: "doubt", start_prob_next_gw: 0.5 }, "out", "✕ 0%"],
    [{ chance: null }, { status: "rotation_risk", start_prob_next_gw: 0.3 }, "fit", null],
    [{ chance: 50 }, { status: "fit", start_prob_next_gw: 1 }, "doubt", "! 50%"],
  ] as const)("FPL %o with flag %o -> the worse, %s", (over, f, status, label) => {
    expect(statusOf(player(over), flag(f))).toMatchObject({ status, label });
  });
});

describe("lineup flags", () => {
  it("carries the coach's flag so a flagged-out player shows out", () => {
    const s = snap();
    const p1 = lineup(s, s.current!).bench.find((x) => x.player.id === 1)!;
    expect(statusOf(p1.player, p1.flag)).toMatchObject({ status: "out", label: "✕ 0%" });
  });
});

describe("deadlineLabel", () => {
  const deadline = "2026-11-14T11:00:00+00:00";
  it("shows the local day, time and hours left", () => {
    expect(deadlineLabel(11, deadline, new Date("2026-11-12T21:00:00Z"), "en-GB")).toBe("Gameweek 11 deadline Sat 11:00 (38h)");
  });
  it("says under 1h in the last hour", () => {
    expect(deadlineLabel(11, deadline, new Date("2026-11-14T10:30:00Z"))).toBe("Gameweek 11 deadline Sat 11:00 (under 1h)");
  });
  it("says passed after the deadline", () => {
    expect(deadlineLabel(11, deadline, new Date("2026-11-14T11:00:00Z"))).toBe("Gameweek 11 deadline Sat 11:00 (passed)");
  });
});

describe("issues", () => {
  it("lists squad players with doubts or flags, the flagged-out player first", () => {
    const list = issues(snap());
    expect(list.map((x) => [x.player.name, x.status, x.label, x.news, x.inSquad])).toEqual([
      ["P1", "out", "✕ 0%", "Hamstring, out", true],
      ["P1000", "doubt", "! 75%", "Knock, 75% chance of playing", true],
    ]);
  });
  it("includes a flagged player outside the squad after the squad", () => {
    const s = snap();
    s.players.push(player({ id: 99, name: "STAR", chance: 0, news: "Ankle" }));
    s.flags.push({ ...s.flags[0], player_id: 99 });
    const list = issues(s);
    expect(list.at(-1)).toMatchObject({ player: { name: "STAR" }, status: "out", label: "✕ 0%", inSquad: false });
  });
  it("includes plan week 1 signings and sorts squad issues by chance", () => {
    const s = snap();
    s.players.push(player({ id: 98, name: "NEW", chance: 25, news: "Ill" }));
    s.plan.weeks[0].squad.push(98);
    expect(issues(s).map((x) => x.player.name)).toEqual(["P1", "NEW", "P1000"]);
  });
  it("is empty when nobody has news", () => {
    const s = snap();
    s.flags = [];
    s.players.forEach((p) => { p.chance = null; p.news = ""; });
    expect(issues(s)).toEqual([]);
  });
});

describe("chipsSoon", () => {
  it("keeps the best positive gain per chip, best first", () => {
    const s = snap();
    s.chip_scan!.push({ ...s.chip_scan![0], chip: "3xc", gameweek: 9, gain_vs_none: 5 }, { ...s.chip_scan![0], chip: "wildcard", gain_vs_none: -2 });
    const list = chipsSoon(s);
    expect(list.map((c) => [c.chip, c.gameweek, c.gain.toFixed(1)])).toEqual([["3xc", 9, "5.0"], ["bboost", 10, "3.6"]]);
  });
  it("is empty without a chip scan", () => {
    const s = snap();
    s.chip_scan = null;
    expect(chipsSoon(s)).toEqual([]);
  });
});

describe("transfersSummary", () => {
  const week = () => snap().plan.weeks[0];
  it("says no transfers", () => expect(transfersSummary(week(), 4)).toBe("No transfers"));
  it("counts transfers and the hit", () => {
    expect(transfersSummary({ ...week(), transfers_in: [1, 2], transfers_out: [3, 4], hits: 1 }, 4)).toBe("2 transfers, −4 hit");
    expect(transfersSummary({ ...week(), transfers_in: [1, 2], transfers_out: [3, 4], hits: 1 }, 5)).toBe("2 transfers, −5 hit");
    expect(transfersSummary({ ...week(), transfers_in: [1], transfers_out: [3], hits: 0 }, 4)).toBe("1 transfer");
  });
});

describe("moves", () => {
  it("pairs outgoing and incoming players by position", () => {
    const s = snap();
    const w = { ...s.plan.weeks[0], transfers_out: [1000, 1036], transfers_in: [1001, 1030] };
    expect(moves(s, w).map(([o, i]) => [o.name, i.name])).toEqual([["P1000", "P1001"], ["P1036", "P1030"]]);
    const swapped = { ...w, transfers_in: [1030, 1001] };
    expect(moves(s, swapped).map(([o, i]) => [o.position, i.position])).toEqual([["GKP", "GKP"], ["FWD", "FWD"]]);
  });
});

describe("runLabel", () => {
  it("shows when the coach ran", () => expect(runLabel("2026-11-10T12:00:00+00:00")).toBe("Coach run Tue 10 Nov, 12:00"));
});
