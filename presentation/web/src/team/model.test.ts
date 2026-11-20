import { describe, expect, it } from "vitest";
import fixture from "../test/snapshot.json";
import type { Player, Snapshot } from "../types";
import { lineup, statusOf } from "./model";

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
  ] as const)("%o -> %s", (over, status, label) => {
    expect(statusOf(player(over))).toEqual({ status, label });
  });
});
