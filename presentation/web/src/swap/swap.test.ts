import golden from "./golden.json";
import { netSwaps, options, preview, rescore, type Swap } from "./swap";
import type { Chip, Player, Position, Snapshot2 } from "../types";

interface GoldenPlayer { id: number; position: Position; team: string; price: number; selling_price: number | null; points: number[] }
interface GoldenCase {
  name: string; players: GoldenPlayer[]; bank: number; free_transfers: number | null; chip: Chip | null; swaps: number[][];
  expect: { legal: boolean; reasons: string[]; hits: number; starting: number[] | null; captain: number | null;
    vice_captain: number | null; week_points: number | null; five_week: number | null };
}

const { max_hits, ...rules } = golden.rules;

function snapshotOf(players: GoldenPlayer[], bank: number, free: number | null): Snapshot2 {
  return {
    schema: 2, gameweeks: players[0].points.map((_, i) => i + 1),
    players: players.map((p) => ({ ...p, name: `P${p.id}`, status: "a", chance: null, news: null, p60: null })) as Player[],
    state: { bank, free_transfers: free, chips_available: ["wildcard", "freehit"], hit_cost: 99 },
    rules: rules as Snapshot2["rules"],
    settings: { horizon: 5, discount: 1, bench_weight: 0.1, leftover_values: [], hit_margin: 0, max_hits, summed: false, hold: false },
    fixtures: {},
  } as unknown as Snapshot2;
}

describe("golden cases", () => {
  it.each((golden.cases as GoldenCase[]).map((c) => [c.name, c] as const))("%s", (_name, c) => {
    const snap = snapshotOf(c.players, c.bank, c.free_transfers);
    const p = preview(snap, c.swaps as Swap[], c.chip);
    const e = c.expect;
    expect(p.legality.legal).toBe(e.legal);
    expect(p.legality.reasons).toEqual(e.reasons);
    expect(p.hits).toBe(e.hits);
    if (!e.legal) return;
    expect([...p.xi!.starting].sort((a, b) => a - b)).toEqual(e.starting);
    expect(p.xi!.captain).toBe(e.captain);
    expect(p.xi!.vice).toBe(e.vice_captain);
    expect(p.weekPoints).toBeCloseTo(e.week_points!, 6);
    expect(p.fiveWeek).toBeCloseTo(e.five_week!, 6);
  });
});

describe("netSwaps", () => {
  const owned = [1, 2, 3];
  it("chains A->B, B->C into A->C", () => expect(netSwaps(owned, [[1, 10], [10, 11]])).toEqual({ sold: [1], bought: [11] }));
  it("cancels A->B, B->A", () => expect(netSwaps(owned, [[1, 10], [10, 1]])).toEqual({ sold: [], bought: [] }));
  it("buys back a sold player", () => expect(netSwaps(owned, [[1, 10], [2, 11], [10, 1]])).toEqual({ sold: [2], bought: [11] }));
  it("rejects selling a player not in the squad", () => expect(() => netSwaps(owned, [[9, 10]])).toThrow());
});

describe("options", () => {
  const base = (golden.cases as GoldenCase[])[0].players.map((p) => ({ ...p }));
  const mids = base.filter((p) => p.position === "MID");
  const owned = mids.filter((p) => p.selling_price !== null);
  const [cand1, cand2, cand3] = mids.filter((p) => p.selling_price === null);
  // Three owned from ARS elsewhere in the squad; cand1 from ARS too, so buying it breaks the club cap.
  base.filter((p) => p.position === "DEF" && p.selling_price !== null).slice(0, 3).forEach((p) => (p.team = "ARS"));
  Object.assign(cand1, { team: "ARS", price: 4.0, points: [20, 20, 20, 20, 20] });
  Object.assign(cand2, { team: "ZZA", price: 4.0, points: [15, 1, 1, 1, 1] });
  Object.assign(cand3, { team: "ZZB", price: 4.0, points: [2, 2, 2, 2, 2] });
  const snap = snapshotOf(base, 10, 1);

  it("puts the best legal first and greys the club cap", () => {
    const rows = options(snap, [], null, owned[0].id);
    expect(rows.map((r) => r.player.id)).toEqual([cand2.id, cand3.id, cand1.id]);
    expect(rows[0].reason).toBeNull();
    expect(rows[0].gainWeek).toBeGreaterThan(rows[1].gainWeek);
    expect(rows[2].reason).toBe("already 3 from ARS");
  });
});

describe("options fast path", () => {
  it.each([null, "freehit", "wildcard"] as const)("gains equal full previews (chip %s)", (chip) => {
    for (const c of (golden.cases as GoldenCase[]).filter((c) => c.name.startsWith("random"))) {
      const snap = snapshotOf(c.players, c.bank, c.free_transfers ?? 1);
      const swaps = c.swaps.slice(0, 1) as Swap[];
      const base = preview(snap, swaps, chip);
      for (const out of base.squad) for (const r of options(snap, swaps, chip, out)) {
        const p = preview(snap, [...swaps, [out, r.player.id]], chip);
        expect(r.gainWeek).toBeCloseTo(p.weekPoints! - base.weekPoints!, 9);
        expect(r.gainFive).toBeCloseTo(p.fiveWeek! - base.fiveWeek!, 9);
        expect(r.reason).toBe(p.legality.reasons[0] ?? null);
      }
    }
  });
});

describe("rescore", () => {
  it("scores each known week's squad and skips weeks not in the snapshot", () => {
    const c = (golden.cases as GoldenCase[])[0];
    const snap = snapshotOf(c.players, c.bank, c.free_transfers);
    const squad = c.players.filter((p) => p.selling_price !== null).map((p) => p.id);
    const out = rescore(snap, [{ gameweek: 1, squad }, { gameweek: 99, squad }]);
    expect(out).toHaveLength(1);
    expect(out[0].gameweek).toBe(1);
    expect(out[0].points).toBeCloseTo(c.expect.week_points!, 6);
  });
});

describe("timing", () => {
  it("options over a 700-player snapshot runs under 16 ms", () => {
    const positions: [Position, number][] = [["GKP", 70], ["DEF", 230], ["MID", 280], ["FWD", 120]];
    let seed = 1;
    const rand = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
    const players: GoldenPlayer[] = [];
    const ownedLeft: Record<Position, number> = { ...rules.composition } as Record<Position, number>;
    for (const [pos, n] of positions) for (let k = 0; k < n; k++) {
      const id = players.length + 1, own = ownedLeft[pos]-- > 0;
      players.push({ id, position: pos, team: own ? `O${id}` : `T${id % 20}`, price: 4.5, selling_price: own ? 4.5 : null,
        points: Array.from({ length: 5 }, () => rand() * 9) });
    }
    const snap = snapshotOf(players, 2, 1);
    const out = players.find((p) => p.position === "MID" && p.selling_price !== null)!.id;
    expect(options(snap, [], null, out).length).toBe(275);
    options(snap, [], null, out);   // two warm-up runs
    const time = () => { const t = performance.now(); options(snap, [], null, out); return performance.now() - t; };
    const ms = Array.from({ length: 9 }, time).sort((x, y) => x - y)[4];   // median of nine: a tap's cost, not a GC pause
    expect(ms).toBeLessThan(16);
  });
});
