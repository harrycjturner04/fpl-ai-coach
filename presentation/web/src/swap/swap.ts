/** Instant swap preview in the browser, from a schema-2 snapshot only. Pinned to the Python rules by golden.json
 * (scripts/make_swap_golden.py). Rule values always come from the snapshot. */
import type { Chip, Player, Position, Snapshot2 } from "../types";

export type Swap = [out: number, inn: number];

/** "the next 5 GWs" for the gameweeks a snapshot covers (fewer near the season's end). */
export const nextGws = (snap: { gameweeks: number[] }) => {
  const n = snap.gameweeks.length;
  return `the next ${n} GW${n === 1 ? "" : "s"}`;
};

export interface XI { starting: number[]; bench: number[]; captain: number; vice: number; points: number }

interface Info { byId: Map<number, Player>; owned: number[]; budget: number; positions: Position[]; formations: number[][] }

/** Per-snapshot lookups, built once: player by id, owned ids, budget in tenths (bank + all selling prices), and every
 * XI count per position (in `rules.composition` order) within the XI limits. */
const cache = new WeakMap<Snapshot2, Info>();
function info(snap: Snapshot2): Info {
  let c = cache.get(snap);
  if (!c) {
    const { rules } = snap, owned = snap.players.filter((p) => p.selling_price !== null);
    const positions = Object.keys(rules.composition) as Position[], formations: number[][] = [];
    const walk = (counts: number[], left: number) => {
      const k = counts.length;
      if (k === positions.length) { if (left === 0) formations.push(counts); return; }
      for (let n = rules.xi_min[positions[k]]; n <= Math.min(rules.xi_max[positions[k]], left); n++) walk([...counts, n], left - n);
    };
    walk([], rules.starting_xi);
    c = { byId: new Map(snap.players.map((p) => [p.id, p])), owned: owned.map((p) => p.id), positions, formations,
      budget: Math.round(10 * owned.reduce((s, p) => s + p.selling_price!, snap.state.bank)) };
    cache.set(snap, c);
  }
  return c;
}
const byId = (snap: Snapshot2) => info(snap).byId;
const ownedIds = (snap: Snapshot2) => info(snap).owned;

/** (sold, bought) after applying the swaps in turn, as orchestration/swaps.py net_swaps: A->B, B->C sells A and
 * buys C; A->B, B->A is nothing. */
export function netSwaps(owned: number[], swaps: Swap[]): { sold: number[]; bought: number[] } {
  const own = new Set(owned), squad = new Set(owned);
  for (const [out, inn] of swaps) {
    if (!squad.has(out)) throw new Error(`${out} is not in your squad.`);
    if (squad.has(inn)) throw new Error(`${inn} is already in your squad.`);
    squad.delete(out);
    squad.add(inn);
  }
  const sold = [...new Set(swaps.map(([o]) => o).filter((o) => own.has(o) && !squad.has(o)))];
  const bought = [...new Set(swaps.map(([, i]) => i).filter((i) => !own.has(i) && squad.has(i)))];
  return { sold, bought };
}

export function squadAfter(snap: Snapshot2, swaps: Swap[]): number[] {
  return after(ownedIds(snap), netSwaps(ownedIds(snap), swaps));
}

const after = (owned: number[], { sold, bought }: { sold: number[]; bought: number[] }) =>
  [...owned.filter((i) => !sold.includes(i)), ...bought].sort((a, b) => a - b);

/** Hit count (not points) for `transfers` bought this week. */
export function hits(snap: Snapshot2, transfers: number, chip: Chip | null): number {
  const free = snap.state.free_transfers;
  return chip || free === null ? 0 : Math.max(0, transfers - free);
}

/** Composition, club cap, budget (bank + selling prices of those sold - prices of those bought), then the hit cap. */
export function legality(snap: Snapshot2, squad: number[], nHits: number): { legal: boolean; reasons: string[]; moneyLeft: number } {
  const { rules } = snap, players = squad.map((i) => byId(snap).get(i)!);
  const reasons: string[] = [];
  for (const [pos, n] of Object.entries(rules.composition))
    if (players.filter((p) => p.position === pos).length < n) reasons.push(`needs a ${pos}`);
  const clubs = new Map<string, number>();
  for (const p of players) clubs.set(p.team, (clubs.get(p.team) ?? 0) + 1);
  for (const team of [...clubs.keys()].filter((t) => clubs.get(t)! > rules.max_per_club).sort())
    reasons.push(`already ${rules.max_per_club} from ${team}`);
  const budget = info(snap).budget;
  const cost = Math.round(10 * players.reduce((s, p) => s + (p.selling_price ?? p.price), 0));
  if (cost > budget) reasons.push(`£${((cost - budget) / 10).toFixed(1)}m short`);
  const maxHits = snap.settings.max_hits;
  if (maxHits !== null && nHits > maxHits)
    reasons.push(`At most ${maxHits} hits a week: play a Wildcard or Free Hit, or make fewer swaps.`);
  return { legal: reasons.length === 0, reasons, moneyLeft: (budget - cost) / 10 };
}

/** Best XI of a held squad for one week (index into snap.gameweeks): every formation within the XI limits, captain the
 * top starter (points doubled), vice the second, bench GKP first then by points. Throws when no formation fits. */
export function bestXI(snap: Snapshot2, squad: number[], week: number): XI {
  const { byId: m, formations } = info(snap);
  const rows = ranked(snap, squad, week);
  const prefix = prefixes(snap, rows);
  const best = bestFormation(formations, (j, n) => prefix[j][n]);
  if (!best) throw new Error("No valid formation for this squad.");
  const left = [...best.counts], starting: number[] = [], bench: number[] = [];
  for (const r of rows) (left[r.pos]-- > 0 ? starting : bench).push(r.id);
  bench.sort((a, b) => Number(m.get(b)!.position === "GKP") - Number(m.get(a)!.position === "GKP"));
  return { starting, bench, captain: starting[0], vice: starting[1], points: best.points };
}

/** Squad players with their position index and this week's points, best first. */
function ranked(snap: Snapshot2, squad: number[], week: number) {
  const { byId: m, positions } = info(snap);
  return squad.map((id) => { const p = m.get(id)!; return { id, pos: positions.indexOf(p.position), pts: p.points[week] ?? 0 }; })
    .sort((a, b) => b.pts - a.pts);
}

/** prefix[j][n]: points of the top n at position j (rows best first). */
function prefixes(snap: Snapshot2, rows: { pos: number; pts: number }[]): number[][] {
  const prefix = info(snap).positions.map(() => [0]);
  for (const r of rows) { const pj = prefix[r.pos]; pj.push(pj[pj.length - 1] + r.pts); }
  return prefix;
}

/** Best formation given `top(j, n)`, the points of the top n at position j (undefined when there are fewer): XI points
 * plus the captain's (the best starter, `top(j, 1)` over the positions used). */
function bestFormation(formations: number[][], top: (j: number, n: number) => number | undefined) {
  let best: { counts: number[]; points: number } | null = null;
  for (const counts of formations) {
    let points = 0, captain = -Infinity, ok = true;
    for (let j = 0; j < counts.length && ok; j++) {
      if (counts[j] === 0) continue;
      const v = top(j, counts[j]);
      if (v === undefined) ok = false;
      else { points += v; captain = Math.max(captain, top(j, 1)!); }
    }
    if (ok && (!best || points + captain > best.points)) best = { counts, points: points + captain };
  }
  return best;
}

/** This week and the next five (squad held, XI and captain re-picked each week, bench excluded, undiscounted, minus
 * hits). Free Hit: weeks after the first use the original squad. XI and points are null when no formation fits. */
export function preview(snap: Snapshot2, swaps: Swap[], chip: Chip | null) {
  const net = netSwaps(ownedIds(snap), swaps), squad = after(ownedIds(snap), net);
  const nHits = hits(snap, net.bought.length, chip);
  const legal = legality(snap, squad, nHits);
  const cost = nHits * snap.rules.hit_cost;
  let xi: XI | null = null, weekPoints: number | null = null, fiveWeek: number | null = null;
  if (legal.reasons.every((r) => !r.startsWith("needs a "))) {
    xi = bestXI(snap, squad, 0);
    const later = chip === "freehit" ? ownedIds(snap) : squad;
    weekPoints = xi.points - cost;
    fiveWeek = snap.gameweeks.slice(1).reduce((s, _, w) => s + bestXI(snap, later, w + 1).points, weekPoints);
  }
  return { squad, xi, hits: nHits, weekPoints, fiveWeek, legality: legal };
}

/** Replacements for `out` (same position, not in the squad): legal first by this-week gain, illegal after with the
 * first reason. Fast path: per week, the squad without `out` is ranked once and each candidate inserted into its
 * position's top-n sums (top n of L plus x is max(top n of L, top n-1 of L + x)). */
export function options(snap: Snapshot2, swaps: Swap[], chip: Chip | null, out: number) {
  const base = preview(snap, swaps, chip), { byId: m, positions, formations } = info(snap);
  const position = m.get(out)!.position, q = positions.indexOf(position), cost = snap.rules.hit_cost;
  const rest = base.squad.filter((i) => i !== out);
  const later = chip === "freehit" ? 1 : snap.gameweeks.length;   // Free Hit: weeks 2+ are the original squad
  const weeks = Array.from({ length: later }, (_, w) => prefixes(snap, ranked(snap, rest, w)));
  const rows = snap.players.filter((p) => p.position === position && !base.squad.includes(p.id)).map((player) => {
    const next: Swap[] = [...swaps, [out, player.id]];
    const net = netSwaps(ownedIds(snap), next), nHits = hits(snap, net.bought.length, chip);
    const reason = legality(snap, after(ownedIds(snap), net), nHits).reasons[0] ?? null;
    const pts = weeks.map((prefix, w) => {
      const x = player.points[w] ?? 0, pq = prefix[q];
      return bestFormation(formations, (j, n) => j !== q ? prefix[j][n]
        : n > pq.length ? undefined : Math.max(pq[n] ?? -Infinity, pq[n - 1] + x))?.points ?? NaN;
    });
    const gainWeek = pts[0] - nHits * cost - base.weekPoints!;
    const gainFive = pts.reduce((s, v) => s + v, -nHits * cost) - (later === 1 ? base.weekPoints! : base.fiveWeek!);
    return { player, gainWeek, gainFive, reason };
  });
  return rows.sort((a, b) => Number(a.reason !== null) - Number(b.reason !== null) || b.gainWeek - a.gainWeek);
}

/** Best XI points of a stored plan's squads on this snapshot's predictions, plus the bench on a Bench Boost week (as
 * the plan's points); weeks not in the snapshot are skipped, as are players no longer in it. Throws when no formation
 * fits. */
export function rescore(snap: Snapshot2, weeks: { gameweek: number; squad: number[]; chip?: string | null }[]): { gameweek: number; points: number }[] {
  return weeks.filter((w) => snap.gameweeks.includes(w.gameweek)).map((w) => {
    const week = snap.gameweeks.indexOf(w.gameweek), m = byId(snap);
    const xi = bestXI(snap, w.squad.filter((i) => m.has(i)), week);
    const bench = w.chip === "bboost" ? xi.bench.reduce((s, i) => s + (m.get(i)!.points[week] ?? 0), 0) : 0;
    return { gameweek: w.gameweek, points: xi.points + bench };
  });
}
