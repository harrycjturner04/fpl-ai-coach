import { useMemo, useState } from "react";
import { Pitch } from "../team/Pitch";
import { chipName } from "../team/model";
import { PlayerSheet } from "./PlayerSheet";
import { preview, type Swap } from "./swap";
import type { Chip, Snapshot, Snapshot2 } from "../types";
import "./swap.css";

const CHIPS: Chip[] = ["wildcard", "freehit"];
const signed = (n: number) => `${n >= 0 ? "+" : "−"}${Math.abs(n).toFixed(1)}`;

/** Swaps need schema 2 (rules) and a selling price for every current squad player. */
export function swappable(snap: Snapshot): snap is Snapshot2 {
  const byId = new Map(snap.players.map((p) => [p.id, p]));
  return snap.schema === 2 && !!snap.rules && !!snap.current
    && snap.current.squad.every((id) => byId.get(id)?.selling_price != null);
}

/** Adds out -> inn, kept net so every swap is independent and any one can be undone: selling a player bought earlier
 * extends that swap (A->B, B->C is A->C), buying back one sold earlier joins that swap (A->B, C->A is C->B), and a
 * chain that comes back to its start is nothing (A->B, B->A). */
export function addSwap(swaps: Swap[], out: number, inn: number): Swap[] {
  const prev = swaps.find(([, i]) => i === out), back = swaps.find(([o]) => o === inn);
  const start = prev ? prev[0] : out, end = back && back !== prev ? back[1] : inn;
  const rest = swaps.filter((s) => s !== prev && s !== back);
  return start === end ? rest : [...rest, [start, end]];
}

interface Props {
  snap: Snapshot;
  swaps: Swap[];
  chip: Chip | null;
  onChange: (swaps: Swap[], chip: Chip | null) => void;
  onCheck: (swaps: Swap[], chip: Chip | null) => void;
}

/** Try a swap: the previewed pitch, tally, impact, undo/reset, chip toggle and the check button. State lives in the
 * Team screen so the headline can follow it and the stack survives a tab switch. */
export function SwapTab({ snap, swaps, chip, onChange, onCheck }: Props) {
  const [open, setOpen] = useState<number | null>(null);
  const view = useMemo(() => (swappable(snap) ? { s: snap, base: preview(snap, [], null), now: preview(snap, swaps, chip) } : null),
    [snap, swaps, chip]);
  if (!view) return <p className="swap-empty">Run the coach to enable swaps.</p>;

  const { s, base, now } = view;
  const name = (id: number) => s.players.find((p) => p.id === id)!.name;
  const n = swaps.length, legal = now.legality.legal;
  const week = now.weekPoints === null ? null : now.weekPoints - base.weekPoints!;
  const five = now.fiveWeek === null ? null : now.fiveWeek - base.fiveWeek!;
  const player = open === null ? null : s.players.find((p) => p.id === open)!;

  return (
    <div className="swap">
      <div className="swap-bar">
        {s.state.chips_available.filter((c): c is Chip => CHIPS.includes(c as Chip)).map((c) => (
          <button key={c} type="button" className="toggle" aria-pressed={chip === c} onClick={() => onChange(swaps, chip === c ? null : c)}>
            {chipName(c)}
          </button>
        ))}
        {(n > 0 || chip) && <button type="button" className="reset" onClick={() => onChange([], null)}>Reset</button>}
      </div>
      {chip === "freehit" && <p className="swap-note">Your squad returns next week</p>}

      {now.xi && (
        <Pitch
          snap={s}
          week={{ gameweek: s.gameweeks[0], starting: now.xi.starting, bench: now.xi.bench, captain: now.xi.captain,
            vice_captain: now.xi.vice, transfers_in: swaps.map(([, i]) => i) }}
          onPick={setOpen}
        />
      )}

      <div className="swap-sum" aria-live="polite">
        {n === 0 ? <p className="swap-hint">Tap a player to see their best replacements.</p> : (
          <>
            <p className="tally">{`${n} transfer${n === 1 ? "" : "s"}, ${now.hits ? `−${now.hits * s.rules.hit_cost} hit` : "free"}`}</p>
            {week !== null && five !== null && (
              <div className="impact">
                <p className={`num ${week >= 0 ? "gain" : "loss"}`}>{`${signed(week)} this week`}</p>
                <p className="five">{`${week * five < 0 ? "but " : ""}${signed(five)} over the next 5 GWs (no further transfers)`}</p>
              </div>
            )}
          </>
        )}
        {!legal && (
          <ul className="reasons">
            {now.legality.reasons.map((r) => <li key={r}>{r}</li>)}
          </ul>
        )}
      </div>

      {n > 0 && (
        <ul className="pills" aria-label="Your swaps">
          {swaps.map(([out, inn]) => (
            <li key={out}>
              <button type="button" className="pill" aria-label={`Undo ${name(out)} → ${name(inn)}`} onClick={() => onChange(swaps.filter(([o]) => o !== out), chip)}>
                {`${name(out)} → ${name(inn)}`}<span aria-hidden="true"> ✕</span>
              </button>
            </li>
          ))}
        </ul>
      )}

      <button type="button" className="check" disabled={n === 0 || !legal} onClick={() => onCheck(swaps, chip)}>
        Check against the coach's plan
      </button>

      {player && (
        <PlayerSheet
          snap={s}
          player={player}
          swaps={swaps}
          chip={chip}
          onSwap={(inn) => { onChange(addSwap(swaps, player.id, inn), chip); setOpen(null); }}
          onClose={() => setOpen(null)}
        />
      )}
    </div>
  );
}
