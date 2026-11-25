import { useEffect, useMemo, useRef } from "react";
import { options, type Swap } from "./swap";
import type { Chip, Player, Snapshot2 } from "../types";
import "./swap.css";

const MAX_ROWS = 30;
const signed = (n: number) => `${n >= 0 ? "+" : "-"}${Math.abs(n).toFixed(1)}`;
const tone = (n: number) => (n >= 0 ? "gain" : "loss");

interface Props {
  snap: Snapshot2;
  player: Player;
  swaps: Swap[];
  chip: Chip | null;
  onSwap: (inn: number) => void;
  onClose: () => void;
}

export function PlayerSheet({ snap, player, swaps, chip, onSwap, onClose }: Props) {
  const closeRef = useRef<HTMLButtonElement>(null);
  const rows = useMemo(() => options(snap, swaps, chip, player.id), [snap, swaps, chip, player.id]);
  const best = rows.find((r) => r.reason === null);
  const fixtures = snap.fixtures[player.team] ?? [];

  // Focus the sheet on open; give it back to the disc that opened it on close.
  useEffect(() => {
    const opener = document.activeElement as HTMLElement | null;
    closeRef.current?.focus();
    return () => opener?.focus();
  }, []);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <div className="sheet-backdrop" onClick={onClose}>
      <section className="sheet" role="dialog" aria-modal="true" aria-label={player.name} onClick={(e) => e.stopPropagation()}>
        <header className="sheet-head">
          <div>
            <h2>{player.name}</h2>
            <p>{player.team} {player.position}</p>
          </div>
          <button ref={closeRef} type="button" className="sheet-close" aria-label="Close" onClick={onClose}>Close</button>
        </header>

        <p className="facts">
          {player.selling_price !== null && <span>Sells for £{player.selling_price.toFixed(1)}m</span>}
          <span>{player.p60 === null ? "No minutes estimate" : `${Math.round(player.p60 * 100)}% to play 60+`}</span>
        </p>
        {player.news && <p className="news">{player.news}</p>}

        <table className="xp">
          <caption className="sr-only">Predicted points and fixtures</caption>
          <thead>
            <tr>{snap.gameweeks.map((gw) => <th key={gw} scope="col">GW{gw}</th>)}</tr>
          </thead>
          <tbody>
            <tr>{snap.gameweeks.map((gw, i) => <td key={gw} className="num">{(player.points[i] ?? 0).toFixed(1)}</td>)}</tr>
            <tr className="fix">
              {snap.gameweeks.map((gw) => {
                const f = fixtures.filter((x) => x.gameweek === gw);
                return <td key={gw}>{f.length ? f.map((x, k) => <span key={k} className={`fdr fdr${x.fdr}`}>{x.opponent} ({x.home ? "H" : "A"})</span>) : "-"}</td>;
              })}
            </tr>
          </tbody>
        </table>

        <h3 className="best">
          {best ? <>Best swap: {best.player.name}, <span className={tone(best.gainWeek)}>{signed(best.gainWeek)}</span> this week</> : "No legal swap for this player"}
        </h3>
        <ul className="options">
          {rows.slice(0, MAX_ROWS).map((r) => (
            <li key={r.player.id}>
              <button
                type="button"
                aria-disabled={r.reason !== null || undefined}
                className={r.reason ? "opt off" : "opt"}
                onClick={() => r.reason === null && onSwap(r.player.id)}
              >
                <span className="who">
                  {r.player.name}
                  <span className="sub">{r.player.team} · £{r.player.price.toFixed(1)}m{r.reason && ` · ${r.reason}`}</span>
                </span>
                <span className="gains num">
                  <span className={tone(r.gainWeek)}>{signed(r.gainWeek)}</span>
                  <span className="sub">{signed(r.gainFive)} over 5</span>
                </span>
              </button>
            </li>
          ))}
        </ul>
      </section>
    </div>
  );
}
