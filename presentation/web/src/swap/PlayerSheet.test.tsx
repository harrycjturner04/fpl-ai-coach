import { fireEvent, render, screen, within } from "@testing-library/react";
import { useState } from "react";
import golden from "./golden.json";
import { PlayerSheet } from "./PlayerSheet";
import type { Player, Position, Snapshot2 } from "../types";

const { max_hits, ...rules } = golden.rules;
const layout: [Position, number][] = [["GKP", 2], ["DEF", 5], ["MID", 5], ["FWD", 3]];
let id = 0;
const squad: Player[] = layout.flatMap(([position, n]) =>
  Array.from({ length: n }, () => {
    id += 1;
    return { id, name: `Own${id}`, team: `T${id}`, position, price: 5, selling_price: 5, status: "a", chance: null, news: null,
      points: [2, 2, 2, 2, 2, 2], p60: 0.9 } as Player;
  }));
const cands: Player[] = [
  { id: 101, name: "Cheap", team: "X1", position: "MID", price: 5, selling_price: null, status: "a", chance: null, news: null, points: [6, 3, 3, 3, 3, 3], p60: 0.9 },
  { id: 102, name: "Pricey", team: "X2", position: "MID", price: 14, selling_price: null, status: "a", chance: null, news: null, points: [9, 3, 3, 3, 3, 3], p60: 0.9 },
];
const target = { ...squad[7], news: "Knock, 75%", p60: 0.8, selling_price: 5.5 };
const snap = {
  schema: 2, gameweeks: [12, 13, 14, 15, 16, 17], players: [...squad.map((p) => (p.id === target.id ? target : p)), ...cands],
  state: { bank: 0, free_transfers: 1, chips_available: [], hit_cost: 4 }, rules,
  settings: { horizon: 5, discount: 1, bench_weight: 0.1, leftover_values: [], hit_margin: 0, max_hits, summed: false, hold: false },
  fixtures: { [target.team]: [{ gameweek: 12, opponent: "ARS", home: true, fdr: 4 }, { gameweek: 13, opponent: "CHE", home: false, fdr: 3 }] },
} as unknown as Snapshot2;

function Harness({ onSwap }: { onSwap: (n: number) => void }) {
  const [open, setOpen] = useState(false);
  return (
    <>
      <button onClick={() => setOpen(true)}>disc</button>
      {open && <PlayerSheet snap={snap} player={target} swaps={[]} chip={null} onSwap={onSwap} onClose={() => setOpen(false)} />}
    </>
  );
}

describe("PlayerSheet", () => {
  it("shows the player facts", () => {
    render(<Harness onSwap={() => {}} />);
    fireEvent.click(screen.getByText("disc"));
    const d = screen.getByRole("dialog");
    expect(within(d).getByText("Own8")).toBeInTheDocument();
    expect(within(d).getByText(/£5\.5m/)).toBeInTheDocument();
    expect(within(d).getByText(/80%/)).toBeInTheDocument();
    expect(within(d).getByText("Knock, 75%")).toBeInTheDocument();
    expect(within(d).getByText(/ARS \(H\)/)).toBeInTheDocument();
    expect(within(d).getByText(/CHE \(A\)/)).toBeInTheDocument();
  });

  it("names the best legal swap first and greys the illegal one with its reason", () => {
    render(<Harness onSwap={() => {}} />);
    fireEvent.click(screen.getByText("disc"));
    const d = screen.getByRole("dialog");
    expect(within(d).getByRole("heading", { level: 3 })).toHaveTextContent("Best swap: Cheap, +8.0 this week");
    const rows = within(d).getAllByRole("button", { name: /Cheap|Pricey/ });
    expect(rows[0]).toHaveTextContent("Cheap");
    expect(rows[1]).toHaveAttribute("aria-disabled", "true");
    expect(rows[1]).toHaveTextContent(/short/);
  });

  it("calls onSwap for a legal pick and ignores an illegal one", () => {
    const onSwap = vi.fn();
    render(<Harness onSwap={onSwap} />);
    fireEvent.click(screen.getByText("disc"));
    fireEvent.click(screen.getByRole("button", { name: /Pricey/ }));
    expect(onSwap).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: /Cheap/ }));
    expect(onSwap).toHaveBeenCalledWith(101);
  });

  it("moves focus in, closes on Escape and restores focus", () => {
    render(<Harness onSwap={() => {}} />);
    const disc = screen.getByText("disc");
    disc.focus();
    fireEvent.click(disc);
    expect(screen.getByRole("dialog").contains(document.activeElement)).toBe(true);
    fireEvent.keyDown(document, { key: "Escape" });
    expect(screen.queryByRole("dialog")).toBeNull();
    expect(disc).toHaveFocus();
  });

  it("closes with the close button", () => {
    render(<Harness onSwap={() => {}} />);
    fireEvent.click(screen.getByText("disc"));
    fireEvent.click(screen.getByRole("button", { name: "Close" }));
    expect(screen.queryByRole("dialog")).toBeNull();
  });
});
