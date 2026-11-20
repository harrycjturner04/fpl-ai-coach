import { fireEvent, render, screen, within } from "@testing-library/react";
import { expect, it } from "vitest";
import fixture from "../test/snapshot.json";
import type { SnapshotResponse } from "../types";
import { TeamScreen } from "./TeamScreen";

const snap = () => structuredClone(fixture) as SnapshotResponse;
const NOW = new Date("2026-11-12T21:00:00Z");
const show = (s: SnapshotResponse) => render(<TeamScreen data={s} now={NOW} />);
const CAPTION = "Predicted points for your team now, and with the coach's plan";
const headline = () => screen.getByRole("group", { name: "Predicted points" });
const planTab = () => screen.getByRole("tab", { name: "Coach's plan" });

/** Plan week 1 swaps P1000 for P1001 with a hit. */
const withTransfer = () => {
  const s = snap();
  const w = s.plan.weeks[0];
  w.squad = w.squad.map((id) => (id === 1000 ? 1001 : id));
  w.starting = w.starting.map((id) => (id === 1000 ? 1001 : id));
  w.transfers_in = [1001];
  w.transfers_out = [1000];
  w.hits = 1;
  w.points = 31.46;
  return s;
};

it("shows the header and both headline figures to one decimal", () => {
  show(withTransfer());
  expect(screen.getByRole("heading", { name: "Test FC" })).toBeInTheDocument();
  expect(screen.getByText("Gameweek 9 deadline Sat 11:00 (38h)")).toBeInTheDocument();
  expect(within(headline()).getByText("29.0")).toBeInTheDocument();
  expect(within(headline()).getByText("31.5")).toHaveClass("gain");
  expect(screen.getByText(CAPTION)).toBeInTheDocument();
});

it("colours a loss", () => {
  const s = snap();
  s.plan.weeks[0].points = 27;
  show(s);
  expect(within(headline()).getByText("27.0")).toHaveClass("loss");
});

it("shows a Bench Boost week's points, bench included, not the XI-only figure", () => {
  const s = snap();
  s.plan.weeks[0].chip = "bboost";
  s.plan.weeks[0].projected_points = 30;
  s.plan.weeks[0].points = 38.04;
  show(s);
  expect(within(headline()).getByText("38.0")).toHaveClass("gain");
  expect(within(headline()).queryByText("30.0")).toBeNull();
});

it("switches the pitch to the coach's plan", () => {
  show(withTransfer());
  expect(screen.queryByText("New")).toBeNull();
  expect(screen.queryByText("1 transfer, −4 hit")).toBeNull();
  fireEvent.click(planTab());
  expect(planTab()).toHaveAttribute("aria-selected", "true");
  expect(screen.getByText("New")).toBeInTheDocument();
  expect(screen.getByText("1 transfer, −4 hit")).toBeInTheDocument();
  expect(screen.getByText("P1000 → P1001")).toBeInTheDocument();
});

it("shows no New markers when the plan has no transfers", () => {
  show(snap());
  fireEvent.click(planTab());
  expect(screen.queryByText("New")).toBeNull();
  expect(screen.getByText("No transfers")).toBeInTheDocument();
});

it("moves between tabs with the arrow keys", () => {
  show(snap());
  fireEvent.keyDown(screen.getByRole("tab", { name: "Your team" }), { key: "ArrowRight" });
  expect(planTab()).toHaveAttribute("aria-selected", "true");
  expect(planTab()).toHaveFocus();
});

it("shows the coach's call and its button opens the plan", () => {
  show(snap());
  expect(screen.getByText("Save your free transfer")).toBeInTheDocument();
  expect(screen.getByText("P1 is out.")).toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "See the coach's plan" }));
  expect(planTab()).toHaveAttribute("aria-selected", "true");
});

it("names the chip on the call button", () => {
  const s = snap();
  s.call.chip = "bboost";
  show(s);
  expect(screen.getByRole("button", { name: "See the Bench Boost team" })).toBeInTheDocument();
});

it("lists injuries and chips", () => {
  show(snap());
  const injuries = screen.getByRole("region", { name: "Injuries and doubts" });
  expect(within(injuries).getByText("! 75%")).toBeInTheDocument();
  expect(within(injuries).getByText("Knock, 75% chance of playing")).toBeInTheDocument();
  expect(within(injuries).getByText("Hamstring, out")).toBeInTheDocument();
  const chips = screen.getByRole("region", { name: "Chips worth playing soon" });
  expect(within(chips).getByRole("listitem")).toHaveTextContent("Bench Boost in GW 10, +3.6 over no chip");
});

it("shows when the coach ran when the run is fresh", () => {
  show(snap());
  expect(screen.getByText("Coach run Tue 10 Nov, 12:00")).toBeInTheDocument();
});

it("shows the run-status notice when stale", () => {
  const s = snap();
  s.status = { stale: true, reason: "New data since this run. The figures may have changed." };
  show(s);
  expect(screen.getByText("New data since this run. The figures may have changed.")).toBeInTheDocument();
});

it("handles a run without the AI write-up", () => {
  const s = snap();
  s.sections = {};
  s.chip_scan = null;
  s.flags = [];
  s.players.forEach((p) => { p.chance = null; p.news = ""; });
  s.call.reason = "No write-up this run: the plan is the optimiser's alone.";
  show(s);
  expect(screen.getByText("No write-up this run: the plan is the optimiser's alone.")).toBeInTheDocument();
  expect(screen.getByText("No injury or suspension news for your players.")).toBeInTheDocument();
  expect(screen.getByText("No chip beats holding it in the next 5 gameweeks.")).toBeInTheDocument();
});

it("shows only the plan figure without a current team", () => {
  const s = snap();
  s.current = null;
  show(s);
  expect(within(headline()).getAllByText("29.0")).toHaveLength(1);
  expect(screen.getByText("Predicted points with the coach's plan")).toBeInTheDocument();
  expect(screen.queryByRole("tab", { name: "Your team" })).toBeNull();
  expect(planTab()).toHaveAttribute("aria-selected", "true");
});

it("shows the coach's out flag on the pitch and in the list", () => {
  show(snap());
  const bench = screen.getByRole("group", { name: "Bench" });
  expect(within(bench).getByRole("img", { name: /^P1,.*0% chance of playing/ })).toBeInTheDocument();
  const injuries = screen.getByRole("region", { name: "Injuries and doubts" });
  expect(within(injuries).getAllByRole("listitem")[0]).toHaveTextContent("P1, C1Hamstring, out✕ 0%");
});

it("shows a coach doubt flag as its start probability", () => {
  const s = snap();
  s.flags[0] = { ...s.flags[0], status: "doubt", start_prob_next_gw: 0.5 };
  show(s);
  expect(within(screen.getByRole("group", { name: "Bench" })).getByText("! 50%")).toBeInTheDocument();
  expect(within(screen.getByRole("region", { name: "Injuries and doubts" })).getByText("! 50%")).toBeInTheDocument();
});

it("lists plan moves paired by position", () => {
  const s = withTransfer();
  const w = s.plan.weeks[0];
  w.squad = w.squad.map((id) => (id === 1036 ? 1030 : id));
  w.starting = w.starting.map((id) => (id === 1036 ? 1030 : id));
  w.transfers_out = [1000, 1036];
  w.transfers_in = [1030, 1001];
  show(s);
  fireEvent.click(planTab());
  expect(screen.getByText("P1000 → P1001")).toBeInTheDocument();
  expect(screen.getByText("P1036 → P1030")).toBeInTheDocument();
});

it("falls back to the first tab when the selected one disappears", () => {
  const s = snap();
  const { rerender } = show(s);
  rerender(<TeamScreen data={{ ...s, current: null }} now={NOW} />);
  expect(planTab()).toHaveAttribute("aria-selected", "true");
});
