import { act, fireEvent, render, screen, within } from "@testing-library/react";
import { expect, it, vi } from "vitest";
import fixture from "../test/snapshot.json";
import type { SnapshotResponse } from "../types";
import { TeamScreen } from "../team/TeamScreen";

vi.mock("../api", () => ({ getLatestRun: () => Promise.resolve(null), getRun: vi.fn(), startRun: vi.fn(), ApiError: class extends Error {} }));

const NOW = new Date("2026-11-12T21:00:00Z");
const pts = (s: SnapshotResponse, id: number, points: number[]) => { s.players.find((p) => p.id === id)!.points = points; };
/** P1036 (FWD, starter, 1 then 5) can go for P1030 (3 then 0): up this week, down over the horizon. */
const snap = (edit: (s: SnapshotResponse) => void = () => {}) => {
  const s = structuredClone(fixture) as SnapshotResponse;
  pts(s, 1036, [1, 5]);
  pts(s, 1030, [3, 0]);
  edit(s);
  return s;
};
const show = (s = snap()) => {
  render(<TeamScreen data={s} now={NOW} onSettings={() => {}} onReload={() => {}} />);
  fireEvent.click(screen.getByRole("tab", { name: "Try a swap" }));
};
const headline = () => screen.getByRole("group", { name: "Predicted points" });
const swap = (out: string, inn: string) => {
  fireEvent.click(screen.getByRole("button", { name: new RegExp(`^${out},`) }));
  fireEvent.click(within(screen.getByRole("dialog")).getByRole("button", { name: new RegExp(`^${inn}`) }));
};
const check = () => screen.getByRole("button", { name: "Check against the coach's plan" });

it("updates the headline and the pitch after a swap", () => {
  show();
  expect(within(headline()).getAllByText("29.0")).toHaveLength(2);
  expect(screen.getByText("Predicted points for your team now, and with your swaps")).toBeInTheDocument();
  expect(check()).toBeDisabled();
  swap("P1036", "P1030");
  expect(within(headline()).getByText("31.0")).toHaveClass("gain");
  expect(screen.queryByRole("button", { name: /^P1036,/ })).toBeNull();
  expect(screen.getByRole("button", { name: /^P1030,.*new/ })).toBeInTheDocument();
  expect(screen.getByText("1 transfer, free")).toBeInTheDocument();
  expect(check()).toBeEnabled();
});

it("says but when this week is up and the five weeks are down", () => {
  show();
  swap("P1036", "P1030");
  expect(screen.getByText("+2.0 this week")).toBeInTheDocument();
  expect(screen.getByText("but -2.0 over the next 5 GWs (no further transfers)")).toBeInTheDocument();
});

it("stacks swaps with a hit, undoes one and resets", () => {
  show();
  swap("P1036", "P1030");
  swap("P1037", "P1031");
  expect(screen.getByText("2 transfers, −4 hit")).toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Undo P1036 → P1030" }));
  expect(screen.getByText("1 transfer, free")).toBeInTheDocument();
  expect(screen.getByRole("button", { name: /^P1036,/ })).toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Reset" }));
  expect(screen.queryByText(/^\d+ transfers?,/)).toBeNull();
  expect(within(headline()).getAllByText("29.0")).toHaveLength(2);
});

it("nets a swap of a player bought earlier into one transfer", () => {
  show();
  swap("P1036", "P1030");
  swap("P1030", "P1031");
  expect(screen.getByText("1 transfer, free")).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Undo P1036 → P1031" })).toBeInTheDocument();
});

it("hides the chip toggle without the chip", () => {
  show();
  expect(screen.queryByRole("button", { name: "Wildcard" })).toBeNull();
  expect(screen.queryByRole("button", { name: "Free Hit" })).toBeNull();
});

it("plays a Free Hit: no hits and the squad returns next week", () => {
  show(snap((s) => { s.state.chips_available = ["freehit"]; }));
  expect(screen.queryByRole("button", { name: "Wildcard" })).toBeNull();
  fireEvent.click(screen.getByRole("button", { name: "Free Hit" }));
  expect(screen.getByRole("button", { name: "Free Hit" })).toHaveAttribute("aria-pressed", "true");
  expect(screen.getByText("Your squad returns next week")).toBeInTheDocument();
  swap("P1036", "P1030");
  swap("P1037", "P1031");
  expect(screen.getByText("2 transfers, free")).toBeInTheDocument();
});

it("shows why a stacked set is illegal and disables the check", () => {
  show(snap((s) => { s.state.chips_available = ["wildcard"]; }));
  fireEvent.click(screen.getByRole("button", { name: "Wildcard" }));
  swap("P1036", "P1030");
  swap("P1037", "P1031");
  swap("P1018", "P1019");
  swap("P1021", "P1022");
  expect(check()).toBeEnabled();
  fireEvent.click(screen.getByRole("button", { name: "Wildcard" }));
  expect(screen.getByText(/At most 2 hits a week/)).toBeInTheDocument();
  expect(check()).toBeDisabled();
});

it("asks for a coach run on a schema-1 snapshot", () => {
  show(snap((s) => { s.schema = 1; delete s.rules; }));
  expect(screen.getByText("Run the coach to enable swaps.")).toBeInTheDocument();
  expect(screen.queryByRole("button", { name: /^P1036,/ })).toBeNull();
});

it("asks for a coach run when a squad player has no selling price", () => {
  show(snap((s) => { s.players.find((p) => p.id === 1036)!.selling_price = null; }));
  expect(screen.getByText("Run the coach to enable swaps.")).toBeInTheDocument();
});

it("moves across all three tabs with the keyboard", () => {
  render(<TeamScreen data={snap()} now={NOW} onSettings={() => {}} onReload={() => {}} />);
  const tab = (name: string) => screen.getByRole("tab", { name });
  fireEvent.keyDown(tab("Your team"), { key: "End" });
  expect(tab("Try a swap")).toHaveAttribute("aria-selected", "true");
  expect(tab("Try a swap")).toHaveFocus();
  fireEvent.keyDown(tab("Try a swap"), { key: "ArrowRight" });
  expect(tab("Your team")).toHaveFocus();
  fireEvent.keyDown(tab("Your team"), { key: "ArrowLeft" });
  expect(tab("Try a swap")).toHaveFocus();
  fireEvent.keyDown(tab("Try a swap"), { key: "ArrowLeft" });
  expect(tab("Coach's plan")).toHaveAttribute("aria-selected", "true");
});

it("ticks the headline to the new value, instantly under reduced motion", () => {
  const frames: FrameRequestCallback[] = [];
  const run = (t: number) => { for (const f of frames.splice(0)) act(() => f(t)); };
  const motion = (reduce: boolean) => vi.stubGlobal("matchMedia", () => ({ matches: reduce, addEventListener() {}, removeEventListener() {} }));
  vi.stubGlobal("requestAnimationFrame", (f: FrameRequestCallback) => frames.push(f));
  vi.stubGlobal("cancelAnimationFrame", () => {});
  const plan = () => headline().querySelector(".plan")!;
  motion(false);
  show();
  swap("P1036", "P1030");
  expect(plan()).toHaveTextContent("29.0");
  run(performance.now() + 150);
  expect(plan().textContent).not.toMatch(/^(29\.0|31\.0)$/);
  run(performance.now() + 1000);
  expect(plan()).toHaveTextContent("31.0");
  motion(true);
  fireEvent.click(screen.getByRole("button", { name: "Reset" }));
  expect(plan()).toHaveTextContent("29.0");
  vi.unstubAllGlobals();
});
