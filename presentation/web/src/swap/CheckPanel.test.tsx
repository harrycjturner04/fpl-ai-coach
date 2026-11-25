import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import fixture from "../test/snapshot.json";
import type { PlanCheck, Snapshot2, SnapshotResponse, UserPlan } from "../types";
import { bestXI } from "./swap";
import { TeamScreen } from "../team/TeamScreen";
import { CheckPanel, type CheckState } from "./CheckPanel";

const m = vi.hoisted(() => ({ postPlanCheck: vi.fn(), adoptPlan: vi.fn(), getUserPlan: vi.fn() }));
vi.mock("../api", () => ({
  getLatestRun: () => Promise.resolve(null), getRun: vi.fn(), startRun: vi.fn(),
  postPlanCheck: m.postPlanCheck, adoptPlan: m.adoptPlan, getUserPlan: m.getUserPlan,
  ApiError: class extends Error { constructor(public status: number, public detail: string, public reason?: string) { super(detail); } },
}));
const { ApiError } = await import("../api");

const NOW = new Date("2026-11-12T21:00:00Z");
const snap = (edit: (s: SnapshotResponse) => void = () => {}) => {
  const s = structuredClone(fixture) as SnapshotResponse;
  s.players.find((p) => p.id === 1030)!.points = [3, 0];
  edit(s);
  return s;
};
const week = () => {
  const w = structuredClone(fixture.plan.weeks[0]) as PlanCheck["weeks"][number];
  w.squad = w.squad.map((id) => (id === 1036 ? 1030 : id));
  w.starting = w.starting.map((id) => (id === 1036 ? 1030 : id));
  w.transfers_in = [1030];
  w.transfers_out = [1036];
  w.points = 31;
  return w;
};
const result = (edit: Partial<PlanCheck> = {}): PlanCheck => ({
  id: 7, weeks: [week()], objective: 66.24, points: 31, sold: [1036], bought: [1030], chip: null, chips: {},
  coach: { points: 29, objective: 64.7, exact: true }, seconds: 8.2, ...edit,
});
const userPlan = (edit: Partial<UserPlan> = {}): UserPlan => {
  const { id: _, ...r } = result();
  return {
    id: 7, entry: 42, gameweek: 9, snapshot: "42_gw9_20261110-1200.json", deadline: "2026-11-14T11:00:00+00:00",
    swaps: [[1036, 1030]], chip: null, result: r, checked_at: "2026-11-10T12:05:00Z", adopted_at: "2026-11-10T12:06:00Z",
    replaced_at: null, stale: false, ...edit,
  };
};
const panel = (state: CheckState, s = snap(), props: Partial<Parameters<typeof CheckPanel>[0]> = {}) =>
  render(<CheckPanel snap={s} swaps={[[1036, 1030]]} chip={null} state={state} onRetry={() => {}} onKeep={() => {}} onAdopted={() => {}} {...props} />);
const err = (status: number, detail: string, reason?: string) => ({ kind: "error", error: new ApiError(status, detail, reason) }) as CheckState;

beforeEach(() => {
  localStorage.clear();
  m.getUserPlan.mockResolvedValue(null);
});
afterEach(() => { vi.useRealTimers(); vi.clearAllMocks(); });

it("shows determinate progress over the default 30 s, then says it is nearly there", () => {
  vi.useFakeTimers();
  panel({ kind: "checking", started: Date.now(), estimate: 30 });
  expect(screen.getByRole("status")).toHaveTextContent("Checking against the coach's plan, about 30 s");
  const bar = screen.getByRole("progressbar");
  expect(bar).toHaveAttribute("max", "30");
  act(() => { vi.advanceTimersByTime(15000); });
  expect(Number(bar.getAttribute("value"))).toBeCloseTo(15, 0);
  act(() => { vi.advanceTimersByTime(16000); });
  expect(screen.getByRole("status")).toHaveTextContent("Still solving, nearly there");
});

it("shows the verdict with the planning value never called points", () => {
  panel({ kind: "done", result: result() });
  expect(screen.getByText("This week: 31.0 vs coach 29.0")).toBeInTheDocument();
  expect(screen.getByText("Optimiser's planning value: 66.2 vs coach 64.7")).toBeInTheDocument();
  expect(screen.getByText(/^Quick estimate: [+−]\d+\.\d over the next 2 GWs \(no further transfers\)$/)).toBeInTheDocument();
  expect(screen.getByText(/score for 2 weeks with later transfers planned/)).toBeInTheDocument();
  expect(screen.queryByText(/not exact/)).toBeNull();
  expect(screen.queryByText(/planning value.*points/i)).toBeNull();
  expect(screen.getByRole("button", { name: "Adopt this plan" })).toBeEnabled();
  expect(screen.getByRole("button", { name: "Keep trying" })).toBeEnabled();
});

it("labels the horizon from a 3-GW snapshot", () => {
  panel({ kind: "done", result: result() }, snap((s) => { s.gameweeks = [...s.gameweeks, 11]; }));
  expect(screen.getByText(/^Quick estimate: [+−]\d+\.\d over the next 3 GWs \(no further transfers\)$/)).toBeInTheDocument();
  expect(screen.getByText(/score for 3 weeks with later transfers planned/)).toBeInTheDocument();
});

it("says when the coach's plan is not exact", () => {
  panel({ kind: "done", result: result({ coach: { points: 29, objective: 64.7, exact: false } }) });
  expect(screen.getByText(/\(coach's plan not exact\)/)).toBeInTheDocument();
});

it("asks for the coach's chip to compare like for like", () => {
  panel({ kind: "done", result: result() }, snap((s) => { s.plan.weeks[0].chip = "wildcard"; }));
  expect(screen.getByText("The coach plays a Wildcard this week; turn on Wildcard to compare like for like.")).toBeInTheDocument();
  panel({ kind: "done", result: result({ chip: "wildcard" }) }, snap((s) => { s.plan.weeks[0].chip = "wildcard"; }));
  expect(screen.getAllByText(/compare like for like/)).toHaveLength(1);
});

it.each([
  [err(409, "A check is already running.", "busy"), "A check is already running.", true],
  [err(422, "Swap like for like: P1 is a GKP, P2 a DEF."), "Swap like for like: P1 is a GKP, P2 a DEF.", false],
  [err(429, "Rate limited"), "Too many checks this hour. Try again later.", false],
  [err(0, "Can't reach the coach's server. Check your connection and try again."), "Can't reach the coach's server.", true],
])("shows a designed error state", (state, text, retry) => {
  const onRetry = vi.fn();
  panel(state, snap(), { onRetry });
  expect(screen.getByRole("alert")).toHaveTextContent(text);
  const button = screen.queryByRole("button", { name: "Try again" });
  expect(!!button).toBe(retry);
  if (button) { fireEvent.click(button); expect(onRetry).toHaveBeenCalled(); }
});

it("says a re-checked plan may no longer be legal", () => {
  panel(err(422, "Not affordable with your bank and selling prices."), snap(), { recheck: true });
  expect(screen.getByRole("alert")).toHaveTextContent("This plan may no longer be legal on the latest run");
  expect(screen.getByRole("alert")).toHaveTextContent("Not affordable with your bank and selling prices.");
});

it("adopts once, however often it is tapped", async () => {
  let done!: (p: UserPlan) => void;
  m.adoptPlan.mockReturnValue(new Promise((r) => { done = r; }));
  const onAdopted = vi.fn();
  panel({ kind: "done", result: result() }, snap(), { onAdopted });
  const adopt = screen.getByRole("button", { name: "Adopt this plan" });
  fireEvent.click(adopt);
  fireEvent.click(adopt);
  expect(adopt).toBeDisabled();
  expect(m.adoptPlan).toHaveBeenCalledTimes(1);
  expect(m.adoptPlan).toHaveBeenCalledWith(7);
  await act(async () => done(userPlan()));
  expect(onAdopted).toHaveBeenCalledWith(userPlan());
});

it("shows why an adopt failed", async () => {
  m.adoptPlan.mockRejectedValue(new ApiError(409, "The deadline has passed."));
  panel({ kind: "done", result: result() });
  fireEvent.click(screen.getByRole("button", { name: "Adopt this plan" }));
  expect(await screen.findByRole("alert")).toHaveTextContent("The deadline has passed.");
  expect(screen.getByRole("button", { name: "Adopt this plan" })).toBeEnabled();
});

// On the Team screen
const show = (s = snap()) => render(<TeamScreen data={s} now={NOW} onSettings={() => {}} onReload={() => {}} />);
const swap = () => {
  fireEvent.click(screen.getByRole("tab", { name: "Try a swap" }));
  fireEvent.click(screen.getByRole("button", { name: /^P1036,/ }));
  fireEvent.click(within(screen.getByRole("dialog")).getByRole("button", { name: /^P1030/ }));
};
const check = () => fireEvent.click(screen.getByRole("button", { name: "Check against the coach's plan" }));

it("checks the swaps on this run's snapshot, stores the time, adopts and shows Yours", async () => {
  m.postPlanCheck.mockResolvedValue(result());
  m.adoptPlan.mockResolvedValue(userPlan());
  show();
  swap();
  check();
  expect(m.postPlanCheck).toHaveBeenCalledWith(42, "42_gw9_20261110-1200.json", [[1036, 1030]], null);
  expect(await screen.findByText("This week: 31.0 vs coach 29.0")).toBeInTheDocument();
  expect(localStorage.getItem("planCheckSeconds")).toBe("8.2");
  fireEvent.click(screen.getByRole("button", { name: "Adopt this plan" }));
  await waitFor(() => expect(screen.getByRole("tab", { name: "Coach's plan" })).toHaveAttribute("aria-selected", "true"));
  expect(screen.getByRole("button", { name: "Yours" })).toHaveAttribute("aria-pressed", "true");
  expect(screen.getByText("Your plan, not the coach's")).toBeInTheDocument();
  expect(screen.getByText("P1036 → P1030")).toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Coach's" }));
  expect(screen.getByText("No transfers")).toBeInTheDocument();
});

it("uses the last measured time for the next estimate", async () => {
  localStorage.setItem("planCheckSeconds", "8.2");
  m.postPlanCheck.mockReturnValue(new Promise(() => {}));
  show();
  swap();
  check();
  expect(await screen.findByText("Checking against the coach's plan, about 8 s")).toBeInTheDocument();
});

it("clears the verdict when the swaps change", async () => {
  m.postPlanCheck.mockResolvedValue(result());
  show();
  swap();
  check();
  await screen.findByText("This week: 31.0 vs coach 29.0");
  fireEvent.click(screen.getByRole("button", { name: "Reset" }));
  expect(screen.queryByText(/vs coach/)).toBeNull();
});

it("clears the verdict when a new run arrives", async () => {
  m.postPlanCheck.mockResolvedValue(result());
  const { rerender } = show();
  swap();
  check();
  await screen.findByText("This week: 31.0 vs coach 29.0");
  rerender(<TeamScreen data={snap((s) => { s.generated_at = "2026-11-11T12:00:00+00:00"; })} now={NOW} onSettings={() => {}} onReload={() => {}} />);
  expect(screen.queryByText(/vs coach/)).toBeNull();
});

it("shows a stale plan re-scored beside the coach's new plan and re-checks it on the latest run", async () => {
  m.getUserPlan.mockResolvedValue(userPlan({ stale: true, snapshot: "42_gw9_20261109-1200.json" }));
  m.postPlanCheck.mockRejectedValue(new ApiError(422, "Not affordable with your bank and selling prices."));
  show();
  fireEvent.click(screen.getByRole("tab", { name: "Coach's plan" }));
  fireEvent.click(await screen.findByRole("button", { name: "Yours" }));
  expect(screen.getByText("Based on an older run")).toBeInTheDocument();
  expect(screen.getByText("Re-scored on the latest predictions: 31.0 this week vs the coach's new plan 29.0")).toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Re-check on latest run" }));
  expect(m.postPlanCheck).toHaveBeenCalledWith(42, "42_gw9_20261110-1200.json", [[1036, 1030]], null);
  expect(await screen.findByRole("alert")).toHaveTextContent("This plan may no longer be legal on the latest run");
});

it("has no switch without an adopted plan", async () => {
  show();
  fireEvent.click(screen.getByRole("tab", { name: "Coach's plan" }));
  await waitFor(() => expect(m.getUserPlan).toHaveBeenCalledWith(42));
  expect(screen.queryByRole("button", { name: "Yours" })).toBeNull();
});

it("re-scores a stale plan net of its hits, and hides the line for a gameweek this run does not cover", async () => {
  const p = userPlan({ stale: true });
  p.result.weeks[0].hits = 1;
  m.getUserPlan.mockResolvedValue(p);
  const { unmount } = show();
  fireEvent.click(screen.getByRole("tab", { name: "Coach's plan" }));
  fireEvent.click(await screen.findByRole("button", { name: "Yours" }));
  expect(screen.getByText("Re-scored on the latest predictions: 27.0 this week vs the coach's new plan 29.0")).toBeInTheDocument();
  unmount();
  const q = userPlan({ stale: true });
  q.result.weeks[0].gameweek = 99;
  m.getUserPlan.mockResolvedValue(q);
  show();
  fireEvent.click(screen.getByRole("tab", { name: "Coach's plan" }));
  fireEvent.click(await screen.findByRole("button", { name: "Yours" }));
  expect(screen.getByText("Based on an older run")).toBeInTheDocument();
  expect(screen.queryByText(/Re-scored/)).toBeNull();
});

it("re-scores a stale Bench Boost plan with its bench, as the plan's points", async () => {
  const p = userPlan({ stale: true });
  p.result.weeks[0].chip = "bboost";
  m.getUserPlan.mockResolvedValue(p);
  const s = snap();
  const xi = bestXI(s as Snapshot2, p.result.weeks[0].squad, 0);
  const bench = xi.bench.reduce((t, id) => t + s.players.find((q) => q.id === id)!.points[0], 0);
  expect(bench).toBeGreaterThan(0);
  show(s);
  fireEvent.click(screen.getByRole("tab", { name: "Coach's plan" }));
  fireEvent.click(await screen.findByRole("button", { name: "Yours" }));
  expect(screen.getByText(`Re-scored on the latest predictions: ${(31 + bench).toFixed(1)} this week vs the coach's new plan 29.0`)).toBeInTheDocument();
});

it("hides the re-scored line instead of failing when the stored squad makes no XI", async () => {
  const p = userPlan({ stale: true });
  const gkp = new Set(fixture.players.filter((q) => q.position === "GKP").map((q) => q.id));
  p.result.weeks[0].squad = p.result.weeks[0].squad.filter((id) => !gkp.has(id));
  m.getUserPlan.mockResolvedValue(p);
  show();
  fireEvent.click(screen.getByRole("tab", { name: "Coach's plan" }));
  fireEvent.click(await screen.findByRole("button", { name: "Yours" }));
  expect(screen.getByText("Based on an older run")).toBeInTheDocument();
  expect(screen.queryByText(/Re-scored/)).toBeNull();
});

it("disables Check and Re-check while that check runs", async () => {
  m.postPlanCheck.mockReturnValue(new Promise(() => {}));
  m.getUserPlan.mockResolvedValue(userPlan({ stale: true }));
  show();
  swap();
  check();
  expect(screen.getByRole("button", { name: "Check against the coach's plan" })).toBeDisabled();
  expect(m.postPlanCheck).toHaveBeenCalledTimes(1);
  fireEvent.click(screen.getByRole("tab", { name: "Coach's plan" }));
  fireEvent.click(await screen.findByRole("button", { name: "Yours" }));
  const recheck = screen.getByRole("button", { name: "Re-check on latest run" });
  expect(recheck).toBeEnabled();
  fireEvent.click(recheck);
  expect(recheck).toBeDisabled();
  expect(m.postPlanCheck).toHaveBeenCalledTimes(2);
});

it("drops the plan on a new run until it reloads", async () => {
  m.getUserPlan.mockResolvedValueOnce(userPlan()).mockReturnValueOnce(new Promise(() => {}));
  const { rerender } = show();
  fireEvent.click(screen.getByRole("tab", { name: "Coach's plan" }));
  await screen.findByRole("button", { name: "Yours" });
  rerender(<TeamScreen data={snap((s) => { s.generated_at = "2026-11-11T12:00:00+00:00"; })} now={NOW} onSettings={() => {}} onReload={() => {}} />);
  expect(screen.queryByRole("button", { name: "Yours" })).toBeNull();
  expect(m.getUserPlan).toHaveBeenCalledTimes(2);
});

it("says when the plan could not load and retries", async () => {
  m.getUserPlan.mockRejectedValueOnce(new ApiError(0, "down")).mockResolvedValueOnce(userPlan());
  show();
  fireEvent.click(screen.getByRole("tab", { name: "Coach's plan" }));
  expect(await screen.findByRole("alert")).toHaveTextContent("Couldn't load your plan. Try again shortly.");
  fireEvent.click(screen.getByRole("button", { name: "Try again" }));
  expect(await screen.findByRole("button", { name: "Yours" })).toBeInTheDocument();
  expect(screen.queryByRole("alert")).toBeNull();
});
