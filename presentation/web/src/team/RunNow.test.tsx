import { act, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import type { Run } from "../types";
import { RunNow } from "./RunNow";

const m = vi.hoisted(() => ({ startRun: vi.fn(), getRun: vi.fn(), getLatestRun: vi.fn() }));
vi.mock("../api", () => ({
  startRun: m.startRun,
  getRun: m.getRun,
  getLatestRun: m.getLatestRun,
  ApiError: class extends Error {
    constructor(public status: number, public detail: string, public reason?: string) { super(detail); }
  },
}));

const EXPECTED = ["data", "predict", "plan"];
const run = (over: Partial<Run> = {}): Run => ({
  id: 7, entry: 42, gameweek: 9, kind: "manual", status: "running", steps: [], error: null, cost_usd: null,
  data_pulled_at: null, created_at: "2026-11-24T10:00:00Z", started_at: null, finished_at: null,
  steps_expected: EXPECTED, data_age_hours: 5.2, ...over,
});
const setup = (onDone = vi.fn(), onSettings = vi.fn()) => {
  render(<RunNow entry={42} onDone={onDone} onSettings={onSettings} />);
  return { onDone, onSettings };
};
const tick = (ms = 3000) => act(() => vi.advanceTimersByTimeAsync(ms));
const press = (name = "Run the coach now") => act(async () => { fireEvent.click(screen.getByRole("button", { name })); });

beforeEach(() => {
  vi.useFakeTimers();
  vi.resetAllMocks();
  m.getLatestRun.mockResolvedValue(null);
});
afterEach(() => vi.useRealTimers());

it("progresses through the steps, then stops polling and calls onDone", async () => {
  m.startRun.mockResolvedValue(run({ status: "queued", data_age_hours: 5.2 }));
  m.getRun
    .mockResolvedValueOnce(run({ steps: [{ name: "data", secs: 1.234 }] }))
    .mockResolvedValueOnce(run({ status: "done", steps: EXPECTED.map((name) => ({ name, secs: 2 })) }));
  const { onDone } = setup();
  await press();
  expect(screen.getByText("Waiting for the coach to start...")).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Run the coach now" })).toBeDisabled();
  expect(screen.getByText("Using data pulled 5 h ago.")).toBeInTheDocument();

  await tick();
  expect(screen.getByText("data").closest("li")).toHaveTextContent("1.2 s");
  expect(screen.getByText("predict").closest("li")).toHaveTextContent("in progress");
  expect(screen.getByText("plan").closest("li")).toHaveTextContent("pending");
  expect(onDone).not.toHaveBeenCalled();

  await tick();
  expect(onDone).toHaveBeenCalledOnce();
  await tick(9000);
  expect(m.getRun).toHaveBeenCalledTimes(2);
  expect(screen.getByRole("button", { name: "Run the coach now" })).toBeEnabled();
});

it("resumes an active run on mount", async () => {
  m.getLatestRun.mockResolvedValue(run({ steps: [{ name: "data", secs: 1 }] }));
  m.getRun.mockResolvedValue(run({ steps: [{ name: "data", secs: 1 }] }));
  setup();
  await tick(0);
  expect(screen.getByText("predict").closest("li")).toHaveTextContent("in progress");
  expect(screen.getByRole("button", { name: "Run the coach now" })).toBeDisabled();
  await tick();
  expect(m.getRun).toHaveBeenCalledWith(7);
});

it("keeps polling after a network error", async () => {
  m.getLatestRun.mockResolvedValue(run());
  m.getRun.mockRejectedValueOnce(new Error("offline")).mockResolvedValue(run({ steps: [{ name: "data", secs: 1 }] }));
  setup();
  await tick(0);
  await tick();
  await tick();
  expect(screen.getByText("data").closest("li")).toHaveTextContent("1.0 s");
});

it("shows the cap message with a Settings link on 409", async () => {
  const { ApiError } = await import("../api");
  m.startRun.mockRejectedValue(new ApiError(409, "Monthly cap reached.", "cap"));
  const { onSettings } = setup();
  await press();
  expect(screen.getByText("Monthly cap reached.")).toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Settings" }));
  expect(onSettings).toHaveBeenCalled();
});

it("shows a blocked run's error with a Settings link and no Try again", async () => {
  m.startRun.mockResolvedValue(run({ status: "queued" }));
  m.getRun.mockResolvedValue(run({ status: "blocked", error: "AI is off." }));
  const { onSettings } = setup();
  await press();
  await tick();
  expect(screen.getByText("AI is off.")).toBeInTheDocument();
  expect(screen.queryByRole("button", { name: "Try again" })).toBeNull();
  fireEvent.click(screen.getByRole("button", { name: "Settings" }));
  expect(onSettings).toHaveBeenCalled();
});

it("shows a season block without a Settings link", async () => {
  const { ApiError } = await import("../api");
  m.startRun.mockRejectedValue(new ApiError(409, "The season is over.", "season"));
  setup();
  await press();
  expect(screen.getByText("The season is over.")).toBeInTheDocument();
  expect(screen.queryByRole("button", { name: "Settings" })).toBeNull();
});

it("shows the no-data message with Try again", async () => {
  const { ApiError } = await import("../api");
  m.startRun.mockRejectedValue(new ApiError(503, "No data yet: run the refresh on your PC."));
  setup();
  await press();
  expect(screen.getByText("No data yet: run the refresh on your PC.")).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Try again" })).toBeEnabled();
});

it("shows a failed run's error with Try again", async () => {
  m.startRun.mockResolvedValue(run({ status: "queued" }));
  m.getRun.mockResolvedValue(run({ status: "failed", error: "Solver timed out." }));
  setup();
  await press();
  await tick();
  expect(screen.getByText("Solver timed out.")).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Try again" })).toBeEnabled();
});

it("tells the owner to switch on the PC when the data is over 30 h old", async () => {
  m.getLatestRun.mockResolvedValue(run({ data_age_hours: 31.4 }));
  m.getRun.mockResolvedValue(run({ data_age_hours: 31.4 }));
  setup();
  await tick(0);
  expect(screen.getByText("Using data pulled 31 h ago.")).toBeInTheDocument();
  expect(screen.getByText("Switch on your PC to refresh it.")).toBeInTheDocument();
});

it("omits the data age line when unknown", async () => {
  m.startRun.mockResolvedValue(run({ data_age_hours: null }));
  setup();
  await press();
  expect(screen.queryByText(/Using data pulled/)).toBeNull();
});

it("shows a run the api reports as stopped, with Try again", async () => {
  m.getLatestRun.mockResolvedValue(run());
  m.getRun.mockResolvedValue(run({ status: "failed", error: "The coach stopped without finishing. Try again." }));
  setup();
  await tick(0);
  await tick();
  expect(screen.getByText("The coach stopped without finishing. Try again.")).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Try again" })).toBeEnabled();
  await tick(9000);
  expect(m.getRun).toHaveBeenCalledTimes(1);
});
