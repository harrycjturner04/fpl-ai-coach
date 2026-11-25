import { fireEvent, render, screen } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import type { SnapshotResponse } from "./types";
import App from "./App";
import fixture from "./test/snapshot.json";

vi.mock("./auth", () => ({
  accessToken: () => Promise.resolve("tok"),
  onAuthChange: (cb: (s: boolean) => void) => { cb(true); return () => {}; },
  signOut: vi.fn(),
  configured: true,
}));

afterEach(() => vi.unstubAllGlobals());

const json = (status: number, body: unknown) => new Response(JSON.stringify(body), { status });

it("shows the loading skeleton, then the team name", async () => {
  vi.stubGlobal("fetch", vi.fn((url: string) => Promise.resolve(url === "/api/me" ? json(200, { entry: 42 }) : json(200, fixture))));
  render(<App />);
  expect(screen.getByRole("status")).toBeInTheDocument();
  expect(await screen.findByRole("heading", { name: "Test FC" })).toBeInTheDocument();
});

it("offers Run the coach on a 404 and loads the team when the run is done", async () => {
  let snapshots = 0;
  const run = { id: 7, entry: 42, gameweek: 9, kind: "manual", status: "done", steps: [], error: null, cost_usd: null,
    data_pulled_at: null, created_at: "2026-11-24T10:00:00Z", started_at: null, finished_at: null, steps_expected: [], data_age_hours: null };
  const fetch = vi.fn((url: string, init?: RequestInit) => Promise.resolve(
    url === "/api/me" ? json(200, { entry: 42 })
      : url === "/api/entries/42/snapshot" ? (++snapshots === 1 ? json(404, { detail: "No coach run yet." }) : json(200, fixture))
      : url === "/api/entries/42/runs/latest" ? json(200, null)
      : url === "/api/entries/42/runs" && init?.method === "POST" ? json(202, { ...run, status: "queued" })
      : json(200, run)));
  vi.stubGlobal("fetch", fetch);
  render(<App />);
  expect(await screen.findByRole("heading", { name: "No coach run yet." })).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Settings" })).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Sign out" })).toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Run the coach now" }));
  expect(await screen.findByRole("heading", { name: "Test FC" }, { timeout: 5000 })).toBeInTheDocument();
});

it("shows an error, not a blank page, when the team screen throws", async () => {
  const broken = { ...(fixture as SnapshotResponse), players: [] };  // the pitch cannot find its players
  vi.stubGlobal("fetch", vi.fn((url: string) => Promise.resolve(url === "/api/me" ? json(200, { entry: 42 }) : json(200, broken))));
  vi.spyOn(console, "error").mockImplementation(() => {});
  render(<App />);
  const alert = await screen.findByRole("alert");
  expect(alert).toHaveTextContent("Something went wrong showing your team.");
  expect(alert).toHaveTextContent("Reload the page");
});
