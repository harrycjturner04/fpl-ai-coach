import { render, screen } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import type { SnapshotResponse } from "./types";
import App from "./App";
import fixture from "./test/snapshot.json";

afterEach(() => vi.unstubAllGlobals());

const json = (status: number, body: unknown) => new Response(JSON.stringify(body), { status });

it("shows the loading skeleton, then the team name", async () => {
  vi.stubGlobal("fetch", vi.fn((url: string) => Promise.resolve(url === "/api/me" ? json(200, { entry: 42 }) : json(200, fixture))));
  render(<App />);
  expect(screen.getByRole("status")).toBeInTheDocument();
  expect(await screen.findByRole("heading", { name: "Test FC" })).toBeInTheDocument();
});

it("shows the run-the-coach hint on a 404", async () => {
  vi.stubGlobal("fetch", vi.fn((url: string) =>
    Promise.resolve(url === "/api/me" ? json(200, { entry: 42 }) : json(404, { detail: "No coach run yet. Run: python -m orchestration.coach --entry 42" }))));
  render(<App />);
  expect(await screen.findByRole("alert")).toHaveTextContent("python -m orchestration.coach");
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
