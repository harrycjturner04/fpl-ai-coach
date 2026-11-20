import { render, screen, within } from "@testing-library/react";
import { expect, it } from "vitest";
import fixture from "../test/snapshot.json";
import type { Snapshot } from "../types";
import { Pitch } from "./Pitch";

const snap = () => structuredClone(fixture) as Snapshot;
const captainLabel = () =>
  within(screen.getByRole("group", { name: "Starting eleven" }))
    .getAllByRole("img")
    .map((el) => el.getAttribute("aria-label")!)
    .filter((l) => l.endsWith(", captain"));

it("renders 11 starters and 4 on the bench", () => {
  const s = snap();
  render(<Pitch snap={s} week={s.current!} />);
  expect(within(screen.getByRole("group", { name: "Starting eleven" })).getAllByRole("img")).toHaveLength(11);
  expect(within(screen.getByRole("group", { name: "Bench" })).getAllByRole("img")).toHaveLength(4);
});

it("labels the captain disc", () => {
  const s = snap();
  render(<Pitch snap={s} week={s.current!} />);
  expect(captainLabel()).toEqual(["P2, 7.0 predicted points, 90% chance of 60+ minutes, captain"]);
});

it("moves the armband when the week changes", () => {
  const s = snap();
  s.plan.weeks[0].captain = 3;
  s.plan.weeks[0].vice_captain = 2;
  const { rerender } = render(<Pitch snap={s} week={s.current!} />);
  expect(captainLabel()[0]).toMatch(/^P2,/);
  rerender(<Pitch snap={s} week={s.plan.weeks[0]} />);
  expect(captainLabel()).toHaveLength(1);
  expect(captainLabel()[0]).toMatch(/^P3,/);
});

it("shows the doubtful player's status", () => {
  const s = snap();
  render(<Pitch snap={s} week={s.current!} />);
  expect(screen.getByText("! 75%")).toBeInTheDocument();
});

it("shows the key line", () => {
  const s = snap();
  render(<Pitch snap={s} week={s.current!} />);
  expect(screen.getByText("Ring = chance of playing 60+ minutes. Number = predicted points.")).toBeInTheDocument();
});
