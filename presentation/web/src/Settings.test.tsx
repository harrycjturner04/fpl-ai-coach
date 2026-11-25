import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, expect, it, vi } from "vitest";
import { Settings } from "./Settings";

const m = vi.hoisted(() => ({ getSettings: vi.fn(), patchSettings: vi.fn(), signOut: vi.fn() }));
vi.mock("./api", () => ({ getSettings: m.getSettings, patchSettings: m.patchSettings, ApiError: class extends Error {} }));
vi.mock("./auth", () => ({ signOut: m.signOut }));

const base = { cap_gbp: 5, auto_runs: true, ai_off: false, spent_gbp: 1.234, month: "November 2026" };
beforeEach(() => {
  vi.resetAllMocks();
  m.getSettings.mockResolvedValue(base);
});

it("shows the spend and the backstops", async () => {
  render(<Settings onBack={() => {}} />);
  expect(await screen.findByText("£1.23 spent in November 2026")).toBeInTheDocument();
  expect(screen.getByText(/Google Cloud £1 alert/)).toBeInTheDocument();
});

it("turns AI off", async () => {
  m.patchSettings.mockResolvedValue({ ...base, ai_off: true });
  render(<Settings onBack={() => {}} />);
  fireEvent.click(await screen.findByRole("switch", { name: "AI off" }));
  await waitFor(() => expect(m.patchSettings).toHaveBeenCalledWith({ ai_off: true }));
  await waitFor(() => expect(screen.getByRole("switch", { name: "AI off" })).toHaveAttribute("aria-checked", "true"));
});

it("sends a valid cap and refuses one out of range", async () => {
  m.patchSettings.mockResolvedValue({ ...base, cap_gbp: 7.5 });
  render(<Settings onBack={() => {}} />);
  const cap = await screen.findByLabelText("Monthly cap (£)");
  fireEvent.change(cap, { target: { value: "60" } });
  fireEvent.blur(cap);
  expect(m.patchSettings).not.toHaveBeenCalled();
  expect(screen.getByRole("alert")).toHaveTextContent("0 to 50");
  fireEvent.change(cap, { target: { value: "7.5" } });
  fireEvent.blur(cap);
  await waitFor(() => expect(m.patchSettings).toHaveBeenCalledWith({ cap_gbp: 7.5 }));
});

it("signs out and goes back", async () => {
  const onBack = vi.fn();
  render(<Settings onBack={onBack} />);
  fireEvent.click(await screen.findByRole("button", { name: "Sign out" }));
  expect(m.signOut).toHaveBeenCalled();
  fireEvent.click(screen.getByRole("button", { name: "Back" }));
  expect(onBack).toHaveBeenCalled();
});
