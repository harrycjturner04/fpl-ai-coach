import { expect, test } from "@playwright/test";
import fixture from "../src/test/snapshot.json" with { type: "json" };
import { signedIn } from "./mock";

/** P1030 outscores P1036 this week, so it is the best swap and the headline moves. */
const snapshot = { ...fixture, players: fixture.players.map((p) => (p.id === 1030 ? { ...p, points: [3, 0] } : p)) };
const ME = { status: 200, body: { entry: 42, entries: [42] } };
const week = { ...fixture.plan.weeks[0], squad: fixture.plan.weeks[0].squad.map((id) => (id === 1036 ? 1030 : id)),
  starting: fixture.plan.weeks[0].starting.map((id) => (id === 1036 ? 1030 : id)), transfers_in: [1030], transfers_out: [1036], points: 29.4 };
const result = { weeks: [week], objective: 65.12, points: 29.4, sold: [1036], bought: [1030], chip: null, chips: {},
  coach: { points: 29, objective: 64.7, exact: true }, seconds: 8.2 };
const plan = (stale: boolean) => ({
  id: 7, entry: 42, gameweek: 9, snapshot: stale ? "42_gw9_20261109-1200.json" : "42_gw9_20261110-1200.json",
  deadline: "2026-11-14T11:00:00+00:00", swaps: [[1036, 1030]], chip: null, result, checked_at: "2026-11-10T12:05:00Z",
  adopted_at: "2026-11-10T12:06:00Z", replaced_at: null, stale,
});

test("checks a swap against the coach's plan and adopts it", async ({ page }) => {
  await signedIn(page, ME, { "GET /api/entries/42/snapshot": { body: snapshot }, "POST /api/plan-checks/7/adopt": { body: plan(false) } });
  let sent: unknown;
  await page.route("**/api/entries/42/plan-checks", async (route) => {
    sent = route.request().postDataJSON();
    await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ id: 7, ...result }) });
  });
  await page.getByRole("tab", { name: "Try a swap" }).click();
  const headline = page.getByRole("group", { name: "Predicted points" });
  await expect(headline).toContainText("for the team shown");
  await expect(headline.locator(".points")).toHaveText("29.0");
  await page.getByRole("button", { name: /^P1036,/ }).click();
  await expect(page.getByRole("dialog").locator(".opt").first()).toContainText("P1030");
  await page.getByRole("dialog").locator(".opt").first().click();
  await expect(page.getByText("1 transfer, free")).toBeVisible();
  await expect(headline.locator(".points")).toHaveText("31.0");
  await expect(headline).toContainText("+2.0 vs your team");
  await page.getByRole("button", { name: "Check against the coach's plan" }).click();
  await expect(page.getByText("This week: 29.4 vs coach 29.0")).toBeVisible();
  await expect(page.getByText("Optimiser's planning value: 65.1 vs coach 64.7")).toBeVisible();
  expect(sent).toEqual({ snapshot: "42_gw9_20261110-1200.json", swaps: [[1036, 1030]], chip: null });
  await page.getByRole("button", { name: "Adopt this plan" }).click();
  await expect(page.getByRole("tab", { name: "Coach's plan" })).toHaveAttribute("aria-selected", "true");
  await expect(page.getByRole("button", { name: "Yours" })).toHaveAttribute("aria-pressed", "true");
  await expect(page.getByText("Your plan, not the coach's")).toBeVisible();
  await expect(page.getByText("P1036 → P1030")).toBeVisible();
});

test("shows a stale plan and re-checks it on the latest run", async ({ page }) => {
  const stale = plan(true);
  stale.result = { ...result, weeks: [{ ...week, hits: 1 }] };
  await signedIn(page, ME, {
    "GET /api/entries/42/snapshot": { body: snapshot },
    "GET /api/entries/42/user-plan": { body: stale },
    "POST /api/entries/42/plan-checks": { body: { id: 8, ...result } },
  });
  await page.getByRole("tab", { name: "Coach's plan" }).click();
  await page.getByRole("button", { name: "Yours" }).click();
  await expect(page.getByText("Based on an older run")).toBeVisible();
  await expect(page.getByText("Your plan re-scored on the latest predictions (shown above). The coach's new plan: 29.0 this week")).toBeVisible();
  await expect(page.getByRole("group", { name: "Predicted points" }).locator(".points")).toHaveText("27.0");
  await page.getByRole("button", { name: "Re-check on latest run" }).click();
  await expect(page.getByText("This week: 29.4 vs coach 29.0")).toBeVisible();
  await expect(page.getByRole("button", { name: "Adopt this plan" })).toBeEnabled();
});
