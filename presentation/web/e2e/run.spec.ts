import { expect, test } from "@playwright/test";
import { signedIn } from "./mock";

const json = (body: unknown, status = 200) => ({ status, contentType: "application/json", body: JSON.stringify(body) });
const EXPECTED = ["data", "predict", "plan"];
const run = (status: string, steps: string[]) => ({
  id: 7, entry: 42, gameweek: 9, kind: "manual", status, steps: steps.map((name) => ({ name, secs: 1.5 })), error: null, cost_usd: null,
  data_pulled_at: null, created_at: "2026-11-24T10:00:00Z", started_at: null, finished_at: null, steps_expected: EXPECTED, data_age_hours: 5,
});

test("runs the coach, ticks the steps, reloads the team, then shows the cap block", async ({ page }) => {
  await page.setViewportSize({ width: 320, height: 640 });
  await signedIn(page);
  await expect(page.getByRole("heading", { name: "Test FC" })).toBeVisible();
  let posts = 0, polls = 0, snapshots = 0;
  await page.route("**/api/entries/42/snapshot", (route) => { snapshots++; return route.fallback(); });
  await page.route("**/api/runs/7", (route) => route.fulfill(json(++polls === 1 ? run("running", ["data"]) : run("done", EXPECTED))));
  await page.route("**/api/entries/42/runs", (route) =>
    route.fulfill(++posts === 1 ? json(run("queued", []), 202) : json({ detail: "Monthly cap reached.", reason: "cap" }, 409)));

  await page.getByRole("button", { name: "Run the coach now" }).click();
  await expect(page.getByText("Waiting for the coach to start...")).toBeVisible();
  await expect(page.locator(".steps li.done")).toHaveCount(1, { timeout: 10_000 });
  await expect(page.getByText("Using data pulled 5 h ago.")).toBeVisible();
  await expect.poll(() => snapshots, { timeout: 10_000 }).toBe(1);
  await expect(page.getByRole("button", { name: "Run the coach now" })).toBeEnabled();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);

  await page.getByRole("button", { name: "Run the coach now" }).click();
  await expect(page.getByText("Monthly cap reached.")).toBeVisible();
  await expect(page.getByRole("alert").getByRole("button", { name: "Settings" })).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
});

test("offers Run the coach when there is no run yet, then shows the team", async ({ page }) => {
  await signedIn(page);
  await expect(page.getByRole("heading", { name: "Test FC" })).toBeVisible();
  let snapshots = 0;
  await page.route("**/api/entries/42/snapshot", (route) =>
    ++snapshots === 1 ? route.fulfill(json({ detail: "No coach run yet." }, 404)) : route.fallback());
  await page.route("**/api/entries/42/runs/latest", (route) => route.fulfill(json(null)));
  await page.route("**/api/entries/42/runs", (route) => route.fulfill(json(run("queued", []), 202)));
  await page.route("**/api/runs/7", (route) => route.fulfill(json(run("done", EXPECTED))));
  await page.reload();  // routes added after sign-in take precedence over the mock's catch-all
  await expect(page.getByRole("heading", { name: "No coach run yet." })).toBeVisible();
  await expect(page.getByRole("button", { name: "Sign out" })).toBeVisible();
  await page.getByRole("button", { name: "Run the coach now" }).click();
  await expect(page.getByRole("heading", { name: "Test FC" })).toBeVisible({ timeout: 10_000 });
});
