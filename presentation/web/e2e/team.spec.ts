import { expect, test, type Page } from "@playwright/test";
import fixture from "../src/test/snapshot.json" with { type: "json" };

const serve = (page: Page, me: { status: number; body: unknown }) =>
  page.route("**/api/**", (route) => {
    const me_ = new URL(route.request().url()).pathname === "/api/me";
    return route.fulfill({ status: me_ ? me.status : 200, contentType: "application/json", body: JSON.stringify(me_ ? me.body : fixture) });
  });
const ok = (page: Page) => serve(page, { status: 200, body: { entry: 42 } });
const noScroll = (page: Page) => page.evaluate(() => document.documentElement.scrollWidth <= innerWidth);

test("shows the team without horizontal scroll", async ({ page }) => {
  await ok(page);
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "Test FC" })).toBeVisible();
  await expect(page.getByRole("group", { name: "Predicted points" })).toContainText("29.0");
  await expect(page.locator(".slot")).toHaveCount(15);
  expect(await noScroll(page)).toBe(true);
});

test("has no horizontal scroll at 320 px", async ({ page }) => {
  await page.setViewportSize({ width: 320, height: 640 });
  await ok(page);
  await page.goto("/");
  await expect(page.locator(".slot")).toHaveCount(15);
  expect(await noScroll(page)).toBe(true);
});

test("shows the error state with its fix", async ({ page }) => {
  await serve(page, { status: 503, body: { detail: "Set FPL_COACH_ENTRY to your FPL team ID in .env and restart the API." } });
  await page.goto("/");
  await expect(page.getByRole("alert")).toContainText("Set FPL_COACH_ENTRY");
});

test("switches to the plan with the keyboard", async ({ page }) => {
  await ok(page);
  await page.goto("/");
  await page.getByRole("tab", { name: "Your team" }).focus();
  await page.keyboard.press("ArrowRight");
  await expect(page.getByRole("tab", { name: "Coach's plan" })).toHaveAttribute("aria-selected", "true");
  await expect(page.getByText("No transfers")).toBeVisible();
});
