import { expect, test, type Page } from "@playwright/test";
import { signedIn } from "./mock";

const ok = (page: Page) => signedIn(page);
const noScroll = (page: Page) => page.evaluate(() => document.documentElement.scrollWidth <= innerWidth);

test("shows the team without horizontal scroll", async ({ page }) => {
  await ok(page);
  await expect(page.getByRole("heading", { name: "Test FC" })).toBeVisible();
  await expect(page.getByRole("group", { name: "Predicted points" })).toContainText("29.0");
  await expect(page.locator(".slot")).toHaveCount(15);
  expect(await noScroll(page)).toBe(true);
});

test("has no horizontal scroll at 320 px", async ({ page }) => {
  await page.setViewportSize({ width: 320, height: 640 });
  await ok(page);
  await expect(page.locator(".slot")).toHaveCount(15);
  expect(await noScroll(page)).toBe(true);
});

test("shows the error state with its fix", async ({ page }) => {
  await signedIn(page, { status: 503, body: { detail: "No coach data yet." } });
  await expect(page.getByRole("alert")).toContainText("No coach data yet.");
});

test("switches to the plan with the keyboard", async ({ page }) => {
  await ok(page);
  await page.getByRole("tab", { name: "Your team" }).focus();
  await page.keyboard.press("ArrowRight");
  await expect(page.getByRole("tab", { name: "Coach's plan" })).toHaveAttribute("aria-selected", "true");
  await expect(page.getByText("No transfers")).toBeVisible();
});
