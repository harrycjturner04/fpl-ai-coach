import { expect, test } from "@playwright/test";
import { signedIn } from "./mock";

test("signed out shows the sign-in screen", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("button", { name: "Sign in" })).toBeVisible();
});

test("sign in shows the team, sign out clears the snapshot cache", async ({ page }) => {
  await signedIn(page);
  await expect(page.getByRole("heading", { name: "Test FC" })).toBeVisible();
  await page.evaluate(() => caches.open("snapshots").then((c) => c.put("/x", new Response("x"))));
  await page.getByRole("button", { name: "Settings" }).click();
  await expect(page.getByRole("switch", { name: "AI off" })).toBeVisible();
  await page.getByRole("button", { name: "Sign out" }).click();
  await expect(page.getByRole("button", { name: "Sign in" })).toBeVisible();
  expect(await page.evaluate(() => caches.has("snapshots"))).toBe(false);
});

test("sign-in and settings have no horizontal scroll at 320 px", async ({ page }) => {
  await page.setViewportSize({ width: 320, height: 640 });
  await page.goto("/");
  await expect(page.getByRole("button", { name: "Sign in" })).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
});
