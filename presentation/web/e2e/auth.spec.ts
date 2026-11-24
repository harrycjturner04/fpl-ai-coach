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
  await expect.poll(() => page.evaluate(() => caches.has("snapshots"))).toBe(false);
});

test("sign-in and settings have no horizontal scroll at 320 px", async ({ page }) => {
  await page.setViewportSize({ width: 320, height: 640 });
  await page.goto("/");
  await expect(page.getByRole("button", { name: "Sign in" })).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
});

test("a 401 mid-session returns to sign-in", async ({ page }) => {
  await signedIn(page);
  await expect(page.getByRole("heading", { name: "Test FC" })).toBeVisible();
  await page.route("**/api/settings", (route) =>
    route.fulfill({ status: 401, contentType: "application/json", body: JSON.stringify({ detail: "Sign in again." }) }));
  await page.getByRole("button", { name: "Settings" }).click();
  await expect(page.getByRole("button", { name: "Sign in" })).toBeVisible();
});

test("an account with no team shows why and can sign out", async ({ page }) => {
  await signedIn(page, { status: 403, body: { detail: "This account has no team." } });
  await expect(page.getByRole("alert")).toContainText("This account has no team.");
  await page.getByRole("button", { name: "Sign out" }).click();
  await expect(page.getByRole("button", { name: "Sign in" })).toBeVisible();
});

for (const width of [320, 390, 1280]) {
  test(`gear stays compact at ${width}px`, async ({ page }) => {
    await page.setViewportSize({ width, height: 700 });
    await signedIn(page);
    const gear = page.getByRole("button", { name: "Settings" });
    await expect(gear).toBeVisible();
    expect((await gear.boundingBox())!.width).toBeLessThanOrEqual(44);
    await page.screenshot({ path: `test-results/gear-${width}.png`, clip: { x: 0, y: 0, width, height: 140 } });
  });
}
