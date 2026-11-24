import type { Page } from "@playwright/test";
import fixture from "../src/test/snapshot.json" with { type: "json" };

const json = (body: unknown, status = 200) => ({ status, contentType: "application/json", body: JSON.stringify(body) });

/** Mocks Supabase sign-in and the api (override /api/me with `me`), then signs in through the form. */
export async function signedIn(page: Page, me: { status: number; body: unknown } = { status: 200, body: { entry: 42, entries: [42] } }) {
  await page.route("**/auth/v1/**", (route) => {
    const url = route.request().url();
    if (url.includes("/token")) {
      return route.fulfill(json({
        access_token: "tok", token_type: "bearer", expires_in: 3600, refresh_token: "r",
        user: { id: "u1", email: "me@x.com", aud: "authenticated", app_metadata: {}, user_metadata: {}, created_at: "2026-01-01T00:00:00Z" },
      }));
    }
    return route.fulfill({ status: 204 });
  });
  await page.route("**/api/**", (route) => {
    const path = new URL(route.request().url()).pathname;
    if (path === "/api/me") return route.fulfill(json(me.body, me.status));
    if (path === "/api/settings") return route.fulfill(json({ cap_gbp: 5, auto_runs: true, ai_off: false, spent_gbp: 0.4, month: "November 2026" }));
    return route.fulfill(json(fixture));
  });
  await page.goto("/");
  await page.getByLabel("Email").fill("me@x.com");
  await page.getByLabel("Password").fill("pw");
  await page.getByRole("button", { name: "Sign in" }).click();
}
