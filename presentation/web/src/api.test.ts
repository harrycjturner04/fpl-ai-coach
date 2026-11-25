import { afterEach, expect, it, vi } from "vitest";
import { ApiError, getSnapshot, patchSettings } from "./api";
import fixture from "./test/snapshot.json";

const signOut = vi.hoisted(() => vi.fn());
vi.mock("./auth", () => ({ accessToken: () => Promise.resolve("tok"), signOut }));

afterEach(() => vi.unstubAllGlobals());

const reply = (status: number, body: unknown) =>
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify(body), { status })));

it("turns an error body into ApiError with the server's detail", async () => {
  reply(404, { detail: "No coach run yet. Run: python -m orchestration.coach --entry 42" });
  await expect(getSnapshot(42)).rejects.toMatchObject({ status: 404, detail: expect.stringContaining("orchestration.coach") });
});

it("reports a missing server with the uvicorn hint", async () => {
  vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("Failed to fetch")));
  const err = await getSnapshot(42).catch((e) => e);
  expect(err).toBeInstanceOf(ApiError);
  expect(err.status).toBe(0);
  expect(err.detail).toContain("python -m uvicorn presentation.api.app:app --port 8000");
});

it("rejects another schema version", async () => {
  reply(200, { ...fixture, schema: 3 });
  await expect(getSnapshot(42)).rejects.toThrow(/schema/);
});

it("sends the bearer token", async () => {
  reply(200, fixture);
  await getSnapshot(42);
  const init = vi.mocked(fetch).mock.calls[0][1] as RequestInit;
  expect(new Headers(init.headers).get("Authorization")).toBe("Bearer tok");
});

it("joins FastAPI 422 detail arrays", async () => {
  reply(422, { detail: [{ loc: ["body", "cap_gbp"], msg: "too big", type: "x" }, { loc: [], msg: "bad", type: "y" }] });
  await expect(patchSettings({ cap_gbp: 99 })).rejects.toMatchObject({ status: 422, detail: "too big; bad" });
});

it("maps 401 to ApiError(401)", async () => {
  reply(401, { detail: "Sign in again." });
  await expect(getSnapshot(42)).rejects.toMatchObject({ status: 401, detail: "Sign in again." });
  expect(signOut).toHaveBeenCalled();
});
