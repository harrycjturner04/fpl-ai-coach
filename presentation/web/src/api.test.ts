import { afterEach, expect, it, vi } from "vitest";
import { ApiError, getSnapshot } from "./api";
import fixture from "./test/snapshot.json";

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
  reply(200, { ...fixture, schema: 2 });
  await expect(getSnapshot(42)).rejects.toThrow(/schema/);
});
