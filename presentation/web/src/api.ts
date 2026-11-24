import { accessToken } from "./auth";
import type { Run, SnapshotResponse } from "./types";

export class ApiError extends Error {
  constructor(public status: number, public detail: string, public reason?: string) {
    super(detail);
  }
}

const NO_SERVER = "The coach's server isn't running. Start it with: python -m uvicorn presentation.api.app:app --port 8000";
const BASE = import.meta.env.VITE_API_BASE ?? "";

async function call<T>(path: string, method = "GET", body?: unknown): Promise<T> {
  const token = await accessToken();
  let res: Response;
  try {
    res = await fetch(BASE + path, {
      method,
      headers: { ...(token && { Authorization: `Bearer ${token}` }), ...(body !== undefined && { "Content-Type": "application/json" }) },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
  } catch {
    throw new ApiError(0, NO_SERVER);
  }
  const data = await res.json().catch(() => null);
  if (!res.ok) {
    const d = data?.detail;
    const detail = typeof d === "string" ? d : Array.isArray(d) ? d.map((x) => x.msg).join("; ") : res.statusText;
    throw new ApiError(res.status, detail, data?.reason);
  }
  return data as T;
}

export interface Settings { cap_gbp: number; auto_runs: boolean; ai_off: boolean; spent_gbp: number; month: string }

export const getMe = () => call<{ entry: number; entries: number[] }>("/api/me");
export const getSettings = () => call<Settings>("/api/settings");
export const patchSettings = (p: Partial<Pick<Settings, "cap_gbp" | "auto_runs" | "ai_off">>) => call<Settings>("/api/settings", "PATCH", p);
export const startRun = (entry: number) => call<Run>(`/api/entries/${entry}/runs`, "POST");
export const getRun = (id: number) => call<Run>(`/api/runs/${id}`);
export const getLatestRun = (entry: number) => call<Run | null>(`/api/entries/${entry}/runs/latest`);

export async function getSnapshot(entry: number): Promise<SnapshotResponse> {
  const data = await call<SnapshotResponse>(`/api/entries/${entry}/snapshot`);
  if (data.schema !== 1) {
    throw new ApiError(0, `This snapshot uses schema ${data.schema}, but this app reads schema 1. Update the app.`);
  }
  return data;
}
