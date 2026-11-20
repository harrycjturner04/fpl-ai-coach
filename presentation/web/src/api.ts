import type { SnapshotResponse } from "./types";

export class ApiError extends Error {
  constructor(public status: number, public detail: string) {
    super(detail);
  }
}

const NO_SERVER = "The coach's server isn't running. Start it with: uvicorn presentation.api.app:app --port 8000";

async function get<T>(url: string): Promise<T> {
  let res: Response;
  try {
    res = await fetch(url);
  } catch {
    throw new ApiError(0, NO_SERVER);
  }
  const body = await res.json().catch(() => null);
  if (!res.ok) throw new ApiError(res.status, body?.detail ?? res.statusText);
  return body as T;
}

export const getMe = () => get<{ entry: number }>("/api/me");

export async function getSnapshot(entry: number): Promise<SnapshotResponse> {
  const data = await get<SnapshotResponse>(`/api/entries/${entry}/snapshot`);
  if (data.schema !== 1) {
    throw new ApiError(0, `This snapshot uses schema ${data.schema}, but this app reads schema 1. Update the app.`);
  }
  return data;
}
