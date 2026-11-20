import { useEffect, useState } from "react";
import { ApiError, getMe, getSnapshot } from "./api";
import { ErrorState, Loading } from "./states";
import type { SnapshotResponse } from "./types";

export default function App() {
  const [data, setData] = useState<SnapshotResponse>();
  const [error, setError] = useState<ApiError>();

  useEffect(() => {
    getMe()
      .then(({ entry }) => getSnapshot(entry))
      .then(setData)
      .catch((e) => setError(e instanceof ApiError ? e : new ApiError(0, String(e))));
  }, []);

  if (error) return <ErrorState title="Couldn't load your team" detail={error.detail} />;
  if (!data) return <Loading />;
  return <main><pre>{data.team_name}</pre></main>;
}
