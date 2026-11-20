import { Component, type ReactNode, useEffect, useState } from "react";
import { ApiError, getMe, getSnapshot } from "./api";
import { ErrorState, Loading } from "./states";
import { TeamScreen } from "./team/TeamScreen";
import type { SnapshotResponse } from "./types";

/** A render error shows an error state instead of a blank page. */
class ErrorBoundary extends Component<{ children: ReactNode }, { failed: boolean }> {
  state = { failed: false };
  static getDerivedStateFromError() {
    return { failed: true };
  }
  render() {
    return this.state.failed
      ? <ErrorState title="Something went wrong showing your team." detail="Reload the page to try again." />
      : this.props.children;
  }
}

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
  return <ErrorBoundary><TeamScreen data={data} /></ErrorBoundary>;
}
