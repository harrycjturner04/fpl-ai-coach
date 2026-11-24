import { Component, type ReactNode, useEffect, useState } from "react";
import { ApiError, getMe, getSnapshot } from "./api";
import { onAuthChange, signOut } from "./auth";
import { Settings } from "./Settings";
import { SignIn } from "./SignIn";
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
  const [signedIn, setSignedIn] = useState<boolean>();
  const [data, setData] = useState<SnapshotResponse>();
  const [error, setError] = useState<ApiError>();
  const [settings, setSettings] = useState(false);

  useEffect(() => onAuthChange(setSignedIn), []);

  useEffect(() => {
    setData(undefined);
    setError(undefined);
    setSettings(false);
    if (!signedIn) return;
    let live = true;
    getMe()
      .then(({ entry }) => getSnapshot(entry))
      .then((d) => live && setData(d))
      .catch((e) => {
        if (!live) return;
        if (e instanceof ApiError && e.status === 401) void signOut();
        else setError(e instanceof ApiError ? e : new ApiError(0, String(e)));
      });
    return () => { live = false; };
  }, [signedIn]);

  if (signedIn === undefined) return <Loading />;
  if (!signedIn) return <SignIn />;
  if (settings) return <Settings onBack={() => setSettings(false)} />;
  if (error) return <ErrorState title="Couldn't load your team" detail={error.detail} />;
  if (!data) return <Loading />;
  return <ErrorBoundary><TeamScreen data={data} onSettings={() => setSettings(true)} /></ErrorBoundary>;
}
