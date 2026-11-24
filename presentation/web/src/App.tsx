import { Component, type ReactNode, useEffect, useState } from "react";
import { ApiError, getMe, getSnapshot } from "./api";
import { configured, onAuthChange, signOut } from "./auth";
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
  return configured
    ? <Main />
    : <ErrorState title="Sign-in isn't set up" detail="Set VITE_SUPABASE_URL and VITE_SUPABASE_ANON_KEY in presentation/web/.env.local, then restart." />;
}

function Main() {
  const [signedIn, setSignedIn] = useState<boolean>();
  const [data, setData] = useState<SnapshotResponse>();
  const [error, setError] = useState<ApiError>();
  const [settings, setSettings] = useState(false);
  const [reload, setReload] = useState(0);

  useEffect(() => onAuthChange(setSignedIn), []);
  useEffect(() => { setData(undefined); setSettings(false); }, [signedIn]);

  useEffect(() => {
    setError(undefined);
    if (!signedIn) return;
    let live = true;
    getMe()
      .then(({ entry }) => getSnapshot(entry))
      .then((d) => live && setData(d))
      .catch((e) => live && setError(e instanceof ApiError ? e : new ApiError(0, String(e))));
    return () => { live = false; };
  }, [signedIn, reload]);

  if (signedIn === undefined) return <Loading />;
  if (!signedIn) return <SignIn />;
  if (settings) return <Settings onBack={() => setSettings(false)} />;
  if (error && error.status !== 401) {
    return (
      <ErrorState title="Couldn't load your team" detail={error.detail}>
        <button onClick={() => setSettings(true)}>Settings</button>
        <button onClick={() => void signOut()}>Sign out</button>
      </ErrorState>
    );
  }
  if (!data) return <Loading />;
  return <ErrorBoundary><TeamScreen data={data} onSettings={() => setSettings(true)} onReload={() => setReload((n) => n + 1)} /></ErrorBoundary>;
}
