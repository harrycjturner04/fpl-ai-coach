import { useEffect, useState } from "react";
import { getSettings, patchSettings, type Settings as S } from "./api";
import { signOut } from "./auth";
import { Loading } from "./states";
import "./app.css";

export function Settings({ onBack }: { onBack: () => void }) {
  const [s, setS] = useState<S>();
  const [cap, setCap] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    let live = true;
    getSettings()
      .then((v) => { if (live) { setS(v); setCap(String(v.cap_gbp)); } })
      .catch((e) => { if (live) setError(e.message); });
    return () => { live = false; };
  }, []);

  const save = (p: Parameters<typeof patchSettings>[0]) => {
    setError(null);
    setBusy(true);
    patchSettings(p).then((v) => { setS(v); setCap(String(v.cap_gbp)); }).catch((e) => setError(e.message)).finally(() => setBusy(false));
  };
  const saveCap = () => {
    const n = Number(cap);
    if (busy) return;
    if (cap.trim() === "" || !Number.isFinite(n) || n < 0 || n > 50) return setError("Enter a cap from 0 to 50 pounds.");
    if (n !== s?.cap_gbp) save({ cap_gbp: n });
    else setError(null);
  };

  if (!s) {
    return error
      ? <main className="panel" role="alert"><p>{error}</p><button onClick={onBack}>Back</button></main>
      : <Loading />;
  }

  return (
    <main className="panel">
      <header className="bar"><button onClick={onBack}>Back</button><h1>Settings</h1></header>
      <form className="box settings" onSubmit={(e) => { e.preventDefault(); saveCap(); }}>
        <label>Monthly cap (£)
          <input type="number" min={0} max={50} step={0.5} value={cap} onChange={(e) => setCap(e.target.value)} onBlur={saveCap} />
        </label>
        <p className="muted">£{s.spent_gbp.toFixed(2)} spent in {s.month}</p>
        <Switch busy={busy} label="Automatic runs" on={s.auto_runs} set={(v) => save({ auto_runs: v })} />
        <Switch busy={busy} label="AI off" on={s.ai_off} set={(v) => save({ ai_off: v })} />
        {error && <p className="form-error" role="alert">{error}</p>}
      </form>
      <section className="box settings">
        <h2>Backstops</h2>
        <ul>
          <li>Anthropic monthly limit</li>
          <li>Google Cloud £1 alert</li>
          <li>Supabase free plan</li>
        </ul>
      </section>
      <button onClick={() => signOut()}>Sign out</button>
    </main>
  );
}

function Switch({ label, on, set, busy }: { label: string; on: boolean; set: (v: boolean) => void; busy: boolean }) {
  return (
    <div className="switch-row">
      <span>{label}</span>
      <button type="button" role="switch" aria-checked={on} aria-label={label} disabled={busy} onClick={() => set(!on)} />
    </div>
  );
}
