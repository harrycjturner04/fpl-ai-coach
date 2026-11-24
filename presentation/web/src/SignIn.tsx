import { useState, type FormEvent } from "react";
import { signIn } from "./auth";
import "./app.css";

export function SignIn() {
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const submit = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const f = new FormData(e.currentTarget);
    setBusy(true);
    setError(await signIn(String(f.get("email")), String(f.get("password"))));
    setBusy(false);
  };

  return (
    <main className="panel">
      <h1>FPL AI Coach</h1>
      <form onSubmit={submit}>
        <label>Email<input name="email" type="email" autoComplete="username" required /></label>
        <label>Password<input name="password" type="password" autoComplete="current-password" required /></label>
        {error && <p className="form-error" role="alert">{error}</p>}
        <button className="primary" disabled={busy}>Sign in</button>
      </form>
    </main>
  );
}
