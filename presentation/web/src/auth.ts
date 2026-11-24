import { createClient } from "@supabase/supabase-js";

export const supabase = createClient(import.meta.env.VITE_SUPABASE_URL, import.meta.env.VITE_SUPABASE_ANON_KEY);

const NO_SERVICE = "Can't reach the sign-in service. Check your connection.";

/** Returns the error copy to show, or null on success. */
export async function signIn(email: string, password: string): Promise<string | null> {
  try {
    const { error } = await supabase.auth.signInWithPassword({ email, password });
    if (!error) return null;
    return error.status === 400 ? "Wrong email or password." : NO_SERVICE;
  } catch {
    return NO_SERVICE;
  }
}

export async function signOut() {
  await supabase.auth.signOut();
  try { await caches.delete("snapshots"); } catch { /* no Cache API here */ }
}

export async function accessToken(): Promise<string | null> {
  return (await supabase.auth.getSession()).data.session?.access_token ?? null;
}

/** Calls cb(true or false) now and on every sign-in or sign-out; returns the unsubscribe. */
export function onAuthChange(cb: (signedIn: boolean) => void) {
  const { data } = supabase.auth.onAuthStateChange((_event, session) => cb(!!session));
  return () => data.subscription.unsubscribe();
}
