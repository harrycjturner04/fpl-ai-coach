const boxes = [64, 120, 360, 96];

export function Loading() {
  return (
    <main className="skeleton" role="status" aria-label="Loading your team">
      {boxes.map((h, i) => (
        <div key={i} className="box" style={{ height: h, marginBottom: "var(--s4)" }} />
      ))}
    </main>
  );
}

export function ErrorState({ title, detail }: { title: string; detail: string }) {
  return (
    <main className="error" role="alert">
      <h2>{title}</h2>
      <p>{detail}</p>
    </main>
  );
}
