import type { ReactNode } from "react";
import "./team/pitch.css";
import "./team/team.css";

/** The Team screen's own layout classes with placeholder blocks, so nothing moves when data arrives. */
export function Loading() {
  return (
    <main className="team skeleton" role="status" aria-label="Loading your team">
      <header className="team-head">
        <div className="head-row"><div className="sk sk-head" /><p><span className="sk sk-line">&nbsp;</span></p></div>
        <p className="run"><span className="sk sk-line">&nbsp;</span></p>
      </header>
      <div className="team-grid" aria-hidden="true">
        <div className="headline">
          <p className="figures num"><span className="points sk">00.0</span></p>
          <p className="caption">Predicted points this week for the team shown</p>
        </div>
        <div className="team-pitch">
          <div className="tabs sk" />
          <div className="pitch-wrap">
            <div className="rows sk" />
            <div className="bench">
              <span className="bench-title">Bench</span>
              <div className="row">
                {[0, 1, 2, 3].map((i) => (
                  <div className="slot" key={i}><div className="disc" /><span className="chip"><span className="name">&nbsp;</span></span></div>
                ))}
              </div>
            </div>
            <p className="key">Ring = chance of playing 60+ minutes. Number = predicted points.</p>
          </div>
        </div>
      </div>
    </main>
  );
}

export function ErrorState({ title, detail, children }: { title: string; detail: string; children?: ReactNode }) {
  return (
    <main className="error" role="alert">
      <h2>{title}</h2>
      <p>{detail}</p>
      {children}
    </main>
  );
}
