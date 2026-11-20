import { lineup } from "./model";
import { PlayerDisc } from "./PlayerDisc";
import type { Snapshot, Week } from "../types";
import "./pitch.css";

const W = 300;
const H = 360;
// Penalty box or six-yard box of width w and depth d on the top or bottom goal line.
const box = (top: boolean, w: number, d: number) => {
  const x = (W - w) / 2;
  const [y, e] = top ? [0, d] : [H, H - d];
  return `${x},${y} ${x},${e} ${x + w},${e} ${x + w},${y}`;
};

function Turf() {
  return (
    <svg className="turf" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" aria-hidden="true">
      <defs>
        <radialGradient id="floodlight" cx=".5" cy="0" r="1">
          <stop offset="0" stopColor="#FFFBE0" stopOpacity=".2" />
          <stop offset=".6" stopColor="#FFFBE0" stopOpacity="0" />
        </radialGradient>
      </defs>
      <rect width={W} height={H} fill="#12271D" />
      {[0, 2, 4, 6, 8].map((i) => (
        <rect key={i} y={(i * H) / 10} width={W} height={H / 10} fill="#fff" opacity=".03" />
      ))}
      <g className="chalk">
        <rect x="1" y="1" width={W - 2} height={H - 2} />
        <line x1="0" y1={H / 2} x2={W} y2={H / 2} />
        <ellipse cx={W / 2} cy={H / 2} rx={W * 0.14} ry={W * 0.14} />
        <polyline points={box(true, W * 0.56, H * 0.16)} />
        <polyline points={box(true, W * 0.24, H * 0.06)} />
        <polyline points={box(false, W * 0.56, H * 0.16)} />
        <polyline points={box(false, W * 0.24, H * 0.06)} />
      </g>
      <ellipse cx={W * 0.12} cy="0" rx={W * 0.55} ry={H * 0.42} fill="url(#floodlight)" />
      <ellipse cx={W * 0.88} cy="0" rx={W * 0.55} ry={H * 0.42} fill="url(#floodlight)" />
    </svg>
  );
}

export function Pitch({ snap, week }: { snap: Snapshot; week: Week }) {
  const { rows, bench } = lineup(snap, week);
  return (
    <div className="pitch-wrap">
      <div className="pitch">
        <Turf />
        <div className="rows" role="group" aria-label="Starting eleven">
          {[...rows].reverse().map((row, i) => (
            <div className="row" key={i}>
              {row.map((s) => <PlayerDisc key={s.player.id} slot={s} />)}
            </div>
          ))}
        </div>
      </div>
      <div className="bench" role="group" aria-label="Bench">
        <span className="bench-title">Bench</span>
        <div className="row">
          {bench.map((s) => <PlayerDisc key={s.player.id} slot={s} />)}
        </div>
      </div>
      <p className="key">Ring = chance of playing 60+ minutes. Number = predicted points.</p>
    </div>
  );
}
