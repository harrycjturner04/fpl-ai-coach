export interface Week {
  gameweek: number;
  squad: number[];
  starting: number[];
  bench: number[];
  captain: number;
  vice_captain: number;
  transfers_in: number[];
  transfers_out: number[];
  hits: number;
  chip: string | null;
  projected_points: number;
  /** This week's predicted points (includes the bench on a Bench Boost week); the headline figure. */
  points: number;
  money_left: number;
  free_transfers_next: number;
}

export interface Player {
  id: number;
  name: string;
  team: string;
  position: "GKP" | "DEF" | "MID" | "FWD";
  price: number;
  selling_price: number | null;
  status: string;
  chance: number | null;
  news: string | null;
  points: number[];
  p60: number | null;
}

export interface Flag {
  player_id: number;
  status: "fit" | "doubt" | "out" | "suspended" | "rotation_risk";
  start_prob_next_gw: number;
  return_gw: number | null;
  confidence: "low" | "medium" | "high";
  source_url: string;
  source_date: string | null;
  source_kind: "official" | "media" | "predicted_lineup";
  summary: string;
}

export interface Adjustment {
  player_id: number;
  horizon: number;
  kind: string;
  raw: number;
  proposed: number;
  applied: number;
  capped: boolean;
  reason: string;
}

export interface ChipRow {
  chip: string;
  gameweek: number;
  objective: number;
  gain_vs_none: number;
  week1_same_as_none: boolean;
  exact: boolean;
  note: string;
}

export interface Snapshot {
  schema: 1;
  entry: number;
  team_name: string;
  gameweek: number;
  deadline: string;
  generated_at: string;
  data_pulled_at: string | null;
  report: string;
  model: string;
  exact: boolean;
  state: { bank: number; free_transfers: number; chips_available: string[]; hit_cost: number };
  gameweeks: number[];
  players: Player[];
  current: Week | null;
  plan: { objective: number; weeks: Week[] };
  flags: Flag[];
  adjustments: Adjustment[];
  chip_scan: ChipRow[] | null;
  sections: Record<string, string>;
  notes: string[];
  warnings: string[];
}

export type SnapshotResponse = Snapshot & {
  call: { title: string; reason: string; chip: string | null };
  status: { stale: boolean; reason: string | null };
};

export interface Run {
  id: number;
  entry: number;
  gameweek: number;
  kind: string;
  status: "queued" | "running" | "done" | "failed" | "blocked";
  steps: { name: string; secs: number }[];
  error: string | null;
  cost_usd: number | null;
  data_pulled_at: string | null;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  steps_expected: string[];
  data_age_hours: number | null;
}
