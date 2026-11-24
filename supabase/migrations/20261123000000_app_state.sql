create table app_users (user_id uuid primary key references auth.users on delete cascade, entry integer not null);
create table settings (id boolean primary key default true check (id), cap_gbp numeric not null default 5,
  auto_runs boolean not null default true, ai_off boolean not null default false, usd_per_gbp numeric not null default 1.25);
insert into settings default values;
create table runs (id bigint generated always as identity primary key, entry integer not null, gameweek integer not null,
  kind text not null check (kind in ('manual','auto')),
  status text not null check (status in ('queued','running','done','failed','blocked')),
  steps jsonb not null default '[]', error text, cost_usd numeric, data_pulled_at timestamptz,
  created_at timestamptz not null default now(), started_at timestamptz, finished_at timestamptz);
create unique index one_auto_run_per_gw on runs (entry, gameweek) where kind = 'auto' and status in ('queued','running','done');
create table spend (id bigint generated always as identity primary key, run_id bigint references runs, kind text not null,
  cost_usd numeric not null, at timestamptz not null default now());
create table job_lock (name text primary key, holder text not null, expires_at timestamptz not null);
-- RLS on, no policies: anon/authenticated read nothing
alter table app_users enable row level security;
alter table settings enable row level security;
alter table runs enable row level security;
alter table spend enable row level security;
alter table job_lock enable row level security;
revoke all on app_users, settings, runs, spend, job_lock from anon, authenticated;
