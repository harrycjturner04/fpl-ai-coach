-- Owner's swap checks and adopted plans; rows are never deleted (the adopted history is the owner log)
create table user_plans (id bigint generated always as identity primary key, entry integer not null,
  gameweek integer not null, snapshot text not null, deadline timestamptz not null, swaps jsonb not null, chip text,
  result jsonb not null, checked_at timestamptz not null default now(), adopted_at timestamptz,
  replaced_at timestamptz);
create unique index one_active_plan_per_gw on user_plans (entry, gameweek)
  where adopted_at is not null and replaced_at is null;
-- RLS on, no policies: anon/authenticated read nothing
alter table user_plans enable row level security;
revoke all on user_plans from anon, authenticated;
revoke all on sequence user_plans_id_seq from anon, authenticated;
