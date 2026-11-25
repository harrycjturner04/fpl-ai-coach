# Deploying the coach

The live app is a phone-friendly web app (PWA) behind a login, backed by an API and a cloud job that runs the coach.
This page is the runbook: how the pieces fit, how to set them up once, how to deploy changes, and what to do when
something goes wrong.

## Architecture

```mermaid
flowchart LR
    PC["Owner's PC<br/>scheduled task: fetch FPL"] -->|"push data/"| B["Private bucket<br/>gs://fpl-ai-coach-data"]
    B -->|"pull"| J["Cloud Run Job coach<br/>daily + on demand"]
    J -->|"push reports"| B
    B -->|"read-only mount"| A["api<br/>Cloud Run"]
    A -->|"starts runs"| J
    J & A --> D["Supabase Postgres<br/>runs, settings, spend"]
    A --> W["PWA<br/>Cloudflare Pages"]
    S["Supabase Auth"] --> W
```

- **Why FPL is fetched on the PC.** The FPL API returns 403 to data-centre IP addresses, including Cloud Run's
  (`scripts/check_fpl_reachability.py`). So a scheduled task on the owner's PC fetches FPL data and pushes `data/` to
  the bucket; the cloud never calls FPL.
- **The bucket is the master copy of `data/`.** The PC pulls before fetching and pushes after; the job pulls before
  a run and pushes its report after. Sync never deletes anything, and skips `raw/`, `logs/` and `*.tmp`. A file
  removed locally stays in the bucket: delete it there by hand if it must go.
- **Cloud Run Job `coach`.** Default args `daily`: Cloud Scheduler `daily` (06:00 Europe/London) invokes it with no
  overrides, and it starts at most one automatic run per gameweek in the 36 hours before the deadline. The api starts
  manual runs ("Run the coach now") with overridden args `coach --entry E --run-id R`.
- **api.** FastAPI on Cloud Run, max one instance (rate limiter and start lock are per process). Checks the Supabase
  login token on every data route, reads snapshots from the bucket through a read-only mount, CORS for the Pages
  origin only.
- **Supabase.** Login (email and password) and the app's own state: `app_users`, `settings`, `runs`, `spend`,
  `job_lock`. Row-level security is on with no policies and no grants to `anon`: only the api and job (through
  `DATABASE_URL`) can read them. Pipeline data stays in Parquet.

Service accounts, least privilege (no project-level roles):

| Account | Grants |
|---|---|
| `fpl-api` | `storage.objectViewer` on the bucket; `run.jobsExecutorWithOverrides` on job `coach`; `secretAccessor` on `DATABASE_URL` |
| `fpl-job` | `storage.objectAdmin` on the bucket; `secretAccessor` on `ANTHROPIC_API_KEY`, `FOOTBALL_DATA_API_KEY`, `DATABASE_URL` |
| `fpl-scheduler` | `run.invoker` on job `coach` (cannot override args, hence the job's default `daily`) |

## Where runs happen

Once live, the bucket is the master copy of `data/`. Run the coach from the app ("Run the coach now") or the cloud
job. Local CLI runs (`orchestration.coach`, `prediction.cli`, `optimisation.cli`) are for development: their output
is replaced at the next refresh, since a pull overwrites any local file that differs from the bucket. The
forward-test logs (`prediction_log`, `plan_log`, `coach_log`) accumulate from cloud runs, manual and the automatic
one each gameweek.

Use `python -m orchestration.bucket push` only straight after `python -m orchestration.job refresh` or for the first
upload, never while a run is active. Manual push and pull take the same data lock as the jobs, and stop with "Data is
locked by a running job; try again later." if one holds it.

## One-off setup

1. Accounts: Google Cloud (project `fpl-ai-coach`, region `europe-west2`, billing on), Supabase (free plan),
   Cloudflare, Anthropic, football-data.org. Docker Desktop, Node 24, Python 3.14, `gcloud`.
2. Local `.env` in the repo root (git-excluded; never commit it):
   `ANTHROPIC_API_KEY`, `FOOTBALL_DATA_API_KEY`, `DATABASE_URL`, `SUPABASE_URL`, `SUPABASE_ANON_KEY` (the public
   `sb_publishable_` key), `FPL_COACH_BUCKET`, optional `ALLOWED_ORIGIN`, `PAGES_PROJECT`.
3. Logins: `gcloud auth login`, `gcloud auth application-default login` (the bucket CLI and refresh task use
   Application Default Credentials), `gcloud config set project fpl-ai-coach`, `npx --yes wrangler@4.168.0 login`.
4. `powershell -File deploy\setup.ps1` (safe to re-run; `-DryRun` prints the commands): APIs, bucket, service
   accounts, secrets (byte-exact from `.env`), Pages project.
5. Database:
   ```
   npx supabase@latest login
   npx supabase@latest link --project-ref <ref>
   npx supabase@latest db push --remote
   ```
   Create the owner's login in Supabase (Authentication > Users), then in the SQL editor:
   `insert into app_users (user_id, entry) values ('<user id>', <FPL team ID>);`
6. On the PC install with `pip install -e ".[dev]"` (includes the bucket client). First data upload: `python -m ingestion.cli --history --entry <team ID>`, then `python -m orchestration.bucket push`.
7. `powershell -File deploy\deploy.ps1`.
8. `powershell -File deploy\register_pc_task.ps1` (no admin needed).

## Routine deploy

Commit first (the script refuses a dirty tree; untracked files are ignored), then
`powershell -File deploy\deploy.ps1`. It builds and pushes both images, deploys the api and job, applies the job IAM,
creates or updates the scheduler, builds the web app with the api URL and writes its security headers (CSP,
HSTS, nosniff, Referrer-Policy, Permissions-Policy), and deploys it to Pages. Database changes: a new file in
`supabase/migrations/`, then `npx supabase@latest db push --remote`.

Manual job runs: `gcloud run jobs execute coach --region europe-west2` (daily logic).

## PC refresh task

Task "FPL Coach refresh" runs `python -m orchestration.job refresh` daily at 05:00 and at logon, also on battery
(the PC is a laptop), for the current user only. Under the data lock it pulls the bucket, fetches FPL data for every
entry in `app_users`, and pushes. If finished gameweeks lack ownership and transfer values, it fetches once more with
player history to fill them (no manual `--history` needed).

If the laptop is off, the data goes stale. Once it is 30 hours old, the daily job does not run the coach; it records
a blocked run "Data too old: switch on the PC." (once a day), and the app shows it. Switching the PC on runs the task
at logon, and the next daily run proceeds.

## Pause and resume

```
python -m orchestration.admin pause     # AI off in settings + Cloud Scheduler job paused
python -m orchestration.admin resume
```

## Spend backstops

- Anthropic: monthly spend limit on the API key's workspace.
- Google Cloud: £1 budget alert on the billing account.
- Supabase: free plan (no billing).
- In app (Settings): monthly cap (default £5), automatic-runs switch, AI-off switch. Runs are refused once the cap
  is reached. A coach run costs about $0.08 to $0.10.

## DATABASE_URL

Use the Supabase **session pooler** connection string (IPv4; the direct host is IPv6 only, which Cloud Run and many
home networks can't reach). Percent-encode special characters in the password (for example `[` as `%5B`, `@` as
`%40`), and drop any square brackets around the placeholder. After changing it, re-run `deploy\setup.ps1` (it adds a
new secret version only if the bytes differ) and `deploy\deploy.ps1`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Job fails with `OSError: libgomp.so.1` | LightGBM needs `libgomp1` in the job image (now in `deploy/job.Dockerfile`) |
| `supabase db push` changes nothing on the hosted project | Use `db push --remote` (newer CLIs target the local database by default) |
| `gcloud` or the bucket CLI says credentials are missing | `gcloud auth application-default login` (separate from `gcloud auth login`) |
| wrangler hangs or asks to log in | `npx --yes wrangler@4.168.0 login` (pinned version; unpinned `npx` can hang on a prompt) |
| Password authentication failed | Check `DATABASE_URL` encoding (above) |
| PC task fails at once | `python` resolved to the Microsoft Store stub; install Python or edit the path in `register_pc_task.ps1` |

## Tests

- `pytest`: offline, default.
- `pytest -m db`: against a local Supabase only (`npx supabase@latest start`, then
  `DATABASE_URL=postgresql://postgres:postgres@127.0.0.1:54322/postgres`). The tests skip unless the host is
  `127.0.0.1` or `localhost`, so they never touch the live database.
- `presentation/web`: `npm test`, `npm run e2e`, `npm run build`.

## Local mode

Runs the app against a local Supabase, with runs launched as local processes on local `data/`:

1. `npx supabase@latest start`. In local Studio (http://127.0.0.1:54323) add a user (Authentication > Users), then
   in its SQL editor: `insert into app_users (user_id, entry) values ('<user id>', <FPL team ID>);`
2. api, with these set in the shell:
   `DATABASE_URL=postgresql://postgres:postgres@127.0.0.1:54322/postgres`, `SUPABASE_URL=http://127.0.0.1:54321`,
   `LAUNCHER=local`; then
   `python -m uvicorn presentation.api.app:app --host 127.0.0.1 --port 8000`. The api also loads `.env`, but shell
   values win, so set each of these in the shell to override the hosted ones. With a local `DATABASE_URL`, jobs
   automatically ignore `FPL_COACH_BUCKET` and work on local files only, and `python -m orchestration.bucket`
   refuses to push or pull, because the local database's lock would not protect the live data.
3. Web: `presentation/web/.env.local` with `VITE_SUPABASE_URL=http://127.0.0.1:54321` and
   `VITE_SUPABASE_ANON_KEY=` the local publishable key from `npx supabase status`; then `npm run dev` in
   `presentation/web`.

## Security checks

Before a release, after `npm run build` in `presentation/web`:

- `grep -rE "sk-ant|sb_secret_[A-Za-z0-9]{8,}|service_role|postgresql://" presentation/web/dist` is empty
  (supabase-js itself contains the bare `sb_secret_` prefix, so the pattern needs the characters after it).

## Season end (later)

When the season-end re-tune command exists, it runs on the PC: add the finished season, re-run the archive and
re-tunes, commit, push `data/` and re-deploy. Until then, follow the season-end notes in the README and push the
bucket after.
