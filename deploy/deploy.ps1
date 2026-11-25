# Build and deploy the FPL AI Coach (repeatable). Run after deploy\setup.ps1:
#   powershell -File deploy\deploy.ps1 [-DryRun]
# .env keys needed: SUPABASE_URL, SUPABASE_ANON_KEY (public publishable key).
# Optional: PAGES_PROJECT (default fpl-ai-coach), ALLOWED_ORIGIN (default https://<PAGES_PROJECT>.pages.dev).
# Secrets are not read here: Cloud Run takes them from Secret Manager (setup.ps1).
param([switch]$DryRun)
$ErrorActionPreference = "Stop"

$Project = "fpl-ai-coach"
$Region = "europe-west2"
$Bucket = "fpl-ai-coach-data"
$Registry = "$Region-docker.pkg.dev/$Project/fpl"
$Repo = Split-Path -Parent $PSScriptRoot
$Web = Join-Path $Repo "presentation\web"
$Wrangler = "wrangler@4.168.0"

function Read-DotEnv([string]$Path) {
    # Same rules as ingestion/env.py: NAME=value, skip blanks and comments, strip quotes, shell values win.
    $vars = @{}
    if (Test-Path $Path) {
        foreach ($line in Get-Content -Encoding UTF8 $Path) {
            $line = $line.Trim()
            if (-not $line -or $line.StartsWith("#") -or -not $line.Contains("=")) { continue }
            $name, $value = $line.Split("=", 2)
            $value = $value.Trim().Trim('"', "'")
            if ($value) { $vars[$name.Trim()] = $value }
        }
    }
    foreach ($name in @($vars.Keys)) {
        $shell = [Environment]::GetEnvironmentVariable($name)
        if ($shell) { $vars[$name] = $shell }
    }
    $vars
}

function Invoke-Native([string]$Exe, [string[]]$Arguments) {
    Write-Host "> $Exe $($Arguments -join ' ')"
    if ($DryRun) { return }
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Exe failed (exit $LASTEXITCODE)" }
}

function Test-Native([string]$Exe, [string[]]$Arguments) {
    # Read-only probe (runs in dry-run too); stderr hidden, true when the command succeeds.
    $ErrorActionPreference = "Continue"
    & $Exe @Arguments *> $null
    $LASTEXITCODE -eq 0
}

function Step([string]$Text) { Write-Host "`n== $Text" -ForegroundColor Cyan }

$envVars = Read-DotEnv (Join-Path $Repo ".env")
foreach ($key in @("SUPABASE_URL", "SUPABASE_ANON_KEY")) {
    $v = $envVars[$key]
    if (-not $v) { $v = [Environment]::GetEnvironmentVariable($key) }
    if (-not $v) { throw "Missing $key in .env (or the shell)." }
    $envVars[$key] = $v
}
$pagesProject = $envVars["PAGES_PROJECT"]
if (-not $pagesProject) { $pagesProject = "fpl-ai-coach" }
$allowedOrigin = $envVars["ALLOWED_ORIGIN"]
if (-not $allowedOrigin) { $allowedOrigin = "https://$pagesProject.pages.dev" }
$supabaseUrl = $envVars["SUPABASE_URL"].TrimEnd("/")

$ErrorActionPreference = "Continue"
$active = (& gcloud.cmd config get-value project 2>$null)
$ErrorActionPreference = "Stop"
if ($active -ne $Project) { throw "gcloud project is '$active'; run: gcloud config set project $Project" }
if (& git -C $Repo status --porcelain --untracked-files=no) { throw "Uncommitted changes: commit first so the image tag (git sha) matches the code." }
$tag = (& git -C $Repo rev-parse --short HEAD)
if ($LASTEXITCODE -ne 0) { throw "git rev-parse failed" }
if ($DryRun) { Write-Host "DRY RUN: printing build and cloud-changing commands, not running them." -ForegroundColor Yellow }

$coachJob = "projects/$Project/locations/$Region/jobs/coach"
$apiSa = "fpl-api@$Project.iam.gserviceaccount.com"
$jobSa = "fpl-job@$Project.iam.gserviceaccount.com"
$schedulerSa = "fpl-scheduler@$Project.iam.gserviceaccount.com"

Step "Build and push images ($tag)"
foreach ($name in @("api", "job")) {
    Invoke-Native docker @("build", "--platform=linux/amd64", "--provenance=false", "-f", (Join-Path $Repo "deploy\$name.Dockerfile"), "-t", "$Registry/${name}:$tag", $Repo)
    Invoke-Native docker @("push", "$Registry/${name}:$tag")
}

Step "Cloud Run Job coach"
Invoke-Native gcloud.cmd @("run", "jobs", "deploy", "coach", "--image=$Registry/job:$tag", "--region=$Region",
    "--service-account=$jobSa", "--cpu=4", "--memory=8Gi", "--task-timeout=3600", "--max-retries=0", "--args=daily",
    "--set-env-vars=FPL_COACH_BUCKET=$Bucket,FPL_COACH_DATA_DIR=/tmp/data",
    "--set-secrets=ANTHROPIC_API_KEY=ANTHROPIC_API_KEY:latest,FOOTBALL_DATA_API_KEY=FOOTBALL_DATA_API_KEY:latest,DATABASE_URL=DATABASE_URL:latest")

Step "Job IAM (api runs it with overrides, scheduler invokes it)"
Invoke-Native gcloud.cmd @("run", "jobs", "add-iam-policy-binding", "coach", "--region=$Region",
    "--member=serviceAccount:$apiSa", "--role=roles/run.jobsExecutorWithOverrides")
Invoke-Native gcloud.cmd @("run", "jobs", "add-iam-policy-binding", "coach", "--region=$Region",
    "--member=serviceAccount:$schedulerSa", "--role=roles/run.invoker")

Step "Scheduler job daily (06:00 Europe/London, runs the job's default args: daily)"
# Empty body, no overrides: run.invoker does not allow overrides. From a file: PowerShell 5.1 strips quotes in native args.
$body = Join-Path $env:TEMP "fpl-scheduler-body.json"
[IO.File]::WriteAllText($body, '{}')
$schedule = @("--location=$Region", "--schedule=0 6 * * *", "--time-zone=Europe/London",
    "--uri=https://run.googleapis.com/v2/$($coachJob):run", "--http-method=POST", "--message-body-from-file=$body",
    "--oauth-service-account-email=$schedulerSa")
if (Test-Native gcloud.cmd @("scheduler", "jobs", "describe", "daily", "--location=$Region")) {
    Invoke-Native gcloud.cmd (@("scheduler", "jobs", "update", "http", "daily", "--update-headers=Content-Type=application/json") + $schedule)
} else {
    Invoke-Native gcloud.cmd (@("scheduler", "jobs", "create", "http", "daily", "--headers=Content-Type=application/json") + $schedule)
}

Step "Cloud Run service api"
Invoke-Native gcloud.cmd @("run", "deploy", "api", "--image=$Registry/api:$tag", "--region=$Region",
    "--service-account=$apiSa", "--max-instances=1", "--allow-unauthenticated", "--execution-environment=gen2",
    "--set-env-vars=SUPABASE_URL=$supabaseUrl,ALLOWED_ORIGIN=$allowedOrigin,LAUNCHER=cloudrun,COACH_JOB=$coachJob,FPL_COACH_DATA_DIR=/data",
    "--set-secrets=DATABASE_URL=DATABASE_URL:latest",
    "--add-volume=name=data,type=cloud-storage,bucket=$Bucket,readonly=true,mount-options=implicit-dirs",
    "--add-volume-mount=volume=data,mount-path=/data")
$ErrorActionPreference = "Continue"
$apiUrl = (& gcloud.cmd run services describe api "--region=$Region" "--format=value(status.url)" 2>$null)
$ErrorActionPreference = "Stop"
if (-not $apiUrl) {
    if (-not $DryRun) { throw "Could not read the api URL." }
    $apiUrl = "https://api-PLACEHOLDER.$Region.run.app"
}
Write-Host "api: $apiUrl"

Step "Web build"
$env:VITE_API_BASE = $apiUrl
$env:VITE_SUPABASE_URL = $supabaseUrl
$env:VITE_SUPABASE_ANON_KEY = $envVars["SUPABASE_ANON_KEY"]
Push-Location $Web
try {
    Invoke-Native npm.cmd @("run", "build")
    $headers = (Get-Content -Raw (Join-Path $PSScriptRoot "headers.template")).Replace("{{API_ORIGIN}}", $apiUrl).Replace("{{SUPABASE_ORIGIN}}", $supabaseUrl)
    Write-Host "> write dist\_headers (connect-src $apiUrl $supabaseUrl)"
    if (-not $DryRun) { [IO.File]::WriteAllText((Join-Path $Web "dist\_headers"), $headers) }
    Invoke-Native npx.cmd @("--yes", $Wrangler, "pages", "deploy", "dist", "--project-name=$pagesProject", "--branch=main")
} finally { Pop-Location }

Write-Host "`nDone. App: https://$pagesProject.pages.dev  api: $apiUrl"
Write-Host "api ALLOWED_ORIGIN is $allowedOrigin; if the app is served elsewhere, set ALLOWED_ORIGIN in .env and re-run."
Write-Host "Manual runs: gcloud run jobs execute coach --region $Region (daily); a coach run: --args=coach,--entry,E,--run-id,R"
