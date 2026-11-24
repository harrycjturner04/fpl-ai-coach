# One-off cloud setup for the FPL AI Coach (safe to re-run). Run from any folder:
#   powershell -File deploy\setup.ps1 [-DryRun]
# .env keys needed: ANTHROPIC_API_KEY, FOOTBALL_DATA_API_KEY, DATABASE_URL (secrets), SUPABASE_URL (project ref)
# Optional: PAGES_PROJECT (default fpl-ai-coach).
# Also needs: gcloud signed in with project fpl-ai-coach, and `npx --yes wrangler@4.168.0 login` done once.
param([switch]$DryRun)
$ErrorActionPreference = "Stop"

$Project = "fpl-ai-coach"
$Region = "europe-west2"
$Bucket = "fpl-ai-coach-data"
$Secrets = @("ANTHROPIC_API_KEY", "FOOTBALL_DATA_API_KEY", "DATABASE_URL")
$Wrangler = "wrangler@4.168.0"
$Repo = Split-Path -Parent $PSScriptRoot

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

function Write-Stdin([string]$Exe, [string]$Arguments, [string]$Value) {
    # Sends $Value on stdin with no newline and no BOM (a PowerShell pipe adds a newline; .NET adds a
    # UTF-8 BOM when the console input code page is 65001). Returns the exit code.
    $saved = [Console]::InputEncoding
    try {
        [Console]::InputEncoding = New-Object System.Text.UTF8Encoding $false
        $psi = New-Object System.Diagnostics.ProcessStartInfo
        $psi.FileName = $Exe
        $psi.Arguments = $Arguments
        $psi.UseShellExecute = $false
        $psi.RedirectStandardInput = $true
        $proc = [System.Diagnostics.Process]::Start($psi)
        $proc.StandardInput.Write($Value)
        $proc.StandardInput.Close()
        $proc.WaitForExit()
        $proc.ExitCode
    } finally { [Console]::InputEncoding = $saved }
}

function Step([string]$Text) { Write-Host "`n== $Text" -ForegroundColor Cyan }

$envVars = Read-DotEnv (Join-Path $Repo ".env")
foreach ($key in $Secrets + @("SUPABASE_URL")) {
    $v = $envVars[$key]
    if (-not $v) { $v = [Environment]::GetEnvironmentVariable($key) }
    if (-not $v) { throw "Missing $key in .env (or the shell)." }
    $envVars[$key] = $v
}
$ErrorActionPreference = "Continue"
$active = (& gcloud.cmd config get-value project 2>$null)
$ErrorActionPreference = "Stop"
if ($active -ne $Project) { throw "gcloud project is '$active'; run: gcloud config set project $Project" }
if ($DryRun) { Write-Host "DRY RUN: printing cloud-changing commands, not running them." -ForegroundColor Yellow }

Step "Enable APIs"
Invoke-Native gcloud.cmd @("services", "enable", "run.googleapis.com", "artifactregistry.googleapis.com",
    "secretmanager.googleapis.com", "cloudscheduler.googleapis.com", "storage.googleapis.com", "iam.googleapis.com")

Step "Artifact Registry repo fpl"
if (-not (Test-Native gcloud.cmd @("artifacts", "repositories", "describe", "fpl", "--location=$Region"))) {
    Invoke-Native gcloud.cmd @("artifacts", "repositories", "create", "fpl", "--repository-format=docker", "--location=$Region")
}
Invoke-Native gcloud.cmd @("auth", "configure-docker", "$Region-docker.pkg.dev", "--quiet")

Step "Bucket gs://$Bucket"
if (-not (Test-Native gcloud.cmd @("storage", "buckets", "describe", "gs://$Bucket"))) {
    Invoke-Native gcloud.cmd @("storage", "buckets", "create", "gs://$Bucket", "--location=$Region",
        "--uniform-bucket-level-access", "--public-access-prevention")
}
Invoke-Native gcloud.cmd @("storage", "buckets", "update", "gs://$Bucket", "--uniform-bucket-level-access", "--public-access-prevention")

Step "Secrets (values from .env, sent on stdin, never shown)"
foreach ($name in $Secrets) {
    if (-not (Test-Native gcloud.cmd @("secrets", "describe", $name))) {
        Invoke-Native gcloud.cmd @("secrets", "create", $name, "--replication-policy=automatic")
    }
    $ErrorActionPreference = "Continue"
    $current = (& gcloud.cmd secrets versions access latest "--secret=$name" 2>$null) -join "`n"
    $ErrorActionPreference = "Stop"
    if ($current -ceq $envVars[$name]) { Write-Host "$name unchanged"; continue }
    Write-Host "> gcloud secrets versions add $name --data-file=-  (value on stdin)"
    if ($DryRun) { continue }
    $code = Write-Stdin (Get-Command gcloud.cmd).Source "secrets versions add $name --data-file=-" $envVars[$name]
    if ($code -ne 0) { throw "gcloud secrets versions add $name failed (exit $code)" }
    $ErrorActionPreference = "Continue"
    $stored = (& gcloud.cmd secrets versions access latest "--secret=$name" 2>$null) -join "`n"
    $ErrorActionPreference = "Stop"
    if ($stored -cne $envVars[$name]) { throw "$name read back from Secret Manager does not match .env (value not shown)." }
}

Step "Service accounts"
foreach ($sa in @("fpl-api", "fpl-job", "fpl-scheduler")) {
    if (-not (Test-Native gcloud.cmd @("iam", "service-accounts", "describe", "$sa@$Project.iam.gserviceaccount.com"))) {
        Invoke-Native gcloud.cmd @("iam", "service-accounts", "create", $sa, "--display-name=$sa")
    }
}
$api = "serviceAccount:fpl-api@$Project.iam.gserviceaccount.com"
$job = "serviceAccount:fpl-job@$Project.iam.gserviceaccount.com"

Step "Bucket IAM (api reads, job reads and writes)"
Invoke-Native gcloud.cmd @("storage", "buckets", "add-iam-policy-binding", "gs://$Bucket", "--member=$api", "--role=roles/storage.objectViewer")
Invoke-Native gcloud.cmd @("storage", "buckets", "add-iam-policy-binding", "gs://$Bucket", "--member=$job", "--role=roles/storage.objectAdmin")

Step "Secret IAM (api: DATABASE_URL only; job: all three)"
Invoke-Native gcloud.cmd @("secrets", "add-iam-policy-binding", "DATABASE_URL", "--member=$api", "--role=roles/secretmanager.secretAccessor")
foreach ($name in $Secrets) {
    Invoke-Native gcloud.cmd @("secrets", "add-iam-policy-binding", $name, "--member=$job", "--role=roles/secretmanager.secretAccessor")
}
Write-Host "Job-level roles (api: run.jobsExecutorWithOverrides, scheduler: run.invoker) are applied by deploy.ps1 once job coach exists."

$pagesProject = $envVars["PAGES_PROJECT"]
if (-not $pagesProject) { $pagesProject = "fpl-ai-coach" }
Step "Cloudflare Pages project $pagesProject"
$ErrorActionPreference = "Continue"
$pages = (& npx.cmd --yes $Wrangler pages project list 2>$null) -join "`n"
$listed = $LASTEXITCODE -eq 0
$ErrorActionPreference = "Stop"
if (-not $listed) { throw "npx $Wrangler pages project list failed; run: npx --yes $Wrangler login" }
$cell = '[\s|' + [char]0x2502 + ']'  # wrangler prints a box-drawn table
if ($pages -notmatch "(^|$cell)$([regex]::Escape($pagesProject))($cell|$)") {
    Invoke-Native npx.cmd @("--yes", $Wrangler, "pages", "project", "create", $pagesProject, "--production-branch=main")
}

Step "Supabase (run these yourself)"
$ref = ([Uri]$envVars["SUPABASE_URL"]).Host.Split(".")[0]
Write-Host "  npx supabase@latest link --project-ref $ref"
Write-Host "  npx supabase@latest db push"
Write-Host "Then in the Supabase dashboard SQL editor (user id from Authentication > Users):"
Write-Host "  insert into app_users (user_id, entry) values ('<your-user-id>', 8298351);"
Write-Host "`nNext: python -m orchestration.bucket push (first upload), then deploy\deploy.ps1"
