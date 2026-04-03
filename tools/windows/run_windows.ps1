param(
    [ValidateSet("doctor", "auth", "run", "cycle", "digest", "plan", "review", "daily", "feed", "fix", "seed", "test", "careful", "smart", "light", "full", "loop")]
    [string]$Action = "feed"
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = Resolve-Path (Join-Path $ScriptDir "..\..")
Set-Location $Root

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    Write-Host "Python launcher 'py' was not found in PowerShell." -ForegroundColor Red
    Write-Host "Install Python on Windows, then reopen PowerShell." -ForegroundColor Yellow
    Write-Host "After that, verify with: py --version" -ForegroundColor Yellow
    exit 1
}

function Refresh-Artifacts {
    if ((Test-Path "simulation_results.csv") -or (Test-Path "simulations.csv")) {
        if (-not (Test-Path "artifacts")) {
            New-Item -ItemType Directory -Path "artifacts" | Out-Null
        }
        py scripts/results_digest.py --format markdown | Tee-Object -FilePath "artifacts/tom_tat_moi_nhat.md"
        py scripts/plan_next_batch.py --format markdown --memory "artifacts/bo_nho_nghien_cuu.json" --write-memory "artifacts/bo_nho_nghien_cuu.json" --write-batch "artifacts/bieu_thuc_ung_vien.txt" --write-plan "artifacts/lo_tiep_theo.json" | Tee-Object -FilePath "artifacts/lo_tiep_theo.md"
        py scripts/render_cycle_report.py --output "artifacts/bao_cao_moi_nhat.md" | Out-Null
        py scripts/manual_review.py --input "artifacts/lo_tiep_theo.json" --output "artifacts/duyet_tay.md" | Out-Null
        py scripts/daily_best.py --input "artifacts/lo_tiep_theo.json" --output "artifacts/alpha_tot_nhat_hom_nay.md" | Out-Null
        py scripts/alpha_feed.py --input "artifacts/lo_tiep_theo.json" --output "artifacts/bang_tin_alpha.md" | Out-Null
    }
    else {
        Write-Host "No simulation CSV found yet, skipping digest and next-batch planning." -ForegroundColor Yellow
    }
}

switch ($Action) {
    "doctor" {
        py scripts/doctor.py --mode windows
    }
    "auth" {
        py scripts/doctor.py --mode windows --require-env --require-deps --check-auth
    }
    "digest" {
        py scripts/results_digest.py --format markdown
    }
    "plan" {
        if (-not (Test-Path "artifacts")) {
            New-Item -ItemType Directory -Path "artifacts" | Out-Null
        }
        py scripts/plan_next_batch.py --format markdown --memory "artifacts/bo_nho_nghien_cuu.json" --write-memory "artifacts/bo_nho_nghien_cuu.json" --write-batch "artifacts/bieu_thuc_ung_vien.txt" --write-plan "artifacts/lo_tiep_theo.json"
        py scripts/render_cycle_report.py --output "artifacts/bao_cao_moi_nhat.md" | Out-Null
        py scripts/manual_review.py --input "artifacts/lo_tiep_theo.json" --output "artifacts/duyet_tay.md" | Out-Null
    }
    "review" {
        Refresh-Artifacts
        if (Test-Path "artifacts/duyet_tay.md") {
            Get-Content "artifacts/duyet_tay.md"
        }
    }
    "daily" {
        Refresh-Artifacts
        if (Test-Path "artifacts/alpha_tot_nhat_hom_nay.md") {
            Get-Content "artifacts/alpha_tot_nhat_hom_nay.md"
        }
    }
    "feed" {
        Refresh-Artifacts
        if (Test-Path "artifacts/bang_tin_alpha.md") {
            Get-Content "artifacts/bang_tin_alpha.md"
        }
    }
    "fix" {
        py scripts/fix_alpha.py
    }
    "seed" {
        py scripts/approve_seeds.py --input "artifacts/lo_tiep_theo.json" --top 4 --format markdown
    }
    "test" {
        py scripts/doctor.py --mode windows --require-env --require-deps
        if ($LASTEXITCODE -ne 0) {
            exit $LASTEXITCODE
        }
        py main.py --mode test
        if ($LASTEXITCODE -ne 0) {
            exit $LASTEXITCODE
        }
        Refresh-Artifacts
    }
    "careful" {
        py scripts/doctor.py --mode windows --require-env --require-deps
        if ($LASTEXITCODE -ne 0) {
            exit $LASTEXITCODE
        }
        py scripts/research_cycle.py --profile careful --rounds 1 --cooldown-seconds 300
        if ($LASTEXITCODE -ne 0) {
            exit $LASTEXITCODE
        }
    }
    "smart" {
        py scripts/doctor.py --mode windows --require-env --require-deps
        if ($LASTEXITCODE -ne 0) {
            exit $LASTEXITCODE
        }
        py scripts/research_cycle.py --profile smart --rounds 1 --cooldown-seconds 120
        if ($LASTEXITCODE -ne 0) {
            exit $LASTEXITCODE
        }
    }
    "light" {
        py scripts/doctor.py --mode windows --require-env --require-deps
        if ($LASTEXITCODE -ne 0) {
            exit $LASTEXITCODE
        }
        py scripts/research_cycle.py --profile light --rounds 1 --cooldown-seconds 180
        if ($LASTEXITCODE -ne 0) {
            exit $LASTEXITCODE
        }
    }
    "run" {
        py scripts/doctor.py --mode windows --require-env --require-deps
        if ($LASTEXITCODE -ne 0) {
            exit $LASTEXITCODE
        }
        py scripts/research_cycle.py --profile light --rounds 1 --cooldown-seconds 180
        if ($LASTEXITCODE -ne 0) {
            exit $LASTEXITCODE
        }
    }
    "cycle" {
        py scripts/doctor.py --mode windows --require-env --require-deps
        if ($LASTEXITCODE -ne 0) {
            exit $LASTEXITCODE
        }
        py scripts/research_cycle.py --profile light --rounds 1 --cooldown-seconds 180
        if ($LASTEXITCODE -ne 0) {
            exit $LASTEXITCODE
        }
    }
    "full" {
        py scripts/doctor.py --mode windows --require-env --require-deps
        if ($LASTEXITCODE -ne 0) {
            exit $LASTEXITCODE
        }
        py main.py --mode full
        if ($LASTEXITCODE -ne 0) {
            exit $LASTEXITCODE
        }
        Refresh-Artifacts
    }
    "loop" {
        py scripts/doctor.py --mode windows --require-env --require-deps
        if ($LASTEXITCODE -ne 0) {
            exit $LASTEXITCODE
        }
        py scripts/research_cycle.py --profile light --rounds 2 --cooldown-seconds 180
        if ($LASTEXITCODE -ne 0) {
            exit $LASTEXITCODE
        }
    }
}
