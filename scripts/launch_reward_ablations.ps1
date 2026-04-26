param(
    [switch]$AllowActive
)

$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

$appList = uv run --extra modal modal app list | Out-String
Write-Host $appList
if (-not $AllowActive -and $appList -match "CyberSecur" -and $appList -match "ephemeral") {
    throw "Active CyberSecurity_OWASP Modal apps are present. Re-run with -AllowActive only if overlapping L4 jobs are intentional."
}

$runs = @(
    @{
        Variant = "abl-a0-sparse"
        Config = "training/configs/reward_ablations/A0_sparse_terminal_only.yaml"
        Seed = 110000
    },
    @{
        Variant = "abl-a2-shape035"
        Config = "training/configs/reward_ablations/A2_reduced_shaping.yaml"
        Seed = 120000
    },
    @{
        Variant = "abl-a6-visgate"
        Config = "training/configs/reward_ablations/A6_visible_gate.yaml"
        Seed = 130000
    },
    @{
        Variant = "abl-a7-evid045"
        Config = "training/configs/reward_ablations/A7_evidence045.yaml"
        Seed = 140000
    },
    @{
        Variant = "abl-a3-nospeed"
        Config = "training/configs/reward_ablations/A3_no_speed_token.yaml"
        Seed = 150000
    }
)

foreach ($run in $runs) {
    Write-Host "Launching $($run.Variant) with $($run.Config) seed $($run.Seed)"
    uv run --extra modal modal run --detach scripts/modal_train_grpo.py `
        --mode train `
        --max-steps 60 `
        --dataset-size 32 `
        --num-generations 4 `
        --max-completion-length 768 `
        --difficulty 0 `
        --split train `
        --source-mode local `
        --trace-log-every 5 `
        --seed-start $run.Seed `
        --reward-config $run.Config `
        --reward-variant $run.Variant `
        --detach
}
