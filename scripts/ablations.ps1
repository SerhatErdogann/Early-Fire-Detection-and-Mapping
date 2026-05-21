# Lightweight training presets for flame_fire_project (production gated fusion only).
#
# Run from the project root (activate .venv first):
#   cd C:\Users\Vıctus\Desktop\bitirme\flame_fire_project
#   .venv\Scripts\activate
#   powershell -ExecutionPolicy Bypass -File scripts\ablations.ps1 -Ablation gated
#
# Available runs:
#   gated              Standard dual_branch_gated_fusion train -> models/dual_branch.pt
#   hard_neg_retrain   Retrain with hard negatives CSV (if present)
#   all                gated then hard_neg_retrain (if CSV exists)
#

param(
    [string]$Ablation = "all",
    [int]$Epochs = 20
)

function Run-Cmd($cmd) {
    Write-Host ""
    Write-Host ">>> $cmd" -ForegroundColor Cyan
    Invoke-Expression $cmd
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Command failed: $cmd" -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

Run-Cmd "python src/01_build_master_index.py"

function Ablation-Gated() {
    Run-Cmd "python src/02_train.py --mode fusion --model_family dual_branch_gated_fusion --epochs $Epochs"
}

function Ablation-HardNegRetrain() {
    $fp = "outputs/val_false_positives_fusion_dual_branch_gated_fusion.csv"
    if (!(Test-Path $fp)) {
        Write-Host "Hard-negative CSV not found: $fp. Create it or skip this step." -ForegroundColor Yellow
        return
    }
    Run-Cmd "python src/02_train.py --mode fusion --model_family dual_branch_gated_fusion --hard_negative_csv `"$fp`" --epochs $Epochs"
}

switch ($Ablation.ToLower()) {
    "gated"            { Ablation-Gated }
    "hard_neg_retrain" { Ablation-HardNegRetrain }
    "all" {
        Ablation-Gated
        Ablation-HardNegRetrain
    }
    default {
        Write-Host "Unknown ablation '$Ablation'. Use one of: gated, hard_neg_retrain, all" -ForegroundColor Red
        exit 1
    }
}
