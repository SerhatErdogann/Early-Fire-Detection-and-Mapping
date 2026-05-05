# Ablation / experiment runner for flame_fire_project.
#
# Run from the project root (activate .venv first):
#   cd C:\Users\Vıctus\Desktop\bitirme\flame_fire_project
#   .venv\Scripts\activate
#   powershell -ExecutionPolicy Bypass -File scripts\ablations.ps1 -Ablation all
#
# Available ablations:
#   rgb                 RGB baseline (mode=rgb, family=rgb_baseline)
#   thermal             Thermal baseline (mode=thermal, family=thermal_baseline)
#   early_fusion        Early fusion (single 4-ch encoder)
#   dual_branch         Dual-branch fusion (default for --mode fusion)
#   hard_neg_retrain    Dual-branch fusion retrained with last val FPs as hard negatives
#   all                 Run everything in sequence

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

# 0. Always refresh the master index before training.
Run-Cmd "python src/01_build_master_index.py"

function Ablation-RGB() {
    Run-Cmd "python src/02_train.py --mode rgb --model_family rgb_baseline --epochs $Epochs"
}

function Ablation-Thermal() {
    Run-Cmd "python src/02_train.py --mode thermal --model_family thermal_baseline --epochs $Epochs"
}

function Ablation-EarlyFusion() {
    Run-Cmd "python src/02_train.py --mode fusion --model_family early_fusion --epochs $Epochs"
}

function Ablation-DualBranch() {
    # New default when --model_family is omitted, but we pass it explicitly for clarity.
    Run-Cmd "python src/02_train.py --mode fusion --model_family dual_branch_fusion --epochs $Epochs"
}

function Ablation-HardNegRetrain() {
    $fp = "outputs/val_false_positives_fusion_dual_branch_fusion.csv"
    if (!(Test-Path $fp)) {
        Write-Host "Hard-negative CSV not found: $fp. Run the dual_branch ablation first." -ForegroundColor Yellow
        return
    }
    Run-Cmd "python src/02_train.py --mode fusion --model_family dual_branch_fusion --hard_negative_csv `"$fp`" --epochs $Epochs"
}

switch ($Ablation.ToLower()) {
    "rgb"              { Ablation-RGB }
    "thermal"          { Ablation-Thermal }
    "early_fusion"     { Ablation-EarlyFusion }
    "dual_branch"      { Ablation-DualBranch }
    "hard_neg_retrain" { Ablation-HardNegRetrain }
    "all" {
        Ablation-RGB
        Ablation-Thermal
        Ablation-EarlyFusion
        Ablation-DualBranch
        Ablation-HardNegRetrain
    }
    default {
        Write-Host "Unknown ablation '$Ablation'. Use one of: rgb, thermal, early_fusion, dual_branch, hard_neg_retrain, all" -ForegroundColor Red
        exit 1
    }
}
