# reproduce_all.ps1 —— 完整复现脚本（覆盖论文全部实验）
# 用法：  cd <仓库根目录> ;  .\reproduce_all.ps1
# 说明：  分阶段执行；已完成的阶段可注释掉。GPU 串行，总时长约 6–8 小时。
# 依据：  REPRODUCE.md（含数据获取、已知偏差）

$ErrorActionPreference = "Stop"
$r  = $PSScriptRoot
$py = Join-Path $r "venv\Scripts\python.exe"
$env:PYTHONIOENCODING = "utf-8"

# 各案例的评估点与（案例内统一的）轮数预算
$EV40 = @("--eval-epochs","1","2","4","8","16","24","32","40")
$EV24 = @("--eval-epochs","1","2","4","8","16","24")
$EV16 = @("--eval-epochs","1","2","4","8","16")
$S10  = @("--seeds","0","1","2","3","4","5","6","7","8","9")
$TAGS = @(@("--lr","3e-4",""), @("--lr","1e-3","--tag","_strong"), @("--lr","1e-4","--tag","_lr1e4"))

function Run-Case($case, $size, $epochs, $ev, $extra) {
    foreach ($t in $TAGS) {
        $a = @((Join-Path $r "scripts\run_experiment.py"), "--case", $case, "--size", $size,
               "--epochs", $epochs) + $ev + $S10 + $extra + $t
        Write-Host ("[{0}] {1} {2}" -f (Get-Date -Format HH:mm:ss), $case, ($t -join " ")) -ForegroundColor Cyan
        & $py @a
    }
}

# ---------------- 阶段 1：数据准备（原始数据见 REPRODUCE.md 第 1 节） ----------------
Write-Host "== 阶段 1：构建缓存 ==" -ForegroundColor Yellow
foreach ($c in @("medmnist","digits","pacs","officehome","core50","core50all")) {
    & $py (Join-Path $r "scripts\prepare_data.py") $c
}

# ---------------- 阶段 2：主系列三案例（10 种子，batch 64） ----------------
Write-Host "== 阶段 2：主系列 ==" -ForegroundColor Yellow
Run-Case "pacs"   128 40 $EV40 @("--batch","64")
Run-Case "organ"   64 24 $EV24 @("--batch","64","--max-train","12000")
Run-Case "digits"  64 16 $EV16 @("--batch","64","--max-train","12000")

# ---------------- 阶段 3：OfficeHome 外部验证（16 轮 + 40 轮附加条件） ----------------
Write-Host "== 阶段 3：OfficeHome ==" -ForegroundColor Yellow
Run-Case "officehome" 128 16 $EV16 @("--batch","64")
foreach ($t in @(@("--lr","3e-4","--tag","_e40"), @("--lr","1e-3","--tag","_strong_e40"),
                 @("--lr","1e-4","--tag","_lr1e4_e40"))) {
    $a = @((Join-Path $r "scripts\run_experiment.py"), "--case","officehome","--size","128",
           "--epochs","40") + $EV40 + $S10 + @("--batch","64") + $t
    Write-Host ("[40 轮] officehome {0}" -f ($t -join " ")) -ForegroundColor Cyan
    & $py @a
}

# ---------------- 阶段 4：CORe50（官方测试会话）+ 全 11 会话稳健性 ----------------
Write-Host "== 阶段 4：CORe50 ==" -ForegroundColor Yellow
Run-Case "core50"    64 16 $EV16 @("--batch","64","--max-train","12000")
Run-Case "core50all" 64 16 $EV16 @("--batch","64","--max-train","12000")

# ---------------- 阶段 5：分析与校验（确定性；顺序敏感） ----------------
Write-Host "== 阶段 5：分析链与核查 ==" -ForegroundColor Yellow
foreach ($s in @("analyze.py","extra_stats.py","theory_fit.py","cross_case_model.py",
                 "regularities.py","external_validation.py","external_validation_core50.py")) {
    & $py (Join-Path $r "scripts\$s")
}
& $py (Join-Path $r "scripts\external_validation_any.py") core50all 10 external_validation3.json
& $py (Join-Path $r "scripts\audit_reproducibility.py")
& $py (Join-Path $r "scripts\audit_datasets.py")
Write-Host "== 完成：正文 DOCX/PDF 见工作目录；核查结果见 results/ 与 _check/ ==" -ForegroundColor Green
