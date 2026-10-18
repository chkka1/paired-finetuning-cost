# 复现说明（REPRODUCE）

> 目标：从原始数据重建全部缓存、结果与图表数据。

## 0. 环境

- Python 解释器：`venv\Scripts\python.exe`（依赖见仓库 `requirements.txt`：torch/torchvision、pyarrow、pandas、scipy、matplotlib、Pillow）
- GPU：单卡（本机 RTX 4060 Laptop，8 GB）；`--batch 64` 为统一值
- **正文数据均为实验代码跑出的结果**，各步骤结果保存在 `results/` 下

## 1. 原始数据获取（三类来源）

| 数据集 | 获取方式 | 本机状态 |
|---|---|---|
| PACS | HuggingFace `flwrlabs/pacs`（经 hf-mirror.com）→ `data/pacs.parquet` | **原始文件已不在工作目录** ✗（缓存仍在）→ 需按 `download_hf_dataset.py` 重下 |
| MedMNIST 器官数据 | `torchvision.datasets`（`organamnist/organcmnist/organsmnist`），下载进 torchvision 缓存目录 | 缓存不在项目内 ✗（可由 torchvision 重新下载） |
| 数字（MNIST/EMNIST/SVHN/USPS） | `torchvision.datasets` + 手工 USPS fallback（`prepare_data.py::_usps_manual`） | 同上 |
| OfficeHome | HuggingFace `flwrlabs/office-home` → `data/officehome/data/*.parquet`（1 159 MB） | ✓ 在库 |
| CORe50 | HuggingFace `adrake17/core50` → `data/core50/data/*.parquet`（5 861 MB） | ✓ 在库 |

```powershell
$py = "venv\Scripts\python.exe"
# 下载（hf-mirror 可达；官方端亦可达，见脚本注释）
& $py scripts\download_hf_dataset.py flwrlabs/pacs       data            # PACS（如需重建）
& $py scripts\download_hf_dataset.py flwrlabs/office-home data/officehome
& $py scripts\download_hf_dataset.py adrake17/core50      data/core50
# 由原始数据构建缓存（data/cases/*.npz）
& $py scripts\prepare_data.py medmnist
& $py scripts\prepare_data.py digits
& $py scripts\prepare_data.py pacs
& $py scripts\prepare_data.py officehome
& $py scripts\prepare_data.py core50        # 官方测试会话 s3/s7/s10
& $py scripts\prepare_data.py core50all     # 全 11 会话（稳健性检验）
```

## 2. 运行（`reproduce_all.ps1` 已含全部命令）

主系列三案例（10 种子、batch 64、cap 12 000/5 000）：
PACS 128/40 轮、器官 64/24 轮、数字 64/16 轮；每案例 3 档学习率。

外部验证：
- OfficeHome 128/16 轮（3 档 ×10 种子）+ 128/40 轮（标签 `_e40` 等，附加预算条件）
- CORe50 64/16 轮（3 档 ×10 种子）；`core50all` 同设置（11 会话稳健性）

```powershell
& .\reproduce_all.ps1        # 预计总时长 6–8 小时（GPU 串行）
```

## 3. 分析与校验（确定性）

```powershell
$py = "venv\Scripts\python.exe"
& $py scripts\analyze.py                  # 主系列（10 种子 + 案例内过滤）
& $py scripts\extra_stats.py              # 关键增量、判据、标准化诊断
& $py scripts\theory_fit.py               # 距离-代价模型拟合（保留在仓库）
& $py scripts\cross_case_model.py         # 跨案例检验（保留在仓库）
& $py scripts\regularities.py             # 五数据集四条规律
& $py scripts\external_validation.py      # OfficeHome（OfficeHome 外部验证）
& $py scripts\external_validation_core50.py   # CORe50（CORe50 外部验证）
& $py scripts\external_validation_any.py core50all 10 external_validation3.json   # 全 11 会话
& $py scripts\audit_reproducibility.py    # 复现性审计（数据/分析/训练/脚本覆盖）
```

## 4. 已实测的复现性结论（2026-10 实测，含对早期判断的更正）

| 环节 | 结果 |
|---|---|
| **原始数据** | **全部在库** ✓：`data/pacs.parquet`（191.4 MB，hf-mirror）、`data/officehome/data/*.parquet`（1 159 MB）、`data/core50/data/*.parquet`（5 861 MB）、`data/raw/digits/`（MNIST/EMNIST/SVHN，562 MB）、`data/raw/medmnist/`（organ 三切面 npz） |
| **缓存可重建性** | **六个缓存全部逐位一致重建** ✓✓（`prepare_data.py` 重跑后 npz 的 sha256 与备份完全相同）→ 见 `results/reproducibility_rebuild.json` |
| **分析链确定性** | **7/7 脚本重跑后 JSON 哈希完全一致** ✓✓ |
| **训练确定性** | **同配置连跑两次，逐位一致（最大绝对差 0.000e+00）** ✓✓ → 脚本内已设 `random/numpy/torch` 全部种子 + `cudnn.deterministic=True`、`cudnn.benchmark=False` |
| 更正说明 | 早期审计曾报告"训练数值非位级一致" ✗ —— 实测表明该差异**全部来自已存结果按两位小数写入的舍入**，并非随机性 ✓（重新跑两次的可比数值完全一致） |

## 5. 位级可复现的边界与方案

**已做到**：同机、同环境、同命令 → **位级一致** ✓（本机实测）。

**未保证**：跨机型/跨驱动的位级一致（CUDA 内核实现与 cuDNN 版本差异）✗。若需进一步收紧，按以下顺序加：

1. `torch.use_deterministic_algorithms(True, warn_only=True)`（可能在个别算子上报警并回退）；
2. 环境变量 `CUBLAS_WORKSPACE_CONFIG=:4096:8`（CUDA ≥ 10.2 使用 cuBLAS 确定性路径）；
3. 固定并记录 **torch/CUDA/cuDNN/驱动版本与 GPU 型号**（本机：torch 2.5.1+cu121、CUDA 12.1、**RTX 4060 Laptop**）；
4. 仓库写明：**允许的复现容差**（建议：精度变化 |Δ| ≤ 0.2 个百分点；本机实测为 0）✓。

## 6. 已知偏差（随稿说明）

1. **精度记录**：既有结果的 `acc_ridge` 按两位小数写入 → 每种子的 Δacc 有 **1 个百分点**量化粒度（10 种子均值 0.1 个百分点）。写入格式已修为四位小数，但**既有运行未重跑** ✗。
2. **测试集抽样上限**：每域 5 000 张（分层抽样，种子 0）→ 与全量评测的绝对精度可能略有差别。
3. **USPS 未纳入**：数字案例最终使用 mnist/emnist/svhn 三域；`_usps_manual` 的 TLS 校验失败 ✗（不影响最终结果，但复现脚本会打印该错误）。
4. **数据获取依赖镜像**：`hf-mirror.com` 为第三方镜像（官方端亦可达）；`huggingface.co`、`madm.dfki.de` 在本机不可达。
