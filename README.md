# Paired Fine-tuning Cost

**Measurement of Fine-tuning Gains and Costs Based on a Paired Cross-domain Protocol**

本仓库为论文《基于配对跨域协议的微调收益与代价测量》（中文）的**脚本、结果与核查工具**。
英文题名：MEASUREMENT OF FINE-TUNING GAINS AND COSTS BASED ON A PAIRED CROSS-DOMAIN PROTOCOL

作者：常开开，罗小龙，相龙伟（长江大学地球科学学院）

---

## 1. 目录结构

| 目录/文件 | 内容 |
|---|---|
| `scripts/` | 数据构建、训练、统计、绘图与成稿数字注入脚本（分析链互不覆盖） |
| `results/` | 全部中间结果（JSON/CSV），正文数字均由脚本从这些文件注入 |
| `REPRODUCE.md` | 环境、数据来源、复现命令与已知偏差 |
| `reproduce_all.ps1` | 一键复现全部实验 |
| `_check/` | 逐格核查与参考文献核实清单 |

> 训练缓存（`data/*.npz`，约 12 GB）**不入库**，可用 `prepare_data.py` 按 `REPRODUCE.md` 说明重建（数据源为公开数据集）。

## 2. 环境

- Python 3.9.23（venv）
- torch 2.5.1+cu121，单卡 RTX 4060 Laptop 8 GB
- ResNet-18（ImageNet 预训练）

## 3. 复现

`powershell
python scripts\prepare_data.py      # 构建缓存（需先下载公开数据集）
powershell -File scripts\reproduce_all.ps1
python scripts\audit_tables.py      # 逐格核查
`

复现结论与已知偏差见 `REPRODUCE.md`。正文数字**不手写**，全部由脚本注入并核查。

## 4. 引用

若使用本仓库，请引用上述论文。

## 5. 许可

MIT License（见 `LICENSE`）。
