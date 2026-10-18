# paired-finetuning-cost

《基于配对跨域协议的微调收益与代价测量》的代码与结果。

- scripts/：数据构建、训练、统计、绘图、成稿数字注入与核查脚本
- results/：中间结果（JSON/CSV）
- figures/：论文插图
- REPRODUCE.md：环境、数据来源、复现步骤与已知偏差
- PREREGISTRATION.md：检验前固定的分析口径与证伪条件

环境为 Python 3.9、torch 2.5.1+cu121、单卡 RTX 4060。训练缓存约 12 GB，不入库，
由 prepare_data.py 按说明重建，数据均来自公开数据集。

复现方式：先准备数据，再运行 run_experiment.py 与 analyze.py 等脚本，也可用
reproduce_all.ps1 一次跑完。正文数字均由脚本写入 results/ 后注入，不手工填写。

引用与许可分别见 CITATION.cff 与 LICENSE。
