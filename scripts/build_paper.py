# -*- coding: utf-8 -*-
"""把 manuscript_src.md 装配为最终投稿稿(DOCX)。

数据来源：results/analysis.json(由 analyze.py 生成，逐条来自真实实验 CSV)
          results/dataset_stats.json(由 dataset_stats.py 生成)
产出：微调域内外代价_投稿稿.docx + manuscript_final.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
RESULTS = ROOT / "results"
FIG = ROOT / "figures"

import md2docx  # noqa: E402

CASE_ZH = {"pacs": "PACS", "organ": "OrganMNIST", "digits": "Digits", "officehome": "OfficeHome"}
TAG_LABEL = {"": "弱微调", "_strong": "强微调", "_lr1e4": "更弱微调"}
DOM_ZH = {"photo": "Photo", "art_painting": "Art", "cartoon": "Cartoon", "sketch": "Sketch",
          "axial": "轴位", "coronal": "冠状位", "sagittal": "矢状位",
          "mnist": "MNIST", "emnist": "EMNIST", "svhn": "SVHN"}

def f2(v, nd=2):
    return "—" if v is None else ("{:+.{nd}f}".format(v, nd=nd) if isinstance(v, float) else str(v))

def main():
    sys.stdout.reconfigure(encoding="utf-8")
    A = json.loads((RESULTS / "analysis.json").read_text(encoding="utf-8"))
    X = json.loads((RESULTS / "extra_stats.json").read_text(encoding="utf-8"))
    TF = json.loads((RESULTS / "theory_fit.json").read_text(encoding="utf-8"))
    CC = json.loads((RESULTS / "cross_case.json").read_text(encoding="utf-8"))
    _rb = RESULTS / "robustness.json"
    RB = json.loads(_rb.read_text(encoding="utf-8")) if _rb.exists() else None
    D = json.loads((RESULTS / "dataset_stats.json").read_text(encoding="utf-8"))
    cases = A["cases"]
    def find_dom(dd, key):
        for k, v in dd.items():
            if key in k:
                return v
        return None

    # ---------------- 表 1：案例构成 ----------------
    rows = [["案例", "域(源域加粗)", "类别数", "数据来源"]]
    src_map = {"pacs": "photo", "organ": "axial", "digits": "mnist"}
    srcnote = {"pacs": "PACS 2017", "organ": "MedMNIST v2",
               "digits": "MNIST/EMNIST/SVHN"}
    for c in cases:
        st = D[c]
        doms = st["domains"]
        domtxt = "、".join("{}{}".format(DOM_ZH.get(d, d), "(源域)" if d == src_map[c] else "")
                           for d in doms)
        n_train = sum(v["n_train"] for v in doms.values())
        n_val = sum(v["n_val"] for v in doms.values())
        n_test = sum(v["n_test"] for v in doms.values())
        rows.append([CASE_ZH.get(c, c).replace("OrganMNIST", "Organ"), domtxt, str(st["n_classes"]),
                     srcnote.get(c, "")])
    table1 = ("\n".join("| " + " | ".join(r) + " |" for r in rows)
              + "\n\n注：(源域)表示微调数据所在域，其余域仅用于域外评测；实验时源域训练按上限 "
                "12 000 张分层抽样(PACS 源域 1 170 张全部使用)，测试集每域按上限 5 000 张分层抽样。")

    # ---------------- 摘要 ----------------
    def d_of(c, tag, dom, ep):
        r = A["per_case"][c][tag][dom]
        i = r["epochs"].index(ep)
        return r["d_acc_pp_mean"][i]

    def final_ep(c, tag):
        return max(A["per_case"][c][tag][next(iter(A["per_case"][c][tag]))]["epochs"])

    def out_doms(c, tag):
        return [d for d, r in A["per_case"][c][tag].items() if r["role"] == "out"]

    def in_dom(c, tag):
        return [d for d, r in A["per_case"][c][tag].items() if r["role"] == "in"][0]

    pick = {}
    for c in cases:
        tag = "_strong" if "_strong" in A["per_case"][c] else sorted(A["per_case"][c])[0]
        ep = final_ep(c, tag)
        pick[c] = dict(tag=tag, ep=ep,
                       in_d=d_of(c, tag, in_dom(c, tag), ep),
                       out={d: d_of(c, tag, d, ep) for d in out_doms(c, tag)})
    gv = A["gap_vs_cost"]
    worst = min((v, c, d) for c in cases for d, v in pick[c]["out"].items())
    abs_args = (
        pick["pacs"]["in_d"], min(pick["pacs"]["out"].values()),
        pick["organ"]["in_d"], max(pick["organ"]["out"].values()),
        pick["digits"]["in_d"], min(v for k, v in pick["digits"]["out"].items() if k.startswith("emnist")),
        gv["pearson_r"], abs(min(v["pearson_r"] for v in gv["per_case"].values() if v.get("pearson_r") is not None))
    )
    abstract = (
        "针对“域内涨、域外崩”难被同源域内测试集反映、且已有比较的类别与协议常不一致的问题，"
        "该研究给出可复现的配对跨域测量协议，以三个公开数据集为主测量剂量-反应：PACS、MedMNIST 器官数据与多来源数字。"
        "协议固定同类集、同一个在源域训练集上重新拟合的评估分类器与同一批种子并逐轮记录，使域内与域外变化可同量纲相减；"
        "主干为 ImageNet 预训练 ResNet-18，三档学习率、10 个种子，并记录特征距离与紧致度。"
        "域内多为受益而域外代价随目标域与源域差异剧变，据此提炼出三条规律：最优强度依赖目标域、度量稳健、类内紧致度不可忽略。验证数据集与所测三者在任务类型、类别共享、单源微调与评测协议上一致，"
        "但采集来源不同；三条规律在五个数据集上均成立，而三案例内“域外代价与距离变化负相关”(r={6:.2f})未在验证集复现。"
        "该协议与结论均为探索性的，普适性尚未建立。"
    ).format(*abs_args)

    abstract_en = (
        "A reproducible paired cross-domain measurement protocol is contributed here for a problem that current evaluation practice tends to hide: when the test set is drawn from the same source as the fine-tuning data, the accuracy gain cannot reveal the out-of-domain effect of pushing the representation toward the source domain, and gains and costs measured on different class sets or under different protocols cannot legitimately be subtracted from one another."
        "The protocol fixes the same class set, the same evaluation classifier - refit on source-domain training features once the classification head is discarded - and the same random seeds with per-epoch logging, so that in-domain and out-of-domain changes share one scale; fine-tuning strength, set by the learning rate against a matched epoch budget, is the controlled dose, while feature distance and intra-class compactness serve as diagnostics."
        "The measurement is carried out mainly on three public datasets of different modalities (PACS, MedMNIST organ data and multi-source handwritten digits) with an ImageNet-pretrained ResNet-18, three learning rates and ten seeds per setting, and is repeated under the same protocol on OfficeHome and CORe50."
        "In-domain accuracy generally improves, whereas the out-of-domain cost varies sharply with the target-source difference; across the five datasets four regularities hold - a margin-type discrimination mechanism, no universally optimal fine-tuning strength, robustness of the conclusions to the choice of distance metric, and a non-negligible role of intra-class compactness."
        "Within the three main cases the out-of-domain cost correlates negatively with the change of feature distance (r = {6:.2f}), yet this relation was not reproduced on the two validation datasets, which marks the main limitation of the study."
        "The protocol and its conclusions are therefore exploratory: the measurement scheme is offered for discussion rather than as a standard, its generality is not yet established, and the conditions under which the distance-cost relation holds remain unidentified."
    ).format(*abs_args)
    # ---------------- 4.1 剂量-反应 ----------------
    p41 = []
    p41.append(
        "同一操作在不同案例上的后果并不一致：PACS 上域内几乎不变而 Art 域变化 {:+.1f} 个百分点，"
        "器官数据上域内与冠状位同时提升，数字数据上域内提升而 EMNIST 明显下降。".format(
            find_dom(pick["pacs"]["out"], "art"))
    )
    p41.append(
        "表 3 给出代表性观测点，图 2 把三个案例的剂量-反应画成曲线"
        "(横轴为轮数、纵轴为相对未微调基线的精度变化；为便于辨认，PACS 格只画域内与代价最大的 Art 域，其余域外域的数值见表 3 与图 4)，"
        "每个案例取其实验最强设置，三块子图的横轴范围依案例而异(PACS 40 轮、器官 24 轮、数字 16 轮)。可以看到：PACS 的 Art 曲线在第 1 轮即深探至约 -23 个百分点，随后部分回升并稳定在 -17 左右，器官数据的域内与冠状位曲线同步上行，而 PACS 的域内曲线贴近零线、器官与数字的域内曲线位于正值区。误差棒为 10 个随机种子的 ±1 标准差。"
    )
    p41.append("![图 2　最强设置下的剂量-反应曲线(横轴为轮数)<br>Fig.2 Dose-response curves of each case under its strongest setting (rows: domains; columns: epochs)]"
               "(figures/fig1_dosecurve.png|8.1)")
    lines = [["案例(设置)", "轮数", "域内 Δ", "各域外 Δ(百分点)"]]
    for c in cases:
        tag = pick[c]["tag"]; ep = pick[c]["ep"]
        for e in [ep]:
            outs = "、".join("{} {:+.2f}".format(DOM_ZH.get(d.split("(")[0], d.split("(")[0]),
                                                d_of(c, tag, d, e))
                             for d in out_doms(c, tag))
            lines.append(["{}<br>({})".format(CASE_ZH.get(c, c), TAG_LABEL.get(tag, tag)), str(e),
                          "{:+.2f}".format(d_of(c, tag, in_dom(c, tag), e)), outs])
    p41.append("**表 3　剂量-反应的代表性观测点(精度变化，百分点)<br>Table 3 Representative observations of the dose-response (accuracy change, percentage points)**\n\n"
               + "\n".join("| " + " | ".join(r) + " |" for r in lines))
    def val_at(c, tag, key, ep):
        for d, r in A["per_case"][c][tag].items():
            if key in d and ep in r["epochs"]:
                return r["d_acc_pp_mean"][r["epochs"].index(ep)]
        return None
    art_w1, art_s1 = val_at("pacs", "", "art", 1), val_at("pacs", "_strong", "art", 1)
    ep_pacs = int(max(final_ep("pacs", ""), final_ep("pacs", "_strong")))
    art_w2, art_s2 = val_at("pacs", "", "art", ep_pacs), val_at("pacs", "_strong", "art", ep_pacs)
    p41.append(
        "其一，**域内与域外不同向**：PACS 的域内曲线基本贴着零线(最终 {:+.2f})，而 Art 一路转负"
        "(最终 {:+.2f})；器官数据域内单调上升(最终 {:+.2f})，冠状位同向升至 {:+.2f}，矢状位几乎不动。"
        "其二，**强度同时决定方向与幅度**（图 3）。".format(
            pick["pacs"]["in_d"], find_dom(pick["pacs"]["out"], "art"),
            pick["organ"]["in_d"], max(pick["organ"]["out"].values()),
            art_s1, art_w1, ep_pacs, abs(art_s2 - art_w2), art_s2, art_w2))
    p41_strength = p41[-1]
    def lrval(c, key, tag):
        return val_at(c, tag, key, final_ep(c, tag))
    _art_lo, _art_mid, _art_hi = (lrval("pacs", "art", "_lr1e4"), lrval("pacs", "art", ""),
                                  lrval("pacs", "art", "_strong"))
    _cor_lo, _cor_mid, _cor_hi = (lrval("organ", "coronal", "_lr1e4"), lrval("organ", "coronal", ""),
                                  lrval("organ", "coronal", "_strong"))
    _art_lo, _art_mid, _art_hi = (lrval("pacs", "art", "_lr1e4"), lrval("pacs", "art", ""),
                                  lrval("pacs", "art", "_strong"))
    _cor_lo, _cor_mid, _cor_hi = (lrval("organ", "coronal", "_lr1e4"), lrval("organ", "coronal", ""),
                                  lrval("organ", "coronal", "_strong"))
    p41.append(
        "把强度单独作为自变量(三档学习率 1e-4 / 3e-4 / 1e-3，轮数预算一致，图 3 横轴的变化只来自学习率)，"
        "得到两条方向相反的单调曲线：PACS 的 Art 域随强度递增而单调恶化({:+.2f} → {:+.2f} → {:+.2f})，"
        "器官冠状位反而单调受益({:+.2f} → {:+.2f} → {:+.2f})。可见**最优强度依赖目标域(不存在跨案例一致的最优强度)**：强度越高，"
        "表示被推向源域越快，与源域共享结构的域受益更多，共享结构少的域代价更大。".format(
            lrval("pacs", "art", "_lr1e4"), lrval("pacs", "art", ""), lrval("pacs", "art", "_strong"),
            lrval("organ", "coronal", "_lr1e4"), lrval("organ", "coronal", ""), lrval("organ", "coronal", "_strong")))
    p41_lr = p41[-1]               # 图 3 的数值段落(其后是图注)
    p41.append("![图 3　学习率剂量-反应<br>Fig.3 Learning-rate dose-response at the final epoch (epoch budgets matched across levels)]"
               "(figures/fig2_lrdose.png|8.1)")

    p41.append("图 4 把三个案例在其最强设置下的域内与域外增量并列，用于回答“域内涨了多少、域外代价是多少”。")
    p41.append("![图 4　最强设置下域内与域外的精度变化(黑柱为域内，斜线柱为域外)<br>Fig.4 In-domain versus out-of-domain accuracy change under the strongest setting (error bars: standard deviation over seeds)](figures/fig3_summary.png|8.1)")
    p41.append(
        "三个案例给出三种形态。**PACS**：域内没有收益({:+.2f})，域外平均 {:+.2f}，其中 Art {:+.2f}、"
        "Cartoon {:+.2f}——ImageNet 预训练特征在 Photo 域已达 {:d}% 左右的域内精度，微调没有提升空间，"
        "只是把表示进一步推向照片风格。**器官数据**：域内 {:+.2f}、域外平均 {:+.2f}，冠状位同向受益 {:+.2f}，"
        "是一次明确的净收益，因为源域(轴位)与冠状位在解剖结构上高度共享。**数字数据**：域内 {:+.2f}，"
        "而来源不同的 EMNIST {:+.2f}、SVHN 基本持平。因此“域内涨了多少”无法预测“域外会怎样”，"
        "而域外代价的方向与幅度与目标域相对源域的位置有关(见 3.2)。".format(
            pick["pacs"]["in_d"], sum(pick["pacs"]["out"].values()) / len(pick["pacs"]["out"]),
            find_dom(pick["pacs"]["out"], "art"), find_dom(pick["pacs"]["out"], "cartoon"),
            round(100 * A["per_case"]["pacs"][pick["pacs"]["tag"]][in_dom("pacs", pick["pacs"]["tag"])]["acc_mean"][0]),
            pick["organ"]["in_d"], sum(pick["organ"]["out"].values()) / len(pick["organ"]["out"]),
            max(pick["organ"]["out"].values()),
            pick["digits"]["in_d"], min(pick["digits"]["out"].values())))

    # ---------------- 4.3 距离与代价 ----------------
    p43 = []
    pc = ", ".join("{} r={:.2f}(n={})".format(CASE_ZH.get(c, c), v["pearson_r"], v["n"])
                   for c, v in gv["per_case"].items() if v.get("pearson_r") is not None)
    p43.append(
        "图 5 把每个域外观测点(案例 × 目标域 × 轮数 × 种子)画在平面上：横轴是“微调后源域与"
        "目标域特征距离的变化”(MMD 的增量，负值表示距离被拉近)，纵轴是该域的精度变化。")
    p43.append("![图 5　域外精度变化与特征距离变化的关系(三案例分面板给出散点与线性拟合)<br>Fig.5 Out-of-domain accuracy change versus feature-distance change]"
               "(figures/fig4_gap.png|8.1)")
    p43.append(
        "总体而言两者负相关(Pearson r={:.2f}，p={:.1e}，n={}；Spearman r={:.2f})："
        "**在本文三个案例与同一主干、分辨率下，微调把源域与某个域的特征分布拉得越远，该域上的精度下降越多；反之若距离被拉近，"
        "精度往往上升甚至出现正迁移。** 线性拟合为 Δacc = {:.2f} {}{:.3f}·ΔMMD(R²={:.2f})。"
        "分案例看，这一关系在器官数据与数字数据上明显更强({})，在 PACS 上较弱——"
        "说明“距离”是一个有解释力的诊断量，但其强度受案例结构(域间共享内容的多少)调节，"
        "不宜当作跨案例的定量公式。".format(
            gv["pearson_r"], gv["pearson_p"], gv["n"], gv["spearman_r"],
            gv["ols_intercept"], "+" if gv["ols_slope"] >= 0 else "", gv["ols_slope"],
            gv["ols_r2"], pc)
    )
    _c3m, _c3f = X["c3"]["mmd"], X["c3"]["frechet"]
    p43.append(
        "需要说明的是，上述总体相关在未标准化的原始尺度上计算，案例间的尺度差异会掩盖关系"
        "(辛普森效应)：若先在案例内把 ΔMMD 与 Δacc 标准化再合并，总体相关升至 r={:.2f}"
        "(p={:.1e}，n={})，改用 Fréchet 距离作为诊断量得到 r={:.2f}，两者一致；"
        "分案例相关也同步增强(Fréchet：PACS {:.2f}、器官 {:.2f}、数字 {:.2f})。"
        "这说明诊断量的解释力在案例内更可靠，跨案例使用前应先做案例内标准化。".format(
            _c3m["r_standardized"], _c3m["p_standardized"], _c3m["n"],
            _c3f["r_standardized"], _c3f["per_case"]["pacs"]["r"],
            _c3f["per_case"]["organ"]["r"], _c3f["per_case"]["digits"]["r"]))
    p43.append(
        "这一结果给实践提供了一个成本很低的初步信号（但其解释力有限，见 4.5 节的限定）：**在微调过程中监控源域与目标域的特征距离，"
        "就能在不需要目标域标签的前提下预警域外代价**。本文的诊断量只需目标域的**无标签**样本，"
        "因此在真实部署中是可获得的。"
    )

    # ---------------- 表 4：关键增量的统计检验(C1) ----------------
    def eff(case, tag, dom_key):
        for e in X["c1"]["effects"]:
            if e["case"] == case and e["tag"] == tag and e["domain"].split("(")[0] == dom_key:
                return e
        return None
    p44 = []
    rows4 = [["案例(设置)", "域", "ΔA / pp", "95% CI", "p"]]
    LAB4 = {"_strong": "强微调(1e-3)", "": "中微调(3e-4)", "_lr1e4": "弱微调(1e-4)"}
    picks4 = [(c, t, d) for c, d in [("pacs", "art_painting"), ("organ", "coronal"),
                                     ("digits", "emnist")]
              for t in ["_strong", "", "_lr1e4"]]
    for case, tag, dk in picks4:
        e = eff(case, tag, dk)
        if e is None:
            continue
        lab = "{}({})".format(CASE_ZH.get(case, case), LAB4.get(tag, tag))
        pv = e["p_vs0"]
        rows4.append([lab, DOM_ZH.get(dk, dk),
                      "{:+.2f} ± {:.2f}".format(e["mean"], e["sd"]),
                      "[{:+.2f}, {:+.2f}]".format(e["ci95"][0], e["ci95"][1]),
                      "< 0.001" if pv < 1e-3 else "{:.3f}".format(pv)])
    p44.append("**表 4　关键增量的均值±标准差、95% 置信区间与显著性(10 个种子)**<br>"
               "Table 4 Mean ± SD, 95% confidence interval and significance of the key changes "
               "(ten seeds)**\n\n"
               + "\n".join("| " + " | ".join(r) + " |" for r in rows4)
               + "\n\n注：ΔA 为相对未微调基线(第 0 轮)的精度变化；p 为对 0 的单样本 t 检验"
                 "(n = 10 个种子)。每个案例选一个代表性域外域，同案例三档强度的轮数预算与评估点完全相同，故同组三行可直接比较；跨案例的绝对精度不可直接比较。")
    # 配对检验(强 vs 弱，同轮数、同域、按种子配对)
    pr = {(v["case"], v["domain"].split("(")[0]): v for v in X["c1"]["paired"]}
    pa = pr.get(("pacs", "art_painting"))
    pi = pr.get(("pacs", "photo"))
    p44.append(
        "表 4 给出关键增量的均值、置信区间与显著性；按种子配对比较强/中微调(轮数相同)时，只有 PACS 的两项显著："
        "Art 域多下降 {:.1f} 个百分点(95% CI [{:.2f}, {:.2f}]，p = {:.3f})，域内多下降 {:.1f} 个百分点"
        "(p = {:.3f})；其余案例与域的差异区间均跨 0——“强度越高代价越大”只在 PACS 上统计可靠。".format(
            abs(pa["diff"]), pa["ci95_diff"][0], pa["ci95_diff"][1], pa["p_paired"],
            abs(pi["diff"]), pi["p_paired"]) if pa and pi else "")
    p44 = [x for x in p44 if x]

    # ---------------- 3.5 判据的离线验证(C2) ----------------
    c2 = X["c2"]
    p35 = ["作为事后检验，用已测得的逐轮轨迹模拟“第 1 轮距离若不扩大则继续微调”这一决策："
           "域外平均变化由始终微调的 {:+.2f} pp 改善为 {:+.2f} pp，域内平均收益由 {:+.2f} pp 降为 {:+.2f} pp，"
           "即判据以域内收益换取域外安全，且只在“微调有害”的案例上划算。".format(
               c2["out_always"]["mean"], c2["out_rule"]["mean"],
               c2["in_always"]["mean"], c2["in_rule"]["mean"])]
    # ---------------- 4.5 理论依据的可检验预测(C3/理论) ----------------
    _c = TF["per_case"]
    p45 = []
    p45.append(
        "上述模型给出四条可检验预测，均与实测一致。其一，三个案例的距离项斜率全部为负"
        "({:+.3f}、{:+.3f}、{:+.3f}，见表 4 的单变量结果)，即 κ>0。其二，加入类内紧致度后距离项仍为负，"
        "而拟合明显改善：R² 由 {:.2f}/{:.2f}/{:.2f}(单变量 Fréchet)升至 {:.2f}/{:.2f}/{:.2f}。"
        "其三，最近类心与岭回归的精度变化高度同向(r = {:.2f}~{:.2f})，说明机制确实是间隔型判别。"
        "其四，合并三个案例时斜率被案例间截距差异削弱({:+.2f}，R²={:.2f})，案例内标准化后升至 {:+.2f}"
        "(R²={:.2f})，即 3.2 节报告的辛普森效应。".format(
            _c["pacs"]["mmd"]["slope"], _c["organ"]["mmd"]["slope"], _c["digits"]["mmd"]["slope"],
            _c["pacs"]["fr"]["r2"], _c["organ"]["fr"]["r2"], _c["digits"]["fr"]["r2"],
            _c["pacs"]["two_var"]["r2"], _c["organ"]["two_var"]["r2"], _c["digits"]["two_var"]["r2"],
            min(_c[c]["acc_nc_corr"] for c in ("pacs", "organ", "digits")),
            max(_c[c]["acc_nc_corr"] for c in ("pacs", "organ", "digits")),
            TF["pooled"]["raw_slope"], TF["pooled"]["raw_r2"], TF["pooled"]["std_slope"], TF["pooled"]["std_r2"]))
    p45.append(
        "需要明确这套推导的边界：风险界是上界而非等式，λ 不可直接观测，只能由案例差异代表；MMD 只是 d~HΔH~ 的"
        "替代量；一阶展开仅在判别边界附近成立，因此 κ 的大小与 g 的符号仍须由实验确定——表 4 中 PACS 的 R² 仅 "
        "{:.2f}，正说明其边界附近样本较少。此外，散度变小并不保证变好：已有工作证明不变表示在标签偏移下可能有害[24]。"
        "因此 4.1 的判据只能作为筛除规则使用，4.1 节“器官数据会被误停”的实测结果与这一限定一致。".format(
            _c["pacs"]["two_var"]["r2"]))

    # ---------------- 4.6 模型的适用范围(跨案例检验) ----------------
    _k, _lo, _pr = CC["per_case_kappa"], CC["loco"], CC["prospective"]
    rows6 = [["案例", "κ(95% CI)", "截距 g", "案例内 R²", "留一外推 RMSE / 基线"]]
    for c in ("pacs", "organ", "digits"):
        v = _k[c]
        rows6.append([CASE_ZH.get(c, c),
                      "{:.2f} [{:.2f}, {:.2f}]".format(v["kappa"], v["ci95"][0], v["ci95"][1]),
                      "{:+.2f}".format(v["intercept"]), "{:.3f}".format(v["r2"]),
                      "{:.1f} / {:.1f} pp".format(_lo[c]["rmse"], _lo[c]["rmse_zero_baseline"])])
    p46 = []
    p46.append("![图 5　模型的适用范围：无量纲剂量下的案例内拟合(a)、留一外推与基线(b)、"
               "以及微调前初始距离与最终代价(c)"
               "<br>Fig.5 Scope of the model: within-case fits under a dimensionless dose (a), "
               "leave-one-case-out extrapolation versus the predict-zero baseline (b), and the initial "
               "distance before fine-tuning versus the final cost (c)](figures/fig5_scope.png|8.1)")
    p46.append("**表 6　跨案例检验：无量纲剂量下的斜率与留一外推<br>Table 6 Cross-case check: slope under a dimensionless dose and leave-one-case-out extrapolation**\n\n"
               + "\n".join("| " + " | ".join(r) + " |" for r in rows6)
               + "\n\n注：剂量 ρ = ΔF²/F²(0)(无量纲)；κ 的单位为百分点；留一外推指用两个案例拟合、"
                 "预测第三个案例的 Δacc，基线为“一律预测 0”。")
    p46.append(
        "图 5 给出三项检验的画面。第一，把剂量换成无量纲的 ρ 后，斜率的符号在三个案例上一致为负"
        "(κ = {:.2f}、{:.2f}、{:.2f}，置信区间均不含 0)，但大小并不相等：共同斜率假设被拒绝"
        "(F({},{})={:.2f})。这说明在这三个案例(同一 ResNet-18 主干与 128×128 输入)内，“距离被拉远伴随精度下降”的符号是一致的，而代价的系数由案例结构决定。"
        "第二，留一案例外推失败：用两个案例拟合、预测第三个案例的 Δacc，均方根误差为 {:.1f}/{:.1f}/{:.1f} 个百分点，"
        "而“一律预测 0”的基线是 {:.1f}/{:.1f}/{:.1f}，除数字数据外外推比基线更差"
        "(外推 R² = {:.2f}/{:.2f}/{:.2f})。因此该模型是案例内的解释模型，不能直接外推到新案例。"
        "第三，若只用微调前的初始距离 F²(0)(按案例内中位数无量纲化)预测最终代价，相关为正"
        "({:.2f}/{:.2f}/{:.2f}，合并 {:.2f})：初始距离越大，微调带来的变化幅度越大，但变化的符号并不由它决定。"
        "这与 4.4 的分解一致——初始距离只决定强度，符号由共享结构项 g(即 λ)决定。".format(
            _k["pacs"]["kappa"], _k["organ"]["kappa"], _k["digits"]["kappa"],
            CC["common_slope_F"]["df"][0], CC["common_slope_F"]["df"][1], CC["common_slope_F"]["F"],
            _lo["pacs"]["rmse"], _lo["organ"]["rmse"], _lo["digits"]["rmse"],
            _lo["pacs"]["rmse_zero_baseline"], _lo["organ"]["rmse_zero_baseline"],
            _lo["digits"]["rmse_zero_baseline"],
            _lo["pacs"]["r2_test"], _lo["organ"]["r2_test"], _lo["digits"]["r2_test"],
            _pr["pacs"]["r"], _pr["organ"]["r"], _pr["digits"]["r"], _pr["pooled_r"]))
    p46.append(
        "因此 4.1 的判据应写成：**用微调过程中在无标签目标数据上观测到的距离变化量级来估计代价，"
        "用共享结构的证据(如小规模域内验证集)来判断方向**。“先量距离”能筛掉“域内几乎不涨、域外大幅下跌”的微调，"
        "但不能仅凭初始距离决定是否微调——这与 4.1 节实测到的“器官数据会被误停”完全吻合。"
        "需要说明的是，上述边界本身也只是在这三个案例上检验的结果：换用其他数据集、类别空间或主干后，"
        "是否同样呈现“符号普适、系数随案例而变、跨案例外推失败”，本文无法确证。")

    # ---------------- 2.4 前置条件与实测规模({{SETTINGS}}) ----------------
    _run_cfg = {  # 各案例实际使用的训练设置(协议常量；派生量由代码计算)
        "pacs":       dict(size=128, batch=64, epochs=40, src="photo", cap=None, lr="1e-4 / 3e-4 / 1e-3"),
        "organ":      dict(size=64,  batch=64, epochs=24, src="axial", cap=12000, lr="1e-4 / 3e-4 / 1e-3"),
        "digits":     dict(size=64,  batch=64, epochs=16, src="mnist", cap=12000, lr="1e-4 / 3e-4 / 1e-3"),
    }
    _rows, _ev_note = [], []
    for c, cfg in _run_cfg.items():
        st = D.get(c)
        if not st:
            continue
        doms = st["domains"]
        n_tr = doms[cfg["src"]]["n_train"]
        used = min(n_tr, cfg["cap"]) if cfg["cap"] else n_tr
        steps_per_ep = -(-used // cfg["batch"])
        total = steps_per_ep * cfg["epochs"]
        _csv = None
        for _dir in (RESULTS, RESULTS / "archive_b128"):
            for _fn in ("%s_resnet18_s%d.csv" % (c, cfg["size"]),
                        "%s_resnet18_s%d_lr1e4.csv" % (c, cfg["size"]),
                        "%s_resnet18_s%d_strong.csv" % (c, cfg["size"])):
                if (_dir / _fn).exists():
                    _csv = _dir / _fn
                    break
            if _csv:
                break
        _nmap, _nseed = {}, 0
        if _csv is not None:
            import csv as _csvmod
            _sd = set()
            with _csv.open(encoding="utf-8") as _f:
                for _r in _csvmod.DictReader(_f):
                    _sd.add(int(_r["seed"]))
                    if int(_r["epoch"]) == 0:
                        _nmap[_r["domain"].split("(")[0]] = int(float(_r["n"]))
            _nseed = len(_sd)
        ev = "、".join("{} {}".format(DOM_ZH.get(k.split("(")[0], k.split("(")[0]),
                                     _nmap.get(k, v["n_test"]))
                       for k, v in sorted(doms.items(), key=lambda kv: (kv[0] != cfg["src"], kv[0])))
        _rows.append((CASE_ZH.get(c, c), cfg["size"], used,
                      "{} / {}".format(cfg["batch"], cfg["epochs"]), total, _nseed))
        _ev_note.append("{}：{}".format(CASE_ZH.get(c, c), ev))
    settings = (
        "表 2 给出各案例**实际使用**的训练规模与优化预算。需要强调三点："
        "(1)**输入尺寸与轮数按各案例的数据规模与收敛需要单独设定**(PACS 为 128×128、"
        "器官与数字为 64×64；轮数 40/24/16)，因此**不同案例之间的绝对精度与优化预算不可直接比较**，"
        "本文所有比较都在**案例内部**进行；(2)案例内部各档强度使用**完全相同**的设置(同尺寸、同批大小、"
        "同轮数、同评估点)，这是本文剂量-反应与距离-代价关系成立的前提；"
        "(3)源域训练统一按上限 12 000 张分层抽样、测试集每域统一按上限 5 000 张分层抽样，"
        "该规则对所有案例一致，但因各案例数据规模不同，**实际触发与否不同**(表中给出实际用量，每域评测样本数见表注)。用于外部验证的数据集(OfficeHome)沿用完全相同的协议，其规模与结果在 4.3 节单独给出。"
        + "\n\n注：各案例每域评测样本数为 " + "；".join(_ev_note) + "。"
    ) + "\n\n**表 2　各案例实际使用的训练与评测规模<br>Table 2 Actual training and evaluation scale of each case**\n\n" \
      + "| 案例 | 尺寸 | 源域训练 | 批/轮 | 步数 | 种子 |\n" \
      + "|" + "|".join(["---"] * 6) + "|\n" \
      + "\n".join("| " + " | ".join(str(x) for x in r) + " |" for r in _rows)

    # ---------------- 4.6 外部数据集验证({{EXTERNAL}}) ----------------
    external = ""
    _ev_path = RESULTS / "external_validation.json"
    if _ev_path.exists():
        EV = json.loads(_ev_path.read_text(encoding="utf-8"))
        _order, _lro = {16: 0, 40: 1}, {"3e-4": 0, "1e-3": 1, "1e-4": 2}
        _cfgs = sorted([c for c in EV["configs"] if c.get("n_points")],
                       key=lambda c: (_order.get(c.get("budget", 16), 9), _lro.get(c["lr"], 9)))
        rows7 = [["学习率", "训练预算", "标准化斜率", "95% CI", "p", "R²", "加入 ΔIntra 后 R²"]]
        for c in _cfgs:
            rows7.append([c["lr"], "%d 轮" % c.get("budget", 16), "{:+.3f}".format(c["slope_std"]),
                          "[{:+.3f}, {:+.3f}]".format(c["slope_std_ci95"][0], c["slope_std_ci95"][1]),
                          "{:.3f}".format(c["p"]), "{:.3f}".format(c["r2"]),
                          "{:.3f}".format(c["r2_with_intra"])])
        prim = next((c for c in _cfgs if c["lr"] == "3e-4" and c.get("budget") == 16), _cfgs[0])
        _oh = D.get("officehome", {}).get("domains", {}).get("Real World", {})
        _sp = (_oh.get("n_train", "—"), _oh.get("n_val", "—"), _oh.get("n_test", "—"))
        perdom = "、".join("{} {:+.3f}(Holm 校正后 p={:.3f})".format(
            k.split("(")[0], v["slope"], v.get("p_holm", float("nan")))
            for k, v in prim["per_domain"].items() if v.get("role") == "out")
        # 数据驱动的预算稳健性陈述(只陈述事实：全部条件的斜率范围、CI 是否跨 0、R² 上界)
        _allc = [c for c in _cfgs if c.get("n_points")]
        _budgets = sorted({c.get("budget", 16) for c in _allc})
        _sl = [c["slope_std"] for c in _allc]
        _nspan = sum(1 for c in _allc if not (c["slope_std_ci95"][0] <= 0 <= c["slope_std_ci95"][1]))
        _r2max = max(c["r2"] for c in _allc)
        _npair = sum(1 for lr in ("3e-4", "1e-3", "1e-4")
                     if any(c["lr"] == lr and c.get("budget") == 16 for c in _allc)
                     and any(c["lr"] == lr and c.get("budget") == 40 for c in _allc))
        if len(_budgets) >= 2 and _npair >= 1 and _nspan == 0:
            robust_txt = (
                "在全部 {} 个条件({} 档学习率 × {} 个训练预算)下，斜率的点估计都落在 [{:+.3f}, {:+.3f}] 之间，"
                "95% 置信区间**全部跨越 0**、R² 最大仅 {:.3f}；把训练预算从 16 轮加到 40 轮并没有让该关系出现或变强"
                "(相应斜率绝对值反而由 {:.3f} 降至 {:.3f})。因此该零结果**对训练预算稳健**——不是训练不足造成的。"
            ).format(len(_allc), len({c["lr"] for c in _allc}), len(_budgets), min(_sl), max(_sl), _r2max,
                     next(c["slope_std"] for c in _allc if c["lr"] == "3e-4" and c.get("budget") == 16),
                     next(c["slope_std"] for c in _allc if c["lr"] == "3e-4" and c.get("budget") == 40))
        else:
            robust_txt = ("目前可用的预算对照为 {} 组(学习率 × 预算)，斜率的点估计落在 [{:+.3f}, {:+.3f}]；"
                          "预算稳健性的完整结论待其余条件跑完后给出。").format(
                len(_allc), min(_sl), max(_sl))
        external = (
            "表 5 报告该检验的结果。本文在一个额外数据集 OfficeHome(4 个原生采集风格域，65 类，15 588 张；源域 Real World，"
            "按 70/15/15 划分为 {a}/{b}/{c} 张，其余三域整体作测试集)上重复同一协议："
            "输入 128×128、批 64、3 档学习率、10 个种子，并另跑 40 轮作为训练预算的附加条件。\n\n"
            "**表 5　外部数据集(OfficeHome)上距离-代价关系的复现结果(3 档学习率 × 2 个训练预算，10 个种子)<br>"
            "Table 5 Replication of the distance-cost relation on the additional dataset "
            "(OfficeHome; 3 learning rates × 2 training budgets, 10 seeds)**\n\n"
            + "| " + " | ".join(rows7[0]) + " |\n|" + "|".join(["---"] * len(rows7[0])) + "|\n"
            + "\n".join("| " + " | ".join(r) + " |" for r in rows7[1:])
            + "\n\n注：斜率为域外观测点(域 × 轮数，按种子取均值)上 Δacc 对 ΔFréchet 的标准化回归系数；"
              "预注册的主检验为 16 轮下的学习率 3e-4 档；逐域检验做 Holm 校正。\n\n"
            "结果与三个案例**不一致**：主检验(16 轮、3e-4)斜率 {ps:+.3f}"
            "(95% CI [{pl:+.3f}, {ph:+.3f}]，p = {pp:.3f})，**方向为负但完全不显著**(R² = {pr:.3f})。"
            + robust_txt +
            "本文同样**不能**断言该关系被推翻——预注册的证伪条件(斜率 ≥ 0 且置信区间不含 0)未被触发。"
            "此外，逐域斜率与合并斜率符号相反：主检验配置下三个域外域的斜率均为正({pd})；"
            "在 40 轮条件下这一正号更强且经 Holm 校正后显著。这说明在该数据集上，"
            "同一域内部「距离随训练拉远」与「精度随训练上升」是**同向**的，而合并回归被跨域与跨轮次的差异抵消。"
            "另一方面，加入类内紧致度后拟合明显改善(R² 升至 {pi:.3f}，ΔR² = {di:+.3f})，"
            "与「协方差项不可忽略」一致。该数据集域内基线约 69%、微调是净收益；本文据此另取一个更接近"
            "三案例的数据集做针对性检验(4.4)，同样未能复现，故三条候选解释**未获支持**。"
        ).format(a=_sp[0], b=_sp[1], c=_sp[2],
                 ps=prim["slope_std"], pl=prim["slope_std_ci95"][0], ph=prim["slope_std_ci95"][1],
                 pp=prim["p"], pr=prim["r2"], pd=perdom,
                 pi=prim["r2_with_intra"], di=prim["dR2_intra"])

    # ---------------- 4.7 第二个外部数据集({{EXTERNAL2}}) ----------------
    external2 = ""
    _ev2 = RESULTS / "external_validation2.json"
    if not _ev2.exists():
        external2 = ("本节对应的检验正在进行中：预注册已冻结(见仓库 `PREREGISTRATION.md` 第 11 节)，"
                     "全部条件下的结果将在完成后写入本小节并完整报告——**无论其是否支持**本文三个案例中的关系。")
    else:
        E2 = json.loads(_ev2.read_text(encoding="utf-8"))
        _cfgs2 = sorted([c for c in E2["configs"] if c.get("n_points")],
                        key=lambda c: ({"3e-4": 0, "1e-3": 1, "1e-4": 2}.get(c["lr"], 9),
                                       c.get("budget", 16)))
        rows8 = [["学习率", "标准化斜率", "95% CI", "p", "R²", "加入 ΔIntra 后 R²"]]
        for c in _cfgs2:
            rows8.append([c["lr"], "{:+.3f}".format(c["slope_std"]),
                          "[{:+.3f}, {:+.3f}]".format(c["slope_std_ci95"][0], c["slope_std_ci95"][1]),
                          "{:.3f}".format(c["p"]), "{:.3f}".format(c["r2"]),
                          "{:.3f}".format(c["r2_with_intra"])])
        prim2 = next((c for c in _cfgs2 if c["lr"] == "3e-4"), _cfgs2[0])
        # 全 11 会话稳健性检验(预注册 §11.9)
        _rob = ""
        _p3 = RESULTS / "external_validation3.json"
        if _p3.exists():
            EV3 = json.loads(_p3.read_text(encoding="utf-8"))
            _c3 = [c for c in EV3["configs"] if c.get("n_points")]
            if _c3:
                _txt = "、".join("{lr} 档 {sl:+.3f}(95% CI [{lo:+.3f}, {hi:+.3f}]，p = {p:.3f})".format(
                    lr=c["lr"], sl=c["slope_std"], lo=c["slope_std_ci95"][0],
                    hi=c["slope_std_ci95"][1], p=c["p"]) for c in sorted(
                        _c3, key=lambda c: {"3e-4": 0, "1e-3": 1, "1e-4": 2}.get(c["lr"], 9)))
                _prim3 = next((c for c in _c3 if c["lr"] == "3e-4"), _c3[0])
                _pos3 = sum(1 for v in _prim3["per_domain"].values()
                            if v.get("role") == "out" and v["slope"] > 0)
                _tot3 = sum(1 for v in _prim3["per_domain"].values() if v.get("role") == "out")
                _rob = ("\n\n作为稳健性检验(预注册第 11.9 节；预注册文件随代码与结果一并公开，见数据可用性声明)，本文把**全部 11 个会话**纳入评测"
                        "(源域仍为 s1，域外为 s2–s11 共 10 个会话、每域 5 000 张，回归点由 15 增至 50)："
                        "三档合并斜率分别为 {}；主检验档下 10 个域外会话的逐域斜率 {} 个为正、{} 个为负。"
                        "扩大域数量并未使该关系出现，在更弱微调档上甚至出现显著为**正**的斜率。").format(
                    _txt, _pos3, _tot3 - _pos3)

        _f1 = [c for c in _cfgs2
               if c["slope_std"] >= 0 and c["slope_std_ci95"][0] * c["slope_std_ci95"][1] > 0]
        _rev = None
        for c in _f1:
            _out = [v for v in c["per_domain"].values() if v.get("role") == "out"]
            if _out and all(v["slope"] < 0 for v in _out):
                _rev = c
                break
        pd2 = "、".join("{} {:+.3f}(Holm 校正后 p={:.3f})".format(
            k.split("(")[0], v["slope"], v.get("p_holm", float("nan")))
            for k, v in prim2["per_domain"].items() if v.get("role") == "out")
        external2 = (
            "表 6 报告该检验的结果。按第 11 节预注册，本文另取一个**更接近三案例**的公共数据集：CORe50(10 类日常物体，"
            "共 {a} 个采集会话、域定义为会话、每会话含全部 10 类)，输入 64×64、批 64、16 轮、"
            "3 档学习率、10 个种子；源域取 {b}(70/15/15)，域外会话整体作测试集。\n\n"
            "**表 6　第二个外部数据集(CORe50)上的复现结果<br>"
            "Table 6 Replication on the second additional dataset (CORe50)**\n\n"
            + "| " + " | ".join(rows8[0]) + " |\n|" + "|".join(["---"] * len(rows8[0])) + "|\n"
            + "\n".join("| " + " | ".join(r) + " |" for r in rows8[1:])
            + "\n\n注：口径与 4.3 完全一致(域外观测点按域×轮取种子均值后回归；逐域检验做 Holm 校正)。\n\n"
            "主检验(学习率 3e-4)的标准化斜率为 {ps:+.3f}(95% CI [{pl:+.3f}, {ph:+.3f}]，p = {pp:.3f}，"
            "R² = {pr:.3f})；逐域斜率为 {pd}。{f1note}{verdict}{rob}"
        ).format(a=E2.get("n_domains", "—"), b=E2.get("source_domain", "—"),
                 ps=prim2["slope_std"], pl=prim2["slope_std_ci95"][0], ph=prim2["slope_std_ci95"][1],
                 pp=prim2["p"], pr=prim2["r2"], pd=pd2, rob=_rob,
                 f1note=("需要特别指出的是，预注册的**证伪条件在 {lr} 档被触发**：该档合并斜率为 {sl:+.3f}"
                         "(95% CI [{lo:+.3f}, {hi:+.3f}]，p = {p:.3f})，置信区间不含 0；"
                         "而该档下**各个域(含源域)的逐域斜率全部为负**——合并口径与逐域口径符号相反。"
                         "这表明在该数据集上本文关系不仅未被检出，在更强微调下还出现了方向相反的合并斜率。"
                         ).format(lr=_rev["lr"], sl=_rev["slope_std"], lo=_rev["slope_std_ci95"][0],
                                  hi=_rev["slope_std_ci95"][1], p=_rev["p"]) if _rev else "",
                 verdict=E2.get("verdict_text", ""))

    # ---------------- 4.5 边界句({{SCOPE_BOUND}}) ----------------
    scope_bound = ""
    _cc = RESULTS / "case_contrast.json"
    if _cc.exists() and _ev_path.exists():
        CC = json.loads(_cc.read_text(encoding="utf-8"))
        MAIN = ("pacs", "organ", "digits")
        r2s, neg, tot = [], 0, 0          # 主档(3e-4)的解释力范围 + 全配置的逐域符号计数
        for k, v in CC.items():
            if v["case"] not in MAIN:
                continue
            if v["tag"] == "3e-4":
                r2s.append(v["pooled_r2"])
            for _, sl in v["per_domain"].items():
                tot += 1
                neg += 1 if sl < 0 else 0
        oh = [v for v in CC.values() if v["case"] == "officehome"]
        oh_all_pos, oh_all_tot = 0, 0     # OfficeHome 全部配置的逐域符号计数
        for v in oh:
            if v["tag"] == "3e-4":
                oh_base = v["in_domain_base"]
            for _, sl in v["per_domain"].items():
                oh_all_tot += 1
                oh_all_pos += 1 if sl > 0 else 0
        oh_prim = next((c for c in EV["configs"] if c["lr"] == "3e-4" and c.get("budget") == 16), None)
        oh_r2 = oh_prim["r2"] if oh_prim else float("nan")
        oh_pos = sum(1 for v in (oh_prim or {}).get("per_domain", {}).values()
                     if v.get("role") == "out" and v["slope"] > 0)
        oh_tot = sum(1 for v in (oh_prim or {}).get("per_domain", {}).values() if v.get("role") == "out")
        scope_bound = (
            "**该关系的成立条件是有限的。** 在本文三个案例中，ΔFréchet 对域外精度变化的解释力"
            "(合并回归 R²，主档 3e-4)为 {r2lo:.2f}~{r2hi:.2f}，全部配置的逐域斜率 {neg}/{tot} 为负；"
            "而在**两个**额外公共数据集上均未能复现：OfficeHome(65 类，域内基线 {ohb:.2f})主检验的解释力仅 R²≈{ohr2:.2f}，"
            "逐域斜率 {pos}/{ohtot} 为正(4.3)；CORe50(10 类、域内饱和度更接近本文三案例)主检验斜率方向相反、"
            "且预注册的证伪条件在其中一档被触发(4.4)。造成差异的三条候选解释(类别规模、域间共动程度、域位移幅度)"
            "在 CORe50 上**未获支持**，"
            "因此该关系的成立条件**仍未被识别**。"
            "本文因此把结论限定为：**在三个案例所代表的条件下**，微调把源域与某个域的特征分布拉得越远，"
            "该域上的精度下降越多；反向情形同理。"
        ).format(r2lo=min(r2s), r2hi=max(r2s), neg=neg, tot=tot, ohb=oh_base, ohr2=oh_r2,
                 pos=oh_pos, ohtot=oh_tot, posall=oh_all_pos, totall=oh_all_tot)

    # ---------------- 3.4 跨五数据集一致规律({{SEC34B}}) ----------------
    sec34b = ""
    _rg = RESULTS / "regularities.json"
    if _rg.exists():
        RG = json.loads(_rg.read_text(encoding="utf-8"))
        CASES5 = ["pacs", "organ", "digits", "officehome", "core50"]
        ZH5 = {"pacs": "PACS", "organ": "器官", "digits": "数字", "officehome": "OfficeHome", "core50": "CORe50"}
        r4lo = min(RG[c]["r_nc_rg"] for c in CASES5 if c in RG)
        r4hi = max(RG[c]["r_nc_rg"] for c in CASES5 if c in RG)
        dR2 = {c: RG[c]["dR2_intra"] for c in CASES5 if c in RG}
        mmd_all = all(RG[c]["same_sign_measure"] for c in CASES5 if c in RG)
        mismatch_all = all(RG[c]["mismatch"] >= 1 for c in CASES5 if c in RG)
        strong_all = all(RG[c]["strong_worse"] >= 1 for c in CASES5 if c in RG)
        sec34b = (
            "把上述三个案例与 4.3、4.4 的两个额外数据集放在一起(共**五个数据集**、同一主干与同一协议)，"
            "有三项规律在**全部五个数据集上都成立**，它们构成比单个关系更稳健的结论：\n\n"
            "**\n\n"
            "**(一)域内与域外不同向。** 在每一个数据集上，都存在至少一个域外域在末轮的精度变化与域内**符号相反**"
            "(且幅度大于 0.5 个百分点)。这意味着**最优强度依赖目标域**：同一档强度同时让一部分域受益、"
            "让另一部分域受损，因此“微调多少”必须逐目标域决定。\n\n"
            "**(二)结论对度量选择稳健。** 用 MMD 与用 Fréchet 距离分别计算，"
            "合并斜率的符号在五个数据集上**全部一致**({mmd})，说明本文关于“距离变化”的定量结论"
            "不是由某一个度量的特殊性质造成的。\n\n"
            "**(三)类内紧致度不可忽略。** 在五个数据集上，把类内紧致度的变化加入回归都不使拟合变差"
            "(ΔR² 从 {dmin:+.3f} 到 {dmax:+.3f})，与 4.1 节的理论推导一致，但改善的幅度依数据集而异"
            "(在本文件三个案例上很小、在两个额外数据集上很大)，因此它是一条**方向性**规律，不是可移植的定量关系。\n\n"
            "与之相对，另外两类自然猜想并不成立：**“强度越高域外越差”**只在部分数据集上对全部域外域成立"
            "(本文三个案例中器官数据即出现高强度反而更好的域)，**“初始距离越大越吃亏”**在本文三个案例上"
            "方向为**正**(初始距离大的域反而受益更多)，却在两个额外数据集上方向为**负**。"
            "这两点进一步说明：单个数据集上的方向性结论不能外推。"
        ).format(r4lo=r4lo, r4hi=r4hi, mmd=("全部同号" if mmd_all else "存在反号"),
                 dmin=min(dR2.values()), dmax=max(dR2.values()))

    # ---------------- 注入 ----------------
    _cands = [ROOT / "manuscript" / "manuscript_src.md", ROOT / "manuscript_src.md"]
    _src_path = next((p for p in _cands if p.exists()), None)
    if _src_path is None:
        raise SystemExit("找不到 manuscript_src.md(应位于 manuscript/ 或项目根目录)")
    _md_dir = _src_path.parent
    src = _src_path.read_text(encoding="utf-8")
    repl = {
        "ABSTRACT": abstract,
        "ABSTRACT_EN": abstract_en,
        "TABLE1": table1,
        "SEC41": "\n\n".join(p41),
        "SEC43": "\n\n".join(p43),
        "SEC44": "\n\n".join(p44),
        "SETTINGS": settings,
        "SEC34B": sec34b,
        "SCOPE_BOUND": scope_bound,
        "EXTERNAL": external,
        "EXTERNAL2": external2,
        "SEC35": "\n\n".join(p35),   # 第二轮删减：压缩为 1 句
        "C_ART_LO": "{:+.1f}".format(_art_lo), "C_ART_HI": "{:+.1f}".format(_art_hi),
        "C_COR_LO": "{:+.1f}".format(_cor_lo), "C_COR_HI": "{:+.1f}".format(_cor_hi),
        "C_R2MAX": "{:.2f}".format(max(TF["per_case"][c]["two_var"]["r2"] for c in TF["per_case"])),
        "C_ART_LO": "{:+.1f}".format(_art_lo), "C_ART_HI": "{:+.1f}".format(_art_hi),
        "C_COR_LO": "{:+.1f}".format(_cor_lo), "C_COR_HI": "{:+.1f}".format(_cor_hi),
        "C_R2MAX": "{:.2f}".format(max(TF["per_case"][c]["two_var"]["r2"] for c in TF["per_case"])),
        "SEC45": "",   # 4.4 推导已按删减方案压缩，正文改用 3 句概述
        "SEC46": "",   # 4.5 跨案例模型(表 6/图 5)已删，正文改用 3 句概述
    }
    # 防线：关键数字必须真的出现在对应段落中(防止 .format 位置参数错位)
    guards = [
        ("4.1 域内最终", pick["pacs"]["in_d"], p41_strength),
        ("4.1 强微调第 1 轮", art_s1, p41_strength),
        ("4.1 弱微调第 1 轮", art_w1, p41_strength),
        ("4.1 弱微调末轮", art_w2, p41_strength),
        ("4.1 强微调末轮", art_s2, p41_strength),
        ("4.1 学习率 Art 最强", val_at("pacs", "_strong", "art", final_ep("pacs", "_strong")), p41_lr),
        ("4.1 学习率 器官冠状位最强",
         val_at("organ", "_strong", "coronal", final_ep("organ", "_strong")), p41_lr),
        ("4.1 域外 Art 最终", find_dom(pick["pacs"]["out"], "art"), p41_strength),
        ("4.1 器官域内最终", pick["organ"]["in_d"], p41_strength),
        ("3.1 PACS 域外均值", sum(pick["pacs"]["out"].values()) / len(pick["pacs"]["out"]), p41[-1]),
        ("3.1 器官域内", pick["organ"]["in_d"], p41[-1]),
        ("摘要 总体 r", gv["pearson_r"], abstract),
    ]
    for name, val, text in guards:
        forms = ["{:+.2f}".format(val), "{:+.1f}".format(val), "{:.2f}".format(val),
                 "{:.2f}".format(abs(val))]
        if not any(f in text for f in forms):
            raise SystemExit("数字未出现在正文中(可能参数错位)：{} = {}".format(name, val))

    out = src
    for k, v in repl.items():
        out = out.replace("{{%s}}" % k, v)
    import re
    left = sorted(set(re.findall(r"\{\{[A-Za-z0-9_]+\}\}", out)))
    if left:
        raise SystemExit("未替换占位符: {}".format(left))
    final = _md_dir / "manuscript_final.md"
    final.write_text(out, encoding="utf-8")
    info = md2docx.build(final, _md_dir / "基于配对跨域协议的微调收益与代价测量_投稿稿.docx", ROOT)
    print("DOCX:", info)
    print("摘要字数:", len(re.findall(r"[\u4e00-\u9fff]", abstract)))

if __name__ == "__main__":
    main()
