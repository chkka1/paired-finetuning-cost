# -*- coding: utf-8 -*-
"""极简 Markdown -> DOCX 生成器(中文期刊投稿稿格式，双栏排版)。

设计目标：
1. 稿件正文只有一份人类可读源文件(manuscript_src.md)；
2. 所有 `{{TOKEN}}` 由上游脚本注入，生成时校验是否全部替换，残留即报错；
3. 表格统一输出为三线表(顶线/表头线/底线)，表题在上、图题在下；
   题目/摘要/关键词单栏，正文双栏；宽表与跨栏图自动插入连续分节符改为单栏通栏；
4. 不依赖第三方 markdown 库，只依赖 python-docx。

用法：
    python md2docx.py <src.md> <out.docx>
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
from docx.shared import Cm, Pt, RGBColor, Twips

SONG = "宋体"
HEI = "黑体"
KAI = "楷体"

TOKEN_RE = re.compile(r"\{\{[A-Za-z0-9_]+\}\}")


# --------------------------------------------------------------------------- #
# 基础工具
# --------------------------------------------------------------------------- #
def set_run(run, *, name=SONG, size=10.5, bold=False, italic=False):
    run.font.name = name
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = RGBColor(0, 0, 0)
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    rfonts.set(qn("w:ascii"), "Times New Roman")
    rfonts.set(qn("w:hAnsi"), "Times New Roman")
    rfonts.set(qn("w:eastAsia"), name)
    return run


INLINE_RE = re.compile(r"\*\*(?P<bold>.+?)\*\*|~(?P<sub>[^~]+)~")


def add_rich(par, text, *, name=SONG, size=10.5, bold=False):
    """支持 **加粗**、~下标~ 与 <br> 换行的行内解析(下标用真正的 Word 下标格式)。"""
    if "<br>" in text:
        for li, seg in enumerate(text.split("<br>")):
            if li:
                set_run(par.add_run(), size=size).add_break()
            add_rich(par, seg, name=name, size=size, bold=bold)
        return par
    pos = 0
    for m in INLINE_RE.finditer(text):
        if m.start() > pos:
            set_run(par.add_run(text[pos:m.start()]), name=name, size=size, bold=bold)
        if m.group("bold") is not None:
            set_run(par.add_run(m.group("bold")), name=name, size=size, bold=True)
        else:
            run = set_run(par.add_run(m.group("sub")), name=name, size=size, bold=bold)
            run.font.subscript = True
        pos = m.end()
    if pos < len(text):
        set_run(par.add_run(text[pos:]), name=name, size=size, bold=bold)
    return par


def para(doc, text, *, name=SONG, size=10.5, align=WD_ALIGN_PARAGRAPH.JUSTIFY,
         indent_chars=2.0, space_before=0, space_after=0, line=1.35, bold=False):
    p = doc.add_paragraph()
    p.alignment = align
    pf = p.paragraph_format
    pf.space_before = Pt(space_before)
    pf.space_after = Pt(space_after)
    pf.line_spacing = line
    if indent_chars:
        pf.first_line_indent = Pt(size * indent_chars)
    add_rich(p, text, name=name, size=size, bold=bold)
    return p


def set_cell_border(cell, **kwargs):
    tc_pr = cell._tc.get_or_add_tcPr()
    borders = tc_pr.find(qn("w:tcBorders"))
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        tc_pr.append(borders)
    for edge in ("top", "bottom", "left", "right"):
        spec = kwargs.get(edge)
        if spec is None:
            continue
        tag = "w:{}".format(edge)
        element = borders.find(qn(tag))
        if element is None:
            element = OxmlElement(tag)
            borders.append(element)
        element.set(qn("w:val"), spec.get("val", "single"))
        element.set(qn("w:sz"), str(spec.get("sz", 8)))
        element.set(qn("w:space"), "0")
        element.set(qn("w:color"), spec.get("color", "000000"))


THICK = {"val": "single", "sz": 12}
THIN = {"val": "single", "sz": 6}
NONE = {"val": "nil", "sz": 0}

# A4(21cm × 29.7cm)，左右页边距 2.0cm、栏间距 0.7cm
#   → 通栏宽度 17.0cm；双栏时每栏 (17.0-0.7)/2 = 8.15cm
PAGE_W_CM, PAGE_H_CM = 21.0, 29.7
MARGIN_TB_CM, MARGIN_LR_CM = 2.2, 2.0
COL_SPACE_CM = 0.7
TWIPS_PER_CM = 566.9291
TEXT_TWIPS = int((PAGE_W_CM - 2 * MARGIN_LR_CM) * TWIPS_PER_CM)          # 通栏 17.0 cm
COL_SPACE_TWIPS = int(COL_SPACE_CM * TWIPS_PER_CM)
COL_TWIPS = int((TEXT_TWIPS - COL_SPACE_TWIPS) / 2)                       # 单栏 8.15 cm
FIG_COL_CM = 8.1                                                         # 单栏图推荐宽度
FIG_FULL_CM = 17.0                                                       # 通栏图推荐宽度


def _set_cols(sect_pr, num, space_twips=COL_SPACE_TWIPS):
    """设置该节的栏数(num=1 单栏通栏，num=2 双栏)。"""
    cols = sect_pr.find(qn("w:cols"))
    if cols is None:
        cols = OxmlElement("w:cols")
        sect_pr.insert_element_before(cols, "w:formProt", "w:vAlign", "w:noEndnote",
                                      "w:titlePg", "w:textDirection", "w:bidi",
                                      "w:rtlGutter", "w:docGrid", "w:printerSettings",
                                      "w:sectPrChange")
    cols.set(qn("w:num"), str(num))
    cols.set(qn("w:space"), str(space_twips))
    cols.set(qn("w:equalWidth"), "1")
    return cols


def add_section_break(doc, cols):
    """在当前位置插入连续分节符：其前的段落保持原栏数，其后内容使用 cols 栏。"""
    par = doc.add_paragraph()
    pf = par.paragraph_format
    pf.space_before = Pt(0)
    pf.space_after = Pt(0)
    pf.line_spacing = 1.0
    set_run(par.add_run(""), size=1)
    sect_pr = OxmlElement("w:sectPr")
    typ = OxmlElement("w:type")
    typ.set(qn("w:val"), "continuous")          # 连续分节，不强制换页
    sect_pr.append(typ)
    for tag, w, h in (("w:pgSz", PAGE_W_CM, PAGE_H_CM),):
        el = OxmlElement(tag)
        el.set(qn("w:w"), str(int(w * TWIPS_PER_CM)))
        el.set(qn("w:h"), str(int(h * TWIPS_PER_CM)))
        sect_pr.append(el)
    mar = OxmlElement("w:pgMar")
    mar.set(qn("w:top"), str(int(MARGIN_TB_CM * TWIPS_PER_CM)))
    mar.set(qn("w:bottom"), str(int(MARGIN_TB_CM * TWIPS_PER_CM)))
    mar.set(qn("w:left"), str(int(MARGIN_LR_CM * TWIPS_PER_CM)))
    mar.set(qn("w:right"), str(int(MARGIN_LR_CM * TWIPS_PER_CM)))
    mar.set(qn("w:header"), "720")
    mar.set(qn("w:footer"), "720")
    mar.set(qn("w:gutter"), "0")
    sect_pr.append(mar)
    _set_cols(sect_pr, cols)
    par._p.get_or_add_pPr().append(sect_pr)
    return par


def three_line_table(doc, rows, caption=None):
    """rows[0] 为表头。输出三线表；caption 置于表格上方。"""
    if caption:
        para(doc, caption, name=HEI, size=9, align=WD_ALIGN_PARAGRAPH.CENTER,
             indent_chars=0, space_before=6, space_after=2, line=1.15, bold=False)
    n_cols = len(rows[0])
    table = doc.add_table(rows=0, cols=n_cols)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True
    for r_i, row in enumerate(rows):
        cells = table.add_row().cells
        for c_i, val in enumerate(row):
            cell = cells[c_i]
            cell.text = ""
            p = cell.paragraphs[0]
            # 表头居中；正文中纯数字单元格右对齐(便于按位对齐读数)，其余左对齐
            if r_i == 0:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            elif re.fullmatch(r"[+\-−]?\d+(?:\.\d+)?%?", val.strip()):
                p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
            else:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            pf = p.paragraph_format
            pf.space_before = Pt(1)
            pf.space_after = Pt(1)
            pf.line_spacing = 1.0
            for li, seg in enumerate(str(val).split("<br>")):   # 单元格内 <br> 作为换行
                if li:
                    set_run(p.add_run(), size=8).add_break()
                add_rich(p, seg, name=HEI if r_i == 0 else SONG, size=8, bold=False)
            top = THICK if r_i == 0 else NONE
            bottom = THIN if r_i == 0 else (THICK if r_i == len(rows) - 1 else NONE)
            set_cell_border(cell, top=top, bottom=bottom, left=NONE, right=NONE)
    # 固定为正文宽度，避免宽表溢出页面
    # 表格占单栏宽(与正文同栏)，不再通栏
    # 列宽：先按“最长不可断英文词”给每列下限(避免硬断)，余量再按内容长度分配
    def _vlen(t):
        return sum(2 if ord(ch) > 0x2000 else 1 for ch in str(t))

    def _longest_token(t):
        toks = re.split(r"[^0-9A-Za-z\-+.%]+", str(t))   # 数字与正负号算一个不可断词
        return max((len(x) for x in toks), default=0)

    pad = int(0.13 * TWIPS_PER_CM)                 # 左右内边距余量
    base = [int(0.155 * TWIPS_PER_CM * max(_longest_token(r[ci]) for r in rows)) + pad
            for ci in range(n_cols)]
    if sum(base) >= COL_TWIPS:
        # 列数过多：按比例压缩到栏宽以内(否则最后一列会出现负宽度)
        scale = COL_TWIPS / sum(base)
        widths = [max(int(base[ci] * scale), 1) for ci in range(n_cols)]
        diff = COL_TWIPS - sum(widths)
        if diff > 0:
            widths[-1] += diff
    else:
        extra = COL_TWIPS - sum(base)
        weights = [max(_vlen(r[ci]) for r in rows) if rows else 1 for ci in range(n_cols)]
        widths = [base[ci] + int(extra * weights[ci] / max(sum(weights), 1)) for ci in range(n_cols)]
        widths[-1] += COL_TWIPS - sum(widths)          # 余数补给最后一列
    table.autofit = False
    tbl_pr = table._tbl.tblPr                      # 固定表宽与布局，保证列宽被严格遵守
    tw = tbl_pr.find(qn("w:tblW"))                 # 必须就地更新：python-docx 已插入 tblW(auto)，
    if tw is None:                                 # 再 append 一个会因 schema 顺序被读取器忽略
        tw = OxmlElement("w:tblW")
        tbl_pr.append(tw)
    tw.set(qn("w:w"), str(COL_TWIPS)); tw.set(qn("w:type"), "dxa")
    margins = OxmlElement("w:tblCellMar")
    for side, val in (("left", 28), ("right", 28), ("top", 10), ("bottom", 10)):
        el = OxmlElement("w:" + side)
        el.set(qn("w:w"), str(val)); el.set(qn("w:type"), "dxa")
        margins.append(el)
    tbl_pr.append(margins)
    grid = table._tbl.find(qn("w:tblGrid"))     # 固定布局下列宽由 tblGrid 决定，必须同步
    if grid is not None:
        for ci, gc in enumerate(grid.findall(qn("w:gridCol"))):
            if ci < len(widths):
                gc.set(qn("w:w"), str(widths[ci]))
    grid = table._tbl.find(qn("w:tblGrid"))     # 固定布局下列宽由 tblGrid 决定，必须同步
    if grid is not None:
        for ci, gc in enumerate(grid.findall(qn("w:gridCol"))):
            if ci < len(widths):
                gc.set(qn("w:w"), str(widths[ci]))
    for ci, col in enumerate(table.columns):
        col.width = Twips(widths[ci])
    for row in table.rows:
        for ci, cell in enumerate(row.cells):
            cell.width = Twips(widths[ci])
            tc_pr = cell._tc.get_or_add_tcPr()      # 显式 tcW，保证按该列宽渲染
            el = tc_pr.find(qn("w:tcW"))
            if el is None:
                el = OxmlElement("w:tcW")
                tc_pr.append(el)
            el.set(qn("w:w"), str(widths[ci]))
            el.set(qn("w:type"), "dxa")
    return table


def insert_figure(doc, path, caption, width_cm=9.0):
    """width_cm：图在版面上的最终宽度(期刊通常单栏 8~9 cm、双栏 16~17 cm)。"""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(0)
    run = p.add_run()
    run.add_picture(str(path), width=Cm(width_cm))
    para(doc, caption, name=HEI, size=9, align=WD_ALIGN_PARAGRAPH.CENTER,
         indent_chars=0, space_before=2, space_after=6, line=1.15)



# --------------------------------------------------------------------------- #
# Word 原生公式(OMML)：支持 _{下标} 与 ^{上标}
# --------------------------------------------------------------------------- #
def _m_run(text):
    esc = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return '<m:r><m:t xml:space="preserve">%s</m:t></m:r>' % esc


def _omml_body(spec):
    runs, buf, i = [], "", 0
    while i < len(spec):
        if spec[i] in "_^" and i + 1 < len(spec) and spec[i + 1] == "{":
            j = spec.index("}", i + 2)
            content = spec[i + 2:j]
            base = buf[-1] if buf else ""
            buf = buf[:-1]
            if buf:
                runs.append(_m_run(buf))
                buf = ""
            # 若紧跟 ^{...}，合并为上下标并存(m:sSubSup)
            k2 = j + 1
            sub_txt = sup_txt = None
            if spec[i] == "_":
                sub_txt = content
            else:
                sup_txt = content
            if k2 < len(spec) and spec[k2] in "_^" and k2 + 1 < len(spec) and spec[k2 + 1] == "{":
                j3 = spec.index("}", k2 + 2)
                if spec[k2] == "_":
                    sub_txt = spec[k2 + 2:j3]
                else:
                    sup_txt = spec[k2 + 2:j3]
                k2 = j3 + 1
            if sub_txt is not None and sup_txt is not None:
                runs.append("<m:sSubSup><m:e>%s</m:e><m:sub>%s</m:sub><m:sup>%s</m:sup></m:sSubSup>"
                            % (_m_run(base), _m_run(sub_txt), _m_run(sup_txt)))
            elif sub_txt is not None:
                runs.append("<m:sSub><m:e>%s</m:e><m:sub>%s</m:sub></m:sSub>"
                            % (_m_run(base), _m_run(sub_txt)))
            else:
                runs.append("<m:sSup><m:e>%s</m:e><m:sup>%s</m:sup></m:sSup>"
                            % (_m_run(base), _m_run(sup_txt)))
            i = k2
        else:
            buf += spec[i]
            i += 1
    if buf:
        runs.append(_m_run(buf))
    return "".join(runs)


def add_equation(doc, spec):
    """插入居中、可编辑的 Word 公式对象(OMML)。"""
    par = doc.add_paragraph()
    par.alignment = WD_ALIGN_PARAGRAPH.CENTER
    par.paragraph_format.space_before = Pt(4)
    par.paragraph_format.space_after = Pt(4)
    xml = '<m:oMathPara %s><m:oMath>%s</m:oMath></m:oMathPara>' % (
        nsdecls("m", "w"), _omml_body(spec))
    par._p.append(parse_xml(xml))
    return par


# --------------------------------------------------------------------------- #
# Markdown 解析
# --------------------------------------------------------------------------- #
FIG_RE = re.compile(r"^!\[(?P<cap>[^\]]*)\]\((?P<path>[^)]+)\)\s*$")
H1_RE = re.compile(r"^#\s+(.*)$")
H2_RE = re.compile(r"^##\s+(.*)$")
H3_RE = re.compile(r"^###\s+(.*)$")
TABLE_ROW_RE = re.compile(r"^\|.*\|\s*$")


def strip_table_row(line):
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    return cells


def is_sep_row(cells):
    return all(re.fullmatch(r":?-{2,}:?", c or "") for c in cells)


def build(src_path: Path, out_path: Path, base_dir: Path, allow_missing=False):
    raw = src_path.read_text(encoding="utf-8")
    leftover = sorted(set(TOKEN_RE.findall(raw)))
    if leftover and not allow_missing:
        raise SystemExit("未替换的占位符: {}".format(", ".join(leftover)))

    lines = raw.splitlines()
    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = SONG
    style.font.size = Pt(10.5)
    style.element.rPr.rFonts.set(qn("w:eastAsia"), SONG)
    for section in doc.sections:
        section.page_width = Cm(PAGE_W_CM)
        section.page_height = Cm(PAGE_H_CM)
        section.top_margin = Cm(MARGIN_TB_CM)
        section.bottom_margin = Cm(MARGIN_TB_CM)
        section.left_margin = Cm(MARGIN_LR_CM)
        section.right_margin = Cm(MARGIN_LR_CM)
    # 正文(最后一节)双栏；题目/摘要节在遇到正文标题时用连续分节符切出
    for section in doc.sections:
        _set_cols(section._sectPr, 2)

    i = 0
    in_refs = False
    n_tables = 0
    n_figs = 0
    author_block = False
    pending_caption = None         # 暂存的独立表题(与表格一起排入通栏节)
    header_closed = False          # 题目/摘要单栏节是否已用连续分节符结束
    while i < len(lines):
        line = lines[i].rstrip()
        stripped = line.strip()

        if not stripped or stripped == "---":
            i += 1
            continue

        if stripped.startswith("> "):
            i += 1
            continue

        m = FIG_RE.match(stripped)
        if m:
            target = m.group("path")
            width = FIG_COL_CM              # 默认单栏宽(8.1 cm)
            if "|" in target:               # 形如 figures/fig2_dose.png|17
                target, w = target.split("|", 1)
                try:
                    width = float(w)
                except ValueError:
                    pass
            wide = width > FIG_COL_CM + 1.0      # 明显宽于单栏 → 通栏(跨栏)排
            if wide:
                add_section_break(doc, 2)        # 结束双栏正文节
            insert_figure(doc, base_dir / target, m.group("cap"), width)
            if wide:
                add_section_break(doc, 1)        # 结束通栏节，正文回到双栏
            n_figs += 1
            i += 1
            continue

        if H1_RE.match(stripped):
            para(doc, H1_RE.match(stripped).group(1), name=HEI, size=16,
                 align=WD_ALIGN_PARAGRAPH.CENTER, indent_chars=0,
                 space_before=0, space_after=12, line=1.2)
            i += 1
            continue

        if H3_RE.match(stripped):
            para(doc, H3_RE.match(stripped).group(1), name=HEI, size=10.5,
                 align=WD_ALIGN_PARAGRAPH.LEFT, indent_chars=0,
                 space_before=6, space_after=3)
            i += 1
            continue

        if H2_RE.match(stripped):
            title = H2_RE.match(stripped).group(1)
            in_refs = title.startswith("参考文献")
            author_block = title.startswith("作者信息")
            if not header_closed and re.match(r"^1(\s|\.|、)", title):
                add_section_break(doc, 1)        # 题目/摘要保持单栏，其后正文双栏
                header_closed = True
            para(doc, title, name=HEI, size=12, align=WD_ALIGN_PARAGRAPH.LEFT,
                 indent_chars=0, space_before=10, space_after=4)
            i += 1
            continue

        if TABLE_ROW_RE.match(stripped):
            block = []
            while i < len(lines) and TABLE_ROW_RE.match(lines[i].strip()):
                block.append(strip_table_row(lines[i]))
                i += 1
            rows = [r for r in block if not is_sep_row(r)]
            caption = pending_caption          # 上一行的独立表题
            pending_caption = None
            if rows and len(rows[0]) == 1:
                caption = rows[0][0]
                rows = rows[1:]
            if rows:
                three_line_table(doc, rows, caption=caption)   # 表格在栏内排(单栏宽)
                n_tables += 1
            continue

        if stripped.startswith("$$") and stripped.endswith("$$") and len(stripped) > 4:
            add_equation(doc, stripped[2:-2].strip())     # Word 原生公式对象
            i += 1
            continue

        if re.match(r"^\*{0,2}(摘要|关键词|中图分类号|文献标识码|基金项目|作者简介|通信地址)", stripped):
            # 摘要/关键词/著录项：左对齐、无首行缩进
            para(doc, stripped, name=SONG, size=10.5, align=WD_ALIGN_PARAGRAPH.JUSTIFY,
                 indent_chars=0, space_before=2, space_after=2, line=1.3)
            i += 1
            continue

        if re.match(r"^\*{0,2}表\s*\d+\s*[　:：]", stripped):
            # 表题先暂存：等表格出现时一起放进同一个通栏节，避免题在双栏、表在通栏
            pending_caption = stripped.replace("**", "")
            i += 1
            continue

        if in_refs:
            para(doc, stripped, name=SONG, size=9, align=WD_ALIGN_PARAGRAPH.LEFT,
                 indent_chars=0, space_before=0, space_after=1, line=1.15)
            i += 1
            continue

        if author_block:
            para(doc, stripped, name=SONG, size=10.5, align=WD_ALIGN_PARAGRAPH.LEFT,
                 indent_chars=0, space_before=2, space_after=2, line=1.25)
            i += 1
            continue

        para(doc, stripped)
        i += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(out_path))
    return {"tables": n_tables, "figures": n_figs, "paragraphs": len(doc.paragraphs)}


if __name__ == "__main__":
    src = Path(sys.argv[1])
    out = Path(sys.argv[2])
    info = build(src, out, src.parent, allow_missing="--allow-missing" in sys.argv)
    print("saved:", out, info)
