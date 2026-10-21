# -*- coding: utf-8 -*-
"""把实验结果图插入 PPT（就地修改，不重建整个 deck）。

原则：**不动原有元素的位置**，只删除确实多余的旧图，新图放进空白区。
  - 第 6 页  右下空白区 (7.0, 3.55)-(12.5, 4.75) -> forward_noising.png
  - 第 15 页 右侧说明块位置                       -> samples_over_t.png
  - 第 18 页 左栏（文字压缩到右栏）               -> samples_grid.png
  - 新增页（紧跟第 18 页之后，插入为第 19 页）    -> samples_grid + cfg_sweep

用法: <bundled python> insert_results.py [--dry-run]
"""
import os
import sys

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_AUTO_SIZE
from pptx.enum.shapes import MSO_SHAPE
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
DECK = os.path.join(HERE, "..", "report-deck", "DDPM_Academic_Report_v2.pptx")
RES = os.path.join(HERE, "..", "_demo-run", "results")

NAVY = RGBColor(0x0B, 0x25, 0x45)
TEAL = RGBColor(0x0E, 0x7C, 0x86)
AMBER = RGBColor(0xB4, 0x6A, 0x0A)
GREY = RGBColor(0x5A, 0x66, 0x72)
LIGHT = RGBColor(0xF2, 0xF5, 0xF8)
RULE = RGBColor(0xD5, 0xDD, 0xE5)
FONT, MONO = "Calibri", "Consolas"
SH, MARGIN, E = 7.5, 0.62, 914400.0


def is_pic(sh):
    return sh.shape_type is not None and "PICTURE" in str(sh.shape_type)


def add_pic(slide, path, x, y, w, max_h=None):
    iw, ih = Image.open(path).size
    ar = iw / ih
    dw, dh = w, w / ar
    if max_h and dh > max_h:
        dh, dw = max_h, max_h * ar
    slide.shapes.add_picture(path, Inches(x), Inches(y), Inches(dw), Inches(dh))
    return dw, dh


def add_text(slide, x, y, w, h, text, size, color=NAVY, bold=False,
             italic=False, align=PP_ALIGN.LEFT, font=FONT, ls=1.1):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.auto_size = MSO_AUTO_SIZE.NONE
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    for i, line in enumerate(str(text).split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = ls
        r = p.add_run()
        r.text = line
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.italic = italic
        r.font.color.rgb = color
        r.font.name = font
    return tb


def del_shape(sh):
    sh._element.getparent().remove(sh._element)


def rect(slide, x, y, w, h, fill):
    sp = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y),
                                Inches(w), Inches(h))
    sp.fill.solid(); sp.fill.fore_color.rgb = fill
    sp.line.fill.background(); sp.shadow.inherit = False
    return sp


def slide_title(slide, title, kicker):
    rect(slide, 0, 0, 0.20, SH, TEAL)
    add_text(slide, MARGIN, 0.40, 12.09, 0.26, kicker.upper(), 11.5,
             color=TEAL, bold=True)
    add_text(slide, MARGIN, 0.70, 12.09, 0.62, title, 29, color=NAVY, bold=True)
    rect(slide, MARGIN, 1.36, 1.55, 0.045, AMBER)


def footer(slide, n):
    rect(slide, MARGIN, SH - 0.52, 12.09, 0.012, RULE)
    add_text(slide, MARGIN, 7.05, 11.09, 0.26,
             "Denoising Diffusion Probabilistic Models", 10, color=GREY)
    add_text(slide, 12.01, 7.05, 0.70, 0.26, str(n), 10, color=GREY,
             align=PP_ALIGN.RIGHT)


# ================================================================ 第 6 页
def patch_s6(prs):
    s = prs.slides[5]
    # 删除两张公式图片
    for sh in list(s.shapes):
        if is_pic(sh):
            del_shape(sh)
    # 删除：两个空公式面板、左侧色条、右上角说明、无主的反向说明
    # ⚠️ 必须跳过标题区 (y<1.5)，否则会误删标题框（它也是 w=12.09, h=0.62）
    for sh in list(s.shapes):
        if not sh.has_text_frame:
            continue
        y, w, h = sh.top / E, sh.width / E, sh.height / E
        if y < 1.50:
            continue
        t = sh.text_frame.text.strip()
        if h > 0.5 and w > 10:                       # 两条大面板
            del_shape(sh)
        elif abs(w - 0.05) < 0.01 and h > 0.5:       # 左侧色条
            del_shape(sh)
        elif "Gaussian Distribution" in t:            # 右上角说明
            del_shape(sh)
        elif "reverse process: learned by NN" in t:
            del_shape(sh)
    # 全宽展示前向加噪（尺寸留足下方空间给 caption 与 FORWARD 行）
    dw, dh = add_pic(s, f"{RES}/forward_noising.png", 1.15, 1.62, 11.00,
                     max_h=1.62)
    add_text(s, 1.15, 1.62 + dh + 0.06, 11.00, 0.34,
             "Measured on a real training image (tiger face, 32×32): "
             "q(xₜ|x₀) over t = 0 → 99. Noise is Gaussian at every step and "
             "the schedule is fixed — nothing is learned here.",
             10.5, color=GREY, italic=True, ls=1.10)


# ================================================================ 第 15 页
def patch_s15(prs):
    s = prs.slides[14]
    for sh in list(s.shapes):
        if sh.has_text_frame and "start from pure noise" in sh.text_frame.text:
            del_shape(sh)
    dw, dh = add_pic(s, f"{RES}/samples_over_t.png", 8.07, 1.66, 4.30)
    add_text(s, 8.07, 1.66 + dh + 0.10, 4.64, 1.10,
             "Reverse process p(xₜ₋₁|xₜ): four independent runs from pure "
             "noise x_T down to x_0, using the pretrained weights.",
             11, color=GREY, italic=True, ls=1.15)


# ================================================================ 第 18 页
def patch_s18(prs):
    s = prs.slides[17]
    for sh in list(s.shapes):
        if not sh.has_text_frame:
            continue
        t, x = sh.text_frame.text, sh.left / E
        if x < 7.0 and "DDPM reported" in t:
            # DDPM 论文结果 -> 右栏顶部，缩小字号免得压住下方
            sh.left, sh.top = Inches(7.17), Inches(1.62)
            sh.width, sh.height = Inches(5.00), Inches(1.30)
            for p in sh.text_frame.paragraphs:
                for r in p.runs:
                    r.font.size = Pt(11)
        elif x > 7.0 and "Direct descendants" in t:
            sh.top = Inches(3.20)
        elif x > 7.0 and t.startswith("–"):
            sh.top = sh.top + Inches(1.60)      # 五个条目整体下移
            for p in sh.text_frame.paragraphs:
                for r in p.runs:
                    r.font.size = Pt(11)
    add_text(s, MARGIN, 1.58, 6.05, 0.30, "Sampled from the pretrained model",
             14, color=TEAL, bold=True)
    dw, dh = add_pic(s, f"{RES}/samples_grid.png", MARGIN, 1.96, 4.30)
    add_text(s, MARGIN, 1.96 + dh + 0.08, 5.20, 0.50,
             "25 unconditional samples, T = 100.",
             11, color=GREY, italic=True)


# ================================================================ 新增一页（第 19 页）
def make_results_slide(prs, page_no):
    """新建一页，并把它移动到指定位置（插到"结果"页之后）。"""
    s = prs.slides.add_slide(prs.slide_layouts[6])
    slide_title(s, "Generation results", "Outcome · practical")

    add_text(s, MARGIN, 1.58, 6.05, 0.30, "Unconditional samples",
             14.5, color=TEAL, bold=True)
    dw1, dh1 = add_pic(s, f"{RES}/samples_grid.png", MARGIN, 1.96, 4.30)
    add_text(s, MARGIN, 1.96 + dh1 + 0.06, 4.30, 0.28,
             "25 samples, T = 100", 10.5, color=GREY, italic=True)

    add_text(s, 7.17, 1.58, 5.00, 0.30,
             'Classifier-free guidance — "crying face"',
             14.5, color=TEAL, bold=True)
    dw2, dh2 = add_pic(s, f"{RES}/cfg_sweep.png", 7.17, 1.96, 4.30)
    add_text(s, 7.17, 1.96 + dh2 + 0.06, 5.00, 0.60,
             "w = 0 / 2 / 5 / 10.  Higher w → more faithful to the prompt, "
             "less diverse.", 10.5, color=GREY, italic=True, ls=1.15)

    rect(s, MARGIN, 6.20, 12.09, 0.62, LIGHT)
    add_text(s, MARGIN + 0.22, 6.28, 11.70, 0.28,
             "q_sample 5.1e-07  ·  predict_start_from_noise 2.8e-06  ·  "
             "predict_noise_from_start 1.1e-06",
             11.5, color=NAVY, font=MONO)
    add_text(s, MARGIN + 0.22, 6.52, 11.70, 0.26,
             "U-Net 12.4M params  ·  reproduced from the released pretrained "
             "checkpoint (step 70,000)", 9.5, color=GREY, italic=True)
    footer(s, page_no)
    return s


def move_slide(prs, from_idx, to_idx):
    """把幻灯片从 from_idx 移到 to_idx（0-based）。"""
    xml_slides = prs.slides._sldIdLst
    slides = list(xml_slides)
    xml_slides.remove(slides[from_idx])
    xml_slides.insert(to_idx, slides[from_idx])


def renumber_footers(prs, skip=(0,)):
    """移动幻灯片后，把每页右下角页码改成真实页码（封面跳过）。"""
    fixed = 0
    for i, s in enumerate(prs.slides, 1):
        if i - 1 in skip:
            continue
        for sh in s.shapes:
            if not sh.has_text_frame:
                continue
            if sh.top / E < 6.9 or sh.left / E < 11.5:
                continue
            if sh.text_frame.text.strip() == str(i):
                break
            tf = sh.text_frame
            while len(tf.paragraphs) > 1:
                p = tf.paragraphs[-1]._p
                p.getparent().remove(p)
            para = tf.paragraphs[0]
            for r in list(para.runs):
                r._r.getparent().remove(r._r)
            run = para.add_run()
            run.text = str(i)
            run.font.size = Pt(10)
            run.font.color.rgb = GREY
            run.font.name = FONT
            fixed += 1
            break
    return fixed


def main():
    dry = "--dry-run" in sys.argv
    prs = Presentation(DECK)
    patch_s6(prs)
    patch_s15(prs)
    patch_s18(prs)
    n = len(prs.slides) + 1
    make_results_slide(prs, n)
    # 新页当前在最后 (索引 n-1)，移到第 18 页之后 -> 索引 18
    move_slide(prs, n - 1, 18)
    fixed = renumber_footers(prs)
    print(f"页数: {n-1} -> {len(prs.slides)}；重排页码 {fixed} 处")
    if not dry:
        prs.save(DECK)
        print("已保存:", os.path.abspath(DECK))
    else:
        print("(dry-run，未写盘)")


if __name__ == "__main__":
    main()
