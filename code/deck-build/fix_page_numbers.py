# -*- coding: utf-8 -*-
"""就地修正页脚页码：把所有幻灯片右下角的页码改成真实页码。

背景：用户增删了几页，页脚页码没有同步，导致 2/3 对调、第 5 页起全部少 1、
最后几页缺号。

策略：
  - 只在【页脚区】(y >= 6.9in 且 x >= 11.5in) 找页码框，其余内容一律不动
  - 找到就改文字；找不到就在右下角新建一个（样式与既有页脚一致）
  - 第 1 页（封面）不编号 —— 它是深色满版页，没有页脚

用法:  <bundled python> fix_page_numbers.py [--dry-run]
"""
import os
import sys

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_AUTO_SIZE

HERE = os.path.dirname(os.path.abspath(__file__))
DECK = os.path.join(HERE, "..", "report-deck", "DDPM_Academic_Report_v2.pptx")

GREY = RGBColor(0x5A, 0x66, 0x72)
FONT = "Calibri"

NUM_X, NUM_Y, NUM_W, NUM_H = 12.01, 7.05, 0.70, 0.26   # 与既有页码框一致
NUM_SIZE = 10.0

SKIP_PAGES = {1}      # 封面不编号


def find_page_number_box(slide):
    """在页脚区找右对齐的页码框。"""
    for sh in slide.shapes:
        if not sh.has_text_frame:
            continue
        x = sh.left / 914400.0
        y = sh.top / 914400.0
        if y < 6.9 or x < 11.5:
            continue
        return sh
    return None


def set_number(shape, n):
    """把页码框里的文字换成 n（保留原有字体格式）。"""
    tf = shape.text_frame
    # 只保留第一个段落，删掉多余的
    while len(tf.paragraphs) > 1:
        p = tf.paragraphs[-1]._p
        p.getparent().remove(p)
    para = tf.paragraphs[0]
    for r in list(para.runs):          # 清掉旧 run
        r._r.getparent().remove(r._r)
    run = para.add_run()
    run.text = str(n)
    run.font.size = Pt(NUM_SIZE)
    run.font.color.rgb = GREY
    run.font.name = FONT
    para.alignment = PP_ALIGN.RIGHT


def add_number(slide, n):
    tb = slide.shapes.add_textbox(Inches(NUM_X), Inches(NUM_Y),
                                  Inches(NUM_W), Inches(NUM_H))
    tf = tb.text_frame
    tf.word_wrap = False
    tf.auto_size = MSO_AUTO_SIZE.NONE
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    para = tf.paragraphs[0]
    para.alignment = PP_ALIGN.RIGHT
    run = para.add_run()
    run.text = str(n)
    run.font.size = Pt(NUM_SIZE)
    run.font.color.rgb = GREY
    run.font.name = FONT
    return tb


def build_footer(slide, n):
    """页脚区完全为空时，补齐「页脚文字 + 页码」这一整套。"""
    tb = slide.shapes.add_textbox(Inches(0.62), Inches(NUM_Y),
                                  Inches(11.09), Inches(NUM_H))
    tf = tb.text_frame
    tf.word_wrap = False
    tf.auto_size = MSO_AUTO_SIZE.NONE
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    run = tf.paragraphs[0].add_run()
    run.text = "Denoising Diffusion Probabilistic Models"
    run.font.size = Pt(NUM_SIZE)
    run.font.color.rgb = GREY
    run.font.name = FONT
    add_number(slide, n)


def main():
    dry = "--dry-run" in sys.argv
    prs = Presentation(DECK)
    print(f"{'页':>3} {'原页码':>8} {'新页码':>8}   动作")
    print("-" * 48)
    changes = 0
    for i, slide in enumerate(prs.slides, 1):
        if i in SKIP_PAGES:
            continue
        box = find_page_number_box(slide)
        old = box.text_frame.text.strip() if box else "(无)"
        if box is not None:
            act = "改文字" if old != str(i) else "已正确"
            if old != str(i):
                if not dry:
                    set_number(box, i)
                changes += 1
        else:
            # 判断页脚区是否完全空白
            has_footer_text = any(
                sh.has_text_frame and sh.text_frame.text.strip()
                and sh.top / 914400.0 >= 6.9 and sh.left / 914400.0 < 11.5
                for sh in slide.shapes)
            act = "补页脚+页码" if not has_footer_text else "补页码"
            if not dry:
                if has_footer_text:
                    add_number(slide, i)
                else:
                    build_footer(slide, i)
            changes += 1
        print(f"{i:>3} {old:>8} {str(i):>8}   {act}")

    print("-" * 48)
    print(f"共 {changes} 页需要改动" + ("" if not dry else "（dry-run，未写盘）"))
    if not dry:
        prs.save(DECK)
        print("已保存:", os.path.abspath(DECK))


if __name__ == "__main__":
    main()
