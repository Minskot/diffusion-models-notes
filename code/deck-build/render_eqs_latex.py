# -*- coding: utf-8 -*-
"""一步到位：用真 LaTeX 重新渲染 deck 全部公式，并【就地安装】到 eq/。

设计要点:
  1. 输出可复现: Ghostscript 关闭时间戳等元数据
  2. 只写 eq/ 下 deck 真正需要的 <key>.png，不留中间文件
  3. 【两遍渲染 + 字号自适应】：先量出每个公式的自然宽度，再为"过宽"的
     公式单独指定更小字号，使它在 deck 里占满面板宽度时，与其他公式
     保持一致的物理字号（而不是被 build_deck 的等比缩放压小）。

用法:
    <torch-gpu python> render_eqs_latex.py
"""
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from PIL import Image
from latex_spec import DISPLAY
import latex_render as LR

EQ_DIR = os.path.join(HERE, "eq")                    # build_deck.py 读取这里
BACKUP = os.path.join(HERE, "_eq_mathtext_backup")   # mathtext 版备份

# ---- 与 build_deck.py 的版式保持一致（改了那边要同步改这里）----
SW_IN, SH_IN, MARGIN = 13.333, 7.5, 0.62
BODY_W = SW_IN - 2 * MARGIN
PANEL_PAD = 0.26          # eq_panel 左右各留的边距
CONTENT_W = BODY_W - 2 * PANEL_PAD        # 公式可用宽度（英寸）= 11.573

# 公式目标宽度。build_deck.py 还会再乘 EQ_WIDTH_SAFETY(0.96)。
#
# 这里【不做收缩】，让每个公式按基准字号（26pt）的自然尺寸渲染；
# 最终在幻灯片上的显示尺寸完全由 build_deck.py 的 eq_image 控制：
#   - EQ_WIDTH_SAFETY 给渲染器越界留余量
#   - EQ_SHRINK       统一压缩最终尺寸（治「公式过大」的地方）
# 这样只有一个缩放控制点，改起来可预测。取 0.99 是为了让 eq_image 的
# 宽度适配在正常范围内生效，不引入额外的非预期收缩。
DISPLAY_TARGET_W = CONTENT_W * 0.99       # ≈ 11.46 in（仅作上限，几乎不触发）
BASE_FONTSIZE = LR.BASE_FONTSIZE          # 基准字号

# 渲染 DPI 与 deck 显示 DPI 的关系：
#   图片 px -> 英寸英寸 = px / 400
#   在幻灯片上 1 英寸 = PANEL_PX / CONTENT_W 个图片 px
PANEL_PX = SW_IN * 81.0                   # 110dpi 预览下整页约 1467px -> 81 px/in


def to_body(spec):
    """latex_spec 的一项 -> LaTeX 数学内容。多行用 gathered 纵向排列。"""
    lines = [ln.strip() for ln in spec.split("\n") if ln.strip()]
    lines = [ln[1:-1].strip() if ln.startswith("$") and ln.endswith("$") else ln
             for ln in lines]
    if len(lines) == 1:
        return lines[0]
    return r"\begin{gathered}" + r"\\[6pt]".join(lines) + r"\end{gathered}"


def main():
    keys = list(DISPLAY.keys())
    bodies = [to_body(DISPLAY[k]) for k in keys]
    work = os.path.join(HERE, "_latex_work")
    if os.path.exists(work):
        shutil.rmtree(work)

    # ---------- 第一遍：量自然尺寸 ----------
    print(f"共 {len(keys)} 个公式；第一遍测量自然尺寸 ...")
    dims = LR.measure_ink_widths(bodies, workdir=work)[:]
    if len(dims) != len(keys):
        raise RuntimeError(f"测量页数不符: {len(dims)} vs {len(keys)}")

    # 换算关系：图片按 400dpi 渲染（400px = 1in），在 PPTX 里按 inches = px/400
    # 放置，所以 LaTeX 的 1pt 墨迹 = 1/72 英寸显示宽度。
    max_pt = DISPLAY_TARGET_W * 72.0

    fontsizes = []
    n_scaled = 0
    for (w, h) in dims:
        if w > max_pt:                    # 超出上限 -> 按比例缩字号
            fontsizes.append(BASE_FONTSIZE * max_pt / w)
            n_scaled += 1
        else:                             # 已在范围内 -> 保持基准字号
            fontsizes.append(BASE_FONTSIZE)
    print(f"  基准字号 {BASE_FONTSIZE:g}pt；宽度上限 {DISPLAY_TARGET_W:.2f}in "
          f"({max_pt:.0f}pt)")
    print(f"  超出上限需缩字号: {n_scaled} 个；保持基准: {len(keys) - n_scaled} 个")

    # ---------- 第二遍：按各自字号正式渲染 ----------
    print("第二遍：按自适应字号渲染 ...")
    pdf = LR.compile_pdf(LR.build_document(bodies, fontsizes), work)
    bb = LR.page_bboxes(pdf)
    if len(bb) != len(keys):
        raise RuntimeError(f"页数不匹配: bbox {len(bb)} vs keys {len(keys)}")

    tmpdir = os.path.join(HERE, "_latex_png")
    if os.path.exists(tmpdir):
        shutil.rmtree(tmpdir)
    os.makedirs(tmpdir, exist_ok=True)
    raw = LR.render_pages(pdf, bb, tmpdir, "_t", dpi=400, deterministic=True)

    # ---------- 备份旧 mathtext 版（仅首次） ----------
    if not os.path.exists(BACKUP):
        os.makedirs(BACKUP, exist_ok=True)
        for n in os.listdir(EQ_DIR):
            if n.endswith(".png"):
                shutil.copy2(os.path.join(EQ_DIR, n), os.path.join(BACKUP, n))
        print(f"  旧公式已备份 -> {BACKUP}")

    # ---------- 安装 ----------
    sizes = []
    for k, src in zip(keys, raw):
        dst = os.path.join(EQ_DIR, f"{k}.png")
        shutil.move(src, dst)
        im = Image.open(dst)
        sizes.append((k, im.size, im.width / 400.0))
    shutil.rmtree(tmpdir)

    print(f"\n{'key':20s} {'尺寸':>14s} {'字号':>7s} {'显示宽(in)':>11s} {'占面板':>8s}")
    print("-" * 66)
    total = 0
    for (k, (w, h), disp_in), fs in zip(sizes, fontsizes):
        total += os.path.getsize(os.path.join(EQ_DIR, f"{k}.png"))
        pct = disp_in / CONTENT_W * 100
        print(f"{k:20s} {f'{w}x{h}':>14s} {fs:>6.1f}p {disp_in:>11.2f} {pct:>7.0f}%")

    valid = {f"{k}.png" for k in keys}
    removed = 0
    for n in os.listdir(EQ_DIR):
        if n.endswith(".png") and n not in valid:
            os.remove(os.path.join(EQ_DIR, n))
            removed += 1
    print(f"\neq/ 安装完成: {len(sizes)} 个公式，{total/1024:.0f} KB"
          + (f"（清理了 {removed} 个多余文件）" if removed else ""))


if __name__ == "__main__":
    main()
