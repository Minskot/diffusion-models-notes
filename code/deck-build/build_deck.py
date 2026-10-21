"""
Build an academic 16:9 deck: "Denoising Diffusion Probabilistic Models".
Style: technical blueprint (navy/teal, accent bars, rule-based layout).
All text is native and editable; equations use Unicode math for reliable rendering.
"""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR, MSO_AUTO_SIZE
from pptx.enum.shapes import MSO_SHAPE
from PIL import Image
import os

HERE = os.path.dirname(os.path.abspath(__file__))
EQ_DIR = os.path.join(HERE, "eq")
OUT_PPTX = os.path.join(HERE, "DDPM_Academic_Report_v2.pptx")

# 预览/导出用的 LibreOffice 渲染器对「形状/图片」的横向放置与文字框不完全一致，
# 实测形状与图片会略微超出预期右边界。这里给公式图片留出宽度余量即可
# （渲染器的文字框是准确的，因此面板与文字保持原位不动）。
EQ_WIDTH_SAFETY = 0.96

# 公式的最终整体缩小系数。
# eq_image 会把图片【放大到填满面板】，于是高宽比大的公式（idea_forward ar≈13、
# fwd_reparam ar≈16、kl_gauss ar≈11）会被顶到面板高度上限，字形占满整页、
# 观感过重。此系数在适配完成后再统一压缩最终尺寸：
#   0.85 -> 面积降约 28%，最宽公式的显示宽从 11.11in 降到约 9.44in
EQ_SHRINK = 0.85


# ---------------------------------------------------------------- design system
NAVY = RGBColor(0x0B, 0x25, 0x45)
NAVY_SOFT = RGBColor(0x1B, 0x3A, 0x5C)
TEAL = RGBColor(0x0E, 0x7C, 0x86)
AMBER = RGBColor(0xB4, 0x6A, 0x0A)
GREY = RGBColor(0x5A, 0x66, 0x72)
LIGHT = RGBColor(0xF2, 0xF5, 0xF8)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
RULE = RGBColor(0xD5, 0xDD, 0xE5)

FONT = "Calibri"
MONO = "Consolas"

SW, SH = 13.333, 7.5
MARGIN = 0.62
BODY_W = SW - 2 * MARGIN

# 素材图片（来自 report-deck/newSupply/）
SUPPLY_DIR = os.path.join(HERE, "..", "report-deck", "newSupply")
FIG_CAT = os.path.join(SUPPLY_DIR, "catDiffusion.jpg")     # 前向/反向过程（猫）
FIG_GENMODELS = os.path.join(SUPPLY_DIR, "genModels.jpg")  # 生成模型分类树
FIG_DDPM = os.path.join(SUPPLY_DIR, "DDPM.jpg")            # 反向链 + 为什么不可解


def blank(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def txt(slide, x, y, w, h, text, size=18, bold=False, color=NAVY,
        align=PP_ALIGN.LEFT, italic=False, font=FONT, line_spacing=1.06,
        anchor=MSO_ANCHOR.TOP):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.auto_size = MSO_AUTO_SIZE.NONE      # do NOT let the box grow with text
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = 0
    tf.margin_top = tf.margin_bottom = 0
    first = True
    for line in str(text).split("\n"):
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.alignment = align
        p.line_spacing = line_spacing
        r = p.add_run()
        r.text = line
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.italic = italic
        r.font.color.rgb = color
        r.font.name = font
    return tb


def bullets(slide, x, y, w, h, items, size=17, color=NAVY, gap=7, bullet_char="\u25aa"):
    """items: list of str, or (str, level) tuples."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.auto_size = MSO_AUTO_SIZE.NONE
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    first = True
    for it in items:
        lvl = 0
        if isinstance(it, tuple):
            it, lvl = it
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.line_spacing = 1.06
        p.space_after = Pt(gap)
        mark = "" if lvl else f"{bullet_char}  "
        r = p.add_run()
        r.text = mark + it
        r.font.size = Pt(size - (1 if lvl else 0))
        r.font.color.rgb = color if not lvl else GREY
        r.font.name = FONT
        if lvl:
            p.level = 1
    return tb


def rect(slide, x, y, w, h, fill, line=None, shape=MSO_SHAPE.RECTANGLE):
    """背景面板（保持原位；渲染器的文字框与面板右边界已对齐）。"""
    sh = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.shadow.inherit = False
    if fill is None:
        sh.fill.background()
    else:
        sh.fill.solid()
        sh.fill.fore_color.rgb = fill
    if line is None:
        sh.line.fill.background()
    else:
        sh.line.color.rgb = line
        sh.line.width = Pt(0.75)
    return sh


def slide_title(slide, title, kicker=None):
    rect(slide, 0, 0, 0.20, SH, TEAL)                 # left accent spine
    y = 0.40
    if kicker:
        txt(slide, MARGIN, y, BODY_W, 0.26, kicker.upper(), size=11.5,
            bold=True, color=TEAL)
        y += 0.30
    txt(slide, MARGIN, y, BODY_W, 0.62, title, size=29, bold=True, color=NAVY)
    rect(slide, MARGIN, y + 0.66, 1.55, 0.045, AMBER)   # rule under title
    return y + 0.92


def footer(slide, n):
    rect(slide, MARGIN, SH - 0.52, BODY_W, 0.012, RULE)
    txt(slide, MARGIN, SH - 0.45, BODY_W - 1.0, 0.26,
        "Denoising Diffusion Probabilistic Models", size=10, color=GREY)
    txt(slide, SW - MARGIN - 0.7, SH - 0.45, 0.7, 0.26, str(n),
        size=10, color=GREY, align=PP_ALIGN.RIGHT)


def no_wrap(tb):
    """Disable word wrapping on a text box so each line renders as written."""
    bodyPr = tb.text_frame._txBody.find(
        "{http://schemas.openxmlformats.org/drawingml/2006/main}bodyPr")
    if bodyPr is not None:
        bodyPr.set("wrap", "none")
    return tb


def equation(slide, x, y, w, text, size=21, color=NAVY, align=PP_ALIGN.CENTER):
    tb = txt(slide, x, y, w, 0.62, text, size=size, color=color, align=align,
             line_spacing=1.0)
    return no_wrap(tb)


def eq_image(slide, key, x, y, w, h, safety=EQ_WIDTH_SAFETY):
    """把已渲染的公式 PNG 放进 (x, y, w, h) 区域，等比缩放并居中。

    适配规则：
      1. 先按安全系数收缩可用宽度（safety<1 给渲染器越界留余量）
      2. 等比缩放到恰好放进 (w*safety, h)
      3. 再乘 EQ_SHRINK 做一次【整体缩小】—— 上一步会把高宽比大的公式
         （如 idea_forward ar≈13、fwd_reparam ar≈16）顶到面板高度上限，
         导致字形占满整页、观感过重。EQ_SHRINK 统一压缩最终尺寸。
    """
    path = f"{EQ_DIR}/{key}.png"
    iw, ih = Image.open(path).size
    ar = iw / ih
    w = w * safety
    dw, dh = w, w / ar
    if dh > h:
        dh, dw = h, h * ar
    dw *= EQ_SHRINK
    dh *= EQ_SHRINK
    left = x + (w - dw) / 2
    top = y + (h - dh) / 2
    return slide.shapes.add_picture(path, Inches(left), Inches(top),
                                    Inches(dw), Inches(dh))


def figure(slide, src, x, y, w, max_h=None, align="left"):
    """放置素材图片（等比缩放），返回 (底边 y, 实际宽, 实际高)。

    与 eq_image 分开：公式图有 EQ_SHRINK 的额外压缩，素材图不需要。
    align: left/center —— center 表示在宽度 w 的区域内水平居中。
    """
    iw, ih = Image.open(src).size
    ar = iw / ih
    dw, dh = w, w / ar
    if max_h is not None and dh > max_h:
        dh, dw = max_h, max_h * ar
    left = x if align == "left" else x + (w - dw) / 2
    slide.shapes.add_picture(src, Inches(left), Inches(y),
                             Inches(dw), Inches(dh))
    return y + dh, dw, dh


def eq_panel(slide, x, y, w, h, text, size=21, fill=LIGHT, accent=TEAL):
    """若 text 是公式表里的 key，则放公式图片；否则按文字处理。"""
    rect(slide, x, y, w, h, fill)                       # 背景面板
    rect(slide, x, y, 0.05, h, accent)                  # 左侧色条
    try:
        import latex_spec
        if text in latex_spec.DISPLAY:
            eq_image(slide, text, x + 0.26, y + 0.05, w - 0.52, h - 0.10)
            return None
    except Exception:
        pass
    tb = txt(slide, x + 0.22, y, w - 0.44, h, text, size=size, color=NAVY,
             align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, line_spacing=1.12)
    return no_wrap(tb)


def _code_panel(slide, x, y, w, algo_lines, code_lines, accent=TEAL,
                size=11.0, line_h=0.215, pad=0.14, gap=0.10):
    """算法页的等宽代码面板：上半是伪代码，下半是对应的 PyTorch 实现。

    全部使用纯 ASCII（下标写 [t]、根号写 sqrt()、希腊字母写名字），
    避免 Consolas 缺字形导致的回退与基线错乱。
    返回面板高度（英寸），便于调用方在其下方/右侧继续排版。
    """
    total = len(algo_lines) + len(code_lines)
    panel_h = pad * 2 + total * line_h + gap
    rect(slide, x, y, w, panel_h, LIGHT)
    rect(slide, x, y, 0.05, panel_h, accent)

    yy = y + pad
    for st in algo_lines:
        txt(slide, x + 0.26, yy, w - 0.5, line_h, st, size=size,
            color=NAVY, font=MONO, line_spacing=1.0)
        yy += line_h
    yy += gap
    for st in code_lines:
        txt(slide, x + 0.26, yy, w - 0.5, line_h, st, size=size,
            color=NAVY_SOFT, font=MONO, line_spacing=1.0)
        yy += line_h
    return panel_h


# ---------------------------------------------------------------- content
def build(path):
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(SW), Inches(SH)
    n = 0

    # ---------- 1. Title
    n += 1
    s = blank(prs)
    rect(s, 0, 0, SW, SH, NAVY)
    rect(s, 0, 0, SW, 0.10, TEAL)
    txt(s, MARGIN + 0.25, 1.30, BODY_W - 0.5, 0.35,
        "ACADEMIC SEMINAR  \u00b7  GENERATIVE MODELS", size=12.5, bold=True,
        color=RGBColor(0x6F, 0xC7, 0xC9))
    txt(s, MARGIN + 0.25, 1.78, BODY_W - 0.5, 1.5,
        "Denoising Diffusion\nProbabilistic Models", size=44, bold=True,
        color=WHITE)
    rect(s, MARGIN + 0.25, 3.55, 2.0, 0.05, AMBER)
    txt(s, MARGIN + 0.25, 3.85, BODY_W - 5.2, 1.0,
        "Learning to generate images by reversing a gradual noising process",
        size=17.5, color=RGBColor(0xC9, 0xD6, 0xE2))
    txt(s, MARGIN + 0.25, 5.05, BODY_W - 5.2, 0.9,
        "Ho, Jain & Abbeel  \u00b7  NeurIPS 2020", size=13.5, color=TEAL)
    txt(s, MARGIN + 0.25, 5.35, BODY_W - 5.2, 0.9,
        "12\u201315 minute report  \u00b7  from intuition to equations to implementation",
        size=12, color=GREY)
    txt(s, MARGIN + 0.25, SH - 0.95, BODY_W - 0.5, 0.5,
        "x\u209c = \u221a(\u03b1\u0304\u209c)\u00b7x\u2080 + \u221a(1\u2212\u03b1\u0304\u209c)\u00b7\u03b5",
        size=15, color=RGBColor(0x8F, 0xA6, 0xBD))

    # ---------- 2. Background / intro
    # 素材来自 report-deck/newSupply/：intro.md 的文字（已译为英文并适度压缩）
    # + catDiffusion.jpg（同时展示前向/反向两条链，与本页文字天然配套）。
    n += 1
    s = blank(prs)
    y = slide_title(s, "Background: what is a diffusion model?", "Introduction")
    intro = [
        ("The physical analogy",
         "Diffusion is inspired by a phenomenon we see every day: particles "
         "drift from regions of high concentration to regions of low "
         "concentration, until the whole system reaches equilibrium."),
        ("The mechanism",
         "A diffusion model uses this physics. It progressively adds noise to "
         "an image \u2014 increasing entropy \u2014 and then trains a network to "
         "reverse that process."),
        ("Why it can generate",
         "Once the model can turn noise back into a clear image, we can start "
         "from pure randomness. No existing image is needed, so the model "
         "produces entirely original data \u2014 at very high quality."),
    ]
    yy = y + 0.02
    for i, (head, body) in enumerate(intro, 1):
        txt(s, MARGIN, yy, 0.42, 0.30, f"{i}.", size=14, bold=True, color=TEAL)
        txt(s, MARGIN + 0.42, yy, 6.30, 0.28, head, size=14.5, bold=True,
            color=NAVY)
        box = txt(s, MARGIN + 0.42, yy + 0.32, 6.30, 1.40, body, size=14,
                  color=NAVY, line_spacing=1.12)
        yy += 0.32 + 0.235 * max(2, len(body) // 62 + 1) + 0.24

    # 右栏：前向 / 反向过程示意（猫）
    fig_y = 2.80
    bot, fw, fh = figure(s, FIG_CAT, 7.55, fig_y, 5.40)
    txt(s, 7.55, bot + 0.10, 5.40, 0.60,
        "Forward (top) and reverse (bottom) diffusion chains: the network only "
        "ever learns one small denoising step, x[t] \u2192 x[t-1].",
        size=11, color=GREY, line_spacing=1.10)

    # 关键要点
    rect(s, 7.55, 5.62, 5.40, 1.10, RGBColor(0xE7, 0xF3, 0xF4))
    rect(s, 7.55, 5.62, 0.05, 1.10, AMBER)
    txt(s, 7.80, 5.76, 5.00, 0.86,
        "The process is physics-inspired: we destroy information by adding "
        "noise, then learn to reverse it \u2014 which is what lets us generate "
        "new samples from pure noise.",
        size=12.5, color=NAVY, line_spacing=1.12)
    footer(s, n)

    # ---------- 4. Outline
    n += 1
    s = blank(prs)
    y = slide_title(s, "Outline", "Roadmap")
    txt(s, MARGIN, y - 0.02, 7.10, 0.72,
        "We first frame diffusion against other generative models, then build "
        "it up in three steps: the fixed forward process, the learned reverse "
        "process, and the variational bound that turns the likelihood into a "
        "trainable loss.", size=14, color=GREY, line_spacing=1.14)
    items = [
        ("1", "Why diffusion? Generative modelling landscape"),
        ("2", "Forward process \u2014 the closed-form shortcut"),
        ("3", "Reverse process \u2014 the intractable posterior"),
        ("4", "Variational bound \u2014 deriving the training loss"),
        ("5", "Training and sampling algorithms"),
        ("6", "Design choices, results, and modern extensions"),
    ]
    yy = y + 0.82
    for num, label in items:
        rect(s, MARGIN, yy, 0.42, 0.42, TEAL)
        txt(s, MARGIN, yy + 0.05, 0.42, 0.34, num, size=14, bold=True,
            color=WHITE, align=PP_ALIGN.CENTER)
        txt(s, MARGIN + 0.62, yy + 0.055, 7.10, 0.40, label,
            size=17.5, color=NAVY)
        yy += 0.62
    # 右栏：本报告的一句话定位（与左侧路线图读完后呼应）
    rect(s, 8.30, y + 0.82, 4.41, 3.72, RGBColor(0xE7, 0xF3, 0xF4))
    rect(s, 8.30, y + 0.82, 0.05, 3.72, AMBER)
    txt(s, 8.56, y + 1.00, 3.95, 3.40,
        "What this report covers\n"
        "A single generative model, end to end: how a fixed noising process "
        "defines a sequence of Gaussians, how a network learns to invert one "
        "step of it, and how the resulting objective collapses to a simple "
        "noise-regression loss.\n"
        "\n"
        "Along the way we contrast the two parameterisations (predicting "
        "eps vs. predicting x0), the design choices that matter in practice, "
        "and the direct descendants of DDPM \u2014 DDIM, score-based SDEs, "
        "classifier-free guidance, latent diffusion and diffusion "
        "transformers.",
        size=12.5, color=NAVY, line_spacing=1.16)
    footer(s, n)

    # ---------- 4. Why diffusion / landscape
    n += 1
    s = blank(prs)
    y = slide_title(s, "Why diffusion? The generative modelling landscape",
                    "Motivation")
    # 左栏：Goodfellow 生成模型分类树（素材），点明扩散落在哪一格
    bot, fw, fh = figure(s, FIG_GENMODELS, MARGIN, y + 0.02, 4.85)
    txt(s, MARGIN, bot + 0.08, 4.85, 0.52,
        "Goodfellow's taxonomy: diffusion sits under Explicit \u2192 Approximate "
        "density, reached by an indirect (iterative) sampling procedure.",
        size=11, color=GREY, line_spacing=1.10)
    # 左栏下方：扩散的取舍
    rect(s, MARGIN, 5.35, 4.85, 1.32, RGBColor(0xE7, 0xF3, 0xF4))
    rect(s, MARGIN, 5.35, 0.05, 1.32, AMBER)
    txt(s, MARGIN + 0.24, 5.47, 4.50, 0.55,
        "Simple to train  \u00b7  strong sample quality  \u00b7  slow, iterative "
        "sampling", size=13.5, bold=True, color=NAVY, line_spacing=1.10)
    txt(s, MARGIN + 0.24, 6.06, 4.50, 0.55,
        "Since 2020 it has overtaken GANs on image benchmarks and now underpins "
        "Stable Diffusion, DALL\u00b7E 2 and Sora.", size=11.5, color=GREY,
        line_spacing=1.10)

    # 右栏：四类模型的取舍对比
    txt(s, 7.00, y + 0.02, 5.71, 0.34, "Four families of generative models",
        size=13.5, bold=True, color=TEAL)
    fams = [
        ("GAN", "one-shot, sharp images \u2014 but unstable training, mode "
                "collapse"),
        ("VAE", "stable and principled likelihood \u2014 samples tend to be "
                "blurry"),
        ("Normalising flow", "exact likelihood \u2014 but a restrictive "
                             "invertible architecture"),
        ("Diffusion", "stable training + strong sample quality + good mode "
                      "coverage"),
    ]
    yy = y + 0.38
    for name, desc in fams:
        hi = (name == "Diffusion")
        if hi:
            rect(s, 7.00, yy - 0.06, 5.71, 0.90, RGBColor(0xE7, 0xF3, 0xF4))
        txt(s, 7.08, yy, 5.55, 0.28, name, size=14.5, bold=True,
            color=TEAL if hi else NAVY_SOFT)
        txt(s, 7.08, yy + 0.28, 5.55, 0.58, desc, size=12.5, color=NAVY,
            line_spacing=1.10)
        yy += 0.90
    # 右栏收尾：两列紧凑小结，补齐与左栏的高度，并点明后文的取舍
    txt(s, 7.00, 5.74, 5.71, 0.28, "Where each family wins",
        size=12.5, bold=True, color=TEAL)
    txt(s, 7.08, 5.98, 2.75, 1.00,
        "GAN \u2192 speed / sharpness\n"
        "VAE \u2192 principled likelihood",
        size=11, color=NAVY, line_spacing=1.14)
    txt(s, 9.95, 5.98, 2.75, 1.00,
        "Flow \u2192 exact likelihood\n"
        "Diffusion \u2192 quality + stability",
        size=11, color=NAVY, line_spacing=1.14)
    footer(s, n)

    # ---------- 4. Core idea / two processes
    n += 1
    s = blank(prs)
    y = slide_title(s, "The core idea: destroy information, then learn to restore it",
                    "Intuition")
    eq_panel(s, MARGIN, y, BODY_W, 1.15, "idea_forward", fill=RGBColor(0xE7, 0xF3, 0xF4))
    rect(s, MARGIN, y + 1.20, BODY_W, 0.72, LIGHT)
    rect(s, MARGIN, y + 1.20, 0.05, 0.72, AMBER)
    eq_image(s, "idea_reverse", MARGIN + 0.30, y + 1.26, 5.4, 0.60)
    txt(s, MARGIN + 6.10, y + 1.36, 5.9, 0.40,
        "reverse process: learned, iteratively denoises", size=14, color=GREY)
    rows = [
        ("FORWARD", "fixed, no learning \u2014 gradually add Gaussian noise over T "
                    "steps until the signal becomes pure noise"),
        ("REVERSE", "learned \u2014 train a network to undo one small noising step "
                    "at a time"),
        ("SAMPLE", "inference \u2014 iteratively denoise from random noise to "
                   "obtain a new image"),
    ]
    yy = y + 2.10
    for name, desc in rows:
        rect(s, MARGIN, yy, 1.75, 0.56, LIGHT)
        txt(s, MARGIN, yy + 0.13, 1.75, 0.32, name, size=13, bold=True,
            color=TEAL if name != "REVERSE" else AMBER, align=PP_ALIGN.CENTER)
        txt(s, MARGIN + 1.95, yy + 0.10, BODY_W - 2.05, 0.40, desc, size=14.5,
            color=NAVY)
        yy += 0.68
    txt(s, MARGIN, yy + 0.00, BODY_W, 0.40,
        "Key point: each step is small, so both the forward and the learned "
        "reverse step can be modelled as Gaussians.",
        size=14, color=GREY)
    footer(s, n)

    # ---------- 5. Notation
    n += 1
    s = blank(prs)
    y = slide_title(s, "Notation", "Setup")
    rows = [
        ("x\u2080", "clean data sample drawn from the data distribution q(x\u2080)"),
        ("x\u209c", "noisy sample at time step t, t = 1 \u2026 T"),
        ("\u03b2\u209c \u2208 (0, 1)", "per-step noise variance (the noise schedule)"),
        ("\u03b1\u209c = 1 \u2212 \u03b2\u209c", "per-step signal retention factor"),
        ("\u03b1\u0304\u209c = \u220f\u1d62\u208c\u2081\u1d57 \u03b1\u1d62", "cumulative signal retention factor"),
        ("\u03b5 ~ N(0, I)", "standard Gaussian noise, same shape as x\u2080"),
    ]
    yy = y + 0.05
    for sym, desc in rows:
        txt(s, MARGIN, yy, 3.1, 0.4, sym, size=16, bold=True, color=TEAL)
        txt(s, MARGIN + 3.25, yy + 0.02, BODY_W - 3.25, 0.4, desc, size=15.5,
            color=NAVY)
        yy += 0.60
    eq_panel(s, MARGIN, yy + 0.05, BODY_W, 0.85,
             "\u03b2\u209c increases with t  \u21d2  \u03b1\u0304\u209c decreases with t  "
             "\u21d2  \u03b1\u0304_T \u2248 0  \u21d2  x_T \u2248 N(0, I)",
             size=16)
    footer(s, n)

    # ---------- 6. Forward process
    n += 1
    s = blank(prs)
    y = slide_title(s, "Forward process: a fixed Markov chain that adds noise",
                    "Step 1")
    txt(s, MARGIN, y, BODY_W, 0.4,
        "Each step keeps a shrinking fraction of the signal and injects a "
        "little Gaussian noise:", size=15, color=GREY)
    eq_panel(s, MARGIN, y + 0.42, BODY_W, 0.92, "fwd_def")
    txt(s, MARGIN, y + 1.48, 2.3, 0.4, "Reparameterised:", size=15, bold=True,
        color=TEAL)
    eq_panel(s, MARGIN, y + 1.88, BODY_W, 0.92, "fwd_reparam")
    bullets(s, MARGIN, y + 2.95, BODY_W, 1.0, [
        "The forward process has no learnable parameters \u2014 it is fully specified by the schedule {\u03b2\u209c}.",
        "Because every step is Gaussian, the composition of steps stays Gaussian \u2014 which gives us a closed form (next slide).",
    ], size=15)
    footer(s, n)

    # ---------- 7. Closed form
    n += 1
    s = blank(prs)
    y = slide_title(s, "The closed-form shortcut: jump to any time step",
                    "Step 1 \u00b7 derivation")
    txt(s, MARGIN, y, BODY_W, 0.4,
        "Unrolling two steps and using the fact that the sum of independent "
        "Gaussians is Gaussian:", size=15, color=GREY)
    eq_panel(s, MARGIN, y + 0.42, BODY_W, 0.86, "cf_step1")
    txt(s, MARGIN, y + 1.36, BODY_W, 0.35,
        "Substituting both sides and collecting the noise terms:", size=14.5,
        color=GREY)
    eq_panel(s, MARGIN, y + 1.72, BODY_W, 1.05, "cf_step2")
    txt(s, MARGIN, y + 2.88, BODY_W, 0.35,
        "Repeating to x\u2080 gives the closed form used throughout DDPM:",
        size=14.5, color=GREY)
    eq_panel(s, MARGIN, y + 3.24, BODY_W, 1.10, "cf_final",
             fill=RGBColor(0xE7, 0xF3, 0xF4), accent=AMBER)
    footer(s, n)

    # ---------- 8. Why closed form matters
    n += 1
    s = blank(prs)
    y = slide_title(s, "Why the closed form matters", "Step 1 \u00b7 consequence")
    bullets(s, MARGIN, y, 6.1, 3.2, [
        "One sample for any t \u2014 no need to simulate t sequential steps during training.",
        "In practice we draw one random t per example per gradient step.",
        "x\u209c is a deterministic function of (x\u2080, \u03b5), so the model can be trained with a single random draw.",
        "It makes the entire training objective computable in O(1) per example.",
    ], size=16.5)
    eq_panel(s, MARGIN + 6.6, y + 0.10, 4.95, 0.92, "cf_dist")
    eq_panel(s, MARGIN + 6.6, y + 1.22, 4.95, 0.92, "cf_limit",
             fill=RGBColor(0xFE, 0xF4, 0xE6), accent=AMBER)
    txt(s, MARGIN + 6.6, y + 2.35, 4.95, 0.9,
        "Because x\u209c is available in closed form, the loss needs only one "
        "network evaluation per example.", size=14, color=GREY)
    footer(s, n)

    # ---------- 9. Reverse process / intractability
    n += 1
    s = blank(prs)
    y = slide_title(s, "Reverse process: the posterior we actually need",
                    "Step 2")
    txt(s, MARGIN, y, BODY_W, 0.4,
        "We want the inverse of the forward kernel \u2014 but Bayes' rule "
        "requires the unknown marginals q(x\u209c):", size=15, color=GREY)
    eq_panel(s, MARGIN, y + 0.42, BODY_W, 0.86, "rev_bayes",
             fill=RGBColor(0xFD, 0xEE, 0xEE), accent=RGBColor(0xB0, 0x3A, 0x2B))
    txt(s, MARGIN, y + 1.40, BODY_W, 0.4,
        "Standard fix: condition on x\u2080 as well. Then all three terms are "
        "known Gaussian densities, so the posterior is Gaussian and has a "
        "closed form:", size=15, color=GREY)
    eq_panel(s, MARGIN, y + 1.84, BODY_W, 0.86, "rev_cond")
    txt(s, MARGIN, y + 2.78, BODY_W, 0.4,
        "This is the training target: an explicit Gaussian we can regress to.",
        size=15, bold=True, color=TEAL)

    # 下方：DDPM 反向链示意 —— 图里直接把「q(x_{t-1}|x_t) is unknown」标了出来，
    # 正好对应本页讲的不可解性，并指出出路是变分下界（variational lower bound）。
    fig_y = y + 3.28
    bot, fw, fh = figure(s, FIG_DDPM, MARGIN, fig_y, 6.80)
    txt(s, MARGIN + 7.05, fig_y + 0.12, 4.42, 1.60,
        "The reverse chain we must learn. The true reverse kernel "
        "q(x[t-1] | x[t]) is unknown, so instead we optimise a variational "
        "lower bound on the data likelihood \u2014 that bound is exactly what the "
        "next two steps derive.",
        size=12, color=NAVY, line_spacing=1.14)
    footer(s, n)

    # ---------- 10. Posterior closed form
    n += 1
    s = blank(prs)
    y = slide_title(s, "The posterior is Gaussian \u2014 with known mean and variance",
                    "Step 2 \u00b7 result")
    eq_panel(s, MARGIN, y, BODY_W, 0.86, "post_dist")
    txt(s, MARGIN, y + 1.00, BODY_W, 0.35,
        "Expanding the product of the three Gaussian densities and completing "
        "the square in x\u209c\u208b\u2081:", size=14.5, color=GREY)
    eq_panel(s, MARGIN, y + 1.38, BODY_W, 1.30, "post_mean")
    eq_panel(s, MARGIN, y + 2.82, BODY_W, 0.86, "post_var",
             fill=RGBColor(0xE7, 0xF3, 0xF4), accent=AMBER)
    txt(s, MARGIN, y + 3.72, BODY_W, 0.4,
        "The variance depends only on t \u2014 it is a known constant, not a "
        "quantity we must learn.", size=14.5, color=GREY, align=PP_ALIGN.CENTER)
    footer(s, n)

    # ---------- 11. ELBO
    n += 1
    s = blank(prs)
    y = slide_title(s, "Variational bound: turning likelihood into a sum of KL terms",
                    "Step 3")
    txt(s, MARGIN, y, BODY_W, 0.4,
        "We maximise the data likelihood. Since it is intractable, we maximise "
        "its evidence lower bound:", size=15, color=GREY)
    eq_panel(s, MARGIN, y + 0.42, BODY_W, 0.86, "elbo1")
    txt(s, MARGIN, y + 1.40, BODY_W, 0.4,
        "Writing the joint distributions along the Markov chain and "
        "rearranging gives a sum of three groups of terms:", size=14.5,
        color=GREY)
    eq_panel(s, MARGIN, y + 1.78, BODY_W, 1.30, "elbo2")
    bullets(s, MARGIN, y + 3.30, BODY_W, 0.8, [
        "L_T : prior-matching term \u2014 the forward process is fixed, so it "
        "has no parameters and can be ignored.",
        "L\u209c\u208b\u2081 : the per-step term \u2014 this is where all the learning "
        "happens.",
    ], size=14.5, gap=4)
    footer(s, n)

    # ---------- 12. KL to simple loss
    n += 1
    s = blank(prs)
    y = slide_title(s, "From KL divergence to a simple noise-regression loss",
                    "Step 3 \u00b7 result")
    txt(s, MARGIN, y, BODY_W, 0.4,
        "Both distributions in L\u209c\u208b\u2081 are Gaussian, so the KL has a "
        "closed form. With the variance fixed, only the means matter:",
        size=15, color=GREY)
    eq_panel(s, MARGIN, y + 0.45, BODY_W, 0.80, "kl_gauss")
    txt(s, MARGIN, y + 1.38, BODY_W, 0.4,
        "Substituting x\u2080 = (x\u209c \u2212 \u221a(1\u2212\u03b1\u0304\u209c)\u00b7\u03b5)/\u221a(\u03b1\u0304\u209c) "
        "removes x\u2080 and leaves a prediction of the noise itself:",
        size=15, color=GREY)
    eq_panel(s, MARGIN, y + 1.80, BODY_W, 1.10, "mean_noise")
    eq_panel(s, MARGIN, y + 3.02, BODY_W, 1.02, "loss_final",
             fill=RGBColor(0xE7, 0xF3, 0xF4), accent=AMBER)
    footer(s, n)

    # ---------- 13. Training algorithm
    # 算法页用【纯 ASCII 代码风格】：下标写 [t]，根号写 sqrt()，希腊字母写名字。
    # 原因：Consolas 缺 U+209C(ₜ) 等下标字形，用 Unicode 数学符号会触发字形回退，
    # 导致下标字体与基线错乱；纯 ASCII 则与 gaussian_diffusion.py 的变量名一致。
    n += 1
    s = blank(prs)
    y = slide_title(s, "Training algorithm", "Step 4 \u00b7 Algorithm 1")
    algo = [
        "repeat",
        "    x0    ~ q(x0)                    # sample a clean image",
        "    t     ~ Uniform({1, ..., T})      # pick a random time step",
        "    eps   ~ N(0, I)                   # draw standard Gaussian noise",
        "    gradient-descent step on:",
        "        || eps - eps_theta( sqrt(abar[t]) * x0",
        "                            + sqrt(1 - abar[t]) * eps, t ) ||^2",
        "until converged",
    ]
    code = [
        "# cs231n/gaussian_diffusion.py  ->  p_losses()",
        "t      = torch.randint(0, T, (b,))      # one random t per example",
        "noise  = torch.randn_like(x_start)",
        "x_t    = q_sample(x_start, t, noise)    # sqrt(abar)*x0 + sqrt(1-abar)*eps",
        "target = noise if objective == 'pred_noise' else x_start",
        "loss   = ((model(x_t, t) - target) ** 2).mean()",
    ]
    _code_panel(s, MARGIN, y, 7.15, algo, code, accent=TEAL)
    # 右侧说明刻意保持"一行一条"：盒子宽 4.64in，12pt Consolas 约 4.7in/54 字符，
    # 留足余量避免溢出（LibreOffice 对 auto_size=NONE 的框不会自动折行）。
    bullets(s, MARGIN + 7.45, y + 0.16, BODY_W - 7.45, 3.6, [
        "predict eps from (x[t], t);",
        "one random t per example;",
        "closed form => one net call;",
        "eps and x0 are equivalent.",
    ], size=12, gap=11)
    footer(s, n)

    # ---------- 14. Sampling algorithm
    n += 1
    s = blank(prs)
    y = slide_title(s, "Sampling algorithm", "Step 4 \u00b7 Algorithm 2")
    algo2 = [
        "x[T] ~ N(0, I)",
        "for t = T, ..., 1 do",
        "    z      = randn_like(x[t]) if t > 1 else 0",
        "    x[t-1] = (1/sqrt(alpha[t])) * ( x[t]",
        "               - (beta[t]/sqrt(1 - abar[t])) * eps_theta(x[t], t) )",
        "             + sigma[t] * z",
        "end for",
        "return x[0]",
    ]
    code2 = [
        "# cs231n/gaussian_diffusion.py  ->  p_sample()",
        "model_out = model(x_t, t)",
        "x_start   = predict_start_from_noise(x_t, t, model_out)  # invert x_t",
        "x_start.clamp_(-1.0, 1.0)        # keep generation stable",
        "mean, std = q_posterior(x_start, x_t, t)",
        "return mean + (t != 0) * std * torch.randn_like(x_t)",
    ]
    _code_panel(s, MARGIN, y, 7.15, algo2, code2, accent=AMBER)
    bullets(s, MARGIN + 7.45, y + 0.16, BODY_W - 7.45, 3.6, [
        "start from pure noise;",
        "one net call per step;",
        "T = 1000, so sampling is slow;",
        "sigma[t]*z keeps it stochastic.",
    ], size=12, gap=11)
    footer(s, n)

    # ---------- 15. Design choices
    n += 1
    s = blank(prs)
    y = slide_title(s, "Design choices that matter", "Step 5 \u00b7 practice")
    rows = [
        ("Noise schedule \u03b2\u209c",
         "linear (original DDPM)  \u00b7  cosine (better SNR)  \u00b7  "
         "sigmoid (above 64\u00d764)"),
        ("Reverse variance \u03c3\u209c\u00b2",
         "analytic value  \u00b7  DDPM fixes \u03c3\u209c\u00b2 = \u03b2\u209c  \u00b7  "
         "iDDPM learns a hybrid"),
        ("Network \u03b5\u03b8",
         "U-Net with skips  \u00b7  residual blocks  \u00b7  "
         "self-attention at low resolution"),
        ("Timestep conditioning",
         "sinusoidal embedding of t, injected by scale-and-shift"),
    ]
    yy = y + 0.02
    for name, desc in rows:
        rect(s, MARGIN, yy, BODY_W, 0.72, LIGHT)
        rect(s, MARGIN, yy, 0.05, 0.72, TEAL)
        txt(s, MARGIN + 0.24, yy + 0.11, 3.55, 0.30, name, size=14.5,
            bold=True, color=TEAL)
        txt(s, MARGIN + 3.85, yy + 0.12, BODY_W - 4.05, 0.50, desc, size=13,
            color=NAVY)
        yy += 0.81
    footer(s, n)

    # ---------- 16. Results & extensions
    n += 1
    s = blank(prs)
    y = slide_title(s, "Results and extensions", "Outcome")
    bullets(s, MARGIN, y, 6.05, 3.3, [
        "DDPM reported state-of-the-art FID on CIFAR-10 and high-quality 256\u00d7256 ImageNet samples.",
        "FID 3.17 on CIFAR-10 (unconditional) \u2014 competitive with the best GANs of its time.",
        "Quality improves with model size and with the number of sampling steps.",
    ], size=16, gap=6)
    txt(s, MARGIN + 6.55, y, 5.0, 0.35, "Direct descendants", size=16,
        bold=True, color=TEAL)
    ext = [
        "DDIM \u2014 deterministic, far fewer sampling steps",
        "Score-based SDE \u2014 continuous-time unification",
        "Classifier-free guidance \u2014 conditional control",
        "Latent diffusion \u2014 diffuse in a compressed latent space",
        "Diffusion Transformers \u2014 replace U-Net with a Transformer",
    ]
    yy = y + 0.42
    for e in ext:
        txt(s, MARGIN + 6.55, yy, 5.0, 0.34, "\u2013  " + e, size=14.5, color=NAVY)
        yy += 0.42
    footer(s, n)

    # ---------- 17. Takeaways
    n += 1
    s = blank(prs)
    rect(s, 0, 0, SW, SH, NAVY)
    rect(s, 0, 0, SW, 0.10, TEAL)
    txt(s, MARGIN + 0.25, 0.80, BODY_W - 0.5, 0.7, "Takeaways", size=34,
        bold=True, color=WHITE)
    rect(s, MARGIN + 0.25, 1.62, 1.8, 0.05, AMBER)
    pts = [
        "Diffusion replaces one hard generation problem with many easy denoising problems.",
        "The forward process is fixed; a Gaussian closed form gives x\u209c directly from x\u2080.",
        "The true reverse posterior is intractable, so we condition on x\u2080 to obtain a Gaussian.",
        "The variational bound reduces to a simple noise-regression loss L = E[\u2016\u03b5 \u2212 \u03b5\u03b8\u2016\u00b2].",
        "Training is stable and parallel; the price is slow, iterative sampling.",
    ]
    yy = 2.05
    for i, p_ in enumerate(pts, 1):
        txt(s, MARGIN + 0.25, yy, 0.45, 0.4, f"{i}.", size=17, bold=True,
            color=TEAL)
        txt(s, MARGIN + 0.85, yy, BODY_W - 1.4, 0.72, p_, size=17, color=WHITE)
        yy += 0.80
    footer_dark = txt(s, MARGIN + 0.25, SH - 0.85, BODY_W - 0.5, 0.4,
                      "Thank you \u2014 questions welcome", size=15, color=TEAL)

    # ---------- 18. References
    # 按幻灯片出现顺序列出全部被引文献；出处（会议/年份）已逐条核实。
    # ⚠️ 每条都【手动换行】：renderer 对 auto_size=NONE 的文本框不自动折行，
    #    文字会直接溢出右边界被裁切（这是本 deck 的既有做法，见各页 bullets）。
    n += 1
    s = blank(prs)
    y = slide_title(s, "References", "Sources")
    refs_l = [
        ("[1]", "Ho, Jain & Abbeel. Denoising Diffusion Probabilistic Models.\n"
                "NeurIPS 2020.  arXiv:2006.11239"),
        ("[2]", "Sohl-Dickstein et al. Deep Unsupervised Learning using\n"
                "Nonequilibrium Thermodynamics. ICML 2015.  arXiv:1503.03585"),
        ("[3]", "Song & Ermon. Generative Modeling by Estimating Gradients\n"
                "of the Data Distribution. NeurIPS 2019.  arXiv:1907.05600"),
        ("[4]", "Song, Meng & Ermon. Denoising Diffusion Implicit Models\n"
                "(DDIM). ICLR 2021.  arXiv:2010.02502"),
        ("[5]", "Song et al. Score-Based Generative Modeling through SDEs.\n"
                "ICLR 2021.  arXiv:2011.13456"),
    ]
    refs_r = [
        ("[6]", "Nichol & Dhariwal. Improved Denoising Diffusion\n"
                "Probabilistic Models. ICML 2021.  arXiv:2102.09672"),
        ("[7]", "Ho & Salimans. Classifier-Free Diffusion Guidance.\n"
                "2022.  arXiv:2207.12598"),
        ("[8]", "Rombach et al. High-Resolution Image Synthesis with\n"
                "Latent Diffusion Models. CVPR 2022.  arXiv:2112.10752"),
        ("[9]", "Peebles & Xie. Scalable Diffusion Models with\n"
                "Transformers (DiT). ICCV 2023.  arXiv:2212.09748"),
        ("[10]", "Liu, Gong & Liu. Flow Straight and Fast: Learning to\n"
                 "Generate and Transfer Data with Rectified Flow.\n"
                 "2022.  arXiv:2209.03003"),
    ]
    for col, refs in ((0, refs_l), (1, refs_r)):
        cx = MARGIN + col * 6.10
        yy = y + 0.02
        for tag, body in refs:
            nlines = body.count("\n") + 1
            txt(s, cx, yy + 0.03, 0.42, 0.26, tag, size=10.5, bold=True,
                color=TEAL)
            txt(s, cx + 0.44, yy, 5.30, 0.68, body, size=10.5, color=NAVY,
                line_spacing=0.98)
            yy += 0.26 * nlines + 0.08
    txt(s, MARGIN, y + 3.72, BODY_W, 0.30,
        "All references verified against the official proceedings / arXiv "
        "listings.", size=10.5, italic=True, color=GREY)
    footer(s, n)

    prs.save(path)
    print("saved", path, "slides:", len(prs.slides.__iter__.__self__._sldIdLst))


if __name__ == "__main__":
    build(OUT_PPTX)
