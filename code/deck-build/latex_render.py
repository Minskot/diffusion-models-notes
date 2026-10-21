# -*- coding: utf-8 -*-
"""真 LaTeX 渲染工具（用于 deck 公式）。

链路:
  1. 把多个公式合成一个多页 LaTeX 文档（article + geometry），pdflatex 编译【一次】
  2. Ghostscript -sDEVICE=bbox 取得每页内容的精确边界框
  3. Ghostscript -sDEVICE=pngalpha 按该边界框渲染成 400dpi 透明 PNG

相比 mathtext 的优点: 真 Computer Modern、真数学斜体、真 TeX 间距、真 LaTeX 排版。
"""
import os
import re
import subprocess

TEXBIN = r"D:\CodingPrograms\TeXLive\texliveIso\bin\windows"
GSROOT = r"D:\CodingPrograms\TeXLive\texliveIso\tlpkg\tlgs"
GS = os.path.join(GSROOT, r"bin\gswin64c.exe")
PDFLATEX = os.path.join(TEXBIN, "pdflatex.exe")

# TeX Live 自带的 Ghostscript 是「便携版」，必须显式指定 GS_LIB，
# 否则报 "Can't find initialization file gs_init.ps / kfwin32.ps"
GS_LIB = ";".join([
    os.path.join(GSROOT, "Resource", "Init"),
    os.path.join(GSROOT, "lib"),
    os.path.join(GSROOT, "Resource", "Font"),
    os.path.join(GSROOT, "kanji"),
])


def _env():
    env = dict(os.environ)
    env["PATH"] = TEXBIN + os.pathsep + env.get("PATH", "")
    env["GS_LIB"] = GS_LIB
    return env

# 版面尺寸（pt）。standalone 类没装，用 geometry 自己控制。
# 页宽取得足够大，避免 LaTeX 自动折行（折行只应发生在 gathered 内部）。
PAGE_W = 2000
PAGE_H = 300

BASE_FONTSIZE = 26.0    # 基准字号（pt）；deck 中正文公式的实际显示字号由它决定


def build_document(equations, fontsizes=None, page_w=None, page_h=None):
    """equations: list[str]，每项是完整的 LaTeX 数学内容（不含 $ 定界符）。

    每页一个公式。fontsizes 可给每个公式单独指定字号（pt）——
    用于让"过宽"的公式用更小字号渲染，从而在 deck 里占满面板宽度时
    仍保持与其他公式一致的物理字号。
    """
    if fontsizes is None:
        fontsizes = [BASE_FONTSIZE] * len(equations)
    pw = PAGE_W if page_w is None else page_w
    ph = PAGE_H if page_h is None else page_h
    body = []
    for i, (eq, fs) in enumerate(zip(equations, fontsizes)):
        lead = max(12, int(round(fs * 1.25)))
        sel = r"\fontsize{%g}{%d}\selectfont" % (fs, lead)
        body.append(r"\begin{center}" + sel + r"$\displaystyle " + eq + r"$"
                    r"\end{center}")
        if i != len(equations) - 1:
            body.append(r"\newpage")
    return (
        "\\documentclass[12pt]{article}\n"
        "\\usepackage{amsmath,amssymb,amsfonts}\n"
        "\\usepackage[paperwidth=%dpt,paperheight=%dpt,margin=0pt,"
        "noheadfoot]{geometry}\n" % (pw, ph)
        + "\\pagestyle{empty}\n"
        + "\\setlength{\\parindent}{0pt}\n"
        + "\\begin{document}\n"
        + "\n".join(body) + "\n"
        + "\\end{document}\n"
    )


def measure_ink_widths(equations, fontsizes=None, workdir=None, dpi=100):
    """测量每个公式的真实墨迹宽度（pt）与高度（pt）。

    用于两遍渲染：先量出宽度，再决定每个公式该用什么字号。
    """
    import tempfile
    tmp = workdir or tempfile.mkdtemp(prefix="eqmeasure_")
    pdf = compile_pdf(build_document(equations, fontsizes), tmp, name="measure")
    bb = page_bboxes(pdf)
    return [(x1 - x0, y1 - y0) for (x0, y0, x1, y1) in bb]


def compile_pdf(tex_source, workdir, name="eqs"):
    os.makedirs(workdir, exist_ok=True)
    tex = os.path.join(workdir, name + ".tex")
    with open(tex, "w", encoding="utf-8") as f:
        f.write(tex_source)
    env = _env()
    r = subprocess.run([PDFLATEX, "-interaction=nonstopmode", "-halt-on-error",
                        name + ".tex"],
                       cwd=workdir, capture_output=True, text=True, env=env,
                       encoding="utf-8", errors="replace")
    pdf = os.path.join(workdir, name + ".pdf")
    if r.returncode != 0 or not os.path.exists(pdf):
        raise RuntimeError("pdflatex 失败:\n" + "\n".join(r.stdout.splitlines()[-30:]))
    return pdf


_BBOX_RE = re.compile(r"%%HiResBoundingBox:\s*([\d.eE+-]+)\s+([\d.eE+-]+)\s+"
                      r"([\d.eE+-]+)\s+([\d.eE+-]+)")


def page_bboxes(pdf):
    """返回每页 (x0, y0, x1, y1)，单位 pt。设备坐标 y 向上。"""
    r = subprocess.run([GS, "-dSAFER", "-dBATCH", "-dNOPAUSE", "-q",
                        "-sDEVICE=bbox", pdf],
                       capture_output=True, text=True, env=_env())
    # 注意：gs 的 bbox 信息输出到 stderr，不是 stdout
    text = (r.stdout or "") + "\n" + (r.stderr or "")
    out = []
    for line in text.splitlines():
        m = _BBOX_RE.search(line)
        if m:
            out.append(tuple(float(g) for g in m.groups()))
    return out


def render_pages(pdf, bboxes, outdir, prefix, dpi=400, page_h=PAGE_H,
                 deterministic=False):
    """把每页按 bbox 裁切渲染成透明 PNG，返回文件路径列表。

    实现方式：先用 gs 把【整页】渲染成 400dpi PNG（含透明通道），
    再用 PIL 按 bbox 精确裁切。

    deterministic=True 时传入 -dNOTIMESTAMP 等参数，使同等输入产生
    字节级一致的输出（便于用哈希校验「文件是否真的被更新过」）。
    """
    from PIL import Image
    os.makedirs(outdir, exist_ok=True)
    env = _env()
    paths = []
    scale = dpi / 72.0                      # pt -> px
    extra = ["-dNOTIMESTAMP"] if deterministic else []
    for i, (x0, y0, x1, y1) in enumerate(bboxes, start=1):
        raw = os.path.join(outdir, f"_{prefix}_p{i}_full.png")
        cmd = [GS, "-dSAFER", "-dBATCH", "-dNOPAUSE", "-q",
               "-sDEVICE=pngalpha", f"-r{dpi}",
               "-dGraphicsAlphaBits=4", "-dTextAlphaBits=4",
               *extra,
               f"-dFirstPage={i}", f"-dLastPage={i}",
               f"-sOutputFile={raw}", pdf]
        r = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if not os.path.exists(raw):
            raise RuntimeError(f"第 {i} 页渲染失败: {r.stderr[-800:]}")

        im = Image.open(raw).convert("RGBA")
        # gs bbox 的 y 自底向上；图片 y 自顶向下 -> 翻转
        top = int(round((page_h - y1) * scale))
        bottom = int(round((page_h - y0) * scale))
        left = int(round(x0 * scale))
        right = int(round(x1 * scale))
        pad = 6
        box = (max(0, left - pad), max(0, top - pad),
               min(im.width, right + pad), min(im.height, bottom + pad))
        cropped = im.crop(box)

        # 再按实际非透明像素收紧一次（去掉多余空白）
        bbox_alpha = cropped.split()[3].getbbox()
        if bbox_alpha:
            cropped = cropped.crop(bbox_alpha)

        out = os.path.join(outdir, f"{prefix}_p{i}.png")
        cropped.save(out)
        os.remove(raw)
        paths.append(out)
    return paths


if __name__ == "__main__":
    HERE = os.path.dirname(os.path.abspath(__file__))
    from latex_spec import DISPLAY

    keys = ["idea_forward", "fwd_def", "fwd_reparam", "post_mean", "loss_final"]
    eqs = [DISPLAY[k].replace("$", "") for k in keys]

    work = os.path.join(HERE, "_fonttest", "texfull")
    pdf = compile_pdf(build_document(eqs), work)
    print("PDF:", pdf, os.path.getsize(pdf), "bytes")
    bb = page_bboxes(pdf)
    print(f"取得 {len(bb)} 页 bbox:")
    for i, b in enumerate(bb, 1):
        print(f"   page {i}: x0={b[0]:.1f} y0={b[1]:.1f} x1={b[2]:.1f} y1={b[3]:.1f}"
              f"  -> {b[2]-b[0]:.1f} x {b[3]-b[1]:.1f} pt")
    paths = render_pages(pdf, bb, os.path.join(work, "png"), "eqtest")
    from PIL import Image
    for k, p in zip(keys, paths):
        im = Image.open(p)
        print(f"   {k:14s} {im.size}  mode={im.mode}")
    print("\n输出目录:", os.path.join(work, "png"))
