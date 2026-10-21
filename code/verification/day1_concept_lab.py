# -*- coding: utf-8 -*-
"""
Day 1 概念重建实验 —— 针对摸底测验 A2/A3/A4 的三个误解
================================================================
误解 A2：以为反向过程"参考最终清晰图像 x0"
误解 A3：以为 abar_t 是"噪声水平"（方向反了）
误解 A4：以为去噪就是 x_{t-1} = x_t - eps

本实验用纯数值方式把三件事"打印出来"，不含任何训练。
运行： python day1_concept_lab.py
输出： 终端报告 + figs/*.png
"""
import math
import os
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ---------- 中文字体 ----------
for _f in ["C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf"]:
    if Path(_f).exists():
        font_manager.fontManager.addfont(_f)
        plt.rcParams["font.sans-serif"] = [font_manager.FontProperties(fname=_f).get_name()]
        break
plt.rcParams["axes.unicode_minus"] = False

OUT = Path(__file__).resolve().parent.parent / "figs"
OUT.mkdir(exist_ok=True)
torch.manual_seed(0)
np.random.seed(0)

BAR = "=" * 68


def make_schedule(T=1000, beta_start=1e-4, beta_end=0.02):
    """线性 beta 调度（DDPM 原论文），返回 (betas, alphas, alphas_cumprod)。"""
    betas = torch.linspace(beta_start, beta_end, T, dtype=torch.float64)
    alphas = 1.0 - betas
    abar = torch.cumprod(alphas, dim=0)
    return betas, alphas, abar


def extract(a, t):
    """把 (T,) 的系数按 batch 里的 t 取出来（1D 版）。"""
    return a[t]


# ==================================================================
# 实验 1：abar_t 到底是"信号"还是"噪声"？ —— 修 A3
# ==================================================================
def exp1_signal_vs_noise(betas, abar):
    print(BAR)
    print("实验 1：abar_t 的方向 —— 它衡量【信号还剩多少】，不是【噪声有多少】")
    print(BAR)
    print("核心式子: x_t = sqrt(abar_t) * x0 + sqrt(1-abar_t) * eps")
    print()
    print(f"{'t':>5} {'abar_t':>10} {'信号幅度':>10} {'噪声幅度':>10} "
          f"{'幅度比 噪/信':>13} {'平方和':>9}")
    print("-" * 68)

    rows = [0, 1, 50, 100, 200, 300, 500, 700, 900, 999]
    for t in rows:
        a = float(abar[t])
        sig = math.sqrt(a)          # 信号幅度系数
        noi = math.sqrt(1.0 - a)    # 噪声幅度系数
        ratio = noi / sig if sig > 0 else float("inf")
        print(f"{t:>5} {a:>10.6f} {sig:>10.6f} {noi:>10.6f} "
              f"{ratio:>13.4f} {sig**2 + noi**2:>9.6f}")

    print("-" * 68)
    print("观察：")
    print("  1. abar_t 从 1 递减到 ~0.0000 —— 它衡量【信号保留的方差比例】")
    print("  2. 信号幅度 = sqrt(abar_t)：t=0 时 1.0（信号完好），t=999 时 ~0.006（信号没了）")
    print("  3. 噪声幅度 = sqrt(1-abar_t)：趋势【完全相反】")
    print("  4. 最后一列【平方和恒等于 1】= 方差保持 (Variance Preserving, VP)")
    print()
    print("  => 所以 abar_t 越小 = 噪声越多。你的笔记写的是'累积噪声水平'，")
    print("     而'噪声水平'应该是 1-abar_t。方向刚好反了。")
    print()

    # ---- 图 1 ----
    fig, ax = plt.subplots(figsize=(11, 4.3), dpi=150)
    tt = np.arange(len(abar))
    ax.plot(tt, abar.numpy(), lw=2, color="#1f77b4", label=r"$\bar\alpha_t$  信号保留（方差比例）")
    ax.plot(tt, 1 - abar.numpy(), lw=2, color="#d62728", label=r"$1-\bar\alpha_t$  噪声占比")
    ax.plot(tt, np.sqrt(abar.numpy()), lw=1.5, ls="--", color="#1f77b4",
            label=r"$\sqrt{\bar\alpha_t}$  信号幅度")
    ax.plot(tt, np.sqrt(1 - abar.numpy()), lw=1.5, ls="--", color="#d62728",
            label=r"$\sqrt{1-\bar\alpha_t}$  噪声幅度")
    ax.axhline(1.0, color="gray", lw=0.8, ls=":")
    ax.set_xlabel("时间步 $t$")
    ax.set_ylabel("系数值")
    ax.set_title(r"$\bar\alpha_t$ 递减 = 信号在消失；它与 $1-\bar\alpha_t$ 方向相反")
    ax.legend(fontsize=8, loc="center right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "01_abar_direction.png")
    plt.close(fig)

    # ---- 图 2：1D 双峰数据随 t 的变化（上行=解析密度，下行=按原始归属着色的样品）----
    N = 3000
    x0 = torch.where(torch.rand(N, dtype=torch.float64) < 0.5,
                     torch.tensor(-1.0, dtype=torch.float64),
                     torch.tensor(1.0, dtype=torch.float64))
    eps = torch.randn_like(x0)
    color = np.where(x0.numpy() < 0, "#1f77b4", "#d62728")

    show_t = [0, 5, 100, 300, 600, 999]
    XLO, XHI = -6.0, 6.0
    fig, axes = plt.subplots(2, len(show_t), figsize=(16.5, 5.6), dpi=150,
                             sharex=True, sharey="row",
                             gridspec_kw={"height_ratios": [1.15, 1.0], "hspace": 0.12})
    for k, t in enumerate(show_t):
        a = abar[t]
        sig, nz = math.sqrt(float(a)), math.sqrt(1 - float(a))
        xt = (torch.sqrt(a) * x0 + torch.sqrt(1 - a) * eps).numpy()

        # 上：解析密度（两个高斯的混合）
        gx = np.linspace(XLO, XHI, 900)
        dens = np.zeros_like(gx)
        for mu in (-sig, sig):
            dens += 0.5 / (nz * math.sqrt(2 * math.pi)) * np.exp(-((gx - mu) ** 2) / (2 * nz ** 2))
        ax = axes[0, k]
        ax.fill_between(gx, 0, dens, color="#444444", alpha=0.22)
        ax.plot(gx, dens, color="black", lw=1.3)
        ax.set_title(f"$t={t}$\n" + rf"信号幅度 $\sqrt{{\bar\alpha_t}}={sig:.3f}$"
                     + "\n" + rf"噪声幅度 $\sqrt{{1-\bar\alpha_t}}={nz:.3f}$", fontsize=8.5)
        ax.set_xlim(XLO, XHI)
        ax.grid(alpha=0.2)
        if k == 0:
            ax.set_ylabel("解析密度 $p(x_t)$\n（两个高斯混合）", fontsize=8.5)

        # 下：样品，按原始归属着色 -> 看两簇如何糊成一团
        ax2 = axes[1, k]
        ax2.scatter(xt, np.random.uniform(0, 1, N), s=2.5, c=color, alpha=0.4)
        ax2.set_ylim(0, 1.25)
        ax2.set_yticks([])
        ax2.set_xlabel("$x_t$", fontsize=9)
        ax2.grid(alpha=0.2)
        if k == 0:
            ax2.set_ylabel("样品\n(蓝/红=原始两种数据)", fontsize=8.5)

    fig.suptitle("前向加噪：蓝/红是两种原始数据；$t$ 变大后两簇被噪声吞没（信号消失）\n"
                 "注意：$\\bar\\alpha_t$ 变小时，噪声幅度在【变大】—— 它与噪声方向相反",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(OUT / "02_forward_noising.png")
    plt.close(fig)
    print(f"  [已保存] {OUT / '01_abar_direction.png'}")
    print(f"  [已保存] {OUT / '02_forward_noising.png'}")
    print()


# ==================================================================
# 实验 2：x0 与 eps 是【一一对应】 —— 修 A4
# ==================================================================
def exp2_bijection(abar):
    print(BAR)
    print("实验 2：x0 与 eps 的一一对应 —— 修 A4")
    print(BAR)
    print("给定 x_t 与 t，下面两条式子【互为逆运算】：")
    print("   x0  = (x_t - sqrt(1-abar_t) * eps) / sqrt(abar_t)")
    print("   eps = (x_t - sqrt(abar_t)   * x0 ) / sqrt(1-abar_t)")
    print()
    print("你要写的那条 x_(t-1) = x_t - eps 是【错的】，正确对照如下：")
    print("   你写的:  x_(t-1) = x_t - eps_(t-1)")
    print("   正确的是 x0 = (x_t - sqrt(1-abar_t)*eps) / sqrt(abar_t)")
    print("   差别一: 分母漏了 sqrt(abar_t)")
    print("   差别二: eps 项漏了系数 sqrt(1-abar_t)")
    print("   差别三: 反解出的是 x0，不是 x_(t-1)（那是另一条式子）")
    print()

    x0 = torch.randn(100000, dtype=torch.float64)
    eps = torch.randn_like(x0)
    print(f"{'t':>5} {'abar_t':>10} {'反解x0误差':>14} {'反解eps误差':>14} {'相对误差':>12}")
    print("-" * 68)
    worst = 0.0
    for t in [1, 50, 200, 500, 800, 999]:
        a = abar[t]
        # 前向：造出 x_t
        x_t = torch.sqrt(a) * x0 + torch.sqrt(1 - a) * eps
        # 反解 x0（用真 eps）—— 这就是"预测噪声 ⇒ 得到干净图"
        x0_hat = (x_t - torch.sqrt(1 - a) * eps) / torch.sqrt(a)
        # 反解 eps（用真 x0）—— 这就是"预测干净图 ⇒ 得到噪声"
        eps_hat = (x_t - torch.sqrt(a) * x0) / torch.sqrt(1 - a)
        e_x0 = float((x0_hat - x0).abs().max())
        e_eps = float((eps_hat - eps).abs().max())
        rel = float((x0_hat - x0).norm() / x0.norm())
        worst = max(worst, rel)
        print(f"{t:>5} {float(a):>10.6f} {e_x0:>14.3e} {e_eps:>14.3e} {rel:>12.3e}")
    print("-" * 68)
    print(f"最大相对误差 = {worst:.3e}  →  即机器精度量级，两式【严格】互逆。")
    print()
    print("  结论（A4 的标准答案）：x_t 已知时，x0 与 eps 一一对应，")
    print("  知道一个就能解出另一个 ⇒ '预测 eps' 与 '预测 x0' 是同一问题的")
    print("  两种参数化，信息完全等价，只是输出数值范围不同")
    print("  （x0 在 [-1,1]，eps 是 N(0,1)）。")
    print()

    # ---- 图 3 ----
    t = 300
    a = abar[t]
    x0s = torch.randn(3000, dtype=torch.float64)
    epss = torch.randn_like(x0s)
    x_ts = torch.sqrt(a) * x0s + torch.sqrt(1 - a) * epss
    x0_hat = (x_ts - torch.sqrt(1 - a) * epss) / torch.sqrt(a)
    eps_hat = (x_ts - torch.sqrt(a) * x0s) / torch.sqrt(1 - a)

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.4), dpi=150)
    for ax, (val, name, col) in zip(axes, [
        (x0s.numpy(), r"真实 $x_0$", "#1f77b4"),
        (x_ts.numpy(), r"加噪图 $x_t$", "#7f7f7f"),
        (x0_hat.numpy(), r"由 $\varepsilon$ 反解 $\hat{x}_0$", "#1f77b4"),
        (eps_hat.numpy(), r"由 $x_0$ 反解 $\hat{\varepsilon}$", "#d62728"),
    ]):
        ax.hist(val, bins=60, color=col, alpha=0.8)
        ax.set_title(name, fontsize=10)
        ax.grid(alpha=0.2)
        ax.set_yticks([])
    axes[0].text(0.5, 0.92, f"$t={t}$", transform=axes[0].transAxes,
                 ha="center", fontsize=9, color="gray")
    fig.suptitle("一一对应的数值验证：第1图与第3图完全相同、第2图与第4图可互相还原", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(OUT / "03_bijection.png")
    plt.close(fig)
    print(f"  [已保存] {OUT / '03_bijection.png'}")
    print()


# ==================================================================
# 实验 3：训练期 vs 推理期，网络到底看到什么 —— 修 A2
# ==================================================================
def exp3_train_vs_sample(abar):
    print(BAR)
    print("实验 3：训练期 vs 推理期 —— 网络到底拿到什么输入？（修 A2）")
    print(BAR)

    T = len(abar)
    xt = torch.randn(2, 3, 8, 8)
    t = torch.tensor([231, 700])

    print("【训练期】p_losses 的实际数据流：")
    print("   t                = [231, 700]        <- 随机抽的时间步，是【输入】")
    print("   x_start (x0)     = 干净图            <- 只用来【构造】x_t 和当【监督目标】")
    print("   noise (eps)      = N(0,I)            <- 只用来【构造】x_t 和当【监督目标】")
    print("   x_t              = sqrt(abar_t)*x0 + sqrt(1-abar_t)*eps")
    print("   ------------------------------------------------")
    print("   送进网络 model(x_t, t) 的只有： x_t 和 t      <- 就这两个！")
    print("   x0 和 eps 不进网络，它们生成 target 用于算 loss")
    print()
    print("   loss = ||target - model(x_t, t)||^2")
    print("        target = eps        (objective='pred_noise')")
    print("        target = x0         (objective='pred_x_start')   <- 二选一，两种都合法")
    print()

    print("【推理期 / 采样】p_sample 的实际数据流：")
    print("   x_T = N(0,I)  纯噪声起步")
    print("   for t = T..1:")
    print("       eps_theta = model(x_t, t)                  <- 输入仍是 x_t, t")
    print("       x0_hat    = (x_t - sqrt(1-abar_t)*eps_theta)/sqrt(abar_t)   <- 反解")
    print("       x0_hat    = clamp(x0_hat, -1, 1)")
    print("       x_{t-1}   = 后验均值(x_t, x0_hat) + 后验标准差 * z")
    print()
    print("   >>> 整条循环里【根本没有 x0 输入】！")
    print("   x0 只在训练期作为监督信号存在，生成时它就是我们要求的答案本身。")
    print()

    # 把两条链路显式渲染出来
    fig, ax = plt.subplots(figsize=(12.8, 6.4), dpi=150)
    ax.axis("off")

    def box(x, y, w, h, text, fc, ec="black", fs=8.5, weight="normal"):
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec,
                                   lw=1.3, zorder=2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, zorder=3, weight=weight, linespacing=1.5)

    def arrow(x1, y1, x2, y2, color="black", style="-|>", lw=1.4, ls="-"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                    linestyle=ls, shrinkA=1, shrinkB=1), zorder=1)

    # ============ 训练期 ============
    ax.text(0.5, 0.99, "训练期：$x_0$ 与 $\\varepsilon$ 只当「监督」，不进模型",
            ha="center", fontsize=11.5, weight="bold", color="#1f4e79", va="top")
    box(0.02, 0.755, 0.13, 0.115, "干净图 $x_0$", "#cfe2f3")
    box(0.02, 0.625, 0.13, 0.115, "噪声 $\\varepsilon\\sim N(0,I)$", "#f4cccc")
    box(0.20, 0.69, 0.155, 0.115,
        "前向加噪\n$x_t=\\sqrt{\\bar\\alpha_t}x_0+\\sqrt{1-\\bar\\alpha_t}\\varepsilon$", "#fff2cc", fs=8)
    box(0.405, 0.69, 0.11, 0.115, "时间步 $t$", "#e2efda")
    box(0.585, 0.675, 0.15, 0.145, "噪声预测网络\n$\\varepsilon_\\theta(x_t,t)$",
        "#d9ead3", fs=9, weight="bold")
    box(0.795, 0.69, 0.185, 0.115, "预测 $\\hat\\varepsilon$ 或 $\\hat{x}_0$", "#d9ead3", fs=8)
    box(0.585, 0.50, 0.15, 0.115, "loss\n$\\|target-\\hat{y}\\|^2$", "#fce5cd", fs=8)
    box(0.795, 0.50, 0.185, 0.115, "监督 target\n$\\varepsilon$ 或 $x_0$", "#f4cccc", fs=8)

    arrow(0.15, 0.812, 0.20, 0.762)
    arrow(0.15, 0.682, 0.20, 0.732)
    arrow(0.355, 0.747, 0.585, 0.747, color="#38761d", lw=2.2)
    arrow(0.515, 0.747, 0.585, 0.747, color="#38761d", lw=2.2)
    arrow(0.735, 0.747, 0.795, 0.747, color="black", lw=1.5)
    arrow(0.8875, 0.69, 0.8875, 0.615, color="#cc0000", ls="--")
    arrow(0.795, 0.557, 0.735, 0.557, color="#cc0000")
    arrow(0.660, 0.675, 0.660, 0.615, color="#cc0000")
    ax.text(0.752, 0.588, "对比", fontsize=8.5, color="#cc0000", weight="bold",
            ha="center")
    # 绿色标注：只有 x_t 与 t 进模型（放在方块【上方】的空白处，不压方块）
    ax.text(0.62, 0.885, "只有这两个进模型", fontsize=9, color="#38761d",
            style="italic", ha="left", va="top")

    # ============ 推理期 ============
    ax.text(0.5, 0.425, "推理期：模型只看 $x_t$ 与 $t$，全程没有 $x_0$",
            ha="center", fontsize=11.5, weight="bold", color="#7b2d26")
    box(0.02, 0.205, 0.13, 0.115, "纯噪声\n$x_T\\sim N(0,I)$", "#efefef", fs=8)
    box(0.20, 0.185, 0.15, 0.155, "噪声预测网络\n$\\varepsilon_\\theta(x_t,t)$",
        "#d9ead3", fs=9, weight="bold")
    box(0.40, 0.205, 0.185, 0.115,
        "反解\n$\\hat{x}_0=\\frac{x_t-\\sqrt{1-\\bar\\alpha_t}\\varepsilon_\\theta}{\\sqrt{\\bar\\alpha_t}}$",
        "#fff2cc", fs=7.5)
    box(0.625, 0.205, 0.10, 0.115, "clamp\n$[-1,1]$", "#fff2cc", fs=8)
    box(0.765, 0.185, 0.215, 0.155, "后验采样\n$x_{t-1}=\\tilde\\mu_t+\\tilde\\sigma_t z$",
        "#cfe2f3", fs=8.5)

    arrow(0.15, 0.2625, 0.20, 0.2625, lw=2.2, color="#38761d")
    arrow(0.35, 0.2625, 0.40, 0.2625, lw=2.2, color="#38761d")
    arrow(0.585, 0.2625, 0.625, 0.2625, lw=2.2, color="#38761d")
    arrow(0.725, 0.2625, 0.765, 0.2625, lw=2.2, color="#38761d")
    ax.text(0.152, 0.355, "输入依然是\n$x_t$ 与 $t$", fontsize=8.5, color="#38761d",
            style="italic", ha="left", va="center")
    # 回环箭头：从输出折回输入
    ax.annotate("", xy=(0.275, 0.185), xytext=(0.8725, 0.185),
                arrowprops=dict(arrowstyle="-|>", color="#38761d", lw=1.8,
                                connectionstyle="bar,fraction=-0.18"), zorder=1)
    ax.text(0.575, 0.062,
            "循环 $t=T,\\ldots,1$（每一步都把上一步的 $x_t$ 送回模型）",
            ha="center", fontsize=9, color="#38761d")

    ax.add_patch(plt.Rectangle((0.003, 0.003), 0.994, 0.994, fill=False,
                               edgecolor="#bbbbbb", lw=1))
    fig.savefig(OUT / "04_train_vs_sample.png", bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"  [已保存] {OUT / '04_train_vs_sample.png'}")
    print()


# ==================================================================
# 实验 4：亲手验证你写错的那条式子会崩成什么样 —— 修 A4 补充
# ==================================================================
def exp4_wrong_recursion(abar):
    print(BAR)
    print("实验 4：你写的递推式 x_{t-1} = x_t - eps 会怎样？（修 A4 补充）")
    print(BAR)
    print("正确递推（DDPM Algorithm 2，用 eps 直接构造 x_{t-1}）：")
    print("   x_{t-1} = (1/sqrt(alpha_t)) * (x_t - (beta_t/sqrt(1-abar_t)) * eps_theta)")
    print()
    print("你写的：")
    print("   x_{t-1} = x_t - eps        <- 缺少 1/sqrt(alpha_t) 与 beta_t/sqrt(1-abar_t)")
    print()

    T = 1000
    t = 500
    a_t = float(1 - (1e-4 + (0.02 - 1e-4) * t / (T - 1)))  # beta_t, 线性调度
    beta_t = 1 - a_t
    ab = float(abar[t])
    # 真实采样时 eps ~ N(0,I)，尺度就是 1 左右，所以取 eps = 1.0 是【典型】情形
    eps = 1.0
    x_t = 1.0  # 任取一个 x_t 做演示

    correct = (1 / math.sqrt(a_t)) * (x_t - (beta_t / math.sqrt(1 - ab)) * eps)
    yours = x_t - eps
    print(f"   t = {t},  beta_t = {beta_t:.6f},  abar_t = {ab:.6f},  eps = {eps:+.4f}")
    print(f"   正确的 x_(t-1) = {correct:+.6f}")
    print(f"   你写的 x_(t-1) = {yours:+.6f}")
    print(f"   差值           = {correct - yours:+.6f}   (偏差 {abs(correct-yours)/abs(correct)*100:.1f}%)")
    print()
    print("   可以看到：'减去噪声'这个直觉丢掉了两处关键系数，结果完全对不上。")
    print("   而且每步都错的话，1000 步迭代会累积放大 —— 这就是为什么必须用")
    print("   反解 x0 + 后验采样这套正式流程，而不是凭直觉相减。")
    print()


def main():
    betas, alphas, abar = make_schedule()
    print()
    print("Day 1 概念重建实验  |  T=1000, 线性 beta 调度 (1e-4 → 0.02)")
    print()
    exp1_signal_vs_noise(betas, abar)
    exp2_bijection(abar)
    exp3_train_vs_sample(abar)
    exp4_wrong_recursion(abar)
    print(BAR)
    print("全部实验完成。图已保存到 figs/")
    print("请重点回看：02_forward_noising.png 与 04_train_vs_sample.png")
    print(BAR)


if __name__ == "__main__":
    main()
