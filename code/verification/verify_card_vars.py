# -*- coding: utf-8 -*-
"""
核验背诵卡 §0 中写下的几处数值声明：
  1. 作业配置 T=100 的默认 sigmoid 调度下，abar_T ≈ ?
  2. 经典 DDPM T=1000 线性调度下，abar_T ≈ ?
  3. posterior_mean_coef1 / coef2 是否等于文档所写的两个系数
  4. posterior_std^2 是否等于 beta_t(1-abar_{t-1})/(1-abar_t)
"""
import math
import torch


def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    scale = 1000.0 / timesteps
    return torch.linspace(scale * beta_start, scale * beta_end, timesteps,
                          dtype=torch.float64)


def sigmoid_beta_schedule(timesteps, start=-3, end=3, tau=1, clamp_min=1e-5):
    """严格照抄官方 gaussian_diffusion.py 第 295-310 行
    （注意：签名里的 clamp_min 在函数体内【从未被使用】，属官方遗留问题）"""
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (
        v_end - v_start
    )
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def report(betas, name):
    alphas = 1.0 - betas
    abar = torch.cumprod(alphas, dim=0)
    T = len(betas)
    print(f"\n{name}")
    print(f"   T                = {T}")
    print(f"   beta_0           = {float(betas[0]):.8f}")
    print(f"   beta_T-1         = {float(betas[-1]):.8f}")
    print(f"   abar_0           = {float(abar[0]):.8f}")
    print(f"   abar_T-1  (末态) = {float(abar[-1]):.3e}")
    print(f"   sqrt(abar_T-1)   = {math.sqrt(float(abar[-1])):.3e}   (信号幅度)")
    print(f"   sqrt(1-abar_T-1) = {math.sqrt(1-float(abar[-1])):.8f}   (噪声幅度)")
    return abar


print("=" * 66)
print("核验 1 & 2：末态 abar_T 到底有多接近 0")
print("=" * 66)
report(linear_beta_schedule(1000), "经典 DDPM：T=1000, 线性调度 (Ho et al. 2020)")
report(sigmoid_beta_schedule(100), "作业配置：T=100 (sigmoid)")

# ---------------------------------------------------------------
print()
print("=" * 66)
print("核验 3 & 4：后验系数与后验方差（对照 DDPM 论文 Eq.6/7）")
print("=" * 66)
T = 1000
betas = linear_beta_schedule(T)
alphas = 1.0 - betas
abar = torch.cumprod(alphas, dim=0)
abar_prev = torch.cat([torch.ones(1, dtype=torch.float64), abar[:-1]])

coef1 = betas * torch.sqrt(abar_prev) / (1.0 - abar)
coef2 = (1.0 - abar_prev) * torch.sqrt(alphas) / (1.0 - abar)
pvar = betas * (1.0 - abar_prev) / (1.0 - abar)

print(f"{'t':>5} {'coef1 (x0系数)':>18} {'手算公式':>18} "
      f"{'coef2 (xt系数)':>16} {'手算公式':>16} {'posterior_var':>14} {'手算sigma^2':>14}")
print("-" * 118)
ok = True
for t in [1, 100, 500, 900, 999]:
    a_t = float(alphas[t]); ab = float(abar[t]); abp = float(abar_prev[t])
    b_t = float(betas[t])
    m1 = math.sqrt(abp) * b_t / (1 - ab)
    m2 = math.sqrt(a_t) * (1 - abp) / (1 - ab)
    m3 = b_t * (1 - abp) / (1 - ab)
    d1 = abs(float(coef1[t]) - m1)
    d2 = abs(float(coef2[t]) - m2)
    d3 = abs(float(pvar[t]) - m3)
    ok &= (d1 < 1e-15 and d2 < 1e-15 and d3 < 1e-15)
    print(f"{t:>5} {float(coef1[t]):>18.10f} {m1:>18.10f} "
          f"{float(coef2[t]):>16.10f} {m2:>16.10f} "
          f"{float(pvar[t]):>14.8f} {m3:>14.8f}")
print("-" * 118)
print(f"三种系数的代码实现与手写公式最大偏差 < 1e-15 : {'✅ 通过' if ok else '❌ 不通过'}")

print()
print("=" * 66)
print("核验 5：'方差保持' 恒等式 (sqrt(abar))^2 + (sqrt(1-abar))^2 == 1")
print("=" * 66)
err = ((torch.sqrt(abar) ** 2 + torch.sqrt(1 - abar) ** 2) - 1).abs().max()
print(f"   1000 个时间步上的最大偏差 = {float(err):.3e}  "
      f"{'✅ 恒等于 1' if float(err) < 1e-15 else '❌'}")

print()
print("=" * 66)
print("核验 6：作业 T=100 时 abar_t 的方向（供背诵卡 §0.2 参考）")
print("=" * 66)
ab100 = torch.cumprod(1 - sigmoid_beta_schedule(100), dim=0)
for t in [0, 25, 50, 75, 99]:
    a = float(ab100[t])
    print(f"   t={t:>3}  abar_t={a:.6f}  信号幅度={math.sqrt(a):.4f}  "
          f"噪声幅度={math.sqrt(1-a):.4f}")
