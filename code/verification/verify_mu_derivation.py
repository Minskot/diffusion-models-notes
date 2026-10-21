# -*- coding: utf-8 -*-
"""核验后验均值「代入 x0 消元」这一步的推导。

要验证三件事:
  1. 恒等式  alpha_t*(1-abar_{t-1}) + (1-alpha_t) = 1 - abar_t
  2. 恒等式  (1-alpha_t)*sqrt(1-abar_t) = beta_t*sqrt(1-abar_t) / ... 形式正确性
  3. 第 1 行(原始后验均值, 含 x0) 与 第 3 行(消去 x0 的噪声形式) 数值相等
     并对比「图里第 2 行那个 + 号写法」会差多少
"""
import math
import torch

T = 1000
betas = torch.linspace(1e-4, 0.02, T, dtype=torch.float64)
alphas = 1.0 - betas
abar = torch.cumprod(alphas, dim=0)
abar_prev = torch.cat([torch.ones(1, dtype=torch.float64), abar[:-1]])

print("=" * 72)
print("核验 1：恒等式  alpha_t*(1-abar_{t-1}) + (1-alpha_t) = 1 - abar_t")
print("=" * 72)
print(f"{'t':>5} {'左式':>18} {'1-abar_t':>18} {'差':>12}")
print("-" * 72)
worst1 = 0.0
for t in [1, 50, 200, 500, 800, 999]:
    at, ab, abp = float(alphas[t]), float(abar[t]), float(abar_prev[t])
    lhs = at * (1 - abp) + (1 - at)
    rhs = 1 - ab
    worst1 = max(worst1, abs(lhs - rhs))
    print(f"{t:>5} {lhs:>18.12f} {rhs:>18.12f} {abs(lhs-rhs):>12.2e}")
print("-" * 72)
print(f"最大偏差 = {worst1:.2e}  -> {'✅ 恒等式成立' if worst1 < 1e-12 else '❌'}")

print()
print("=" * 72)
print("核验 2：恒等式  sqrt(abar_{t-1})/sqrt(abar_t) = 1/sqrt(alpha_t)")
print("=" * 72)
worst2 = 0.0
for t in [1, 50, 200, 500, 800, 999]:
    at = float(alphas[t])
    lhs = math.sqrt(float(abar_prev[t])) / math.sqrt(float(abar[t]))
    rhs = 1.0 / math.sqrt(at)
    worst2 = max(worst2, abs(lhs - rhs))
print(f"最大偏差 = {worst2:.2e}  -> {'✅ 恒等式成立' if worst2 < 1e-12 else '❌'}")

print()
print("=" * 72)
print("核验 3：三种写法算出的 mu_t 是否一致")
print("  第1行 = 原始后验均值(含 x0)      -> 基准")
print("  第3行 = 正确形式(减 beta_t/...)")
print("  错的  = 图里第2行的 + 号写法")
print("=" * 72)
x_t = torch.randn(64, dtype=torch.float64)
eps = torch.randn(64, dtype=torch.float64)

print(f"{'t':>5} {'第1行vs第3行误差':>16} {'错号写法误差':>14} "
      f"{'错号多出的幅度':>16} {'占噪声项':>10}")
print("-" * 80)
worst3, worst_bad = 0.0, 0.0
for t in [1, 50, 200, 500, 800, 999]:
    at, ab, abp = float(alphas[t]), float(abar[t]), float(abar_prev[t])
    bt = float(betas[t])
    # 由 x_t 反解 x0（这就是课件里代入的那一步）
    x0 = (x_t - math.sqrt(1 - ab) * eps) / math.sqrt(ab)

    # 第 1 行：两个系数的原始后验均值
    c_x0 = math.sqrt(abp) * bt / (1 - ab)
    c_xt = math.sqrt(at) * (1 - abp) / (1 - ab)
    mu_line1 = c_xt * x_t + c_x0 * x0

    # 第 3 行：正确答案
    mu_ok = (1 / math.sqrt(at)) * (x_t - (bt / math.sqrt(1 - ab)) * eps)

    # 图里第 2 行的 + 号写法
    mu_bad = (1 / math.sqrt(at)) * (x_t + ((1 - at) / math.sqrt(1 - ab)) * eps)

    e_ok = float((mu_ok - mu_line1).abs().max())
    e_bad = float((mu_bad - mu_line1).abs().max())
    # 错号相对正确式多出来的那一项： 2/sqrt(alpha_t) * beta_t/sqrt(1-abar_t) * eps
    extra_coef = 2 / math.sqrt(at) * bt / math.sqrt(1 - ab)
    extra_amp = extra_coef * float(eps.abs().max())
    worst3 = max(worst3, e_ok)
    worst_bad = max(worst_bad, e_bad)
    print(f"{t:>5} {e_ok:>16.2e} {e_bad:>14.4f} {extra_amp:>16.4f} "
          f"{e_bad/extra_amp:>9.0%}")
print("-" * 80)
print(f"第1行 vs 第3行 最大误差 = {worst3:.2e}  "
      f"-> {'✅ 数值上完全等价（推导正确）' if worst3 < 1e-12 else '❌'}")
print(f"第1行 vs +号写法 最大误差 = {worst_bad:.4f}  "
      f"-> {'❌ 差别很大，证实 + 号是错的' if worst_bad > 0.5 else '?'}")
print()
print("  为什么'相对误差'看着小（之前只有 1.9%）？")
print("  因为 mu_t 里 x_t 那一项量级约 0.6，把绝对误差除下去就被稀释了。")
print("  正确看法：错号写法等价于在正确结果上【多加】了")
print("      2/sqrt(alpha_t) * beta_t/sqrt(1-abar_t) * eps")
print("  当 t=999 时 abar_t≈0，这项≈2*eps，量级是 1——与 mu_t 本身同阶，")
print("  生成时会把预测步长放大到完全错误，所以这个符号必须是减号。")

print()
print("=" * 72)
print("结论")
print("=" * 72)
print("  第 1 行 -> 第 3 行 的推导（减号）数值上完全等价 ✅")
print("  图里第 2 行的 '+ (1-alpha_t)/sqrt(1-abar_t) * eps_t' 与正确结果不符 ❌")
print("  应为:  1/sqrt(alpha_t) * ( x_t - beta_t/sqrt(1-abar_t) * eps_t )")
print("  （系数 (1-alpha_t) 与 beta_t 数值相同，错的只是那个加号）")
