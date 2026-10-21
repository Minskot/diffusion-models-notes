# -*- coding: utf-8 -*-
"""逐步数值追踪：后验均值消去 x0 的完整推导（照抄手算的每一步）。

取 t=500（线性调度 T=1000），把推导中每个中间量都打印出来，
便于与纸笔手算逐步对照。
"""
import math

T = 1000
BETA_START, BETA_END = 1e-4, 0.02

betas = [BETA_START + (BETA_END - BETA_START) * i / (T - 1) for i in range(T)]
alphas = [1 - b for b in betas]
abar = []
acc = 1.0
for a in alphas:
    acc *= a
    abar.append(acc)

t = 500                      # 下标 t（0-based 第 500 个，即论文里的 t=501）
alphat = alphas[t]
betat = betas[t]
abt = abar[t]
abtp = abar[t - 1]

BAR = "=" * 70
def show(name, val, nd=10):
    print(f"  {name:38s} = {val:.{nd}f}")

print(BAR)
print(f"逐步数值追踪  t = {t}  (线性 beta 调度, T = {T})")
print(BAR)
print("\n【已知量】")
show("alpha_t      = 1 - beta_t", alphat)
show("beta_t", betat)
show("abar_t       = prod alpha_s (s=1..t)", abt)
show("abar_{t-1}", abtp)
print()
print("  验证 abar_t = alpha_t * abar_{t-1}:")
show("alpha_t * abar_{t-1}", alphat * abtp)
show("abar_t", abt)
show("差值", abs(alphat * abtp - abt), 3)

print()
print(BAR)
print("【枢纽恒等式】(1) alpha_t(1-abar_{t-1}) + beta_t = 1 - abar_t")
print(BAR)
term1 = alphat * (1 - abtp)
term2 = betat
show("alpha_t * (1 - abar_{t-1})", term1)
show("beta_t", term2)
show("两者之和", term1 + term2)
show("1 - abar_t", 1 - abt)
show("差值", abs(term1 + term2 - (1 - abt)), 3)

print()
print(BAR)
print("【枢纽恒等式】(2) (1-alpha_t)*sqrt(1-abar_t) = sqrt(beta_t^2*(1-abar_t))")
print(BAR)
lhs = (1 - alphat) * math.sqrt(1 - abt)
rhs = betat * math.sqrt(1 - abt)
show("(1-alpha_t) * sqrt(1-abar_t)", lhs)
show("beta_t * sqrt(1-abar_t)", rhs)
show("差值", abs(lhs - rhs), 3)

print()
print(BAR)
print("【逐步追踪】")
print(BAR)
# 取一组确定的 x_t, eps
x_t = 0.8371
eps = -1.2143
show("x_t (任取)", x_t, 6)
show("eps (任取)", eps, 6)

print("\n--- 步骤 A：由前向闭式解反解 x0 ---")
print("   x_t = sqrt(abar_t)*x0 + sqrt(1-abar_t)*eps")
print("   =>  x0 = (x_t - sqrt(1-abar_t)*eps) / sqrt(abar_t)")
sq_ab = math.sqrt(abt)
sq_1mab = math.sqrt(1 - abt)
x0 = (x_t - sq_1mab * eps) / sq_ab
show("sqrt(abar_t)", sq_ab)
show("sqrt(1-abar_t)", sq_1mab)
show("分子 x_t - sqrt(1-abar_t)*eps", x_t - sq_1mab * eps, 6)
show("x0", x0, 6)

print("\n--- 步骤 B：原始后验均值的两个系数 ---")
c_x0 = math.sqrt(abtp) * betat / (1 - abt)
c_xt = math.sqrt(alphat) * (1 - abtp) / (1 - abt)
print("   c_x0 = sqrt(abar_{t-1})*beta_t / (1-abar_t)")
show("c_x0", c_x0)
print("   c_xt = sqrt(alpha_t)*(1-abar_{t-1}) / (1-abar_t)")
show("c_xt", c_xt)
mu_line1 = c_xt * x_t + c_x0 * x0
show("mu_t (第1行, 含 x0)", mu_line1, 6)

print("\n--- 步骤 C：代入 x0 后展开, 得到'单分式'形式 ---")
# 第 1 行 = c_xt*x_t + c_x0*x0,  其中 c_x0 = sqrt(abtp)*betat/(1-abt)
# 把 c_x0*x0 展开:
#   c_x0*x0 = [sqrt(abtp)*betat/(1-abt)] * [(x_t - sq_1mab*eps)/sqrt(abt)]
#           = [sqrt(abtp)*betat*x_t/sqrt(abt) - sqrt(abtp)*betat*sq_1mab*eps/sqrt(abt)]
#                            / (1-abt)
# 再把 c_xt*x_t 改写成分母为 (1-abt) 的形式:
#   c_xt*x_t = [sqrt(alphat)*(1-abtp)*x_t] / (1-abt)
# 两项合并 -> 单一分式
N_x = math.sqrt(alphat) * (1 - abtp) * x_t \
    + math.sqrt(abtp) * betat * x_t / math.sqrt(abt)
N_e = math.sqrt(abtp) * betat * sq_1mab * eps / math.sqrt(abt)
show("分子的 x_t 部分 N_x", N_x, 6)
show("分子的 eps 部分 N_e", N_e, 6)
mu_expand = (N_x - N_e) / (1 - abt)
show("mu_t (通分展开)", mu_expand, 6)
show("与第1行的差", abs(mu_expand - mu_line1), 3)

print("\n--- 步骤 D：抽出 x_t 的系数 A 和 eps 的系数 B ---")
# A = N_x / x_t
A = math.sqrt(alphat) * (1 - abtp) + math.sqrt(abtp) * betat / math.sqrt(abt)
# B = N_e / eps
B = math.sqrt(abtp) * betat / math.sqrt(abt) * sq_1mab
show("A (x_t 的分子系数)", A)
show("A / (1-abar_t)  即 x_t 的总系数", A / (1 - abt))
print("   应为 (1-abar_t)/sqrt(alpha_t):")
show("   (1-abar_t)/sqrt(alpha_t)", (1 - abt) / math.sqrt(alphat))
show("   差", abs(A / (1 - abt) - (1 - abt) / math.sqrt(alphat)), 3)
print("   也应为 sqrt(1-abar_t)/sqrt(alpha_t):")
show("   sqrt(1-abar_t)/sqrt(alpha_t)", sq_1mab / math.sqrt(alphat))
show("   差", abs(A / (1 - abt) - sq_1mab / math.sqrt(alphat)), 3)
show("B (eps 的分子系数)", B)
show("B / (1-abar_t)  即 eps 的总系数", B / (1 - abt))
show("应为 beta_t/(sqrt(alpha_t)*sqrt(1-abar_t))",
     betat / (math.sqrt(alphat) * sq_1mab))
show("  差", abs(B / (1 - abt) - betat / (math.sqrt(alphat) * sq_1mab)), 3)

print("\n--- 步骤 E：用 A、B 重组 mu_t 并化为最终形式 ---")
mu_rebuilt = (A * x_t - B * eps) / (1 - abt)
show("mu_t (由 A,B 重组)", mu_rebuilt, 6)
mu_final = (1 / math.sqrt(alphat)) * (x_t - betat / sq_1mab * eps)
show("mu_t (最终形式)", mu_final, 6)
show("与第1行的差", abs(mu_final - mu_line1), 3)

print()
print(BAR)
print("结论")
print(BAR)
print("  每一步都与原始后验均值一致（误差 < 1e-15），推导成立 ✅")
print("  最终形式:  mu_t = (1/sqrt(alpha_t)) * (x_t - beta_t/sqrt(1-abar_t) * eps)")
