# -*- coding: utf-8 -*-
"""沙箱快速验证：语法 + 数值正确性（不加载大模型）。"""
import ast
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

# 语法检查
for f in ("cs231n/gaussian_diffusion.py", "cs231n/unet.py"):
    try:
        ast.parse(open(f, encoding="utf-8").read())
        print(f"  语法 OK: {f}")
    except SyntaxError as e:
        print(f"  ❌ 语法错误 {f} 第 {e.lineno} 行: {e.msg}")
        sys.exit(1)

import numpy as np
import torch
from cs231n.gaussian_diffusion import GaussianDiffusion
from cs231n.unet import Unet


def rel_error(x, y):
    return float(np.max(np.abs(x - y) / np.maximum(1e-10, np.abs(x) + np.abs(y))))


print("\n--- q_sample ---")
d = GaussianDiffusion(model=None, image_size=2, timesteps=1000,
                      beta_schedule="sigmoid")
b, sz = 3, 2
t = torch.tensor([0, 300, 999]).long()
xs = torch.linspace(-0.9, 0.6, b * 3 * sz * sz).view(b, 3, sz, sz)
nz = torch.linspace(-0.7, 0.8, b * 3 * sz * sz).view(b, 3, sz, sz)
xt = d.q_sample(xs, t, nz)
exp = np.array([
 [[[-0.9119949, -0.86840147], [-0.8248081, -0.7812148]],
  [[-0.7376214, -0.694028], [-0.65043473, -0.6068413]],
  [[-0.563248, -0.51965463], [-0.47606122, -0.43246788]]],
 [[[-0.42800453, -0.37039882], [-0.31279305, -0.2551873]],
  [[-0.19758154, -0.1399758], [-0.08237009, -0.024764337]],
  [[0.032841414, 0.090447165], [0.14805292, 0.20565866]]],
 [[[0.32864183, 0.37152246], [0.41440308, 0.45728368]],
  [[0.50016433, 0.5430449], [0.5859255, 0.6288062]],
  [[0.67168677, 0.7145674], [0.757448, 0.8003287]]],
]).astype(np.float32)
print("  x_t error =", f"{rel_error(xt.numpy(), exp):.3e}", " (要求 ~0)")

print("\n--- predict_start_from_noise / predict_noise_from_start ---")
t2 = torch.tensor([1, 300, 998]).long()
xs2 = torch.linspace(-0.91, 0.67, b * 3 * sz * sz).view(b, 3, sz, sz)
nz2 = torch.linspace(-0.73, 0.81, b * 3 * sz * sz).view(b, 3, sz, sz)
xt2 = d.q_sample(xs2, t2, nz2)
print("  x_start error =",
      f"{rel_error(d.predict_start_from_noise(xt2, t2, nz2).numpy(), xs2.numpy()):.3e}",
      " (要求 < 1e-5)")
print("  noise   error =",
      f"{rel_error(d.predict_noise_from_start(xt2, t2, xs2).numpy(), nz2.numpy()):.3e}",
      " (要求 < 1e-5)")

print("\n--- Unet 结构参数量 ---")
m = Unet(dim=2, condition_dim=4, dim_mults=(1, 2, 4))
n = sum(p.numel() for p in m.parameters())
print(f"  dim=2, mult=(1,2,4) 参数量 = {n}   (要求 == 6499)  {'✅' if n == 6499 else '❌'}")

m2 = Unet(dim=48, dim_mults=(1, 2, 4, 8), condition_dim=512)
n2 = sum(p.numel() for p in m2.parameters())
print(f"  训练配置参数量 = {n2:,}   (期望 12,445,863)  {'✅' if n2 == 12445863 else '❌'}")

print("\n--- 权重加载测试 ---")
ck = torch.load("cs231n/exp/pretrained/model-70000.pt", map_location="cpu",
                weights_only=True)
try:
    m2.load_state_dict(ck["model"])
    print("  ✅ 预训练权重加载成功（说明 U-Net 结构完全匹配）")
except Exception as e:
    print("  ❌ 加载失败:", type(e).__name__, str(e)[:300])
