# -*- coding: utf-8 -*-
"""逐键比对 checkpoint 与本地 Unet 的 state_dict，精确定位结构差异。"""
import os
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

from cs231n.unet import Unet

ck = torch.load("cs231n/exp/pretrained/model-70000.pt", map_location="cpu",
                weights_only=True)
sd = ck["model"]

m = Unet(dim=48, dim_mults=(1, 2, 4, 8), condition_dim=512)
mine = m.state_dict()

print(f"checkpoint 参数总量: {sum(v.numel() for v in sd.values()):,}")
print(f"本地 Unet 参数总量 : {sum(v.numel() for v in mine.values()):,}")
print(f"差值               : {sum(v.numel() for v in sd.values()) - sum(v.numel() for v in mine.values()):,}")
print()

ck_keys, my_keys = set(sd), set(mine)
print("=== 只在 checkpoint 里（我缺的）===")
for k in sorted(ck_keys - my_keys):
    print(f"   + {k:60s} {tuple(sd[k].shape)}")
print("\n=== 只在我这（多出来的）===")
for k in sorted(my_keys - ck_keys):
    print(f"   - {k:60s} {tuple(mine[k].shape)}")

print("\n=== 键名相同但形状不同的 ===")
for k in sorted(ck_keys & my_keys):
    a, b = tuple(sd[k].shape), tuple(mine[k].shape)
    if a != b:
        print(f"   ! {k:60s} ckpt={a}  mine={b}")

# 按模块统计参数量差异
def by_module(sd):
    agg = {}
    for k, v in sd.items():
        mod = k.split(".")[0]
        agg[mod] = agg.get(mod, 0) + v.numel()
    return agg

ca, ma = by_module(sd), by_module(mine)
print("\n=== 各顶层模块参数量 ===")
print(f"{'模块':<22}{'checkpoint':>14}{'本地':>14}{'差':>10}")
for k in sorted(set(ca) | set(ma)):
    c, m_ = ca.get(k, 0), ma.get(k, 0)
    flag = "" if c == m_ else "   <<<"
    print(f"{k:<22}{c:>14,}{m_:>14,}{c-m_:>10,}{flag}")
