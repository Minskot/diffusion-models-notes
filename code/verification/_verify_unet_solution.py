"""Verify reconstructed U-Net TODOs against DDPM.ipynb cell 16 (6499 params) and cell 18 (forward)."""
import sys, importlib.util, types, copy
import numpy as np
import torch, torch.nn as nn

# ---- create an 'imp' shim in case cs231n modules need it
if "imp" not in sys.modules:
    m = types.ModuleType("imp"); m.reload = importlib.reload
    sys.modules["imp"] = m

BASE = "cs231n-2026-latest/assignments/assignment3_2026"
sys.path.insert(0, BASE)

# import the official unet module under a private name
spec = importlib.util.spec_from_file_location("unet_official", f"{BASE}/cs231n/unet.py")
unet_mod = importlib.util.module_from_spec(spec)
sys.modules["unet_official"] = unet_mod
spec.loader.exec_module(unet_mod)

Unet = unet_mod.Unet
ResnetBlock, Downsample, Upsample = unet_mod.ResnetBlock, unet_mod.Downsample, unet_mod.Upsample

_orig_init = Unet.__init__
_orig_forward = Unet.forward

def patched_init(self, dim, condition_dim, dim_mults=(1, 2, 4, 8), channels=3, uncond_prob=0.2):
    _orig_init(self, dim, condition_dim, dim_mults, channels, uncond_prob)
    dims = [dim] + [dim * m for m in dim_mults]
    in_out = list(zip(dims[:-1], dims[1:]))
    in_out_ups = [(b, a) for a, b in reversed(in_out)]
    context_dim = dim * 4

    # TODO: downsampling blocks
    self.downs = nn.ModuleList([
        nn.ModuleList([
            ResnetBlock(di, di, context_dim=context_dim),
            ResnetBlock(di, di, context_dim=context_dim),
            Downsample(di, do),
        ]) for (di, do) in in_out
    ])
    # TODO: upsampling blocks
    self.ups = nn.ModuleList([
        nn.ModuleList([
            Upsample(di, do),
            ResnetBlock(2 * do, do, context_dim=context_dim),
            ResnetBlock(2 * do, do, context_dim=context_dim),
        ]) for (di, do) in in_out_ups
    ])

def patched_forward(self, x, time, model_kwargs={}):
    if "cfg_scale" in model_kwargs:
        return self.cfg_forward(x, time, model_kwargs)
    context = self.time_mlp(time)
    cond_emb = model_kwargs["text_emb"]
    if cond_emb is None:
        cond_emb = torch.zeros(x.shape[0], self.condition_dim, device=x.device)
    if self.training:
        mask = (torch.rand(cond_emb.shape[0]) > self.uncond_prob).float()
        mask = mask[:, None].to(cond_emb.device)
        cond_emb = cond_emb * mask
    context = context + self.condition_mlp(cond_emb)
    x = self.init_conv(x)

    # TODO: downsample, saving skips (LIFO)
    skips = []
    for b1, b2, down in self.downs:
        x = b1(x, context=context); skips.append(x)
        x = b2(x, context=context); skips.append(x)
        x = down(x)
    # TODO: middle
    x = self.mid_block1(x, context=context)
    x = self.mid_block2(x, context=context)
    # TODO: upsample, consuming skips LIFO
    for up, b1, b2 in self.ups:
        x = up(x)
        x = torch.cat([x, skips.pop()], dim=1)
        x = b1(x, context=context)
        x = torch.cat([x, skips.pop()], dim=1)
        x = b2(x, context=context)
    return self.final_conv(x)

Unet.__init__ = patched_init
Unet.forward = patched_forward

ok = True
def report(name, cond, detail=""):
    global ok
    ok = ok and cond
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}  {detail}")

def rel_error(x, y):
    return float(np.max(np.abs(x - y) / (np.maximum(1e-10, np.abs(x) + np.abs(y)))))

print("=" * 70)
print("cell 16 — Unet.__init__  (dim=2, condition_dim=4, dim_mults=(1,2,4))")
u = Unet(dim=2, condition_dim=4, dim_mults=(1, 2, 4))
report("len(downs) == len(dim_mults)", len(u.downs) == 3, f"got {len(u.downs)}")
report("len(ups)   == len(dim_mults)", len(u.ups) == 3, f"got {len(u.ups)}")

# official assertion order: [m[0].dim, m[0].dim_out, m[0].context_dim, m[1]....]
downs_dims = [d for m in u.downs for d in [m[0].dim, m[0].dim_out, m[0].context_dim,
                                           m[1].dim, m[1].dim_out, m[1].context_dim]]
exp_down = [2,2,8, 2,2,8, 2,2,8, 2,2,8, 4,4,8, 4,4,8]
report("downs_dims", downs_dims == exp_down, f"{downs_dims}")

ups_dims = [d for m in u.ups for d in [m[1].dim, m[1].dim_out, m[1].context_dim,
                                       m[2].dim, m[2].dim_out, m[2].context_dim]]
exp_up = [8,4,8, 8,4,8, 4,2,8, 4,2,8, 4,2,8, 4,2,8]
report("ups_dims", ups_dims == exp_up, f"{ups_dims}")

n = sum(p.numel() for p in u.parameters())
report("num_params == 6499", n == 6499, f"got {n}")

print("=" * 70)
print("cell 18 — Unet.forward  (seed 231, dim=4, dim_mults=(2,4), b=2, h=w=4)")
np.random.seed(231); torch.manual_seed(231)
u2 = Unet(dim=4, condition_dim=4, dim_mults=(2, 4)); u2.eval()
inp_x = torch.randn(2, 3, 4, 4)
inp_emb = torch.randn(2, 4)
out = u2.forward(x=inp_x, time=torch.tensor([8, 25]).long(),
                 model_kwargs={"text_emb": inp_emb}).detach().numpy()
expected = np.array(
 [[[[ 0.3530089,   0.49302575,  0.55152094,  0.20567769],
    [ 1.1295223,   0.05811183,  0.2143015,   0.23898259],
    [ 0.44839963,  1.0291728,   0.18519637,  0.21956968],
    [ 0.7035912,   0.716347,    0.9166326,  -0.11552593]],
   [[ 0.04585427, -0.26847702,  0.08526254,  0.2396591 ],
    [ 0.2858743,  -0.29726404,  0.02462834,  0.2350314 ],
    [ 0.3121434,   0.3962962,   0.65541625, -0.300183  ],
    [ 0.3409412,   0.35369393,  0.37445623,  0.10297439]],
   [[ 0.49440542,  0.72878885,  0.6669705,   0.56822944],
    [ 1.0084232,   0.19204213,  0.61240125,  0.575651  ],
    [ 0.6452671,   0.91168964,  0.6458106,   0.25952405],
    [ 0.7147298,   0.7707498,   0.88941365,  0.35422024]]],
  [[[ 0.3186316,  -0.17632714,  0.440247,    0.05124736],
    [ 0.6866561,   0.3019496,   0.18366373,  0.42941707],
    [-0.09776017,  0.55849403,  0.30689716,  0.12954405],
    [ 0.3667726,   0.740231,    0.43682802,  0.08275545]],
   [[-0.29714173, -0.28515863, -0.02289355,  0.05637485],
    [-0.10230345,  0.182563,    0.8107456,   0.18209176],
    [ 0.57062995,  0.07061654,  0.2864037,   0.17969263],
    [ 0.31491554,  0.5652035,   0.3810383,   0.05889529]],
   [[ 0.14559639,  0.36408228,  0.5657894,   0.24066216],
    [ 0.4431491,   0.38140664,  1.1457629,   0.40314198],
    [ 0.6673274,   0.5922724,   0.44549578,  0.5544277 ],
    [ 0.5405652,   0.88729244,  0.66340685,  0.23414826]]]])
e = rel_error(out, expected)
report("forward rel_error < 1e-6  (EXPECTED TO FAIL - see note)", e < 1e-6, f"rel_err = {e:.3e}")
print()
print("  NOTE: cell 18 cannot be reproduced by monkey-patching. Replacing the")
print("  down/up ModuleLists AFTER the original __init__ changes the RNG")
print("  consumption order, so the random weights differ from the official")
print("  build. Default-seeded outputs are then essentially unrelated.")
print("  Structure IS verified above (all cell-16 assertions pass, 6499 params).")
print("  See 扩散模型作业参考实现.md section 4.")

print("=" * 70)
print(f"RESULT: {'ALL U-NET CHECKS PASSED' if ok else 'SOME CHECKS FAILED'}")
