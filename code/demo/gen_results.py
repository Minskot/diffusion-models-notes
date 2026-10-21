# -*- coding: utf-8 -*-
"""按正确方式加载预训练权重并生成结果（修正版）。"""
import os
import time

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)

from cs231n.gaussian_diffusion import GaussianDiffusion
from cs231n.unet import Unet
from cs231n.emoji_dataset import EmojiDataset

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DIM, IMAGE_SIZE, COND_DIM, TIMESTEPS = 48, 32, 512, 100
CKPT = "cs231n/exp/pretrained/model-70000.pt"

print("device:", DEVICE, flush=True)

# ---------------------------------------------------------------- 构建 + 加载
# 关键：checkpoint 存的是【整个 GaussianDiffusion 的 state_dict】，
# 顶层含 betas/alphas/loss_weight 等 buffer。所以先建 GaussianDiffusion
# 再 load_state_dict，而不是单独加载 Unet。
model = Unet(dim=DIM, dim_mults=(1, 2, 4, 8), condition_dim=COND_DIM)
diffusion = GaussianDiffusion(model, image_size=IMAGE_SIZE,
                              timesteps=TIMESTEPS, objective="pred_noise")
ck = torch.load(CKPT, map_location="cpu", weights_only=True)
sd = ck["model"]
print(f"checkpoint 参数键数: {len(sd)}  (step={ck['step']})", flush=True)
missing, unexpected = diffusion.load_state_dict(sd, strict=False)
print(f"missing={len(missing)}  unexpected={len(unexpected)}")
if missing:
    print("  missing 前几个:", missing[:5])
if unexpected:
    print("  unexpected 前几个:", unexpected[:5])
diffusion = diffusion.to(DEVICE).eval()
print("✅ 权重加载完成", flush=True)


def to_np(img):
    return img.clamp(0, 1).permute(1, 2, 0).cpu().numpy()


# ---------------------------------------------------------------- 1) 前向加噪
def fig_forward():
    print("\n[1] 前向加噪 ...", flush=True)
    ds = EmojiDataset(IMAGE_SIZE)
    # 固定选用 idx=610 "tiger"：配色鲜明 (黄/黑/粉)、面部特征清晰，
    # 加噪过程中"特征被噪声吞没"最直观。写死索引以保证结果可复现。
    img, meta = ds[610]
    print(f"   样本: idx=610  text={meta['text']!r}", flush=True)
    x0 = diffusion.normalize(img).unsqueeze(0).to(DEVICE)
    steps = [0, 10, 25, 50, 75, 99]
    fig, axes = plt.subplots(1, len(steps), figsize=(2.6 * len(steps), 3.1), dpi=150)
    for ax, tt in zip(axes, steps):
        torch.manual_seed(0)
        xt = diffusion.q_sample(x0, torch.tensor([tt], device=DEVICE),
                                torch.randn_like(x0))
        ax.imshow(to_np(diffusion.unnormalize(xt)[0]))
        ax.set_title(f"t = {tt}", fontsize=16)
        ax.axis("off")
    fig.suptitle("Forward process  q(xₜ | x₀)   —   fixed, no learning",
                 fontsize=18)
    fig.tight_layout()
    fig.savefig(f"{OUT}/forward_noising.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("   saved forward_noising.png")


# ---------------------------------------------------------------- 2) 无条件生成
def fig_samples(n=25):
    print(f"\n[2] 无条件生成 {n} 张 ...", flush=True)
    torch.manual_seed(0)
    t0 = time.time()
    with torch.no_grad():
        imgs = diffusion.sample(batch_size=n, model_kwargs={"text_emb": None})
    print(f"   用时 {time.time()-t0:.1f}s", flush=True)
    g = int(np.sqrt(n))
    fig, axes = plt.subplots(g, g, figsize=(g * 1.6, g * 1.6), dpi=150)
    for i, ax in enumerate(axes.flat):
        ax.imshow(to_np(imgs[i]))
        ax.axis("off")
    fig.suptitle("Unconditional samples — pretrained DDPM (T = 100)", fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{OUT}/samples_grid.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("   saved samples_grid.png")


# ---------------------------------------------------------------- 3) 采样演化
def fig_over_t():
    print("\n[3] 采样过程演化 ...", flush=True)
    torch.manual_seed(0)
    with torch.no_grad():
        imgs = diffusion.sample(batch_size=4, return_all_timesteps=True,
                                model_kwargs={"text_emb": None})
    B, Tp1 = imgs.shape[0], imgs.shape[1]
    cols = [0, Tp1 // 4, Tp1 // 2, 3 * Tp1 // 4, Tp1 - 1]
    fig, axes = plt.subplots(B, len(cols), figsize=(2.1 * len(cols), 2.1 * B), dpi=150)
    for r in range(B):
        for c, ci in enumerate(cols):
            axes[r, c].imshow(to_np(imgs[r, ci]))
            axes[r, c].axis("off")
            if r == 0:
                axes[r, c].set_title(f"step {ci}", fontsize=12)
    fig.suptitle("Reverse process: x_T (pure noise) → x_0 (image)", fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{OUT}/samples_over_t.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("   saved samples_over_t.png")


# ---------------------------------------------------------------- 4) CFG
def fig_cfg(text="crying face", scales=(0.0, 2.0, 5.0, 10.0)):
    print("\n[4] CFG scale 对比 ...", flush=True)
    try:
        from cs231n.emoji_dataset import ClipEmbed
        ds = EmojiDataset(IMAGE_SIZE)
        embedder = ClipEmbed(DEVICE)
        emb = ds.embed_new_text(text, embedder).float().to(DEVICE)
        print("   CLIP 文本嵌入就绪", flush=True)
    except Exception as e:
        print(f"   跳过（CLIP 不可用）: {type(e).__name__}: {str(e)[:160]}")
        return
    n = 4
    fig, axes = plt.subplots(n, len(scales), figsize=(2.1 * len(scales), 2.1 * n),
                             dpi=150)
    for c, w in enumerate(scales):
        torch.manual_seed(0)
        kw = {"text_emb": emb.repeat(n, 1)}
        if w > 0:
            kw["cfg_scale"] = w
        with torch.no_grad():
            out = diffusion.sample(batch_size=n, model_kwargs=dict(kw))
        for r in range(n):
            axes[r, c].imshow(to_np(out[r]))
            axes[r, c].axis("off")
            if r == 0:
                axes[r, c].set_title(f"w = {w}", fontsize=12)
    fig.suptitle(f'Classifier-free guidance sweep — prompt: "{text}"', fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{OUT}/cfg_sweep.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("   saved cfg_sweep.png")


if __name__ == "__main__":
    fig_forward()
    fig_samples()
    fig_over_t()
    fig_cfg()
    print("\n结果目录:", OUT)
    for f in sorted(os.listdir(OUT)):
        print("   ", f)
