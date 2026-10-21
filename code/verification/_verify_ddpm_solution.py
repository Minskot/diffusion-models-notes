"""Verify the DDPM reference implementations against DDPM.ipynb's hard-coded expected values."""
import math
import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------- schedules
def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    return torch.linspace(scale * 1e-4, scale * 2e-2, timesteps, dtype=torch.float64)

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    ab = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    ab = ab / ab[0]
    return torch.clip(1 - (ab[1:] / ab[:-1]), 0, 0.999)

def sigmoid_beta_schedule(timesteps, start=-3, end=3, tau=1):
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    ab = (-(((t * (end - start) + start) / tau).sigmoid()) + v_end) / (v_end - v_start)
    ab = ab / ab[0]
    return torch.clip(1 - (ab[1:] / ab[:-1]), 0, 0.999)

def get_beta_schedule(name, T):
    return {"linear": linear_beta_schedule,
            "cosine": cosine_beta_schedule,
            "sigmoid": sigmoid_beta_schedule}[name](T)

# ---------------------------------------------------------------- helpers
def extract(a, t, x_shape):
    b, *_ = t.shape
    return a.gather(-1, t).reshape(b, *((1,) * (len(x_shape) - 1)))

def rel_error(x, y):
    return np.max(np.abs(x - y) / (np.maximum(1e-10, np.abs(x) + np.abs(y))))

# ---------------------------------------------------------------- minimal GaussianDiffusion
class GD(nn.Module):
    def __init__(self, model=None, image_size=2, timesteps=1000,
                 objective="pred_noise", beta_schedule="sigmoid"):
        super().__init__()
        self.model = model
        self.channels = 3
        self.image_size = image_size
        self.objective = objective
        rb = lambda n, v: self.register_buffer(n, v.float())
        betas = get_beta_schedule(beta_schedule, timesteps)
        self.num_timesteps = int(betas.shape[0])
        alphas = 1.0 - betas
        acp = torch.cumprod(alphas, dim=0)
        rb("betas", betas); rb("alphas", alphas); rb("alphas_cumprod", acp)
        rb("sqrt_alphas_cumprod", torch.sqrt(acp))
        rb("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - acp))
        acp_prev = nn.functional.pad(acp[:-1], (1, 0), value=1.0)
        rb("posterior_mean_coef1", betas * torch.sqrt(acp_prev) / (1.0 - acp))
        rb("posterior_mean_coef2", (1.0 - acp_prev) * torch.sqrt(alphas) / (1.0 - acp))
        pv = betas * (1.0 - acp_prev) / (1.0 - acp)
        rb("posterior_std", torch.sqrt(pv.clamp(min=1e-20)))
        snr = acp / (1 - acp)
        rb("loss_weight", torch.ones_like(snr) if objective == "pred_noise" else snr)

    def normalize(self, img):   return img * 2 - 1
    def unnormalize(self, img): return (img + 1) * 0.5

    # ---- TODO 1 ----  x_0 = (x_t - sqrt(1-abar)*noise) / sqrt(ab)
    def predict_start_from_noise(self, x_t, t, noise):
        return (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise)

    # ---- TODO 2 ----  noise = (x_t - sqrt(ab)*x_0) / sqrt(1-abar)
    def predict_noise_from_start(self, x_t, t, x_start):
        return (x_t - extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_start) \
               / extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)

    def q_posterior(self, x_start, x_t, t):
        c1 = extract(self.posterior_mean_coef1, t, x_t.shape)
        c2 = extract(self.posterior_mean_coef2, t, x_t.shape)
        return c1 * x_start + c2 * x_t, extract(self.posterior_std, t, x_t.shape)

    # ---- TODO 3 ----  x_t = sqrt(ab)*x_0 + sqrt(1-ab)*noise
    def q_sample(self, x_start, t, noise):
        return (extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
                + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    # ---- TODO 4 ----
    def p_losses(self, x_start, model_kwargs={}):
        b, nts = x_start.shape[0], self.num_timesteps
        t = torch.randint(0, nts, (b,), device=x_start.device).long()
        x_start = self.normalize(x_start)
        noise = torch.randn_like(x_start)
        target = noise if self.objective == "pred_noise" else x_start
        loss_weight = extract(self.loss_weight, t, target.shape)
        x_t = self.q_sample(x_start, t, noise)
        pred = self.model(x_t, t, model_kwargs=model_kwargs)
        return (loss_weight * (pred - target) ** 2).mean()

    @torch.no_grad()
    def p_sample(self, x_t, t, model_kwargs={}):
        t = torch.full((x_t.shape[0],), t, device=x_t.device, dtype=torch.long)
        out = self.model(x_t, t, model_kwargs=model_kwargs)
        if self.objective == "pred_noise":
            x_start = self.predict_start_from_noise(x_t, t, out)
        else:
            x_start = out
        x_start.clamp_(-1.0, 1.0)
        mean, std = self.q_posterior(x_start, x_t, t)
        noise = torch.randn_like(x_t)
        nonzero = (t != 0).float().reshape(-1, *((1,) * (x_t.dim() - 1)))
        return mean + nonzero * std * noise

# fix: predict_start_from_noise needs reciprocal buffers
def _add_recip(self):
    acp = self.alphas_cumprod
    self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / acp).float())
    self.register_buffer("sqrt_recipm1_alphas_cumprod",
                         torch.sqrt(1.0 / acp - 1.0).float())

_orig_init = GD.__init__
def _init(self, *a, **k):
    _orig_init(self, *a, **k)
    _add_recip(self)
GD.__init__ = _init

ok = True
def check(name, val, tol):
    global ok
    good = val <= tol
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] {name}: rel_err = {val:.3e}  (tol {tol:g})")

print("=" * 68)
print("cell 9  — q_sample")
d = GD(image_size=2, timesteps=1000, beta_schedule="sigmoid")
t = torch.tensor([0, 300, 999]).long()
x_start = torch.linspace(-0.9, 0.6, 3*3*2*2).view(3, 3, 2, 2)
noise = torch.linspace(-0.7, 0.8, 3*3*2*2).view(3, 3, 2, 2)
x_t = d.q_sample(x_start, t, noise)
expected = np.array([
 [[[-0.9119949, -0.86840147], [-0.8248081, -0.7812148]],
  [[-0.7376214, -0.694028], [-0.65043473, -0.6068413]],
  [[-0.563248, -0.51965463], [-0.47606122, -0.43246788]]],
 [[[-0.42800453, -0.37039882], [-0.31279305, -0.2551873]],
  [[-0.19758154, -0.1399758], [-0.08237009, -0.024764337]],
  [[0.032841414, 0.090447165], [0.14805292, 0.20565866]]],
 [[[0.32864183, 0.37152246], [0.41440308, 0.45728368]],
  [[0.50016433, 0.5430449], [0.5859255, 0.6288062]],
  [[0.67168677, 0.7145674], [0.757448, 0.8003287]]],
], dtype=np.float32)
check("q_sample vs expected_x_t", rel_error(x_t.numpy(), expected), 1e-6)

print("=" * 68)
print("cell 12 — predict_noise_from_start / predict_start_from_noise")
d = GD(image_size=2, timesteps=1000, beta_schedule="sigmoid")
t = torch.tensor([1, 300, 998]).long()
x_start = torch.linspace(-0.91, 0.67, 3*3*2*2).view(3, 3, 2, 2)
noise = torch.linspace(-0.73, 0.81, 3*3*2*2).view(3, 3, 2, 2)
x_t = d.q_sample(x_start, t, noise)
pn = d.predict_noise_from_start(x_t, t, x_start)
px = d.predict_start_from_noise(x_t, t, noise)
check("noise recovered", rel_error(pn.numpy(), noise.numpy()), 1e-5)
check("x_start recovered", rel_error(px.numpy(), x_start.numpy()), 1e-5)

print("=" * 68)
print("cell 20 — p_losses (expected_out = 30.0160, objective=pred_x_start)")
print("  (requires the real Unet; skipped — see note)")
print()
print("=" * 68)
print(f"RESULT: {'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'}")
