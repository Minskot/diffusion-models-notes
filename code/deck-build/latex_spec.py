"""Deck 用到的全部公式，集中定义（LaTeX 子集，供 matplotlib mathtext 渲染）。

每项: key -> (latex, kind)
  kind = 'display' : 整行居中的独立公式，渲染成图片
  kind = 'caption' : 与文字同行的公式，保留 Unicode 文本
"""

NAVY = "#0B2545"
AMBER = "#B46A0A"
TEAL = "#0E7C86"

# ---------------------------------------------------------------- 独立公式
DISPLAY = {
    # 第 4 张：核心思想
    "idea_forward": r"$x_0 \rightarrow x_1 \rightarrow \cdots \rightarrow x_T \approx \mathcal{N}(0, I)$",
    "idea_reverse": r"$x_T \rightarrow \cdots \rightarrow x_1 \rightarrow x_0$",

    # 第 6 张：前向过程
    "fwd_def": r"$q(x_t \mid x_{t-1}) = \mathcal{N}\!\left(x_t;\ \sqrt{1-\beta_t}\,x_{t-1},\ \beta_t I\right)$",
    "fwd_reparam": r"$x_t = \sqrt{1-\beta_t}\,x_{t-1} + \sqrt{\beta_t}\,\varepsilon_{t-1}, \qquad \varepsilon_{t-1} \sim \mathcal{N}(0, I)$",

    # 第 7 张：闭式解推导
    "cf_step1": r"$x_{t-1} = \sqrt{\alpha_{t-1}}\,x_{t-2} + \sqrt{1-\alpha_{t-1}}\,\varepsilon_{t-2} \qquad\Longrightarrow\qquad x_t = \sqrt{\alpha_t}\,x_{t-1} + \sqrt{1-\alpha_t}\,\varepsilon_t$",
    "cf_step2": r"$x_t = \sqrt{\alpha_t \alpha_{t-1}}\,x_{t-2} + \sqrt{\alpha_t(1-\alpha_{t-1}) + (1-\alpha_t)}\,\tilde{\varepsilon} \qquad \mathrm{with}\quad \alpha_t(1-\alpha_{t-1}) + (1-\alpha_t) = 1 - \alpha_t\alpha_{t-1}$",
    "cf_final": r"$x_t = \sqrt{\bar{\alpha}_t}\,x_0 + \sqrt{1-\bar{\alpha}_t}\,\varepsilon, \qquad \bar{\alpha}_t = \prod_{i=1}^{t}\alpha_i, \qquad \varepsilon \sim \mathcal{N}(0, I)$",

    # 第 8 张：为什么闭式解重要
    "cf_dist": r"$q(x_t \mid x_0) = \mathcal{N}\!\left(x_t;\ \sqrt{\bar{\alpha}_t}\,x_0,\ (1-\bar{\alpha}_t) I\right)$",
    "cf_limit": r"$t = T \quad\Longrightarrow\quad x_T \approx \varepsilon$",

    # 第 9 张：反向过程的困难
    "rev_bayes": r"$q(x_{t-1} \mid x_t) = q(x_t \mid x_{t-1})\,\frac{q(x_{t-1})}{q(x_t)}$",
    "rev_cond": r"$q(x_{t-1} \mid x_t, x_0) = q(x_t \mid x_{t-1}, x_0)\,\frac{q(x_{t-1} \mid x_0)}{q(x_t \mid x_0)}$",

    # 第 10 张：后验闭式解
    "post_dist": r"$q(x_{t-1} \mid x_t, x_0) = \mathcal{N}\!\left(x_{t-1};\ \tilde{\mu}_t,\ \tilde{\sigma}_t^2 I\right)$",
    "post_mean": r"$\tilde{\mu}_t = \frac{\sqrt{\bar{\alpha}_t}\,(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}\,x_t \;+\; \frac{\sqrt{\bar{\alpha}_{t-1}}\,\beta_t}{1-\bar{\alpha}_t}\,x_0$",
    "post_var": r"$\tilde{\sigma}_t^2 = \frac{\beta_t\,(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}$",

    # 第 11 张：变分下界
    "elbo1": r"$-\log p_\theta(x_0) \;\leq\; -\log p_\theta(x_0) + D_{\mathrm{KL}}\!\left(q(x_{1:T} \mid x_0)\,\|\,p_\theta(x_{1:T} \mid x_0)\right)$",
    "elbo2": (r"$-\log p_\theta(x_0) \;\leq\; \mathbb{E}_q\!\left[D_{\mathrm{KL}}\!\left(q(x_T \mid x_0)\,\|\,p_\theta(x_T)\right)\right]$" "\n"
              r"$+\; \sum_{t=2}^{T} D_{\mathrm{KL}}\!\left(q(x_{t-1} \mid x_t, x_0)\,\|\,p_\theta(x_{t-1} \mid x_t)\right) \;-\; \log p_\theta(x_0 \mid x_1)$"),

    # 第 12 张：KL → 噪声回归
    "kl_gauss": r"$D_{\mathrm{KL}}\!\left(\mathcal{N}(\mu_1,\Sigma)\,\|\,\mathcal{N}(\mu_2,\Sigma)\right) = \frac{1}{2}(\mu_1-\mu_2)^{\top}\Sigma^{-1}(\mu_1-\mu_2)$",
    "kl_argmin": r"$\Longrightarrow\quad \arg\min_\theta\ \frac{1}{2\tilde{\sigma}_t^2}\left\|\mu_\theta - \tilde{\mu}_t\right\|^2$",
    "sub_x0": r"$x_0 = \frac{x_t - \sqrt{1-\bar{\alpha}_t}\;\varepsilon}{\sqrt{\bar{\alpha}_t}}$",
    "mean_noise": r"$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\,\varepsilon_\theta(x_t, t)\right)$",
    "loss_final": r"$\mathcal{L} = \mathbb{E}_{t,\,x_0,\,\varepsilon}\!\left[\left\|\,\varepsilon - \varepsilon_\theta\!\left(\sqrt{\bar{\alpha}_t}\,x_0 + \sqrt{1-\bar{\alpha}_t}\,\varepsilon,\ t\right)\right\|^2\right]$",

    # 第 13 张：训练算法（注：该页已改为纯 ASCII 代码风格，
    # 不再使用独立的公式图片，故移除 train_grad）

    # 第 17 张：总结
    "takeaway_loss": r"$\mathcal{L} = \mathbb{E}\!\left[\left\|\varepsilon - \varepsilon_\theta\right\|^2\right]$",
}

# ---------------------------------------------------------------- 行内公式
INLINE = {
    "x0": r"$x_0$",
    "xt": r"$x_t$",
    "xT": r"$x_T$",
    "eps": r"$\varepsilon$",
    "beta_t": r"$\beta_t$",
    "alpha_t": r"$\alpha_t$",
    "abar_t": r"$\bar{\alpha}_t$",
    "eps_theta": r"$\varepsilon_\theta$",
    "p_theta": r"$p_\theta$",
    "N01": r"$\mathcal{N}(0, I)$",
    "q_xt_x0": r"$q(x_t \mid x_0)$",
}
