# DDPM 汇报 Q&A 准备手册

> 配套演示文稿：[`slides/DDPM_Academic_Report_v2.pptx`](../slides/DDPM_Academic_Report_v2.pptx)（22 页）
> 用法：**加粗的是必背**；📖 是完整推导；🎤 是口语化说法（可直接念）；
> ✅ 是加分回答（主动说出来会显得你真的懂）。
> 英文部分是**答辩时可以直接用的原句**。

---

## 📋 速查表：30 秒能答完的 12 个问题

| # | 问题 | 一句话答案 |
|---|---|---|
| 1 | 扩散模型属于哪一类生成模型？ | **Implicit density → Indirect sampling**（Goodfellow 口径）。DDPM 论文自称"显式密度"，因它优化似然下界 —— 两套判据不同 |
| 2 | 和 GAN / VAE 比优劣？ | 训练最稳 + 质量最高 + 模式覆盖好；代价是采样慢 |
| 3 | 前向过程为什么不用学习？ | $\beta_t$ 是人工超参数，**没有任何待优化参数** |
| 4 | 反向过程为什么难？ | $q(x_{t-1}\mid x_t)$ 没有闭式解，要用网络近似 |
| 5 | 闭式解怎么来？ | 高斯递归复合，均值方差可加 |
| 6 | 为什么后验要条件在 $x_0$ 上？ | 加上 $x_0$ 后三项都是已知高斯 → 后验有闭式解 |
| 7 | 网络为什么只输出一个量？ | 后验方差只依赖 $t$、是常数，只用学均值 |
| 8 | $\mathcal{L}_{\text{simple}}$ 和 ELBO 什么关系？ | 把 ELBO 的 SNR 权重丢掉，**不再是下界但效果更好** |
| 9 | 为什么是 MSE？ | L2 的最优解 = 条件均值，正是我们要估计的量 |
| 10 | 预测 $\varepsilon$ 还是 $x_0$？ | 数学等价，$\varepsilon$ 尺度更稳（$\mathcal N(0,I)$，不随 $t$ 变） |
| 11 | 为什么要 clamp？ | 高噪声步 $\hat x_0$ 误差大，不截断会在采样中累积放大 |
| 12 | CFG 的代价？ | 每步要跑两次前向，采样开销翻倍 |

---

# 一、定位与直觉（第 3–6 页）

## Q1. 扩散模型在生成模型分类里属于哪一支？

**核心答案（本演示文稿采用的口径）**：属于 **`Implicit density` → `Indirect sampling`** —— 即**间接迭代采样、不直接计算 $p(x)$** 的那一支。

> 📖 **必须分清两套体系的用词**（这是本页最容易被追问的点）：
>
> **Goodfellow 图里的分支与子节点对应关系**：
>
> | 主分支 | 其下的子节点 |
> |---|---|
> | `Explicit density` | `Tractable density` / **`Approximate density`** |
> | `Implicit density` | **`Direct`**（GAN） / **`Indirect`**（**Diffusion**） |
>
> 所以在 Goodfellow 的口径下，扩散模型是 **`Implicit density → Indirect`**；而 **`Approximate density` 属于 Explicit 分支，不能和 `Implicit density` 并列写在一起**。
>
> **DDPM 论文自己的说法则不同**：论文把扩散称为**显式密度**模型，因为它确实去优化 $\log p_\theta(x)$ 的变分下界（ELBO）。**两套说法用的是不同判据，不矛盾。**

🎤 **中文口语版**：
> "按 Goodfellow 那张图的判据 —— **能否直接计算 $p(x)$** —— 扩散模型属于隐式密度，通过间接的迭代采样得到样本；DDPM 论文本身把它称作显式密度模型，因为它优化的是似然的下界。两者判据不同。"

> ✅ **加分**：主动补一句 —— "更实用的分类维度是**单步 vs 多步**：GAN 一步出图，扩散要迭代上百步。"

**Key English**:
> "In Goodfellow's taxonomy the criterion is whether $p(x)$ can be computed directly. Diffusion models fall under **implicit density, reached by indirect sampling**. The DDPM paper calls them explicit because they optimise a variational lower bound on the likelihood — the two statements use **different criteria**."

---

## Q2. 和 GAN、VAE 相比，扩散模型好在哪、差在哪？

| | GAN | VAE | **Diffusion** |
|---|---|---|---|
| 训练稳定性 | 差（对抗博弈、无损失曲线） | 好 | **好** |
| 样本质量 | 高 | 偏模糊 | **最高** |
| 采样速度 | **快（一步）** | 快 | **慢（迭代 $T$ 步）** |
| 模式覆盖 | 差（模式崩塌） | 中 | **好** |

🎤 **口语版**：
> "扩散在训练稳定性、样本质量、模式覆盖三项上都是最好的，唯一代价是采样慢 —— 需要迭代上百步。2016–2021 是 GAN 的时代，2021 之后扩散在图像基准上超过了 GAN。"

**Key English**:
> "Diffusion offers the best combination of training stability, sample quality and mode coverage; the price is slow, iterative sampling."

---

## Q3. 为什么前向过程不需要学习？

**核心**：因为 $\beta_t$ 是**我们自己设定的超参数**（noise schedule），前向过程**没有任何可优化的参数**。

> ⚠️ **常见错误**：不要说"因为前向是确定的"。**前向其实是随机的** —— 给定 $x_{t-1}$，采样出的 $x_t$ 每次都不同。判据是「**有没有待学参数**」，不是「确不确定」。

🎤 **口语版**：
> "前向完全由 schedule $\{\beta_t\}$ 指定，没有参数要学；反向的 $q(x_{t-1}\mid x_t)$ 是 intractable 的，所以要用神经网络去近似。"

**Key English**:
> "The forward process has **no learnable parameters** — it is fully specified by the schedule $\{\beta_t\}$. The reverse distribution is intractable, so we approximate it with a network."

---

## Q4. 为什么反向过程难？

**两层困难**：

1. **分布本身不可解**：贝叶斯公式 $q(x_{t-1}\mid x_t)=q(x_t\mid x_{t-1})\dfrac{q(x_{t-1})}{q(x_t)}$ 里，$q(x_{t-1})$ 和 $q(x_t)$ 是**未知的边缘分布**，没有闭式解。
2. **直接建模任意分布太难**：前向是高斯，但 $p(x_{t-1}\mid x_t)$ 可以是任意复杂形状。

**破解办法**：把 $x_0$ 也作为条件加进去 —— 因为 $q(x_t\mid x_0)$ 有闭式解，于是

$$q(x_{t-1}\mid x_t,x_0)=\frac{q(x_t\mid x_{t-1})\,q(x_{t-1}\mid x_0)}{q(x_t\mid x_0)}$$

三项都是已知高斯 → **后验是高斯，有闭式解**。

> ✅ **加分**：强调 "$x_0$ 只是**训练期的监督信号**，推理时模型看不到它。" 这是第 11–12 页的关键。

**Key English**:
> "We condition on $x_0$ as well. Then all three terms are known Gaussian densities, so the posterior becomes Gaussian and has a closed form."

---

## Q5. "每步噪声够小 ⇒ 反向也近似高斯" 这个假设怎么理解？

前向是**渐进地、小步地**破坏信息，所以反向恢复也应当是渐进的、每步近似高斯。**这个假设依赖的条件就是步长足够小**（$\beta_t$ 较小、$T$ 足够大）。

正因为每步近似高斯，我们**只需要估计均值和方差两个量**，而不是一个任意分布。

> ✅ **加分**：实践中**直接把方差取成 $\beta_t$**（DDPM 的做法），所以反向学习最终**只归结为学一个均值**。这就是网络只输出一个量的根本原因。

---

## Q6. 前向和反向是对称的吗？

都是马尔可夫链，但**性质完全不同**：

| | 前向 $q$ | 反向 $p_\theta$ |
|---|---|---|
| 参数 | 无（固定） | 有（要学） |
| 是否随机 | 是 | 是（每步重新注入 $\sigma_t z$） |
| 需要模拟 $T$ 步吗 | **不需要**（有闭式解） | **需要**（必须迭代） |

> ✅ **加分**：反向每步仍要加噪声 $\sigma_t z$ —— 因为它要建模前向分布的**逆**，前向是随机的，逆过程也必须是随机映射。

---

# 二、核心数学（第 7–14 页）

## Q7. 时间步的记号是怎么定义的？

- $\beta_t\in(0,1)$：单步噪声方差，也叫 noise schedule
- $\alpha_t=1-\beta_t$：单步**保留**系数
- $\bar\alpha_t=\prod_{s=1}^{t}\alpha_s$：累积**保留**系数

> ⚠️ **最容易记反的一点**：**$\bar\alpha_t$ 衡量"信号还剩多少"，不是"噪声有多少"**。
> - $\bar\alpha_t=1$ → 信号完好（$t=0$）
> - $\bar\alpha_t\to0$ → 信号没了、全是噪声（$t=T$）
> - **噪声占比是 $1-\bar\alpha_t$**，与 $\bar\alpha_t$ **方向相反**
>
> ✅ **很好用的自检恒等式**：$\left(\sqrt{\bar\alpha_t}\right)^2+\left(\sqrt{1-\bar\alpha_t}\right)^2=1$ —— 系数平方和恒为 1，这叫 **Variance Preserving (VP)**。

---

## Q8. 闭式解 $q(x_t\mid x_0)$ 是怎么推出来的？

**从单步出发**：$x_t=\sqrt{\alpha_t}\,x_{t-1}+\sqrt{1-\alpha_t}\,\varepsilon_{t-1}$

**递归展开**：

$$
\begin{aligned}
x_t&=\sqrt{\alpha_t}\,x_{t-1}+\sqrt{1-\alpha_t}\,\varepsilon_{t-1}\\
&=\sqrt{\alpha_t}\left(\sqrt{\alpha_{t-1}}x_{t-2}+\sqrt{1-\alpha_{t-1}}\varepsilon_{t-2}\right)+\sqrt{1-\alpha_t}\varepsilon_{t-1}\\
&=\sqrt{\alpha_t\alpha_{t-1}}\,x_{t-2}+\underbrace{\left(\sqrt{\alpha_t(1-\alpha_{t-1})}\,\varepsilon_{t-2}+\sqrt{1-\alpha_t}\,\varepsilon_{t-1}\right)}_{\text{两个独立高斯之和}}
\end{aligned}
$$

**关键引理**：若 $X\sim\mathcal N(\mu_X,\sigma_X^2)$、$Y\sim\mathcal N(\mu_Y,\sigma_Y^2)$ 且独立，则 $X+Y\sim\mathcal N(\mu_X+\mu_Y,\ \sigma_X^2+\sigma_Y^2)$。

**方差相加**：$\alpha_t(1-\alpha_{t-1})+(1-\alpha_t)=1-\alpha_t\alpha_{t-1}$

**继续嵌套到 $x_0$**：

$$\boxed{\;x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\varepsilon,\qquad \bar\alpha_t=\prod_{i=1}^{t}\alpha_i\;}$$

**自洽性检验**（很好的收尾）：$t=T$ 时 $\bar\alpha_T\to0$，故 $x_T\approx\varepsilon_T$ —— **加噪到纯噪声的目标确实达成**。

---

## Q9. 为什么闭式解这么重要？

**四个好处**：

1. **一步到位**：任意 $t$ 都能直接从 $x_0$ 采样，**不用模拟 $t$ 步**
2. **训练高效**：一个 batch 里每个样本可以用**不同的 $t$**（随机抽）
3. **每样本只需一次网络前向** → 整个训练目标 $O(1)$ 可算
4. **训练可并行** ← 这是扩散能扩展到 12B 参数的根本原因

**Key English**:
> "Because $x_t$ is available in closed form, the loss needs only **one network evaluation per example**, and $t$ can be sampled randomly — this is why diffusion training parallelises so well."

---

## Q10. 后验均值 $\tilde\mu_t$ 的两个系数是怎么来的？

**原始后验**（DDPM 论文 Eq.7）：

$$\tilde\mu_t=\frac{\sqrt{\bar\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\,x_t+\frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{1-\bar\alpha_t}\,x_0$$

**推导三步**（大工课件的"配方"路线）：

1. **通分**：两项分母都是 $1-\bar\alpha_t$，合并后按 $x_t$、$\varepsilon_t$ 归并
2. **抓枢纽恒等式**：用 $\beta_t=1-\alpha_t$ 凑出
   $$\alpha_t(1-\bar\alpha_{t-1})+\beta_t=1-\alpha_t\bar\alpha_{t-1}=1-\bar\alpha_t$$
3. **提公因子**：$\dfrac{\sqrt{\bar\alpha_{t-1}}}{\sqrt{\bar\alpha_t}}=\dfrac{1}{\sqrt{\alpha_t}}$，提出 $\dfrac{1}{\sqrt{\alpha_t}}$

**结果**：

$$\tilde\mu_t=\frac{1}{\sqrt{\alpha_t}}\left(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\varepsilon_t\right)$$

> 📌 **枢纽恒等式 $1-\alpha_t\bar\alpha_{t-1}=1-\bar\alpha_t$ 是整条推导的关键** —— 看到 $1-\alpha_t\bar\alpha_{t-1}$ 就要条件反射认出它。答辩时被问"哪一步最关键"，答这个。

---

## Q11. 后验方差为什么不用学？

$$\tilde\sigma_t^2=\frac{\beta_t(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}$$

它**只依赖 $t$**，与 $x_t$、$x_0$ 都无关 —— 是**常数**。代码里 `posterior_std` 是预先算好的 buffer，不是网络输出。

> ✅ **加分**：四种方差方案对比（第 17 页也提到）
>
> | 方案 | $\sigma_t^2$ | 出处 |
> |---|---|---|
> | Fixed-small | $\dfrac{\bar\beta_{t-1}}{\bar\beta_t}\beta_t$ | GLIGEN |
> | **Fixed-large** | $\beta_t$ | **DDPM 用的** |
> | hybrid | $\exp(v\log\beta_t+(1-v)\log\tilde\beta_t)$ | iDDPM（学 $v$） |
> | optimal | 解析式 | Analytic DPM |

---

## Q12. ELBO 是怎么来的？三项分别是什么？

**起点**：最大化似然 $\log p_\theta(x_0)$，但不可直接算 → 优化它的下界。

**Jensen 不等式**给出：

$$\log p_\theta(x_0)\ \ge\ \mathbb E_{q(x_{1:T}\mid x_0)}\!\left[\log\frac{p_\theta(x_{0:T})}{q(x_{1:T}\mid x_0)}\right]$$

**沿马尔可夫链拆开**，得到三项：

$$-\log p_\theta(x_0)\le\underbrace{L_T}_{①}+\sum_{t=2}^{T}\underbrace{L_{t-1}}_{②}-\underbrace{L_0}_{③}$$

| 项 | 形式 | 含义 |
|---|---|---|
| ① $L_T$ | $D_{\mathrm{KL}}\big(q(x_T\mid x_0)\,\Vert\,p(x_T)\big)$ | **先验匹配**：前向固定、$p(x_T)=\mathcal N(0,I)$，**是常数，无参数，直接忽略** |
| ② $L_{t-1}$ | $D_{\mathrm{KL}}\big(q(x_{t-1}\mid x_t,x_0)\,\Vert\,p_\theta(x_{t-1}\mid x_t)\big)$ | **学习发生在这里** |
| ③ $L_0$ | $-\log p_\theta(x_0\mid x_1)$ | **重构项** |

**Key English**:
> "$L_T$ is a constant because the forward process is fixed, so it has no parameters and can be ignored. **All the learning happens in the per-step term $L_{t-1}$.**"

---

## Q13. ⭐ 从 ELBO 到 $\mathcal L_{\text{simple}}$，中间发生了什么？（**最可能被问**）

**第 1 步：两个高斯之间的 KL 有闭式解**

$$D_{\mathrm{KL}}\big(\mathcal N(\mu_1,\Sigma)\,\|\,\mathcal N(\mu_2,\Sigma)\big)=\frac{1}{2}(\mu_1-\mu_2)^{\mathsf T}\Sigma^{-1}(\mu_1-\mu_2)$$

因为**方差被固定成同一个 $\Sigma$**（不学），协方差项与 $\theta$ 无关 → **只剩下均值的差**。

**第 2 步：把均值换成 $\varepsilon$ 形式**

代入 $\tilde\mu_t$ 的表达式，得到**带 SNR 权重**的 MSE：

$$L_{t-1}=\frac{\text{SNR}_t}{2\tilde\sigma_t^2}\big\|\varepsilon-\varepsilon_\theta(x_t,t)\big\|^2,\qquad \text{SNR}_t=\frac{\bar\alpha_t}{1-\bar\alpha_t}$$

> ⚠️ **SNR 随 $t$ 跨好几个数量级** —— DDPM 论文原话是 *"span several orders of magnitude"*。

**第 3 步：把权重丢掉**（Ho et al. 的关键选择）

$$\boxed{\;\mathcal L_{\text{simple}}=\mathbb E_{t,x_0,\varepsilon}\Big[\big\|\varepsilon-\varepsilon_\theta(x_t,t)\big\|^2\Big]\;}$$

**性质对比**：

| | 目标 | 是否似然下界 |
|---|---|---|
| $L_{\text{vlb}}$（完整） | $\sum_t\frac{\text{SNR}_t}{2\tilde\sigma_t^2}\Vert\varepsilon-\varepsilon_\theta\Vert^2$ | ✅ 是 |
| $\mathcal L_{\text{simple}}$ | $\sum_t\Vert\varepsilon-\varepsilon_\theta\Vert^2$ | ❌ **不是** |

> ✅ **必须主动说的一句**（这是理解深度的分水岭）：
> **"$\mathcal L_{\text{simple}}$ 丢掉了 SNR 权重，所以它不再是变分下界；但 Ho et al. 发现这样训出来的样本质量更好。"**

**Key English**:
> "With the covariance fixed, the Gaussian KL reduces to a squared error between the means. Ho et al. then **dropped the SNR weights entirely** to obtain $L_{\text{simple}}$. It is therefore **no longer a valid lower bound** on the likelihood — but it empirically yields **better sample quality**."

---

## Q14. ⭐ 为什么训练目标是一个 L2 loss / MSE？

**三层理由**：

1. **L2 的最优解是条件均值**：最小化 $\mathbb E[\|y-f(z)\|^2]$ 得 $f^*(z)=\mathbb E[y\mid z]$，而我们要网络逼近的 $\mathbb E[\varepsilon\mid x_t,t]$ **正好就是这个形式**。
2. **它由 ELBO 严格推出**：不是"我们决定用 MSE"，而是变分下界在高斯假设下**化简出来就是均值的 L2**。
3. **高斯噪声的负对数似然本身就是 L2**：$-\log p(\varepsilon)=\frac{1}{2}\|\varepsilon\|^2+\text{const}$。

**为什么不用 L1？** L1 的最优解是条件**中位数**，与后验高斯的结构不符；而且 L1 在 0 点不可导。

**Key English**:
> "The L2 loss is not an arbitrary choice: its Bayes-optimal solution **is** the conditional mean, which is exactly the posterior mean we are trying to estimate. It also follows directly from the Gaussian KL."

---

## Q15. 预测 $\varepsilon$ 和预测 $x_0$ 有什么区别？

**数学上完全等价**：$x_t$ 已知时，$x_0$ 与 $\varepsilon$ 通过闭式解**一一对应**

$$x_0=\frac{x_t-\sqrt{1-\bar\alpha_t}\,\varepsilon}{\sqrt{\bar\alpha_t}}\quad\Longleftrightarrow\quad \varepsilon=\frac{x_t-\sqrt{\bar\alpha_t}\,x_0}{\sqrt{1-\bar\alpha_t}}$$

**但有隐含权重差异**：

$$\|\varepsilon-\hat\varepsilon\|^2=\frac{\alpha_t}{1-\bar\alpha_t}\|x_0-\hat x_0\|^2=\text{SNR}_t\cdot\|x_0-\hat x_0\|^2$$

**实践选择**：`pred_noise` 更常用（尺度稳定，$\varepsilon\sim\mathcal N(0,I)$ 不随 $t$ 剧变）；高分辨率时 $v$-prediction 更稳。

**Key English**:
> "Predicting $\varepsilon$ and predicting $x_0$ are **equivalent** — $x_t$ determines both. The difference is numerical: $\varepsilon$ always has unit scale, whereas $x_0\in[-1,1]$ makes the gradient scale vary a lot with $t$."

---

# 三、U-Net 与架构（第 15–17 页）

## Q16. 为什么用 U-Net 不用普通 CNN？

**需要同时满足两个要求**：

| 要求 | 谁提供 |
|---|---|
| **全局上下文**（"这张图整体噪声多大"） | 下采样路径 + 中间块 |
| **空间细节**（"这个像素该改成什么"） | **跳连接**（skip connections） |

**去掉跳连接的后果**：深层特征经过多次下采样后空间信息已丢失，上采样无法恢复精细结构 → 输出会**模糊、细节错位**。

> ✅ **加分**：U-Net 是从**医学图像分割**（Ronneberger et al. 2015）借来的架构，扩散沿用；区别是扩散的 U-Net 还要注入**时间步条件**。

---

## Q17. 时间步 $t$ 是怎么进入网络的？为什么要正弦编码？

**路径**：标量 $t$ → **正弦位置编码** → MLP → context 向量 → 通过 **scale-and-shift（FiLM）** 调制每个 ResNet block。

$$\text{scale, shift}=\text{MLP}(\text{context});\qquad x\leftarrow x\cdot(\text{scale}+1)+\text{shift}$$

**为什么用正弦而不是直接输入一个数？**
**高频编码让网络能分辨相近的 $t$** —— 如果直接输入标量，网络不容易区分 $t=500$ 和 $t=501$；正弦编码用一组不同频率的基函数把 $t$ 展开成向量，$t$ 的微小变化在高频分量上会被放大。

> ✅ **加分**：这用的是 Transformer **位置编码的同一套方法**（不同频率的正弦/余弦）。

---

## Q18. 文本条件是怎么注入的？

**两条独立路径，最后相加**：

```python
context = time_mlp(time_emb) + condition_mlp(text_emb)
```

- 时间步和文本都编码到同一空间（`context_dim = dim*4 = 192`）
- 然后**一起**送进每个 ResNetBlock，用 scale-and-shift 调制

**训练时随机丢条件**（`uncond_prob = 0.2`）：这是 CFG 的训练基础。

---

## Q19. U-Net 参数量多少？参数集中在哪？

| 配置 | 参数量 |
|---|---|
| 测试用（`dim=2, mult=(1,2,4)`） | **6,499** |
| **真实训练用（`dim=48, mult=(1,2,4,8)`）** | **12,444,963 ≈ 12.4M** |

**参数分布**（真实配置）：

| 组 | 参数量 | 占比 |
|---|---|---|
| `downs`（4 stage） | 2,521,296 | 20.3% |
| **`mid`（仅 2 个 block）** | **5,607,936** | **45.1%** |
| `ups`（4 stage） | 4,132,224 | 33.2% |
| 其余（conv / mlp） | ~183K | 1.5% |

> ✅ **加分点**：**中间块只占 2 个 ResNetBlock 却占 45% 参数** —— 因为它在**最高通道数（384）**上做 3×3 卷积。这解释了为什么扩散 U-Net 的参数量远大于同深度分类网络。

---

## Q20. 反向方差 $\sigma_t^2$ 怎么选的？

| 方案 | 取值 | 出处 |
|---|---|---|
| Fixed-small | $\tilde\beta_t=\dfrac{\bar\beta_{t-1}}{\bar\beta_t}\beta_t$ | GLIGEN |
| **Fixed-large** | $\beta_t$ | **DDPM 原始做法** |
| hybrid | 可学习插值 | iDDPM |
| optimal | 解析式 | Analytic DPM |

**DDPM 选 $\beta_t$ 的理由**：理论上界是 $\tilde\beta_t$，但实验发现直接用 $\beta_t$ 效果就很好 —— 又一个"理论最优 ≠ 实践最优"的例子。

---

# 四、训练与采样（第 15–16 页）

## Q21. 训练一次梯度下降要做什么？为什么这么快？

**算法**（4 步）：

```
1. 抽干净图 x0 ~ q(x0)
2. 抽时间步 t ~ Uniform({1,...,T})
3. 抽噪声 eps ~ N(0, I)
4. 用闭式解造 x_t = sqrt(abar_t)*x0 + sqrt(1-abar_t)*eps
   对 || eps - eps_theta(x_t, t) ||^2 做梯度下降
```

**为什么快**：因为有闭式解，**不需要真的模拟 $t$ 步加噪**。每个样本**只需一次网络前向**，而且一个 batch 里每个样本可以用**不同的** $t$。

**Key English**:
> "One random $t$ per example per gradient step, and because $x_t$ has a closed form there is **no sequential simulation** — the loss needs only one network evaluation."

---

## Q22. 采样为什么要迭代 $T$ 步？怎么生成？

```
x_T ~ N(0, I)
for t = T, ..., 1:
    z = randn_like(x_t) if t > 1 else 0
    x_{t-1} = 1/sqrt(alpha_t) * ( x_t - (beta_t/sqrt(1-abar_t)) * eps_theta(x_t, t) ) + sigma_t * z
return x_0
```

**为什么必须迭代**：反向过程是**学习出来的近似**，没有闭式解 —— 只能一步一步往回走。这也是采样慢的根本原因。

**每步在做什么**：
1. 网络预测 $x_t$ 里的噪声 $\varepsilon_\theta$
2. **减去**这个噪声估计（系数 $\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}$）
3. 乘 $\frac{1}{\sqrt{\alpha_t}}$ 做**尺度校正**
4. 加 $\sigma_t z$ **重新注入随机性**

> ✅ **加分**：$t=1$ 时 $z=0$ —— **最后一步不再加噪**，输出干净的 $x_0$。

---

## Q23. 为什么每步还要加噪声 $\sigma_t z$？

因为反向过程要**模拟前向分布的逆**。前向每步都是随机的，所以逆过程也必须是**随机映射**；否则 $p_\theta(x_{t-1}\mid x_t)$ 会退化成确定性函数，偏离真实的 $q(x_{t-1}\mid x_t,x_0)$。

> 📌 **顺便引出 DDIM**：DDIM 发现可以**去掉**这个随机项（$\sigma_t=0$），把采样变成确定性 ODE，**从而允许跳步加速**。

---

## Q24. ⭐ 采样时为什么要把 $\hat x_0$ clamp 到 $[-1,1]$？

**原因**：网络在**早期（$t$ 大）**预测的 $\hat x_0$ 误差很大（那时输入几乎是纯噪声，信息太少）。如果不截断，这些大误差会进入后验均值，并在 **$T$ 步迭代中累积放大** → 生成结果崩坏。

**注意范围是 $[-1,1]$ 而非 $[0,1]$**：因为数据预处理把像素从 $[0,1]$ 归一化到了 $[-1,1]$（`normalize: img*2-1`）。

> ✅ **加分**：可以对比论文原版形式 —— DDPM Algorithm 2 直接用 $\varepsilon_\theta$ 构造 $x_{t-1}$，**没有天然的位置插 clamp**；CS231n 作业用"先反解 $\hat x_0$ 再走 `q_posterior`"，**就是为了方便 clamp**。

**Key English**:
> "At high noise levels $\hat x_0$ is a poor estimate. Without clamping, those errors **accumulate over the $T$ sampling steps** and the generation blows up. Since the data is normalised to $[-1,1]$, that is the range we clamp to."

---

# 五、实验结果（第 18–19 页）

## Q25. 你的验证结果是什么？

| 测试项 | 实测值 | 要求 |
|---|---|---|
| `q_sample` | **5.104e-07** | ~0 |
| `predict_start_from_noise` | **2.798e-06** | < 1e-5 |
| `predict_noise_from_start` | **1.060e-06** | < 1e-5 |
| U-Net 参数量（`dim=2`） | **6,499** | == 6,499 |

**这些是相对误差**，量级都在 float32 的数值精度极限上 —— 说明实现与官方期望值**逐位吻合**。

---

## Q26. 生成的图片是怎么来的？

**来源**：官方发布的预训练权重 `model-70000.pt`（step = 70,000），不是我自己从零训练的。

**配置**：$T=100$ 步、32×32 emoji、`pred_noise` objective、`dim=48`（12.4M 参数）。

> ⚠️ **诚实说明的说法**（很重要，被问到不要含糊）：
> **"These samples are generated by loading the released pretrained checkpoint into my implementation of the diffusion process."**
>
> 📌 **务必想清楚自己的边界**：如果你的实现只完成了部分函数（例如 `q_sample` 已完成，其余是参考实现），就说清"我完成了哪些、其余参考了官方实现"。**含糊其辞比承认更危险** —— 老师追问一句就会露。

**Key English**:
> "These results come from **loading the released pretrained checkpoint** into the diffusion pipeline I implemented. I did not train from scratch — the assignment explicitly says training is not required."

---

## Q27. 训练用多少步？为什么不是 1000？

| 参数 | 值 |
|---|---|
| `timesteps` | **100**（不是经典的 1000） |
| `beta_schedule` | **sigmoid**（`GaussianDiffusion` 默认值） |
| `objective` | `pred_noise` |
| `train_batch_size` / `lr` | 256 / 1e-3（Adam） |
| `train_num_steps` | 50,000 |

**为什么 100 步够了**：32×32 的小图信息量低，100 步已足以把信号破坏成噪声；少步数**训练更快、采样也快 10 倍**。

> ✅ **加分**：这正是"少步采样"思路的体现 —— 也是 DDIM、蒸馏等加速方法的方向。

---

## Q28. CFG 的效果说明了什么？（第 19 页那张图）

**实验设置**：prompt `"crying face"`，$w = 0,\ 2,\ 5,\ 10$。

**观察**：
- $w=0$：**部分样本没有眼泪**（贴合度不够）
- $w\ge2$：**每一列都稳定出现蓝色泪滴**，词跟随性显著提升
- $w$ 越大：越贴合 prompt，但**样本多样性下降**（不同样本越来越像）

**结论**：这正好实证了那一句 *"A higher w makes the generations more faithful to the condition but tends to reduce their diversity."*

**CFG 公式**：

$$\varepsilon_{\text{cfg}}=(1+w)\,\varepsilon_{\text{cond}}-w\,\varepsilon_{\text{uncond}}=\varepsilon_{\text{uncond}}+(1+w)(\varepsilon_{\text{cond}}-\varepsilon_{\text{uncond}})$$

**为什么叫 "classifier-free"**：早期方法要额外训一个分类器 $p(y\mid x)$ 并算 $\nabla_x\log p(y\mid x)$；CFG 用**同一个模型**的条件/无条件预测之差替代它，**不需要分类器**。

**代价**：每步要跑**两次前向** → **采样开销翻倍**。

---

## Q29. 有没有更快的采样方法？

| 方法 | 思路 | 步数 |
|---|---|---|
| **DDIM** | 去掉随机项 $\sigma_t z$，变成确定性 ODE，允许跳步 | 几十步 |
| **Progressive Distillation** | 训练学生模型，用更少步复现教师 | 4–8 步 |
| **Consistency Models** | 直接学"任意噪声 → 干净图"的映射 | **1–2 步** |
| **ADD**（对抗扩散蒸馏） | 加判别器，把 CFG 烘焙进学生 | 1–4 步 |

> 📌 如果要一句话总结：**"扩散的采样慢是结构性的，加速是这个领域最活跃的方向之一。"**

---

# 六、延伸与对比（第 18、21 页）

## Q30. Latent Diffusion 为什么要用 VAE 的编码器？

**动机**：直接在**像素空间**做扩散，在高分辨率数据上不 work（计算量随像素数平方增长）。

**方案**：先把图像压到低维 latent，在 latent 上做扩散。

$$256\times256\times3\ \xrightarrow{\text{Encoder}}\ 32\times32\times16\ \xrightarrow{\text{扩散}}\ \xrightarrow{\text{Decoder}}\ 256\times256\times3$$

**典型配置**：$D=8,\ C=16$ —— 空间维度缩小 8 倍。

**为什么要另加一个判别器？** 因为 VAE 解码器输出**偏模糊**，加一个 GAN 判别器改善清晰度。

> ✅ **加分**：所以现代 LDM 管线是 **"VAE + GAN + diffusion"** 三种生成模型的混合体 —— 这颠覆了"选一个最好的生成模型"的直觉。

---

## Q31. DiT 和 U-Net 的区别？

**核心变化**：把 U-Net 换成标准 Transformer（Peebles & Xie, ICCV 2023）。**好处是可扩展性** —— Transformer 的缩放规律更成熟，能推到 12B 参数。

**要解决的新问题**：怎么把条件注入 Transformer？两种做法：
1. **adaLN / FiLM**：预测 scale & shift —— 时间步条件最常用
2. **Cross-Attention / Joint Attention** —— 文本、图像条件常用

---

## Q32. Rectified Flow 和 DDPM 什么关系？

**都属于"广义扩散"框架**：$x_t=a(t)x+b(t)z$，网络预测 $y_{\text{gt}}=c(t)x+d(t)z$。

| | $a(t)$ | $b(t)$ | 目标 |
|---|---|---|---|
| **Rectified Flow** | $1-t$ | $t$ | $z-x$（速度） |
| **DDPM（VP）** | $\sqrt{\bar\alpha_t}$ | $\sqrt{1-\bar\alpha_t}$ | $z$（噪声） |

**为什么 Rectified Flow 只要约 50 步？** 因为它把噪声和数据用**直线**连接，路径更直，欧拉法积分的误差小；DDPM 的路径是弯曲的，需要很多小步。

> 📌 **关键认识**：两者**不是对立的**，DDPM 是广义扩散框架在方差保持（VP）参数化下的一个特例。

---

# 七、⚠️ 可能的刁难问题与应对

## Q33.（刁难）你说 ELBO 化简成 MSE，中间具体怎么化的？

**这是最可能被追问、也最容易露馅的一题。** 按这个顺序答：

1. **两个高斯 KL 有闭式解**，且因为方差固定成同一个 $\Sigma$，只剩均值的差
2. **把均值换成 $\varepsilon$ 形式** → 得到带 SNR 权重的 MSE
3. **丢掉权重** → $\mathcal L_{\text{simple}}$

> ✅ **如果被追问"你推过吗"**：诚实说"完整推导我参考了 DDPM 论文 Appendix A 和课程材料的整理，核心是第 13 题那三步"。**不要硬撑说"我独立推的"。**

---

## Q34.（刁难）学 MSE 为什么不叫它 MLE？它俩什么关系？

- **MLE**（最大似然）是**目标**：最大化 $\log p_\theta(x_0)$
- **ELBO** 是不可解时的**替代下界**
- **MSE** 是这个下界在**高斯假设下化简后的形式**

所以准确的链条是：**MLE → ELBO → （丢权重）→ MSE**。三者不是同一件事，**MSE 已经不是 MLE 的下界了**。

---

## Q35.（刁难）扩散模型到底算显式还是隐式密度？

**先说结论**：**你的演示文稿按 Goodfellow 的口径，把它归入 `Implicit density → Indirect sampling`，这是对的**（见下表的对照）。

**关键是要分清两套体系的用词**（这是最容易混、也最容易被抓的地方）：

| 分类体系 | 判据 | 扩散模型的位置 |
|---|---|---|
| **Goodfellow 图（你的第 4 页用的）** | **能否直接算出 $p(x)$** | `Implicit density` → 其下分 **Direct / Indirect**；扩散标在 **Indirect** |
| **DDPM 论文自己的说法** | **是否定义了 $p_\theta(x)$ 并优化它** | **显式密度**，因为优化 $\log p(x)$ 的变分下界 |

> ⚠️ **一个必须记住的对应关系**：
> 在 Goodfellow 图里，分支和子节点是这么对应的 ——
> - `Explicit density` → `Tractable density` / **`Approximate density`**
> - `Implicit density` → **`Direct`** / **`Indirect`**
>
> 所以 **`Approximate density` 属于 Explicit 分支**。如果说扩散在 `Implicit density` 下，那它的子节点应当是 **`Indirect`**，**不能同时写 `Approximate density`** —— 这两个词来自不同的分支，放在一起是自相矛盾的。

✅ **已核对演示文稿**：第 4 页现在三处口径统一 ——
- 图上标注：`Indirect` 分支下的 `Diffusion Models`
- 上方正文：`Implicit density → Indirect sampling`
- 右下分类框：`Implicit density / Indirect sampling`

🎤 **如果被追问"那它到底是显式还是隐式？"**，最稳的答法：
> "按 Goodfellow 那张图的判据 —— **能否直接计算 $p(x)$** —— 扩散属于隐式密度，因为它只能通过迭代采样近似样本。不过 DDPM 论文本身把它称作显式密度模型，因为它优化的是似然的下界。**两套说法用的是不同的判据，指的不是同一件事。**"

**Key English**:
> "In Goodfellow's taxonomy the criterion is whether $p(x)$ can be computed directly, and diffusion falls under **implicit density via indirect sampling**. The DDPM paper, however, calls it explicit because it optimises a variational lower bound on the likelihood. The two statements use **different criteria**."

---

## Q35b.（追问）为什么说扩散"学习近似密度"？

因为**变分下界本身就是对 $\log p_\theta(x_0)$ 的近似**：

$$\log p_\theta(x_0)\ \ge\ \text{ELBO}$$

- 我们**没有**精确计算 $\log p(x)$（那是不可解的）
- 我们优化的是它的**下界**，所以是"**近似**密度"
- 并且**采样也是近似的** —— 反向过程是学出来的，不是精确逆

> 📌 这也解释了为什么 $\mathcal{L}_{\text{simple}}$ 丢掉权重后仍能用：**它本来就不是在精确优化似然。**

---

## Q36.（刁难）为什么 $\beta_t$ 用 sigmoid 而不是经典 linear？

**诚实答案**：这里用的是 `GaussianDiffusion` 的**默认值**（sigmoid），主要考虑了 32×32 小图的收敛性和训练速度。

**完整的取舍**：
- **linear**：DDPM 原论文方案，最经典，适合小图
- **cosine**：改善 SNR 分布，避免后期信息过早丢失
- **sigmoid**：论文声称**更适合 >64×64** 的图像

> ✅ **加分**：主动承认"对 32×32 而言这个选择更多是默认值而非精心调优"，并说"如果要严格比较，应该固定其他因素做一次消融"。

---

## Q37.（刁难）clamp 会不会损害生成质量？

**会有一点点影响，但收益远大于代价。**

- **不 clamp**：少数极端值会在 $T$ 步里被反复放大 → **整批生成崩坏**（这是灾难性的）
- **clamp**：把单步的 $\hat x_0$ 约束在合法像素范围内 → 换来**稳定性**

**本质上是一个"单步精度 vs 长链稳定性"的权衡** —— 在 $T$ 步的长链里，**稳定性压倒单步精度**。

---

## Q38.（刁难）你说 $\varepsilon$ 和 $x_0$ 一一对应，那为什么还需要两个函数？

**因为方向不同、数值条件不同**：

| 函数 | 输入 | 输出 | 用途 |
|---|---|---|---|
| `predict_start_from_noise` | $x_t,\varepsilon$ | $x_0$ | `pred_noise` 采样时反解干净图 |
| `predict_noise_from_start` | $x_t,x_0$ | $\varepsilon$ | `pred_x_start` 时转成噪声 |

数学上互逆，但**工程上需要两个方向的转换**，因为 `objective` 可以是两者之一。

---

# 八、兜底话术

## 遇到完全不会的问题

🎤 **诚实且体面的说法**：
> "这个问题我目前没有想清楚。我的理解是……（说出你知道的部分），但完整的推导/实现我还没有验证过，回去我会补上。"

> ⚠️ **绝对不要做的事**：硬编一个听起来合理的答案。老师追问一句就会暴露，那时**信任损失远大于承认不知道**。

## 遇到质疑"这些结果是你做的吗"

🎤 **准确说明边界**：
> "预训练权重是官方提供的，作业也明确说了不要求自己训练。我完成的是（列出你实际完成的部分，例如 `q_sample`、`loss` 等），以及把整个流程跑通、做了 CFG 的对比实验。其余部分我参考了课程提供的实现。"

**Key English**:
> "The pretrained weights are provided by the assignment. What I implemented myself is [X]; I reproduced the full sampling pipeline and ran the CFG comparison."

---

# 九、答辩前自查清单

- [ ] 能用一句话说清**扩散属于哪一类生成模型**（并知道两套口径的差别）
- [ ] 能写出**核心式** $x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\varepsilon$
- [ ] 能说出**枢纽恒等式** $\alpha_t(1-\bar\alpha_{t-1})+\beta_t=1-\bar\alpha_t$
- [ ] 能说清 **ELBO 三项**各是什么、哪一项是学习发生的地方
- [ ] 能说清 **$\mathcal L_{\text{simple}}$ ≠ 下界**（丢权重）
- [ ] 能解释 **L2 为什么是最优解**（条件均值）
- [ ] 能说清**为什么要 clamp**、范围为什么是 $[-1,1]$
- [ ] 能说清 **CFG 的公式、代价、$w$ 的权衡**
- [ ] 能**诚实说明结果来源**（预训练权重 + 自己的实现范围）
- [ ] 演示文稿**分类口径前后一致**

---

*本手册基于当前 22 页演示文稿整理。题目按「被问到的概率 × 答不好的代价」排序，第七节的 6 题最值得提前演练。*
