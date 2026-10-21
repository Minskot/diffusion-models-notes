# Diffusion Model 知识点整合（ML / CV / cs231n 三源梳理）

> 任务：在工作区（DSAI4203 ML、COMP 4423 CV、cs231n 资料包）中定位所有与 **Diffusion Model（扩散模型）** 相关的出现位置，并整合知识点与考点。
>
> 检索范围：`_text\`（ML 全部 19 份 PDF 提取）、`_text_cv\`（CV 11 讲讲义 + Tutorial）、`_text_cs231n\`（cs231n 24 份 slides + 一份中文笔记）、`cs231n\assignments\`、原始 PDF 文件名。
> 检索词：`diffusion`、`DDPM`、`DDIM`、`denoising/de-noising`、`score matching`、`noise schedule`、`stable diffusion`、`latent diffusion`、`reparameterization`。

---

## 1. 结论先说：全工作区只有 3 处提及，且只有 1 处是实质内容

| # | 位置 | 形式 | 实质内容量 |
|---|---|---|---|
| 1 | `4423 CV/cs231n/slides/cs231n_lszy.pdf` 第 10 节 | **完整技术讲义** | ★★★★★ **唯一实质来源** |
| 2 | `4423 CV/L11-Lecture-Learning,paradigms-v4.5.pdf` p76 | Reading List 一行 | ☆ 仅一个词 |
| 3 | `4423 CV/L1-Lecture-Introduction-v5.1.pdf` p28-30、p37 | 两个 demo 网址 + 一篇自己的论文引用 | ☆ 仅名称 |

**DSAI4203 ML 课程：完全没有扩散模型。** 全 19 份 PDF 中 `diffusion` 零命中。唯一相关的只有 L10 自编码器里的 **denoising autoencoder（去噪自编码器）** —— 那是不同的东西，但思想上有联系（见第 5 节）。

---

## 2. 逐处定位（含原始上下文）

### 2.1 位置 1（唯一实质来源）：`cs231n_lszy.pdf` 第 10 节

**文件**：`4423 CV\cs231n\slides\cs231n_lszy.pdf`（27 页，中文笔记）
**性质**：一份**第三方中文学习笔记**，记录作者手写完成 cs231n 全部三个 Assignment 的过程。不是官方课件。

**它在全文中的定位**（p4 的学习路线表）：

| 阶段 | 核心内容 |
|---|---|
| 线性分类器 | kNN → SVM → Softmax → 梯度推导 |
| 浅层网络 | 两层网络 → 反向传播 → 梯度检查 |
| 深层网络 | 任意深度全连接 → BN / LN → Dropout |
| 卷积网络 | Conv → Pool → Spatial BN → 三层 CNN |
| 序列模型 | RNN → LSTM → Image Captioning |
| 注意力机制 | Transformer Decoder → ViT |
| 自监督学习 | SimCLR 对比学习 → CLIP → DINO |
| **生成模型** | **DDPM → U-Net → Classifier-Free Guidance** |

**关键说明**（p4 原文）：Assignment 3 的内容是 *"Transformer Captioning、Vision Transformer、SimCLR 自监督学习、CLIP/DINO、DDPM 扩散模型"* —— 注意这**比官方 2020 版 assignment3 更"新"**：官方 `assignment3.md` 里根本没有 DDPM（见 2.4 节）。

**第 10 节全文结构**（p26-27，正文实际只占 2 页）：
- 10.1 DDPM 理论框架
- 10.2 U-Net 架构
- 10.3 噪声调度

> ⚠️ **重要提醒**：这一节是**第三方笔记**，且篇幅仅 2 页。它的公式记号（如 $\bar{\alpha}_t$、$\beta _t$）与 DDPM 原论文一致，但**没有推导过程**，也不代表任课教师的授课范围。第 4 节的公式整理已与 DDPM 原论文（Ho et al. 2020）交叉核对。

---

### 2.2 位置 2：CV 课程 L11 p76 Reading List

**文件**：`4423 CV\L11-Lecture-Learning,paradigms-v4.5.pdf`，第 76 页（共 77 页，倒数第 2 页）

**原文**：
```
Reading List
• Multimodal learning
• Pretraining and large models
• Diffusion models
```

**上下文**：这一页是整门 CV 课的**最后一页实质内容**（p77 是 "Thank you!"）。L11 讲的是 Learning Paradigms（多任务、N-shot、迁移、度量、元学习、VAE、GAN、强化学习、半监督、弱监督、自监督、主动学习），扩散模型**不在正课内容里**，只出现在结尾的阅读清单。

**这意味着什么**：讲师把 Diffusion 与 Multimodal learning、Pretraining/large models 并列列为**拓展阅读**，属于"知道有这回事"级别，**不是必考内容**。但同时也是整门课**唯一被官方点名、却完全没讲**的三大主题之一 —— 若出"课程拓展"类型的题，这三个词是唯一的官方线索。

---

### 2.3 位置 3：CV 课程 L1 导论中的 Stable Diffusion

**文件**：`4423 CV\L1-Lecture-Introduction-v5.1.pdf`

三处引用：

| 页 | 内容 |
|---|---|
| p28 | `https://stablediffusionweb.com/` |
| p29 | 同网址 + **四段 Stable Diffusion 提示词（prompt）示例** |
| p37 | 论文引用：*"Compositional Inversion for Stable Diffusion Models, The 38th Annual AAAI Conference on Artificial Intelligence (AAAI), 2024"* |

p29 的四段 prompt 原文（用于演示文生图）：
1. `interior design, frank lloyd wright house cave with forest canopy, dark wood, streaks of light, light fog, living room :: bubbletech --test --ar 9:16`
2. `cyberpunk techwear streetwear look and clothes, ... Errolson Hugh, Sacai, Nike ACG, Yohji Yamamoto, Y3, ACRNYM, matte painting --w 2176 --h 3840`
3. `woman, warrior, detailed illustration, digital art, ... Dan Mumford, Krzysztof Maziarz, trending on artstation`
4. `a full page design of spaceship engine, black and bronze paper, intricate, highly detailed, epic, infographic, marginalia --ar 9:16`

**意义**：L1 里扩散模型是作为**"CV 很酷的应用"**来展示的（与 thispersondoesnotexist、Quick Draw、sketch-rnn、风格迁移并列），**没有任何技术讲解**。p37 那篇 AAAI 2024 论文是讲师自己团队（魏骁勇组）的工作，属于研究展示。

---

### 2.4 明确"没有"的地方（避免误判）

这些地方**查过、确认没有**，列出来是为了让你放心：

| 位置 | 检索结果 |
|---|---|
| **DSAI4203 ML 全部 19 份 PDF** | `diffusion` **0 命中** |
| **CV L2–L10 全部讲义** | `diffusion` **0 命中** |
| **cs231n `assignment3.md`**（官方作业说明） | **0 命中**。官方 2020 版共 5 题：RNN Captioning / LSTM Captioning / Network Visualization / Style Transfer / **GAN**（15 分）—— 最接近的是 GAN，没有扩散 |
| **cs231n 官方课件 24 份** | 除 `cs231n_lszy.pdf` 外 **全部 0 命中** |
| **cs231n `README.md`** | 0 命中 |
| **ML 的 ICML 论文**（Polynomial Simplicity） | 0 命中 |

> **关键洞察**：cs231n 官方课件的年份是 **2017–2020**（作业截止日 2020-05-27），正好**早于扩散模型爆发**（DDPM 2020 年 6 月才发 arXiv）。所以官方课件里只有 VAE 和 GAN，**没有扩散模型是正常的**，不是资料缺失。
> 而 `cs231n_lszy.pdf` 是 2025 年前后写的第三方笔记，作者按**新版** cs231n 作业补上了 DDPM，这才有了唯一的技术来源。

---

## 3. cs231n 官方课件里的"生成模型"位置（作为扩散模型的对照背景）

虽然官方课件没有扩散模型，但**有一节完整的生成模型课**，是理解扩散模型的最佳铺垫：

**`lecture_11.pdf`（130 页）** —— 主题是 **Generative Modeling**，含 46 次 "GAN"、65 次 "generative"、85 次 "variational"。

**核心内容**：Goodfellow 2017 教程的**生成模型分类图**（在课件中重复出现 4 次以上）：

```
Generative models
├── Explicit density          ├── Implicit density
│   ├── Tractable density         └── GAN
│   │   ├── Fully visible belief net
│   │   └── (PixelRNN / PixelCNN)
│   └── Approximate density
│       ├── Variational  — Variational Autoencoder (VAE)
│       └── Markov Chain — Boltzmann Machine
```

**这张图正是理解扩散模型位置的关键** —— 在 Goodfellow 的原始分类里，**Diffusion 属于 "Approximate density → Markov Chain" 这一格**（与 Boltzmann Machine 同格，因为扩散的前向/反向过程都是马尔可夫链）。

课件还包含：
- **VAE**（Kingma & Welling, "Auto-Encoding Variational Bayes", ICLR 2014）：不可 tractable 的密度函数、变分推断、重参数化、生成数据；优缺点中明确写 *"Samples blurrier and lower quality compared to state-of-the-art (GANs)"*
- **GAN**（Goodfellow et al.）：*"GANs: not modeling any explicit density function!"*、two-player game、训练算法、DCGAN、可解释向量运算、2017 年 "GAN Zoo" 爆发、BigGAN 等

---

## 4. 扩散模型知识点整合（以 `cs231n_lszy.pdf` 第 10 节为主干）

> 以下公式已与 **DDPM 原论文**（Ho, Jain & Abbeel, *Denoising Diffusion Probabilistic Models*, NeurIPS 2020, arXiv:2006.11239）交叉核对，符号沿用原文，便于你直接对照教材。

### 4.1 核心思想

DDPM 定义了**一条从数据逐步加噪到纯噪声的马尔可夫链**（前向过程），再学习**一条反向的去噪链**，从纯高斯噪声逐步恢复出图像。

```
x₀ ──加噪──> x₁ ──> x₂ ──> … ──> x_T ≈ N(0, I)      前向过程 q（固定，无需学习）
x_T ──去噪──> … ──> x₁ ──> x₀                        反向过程 p（学习，即生成）
```

### 4.2 前向过程（加噪）

单步：

$$
q(x_t \mid  x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta _t} x_{t-1}, \beta _t I)
$$

**任意步的闭式解**（可一步从 x₀ 跳到 x_t，是训练高效的关键）：
$
q(x_t\mid x_0)=\mathcal N\left(x_t;\sqrt{\bar\alpha_t}\,x_0,(1-\bar\alpha_t)I\right)
$
即

$$
x_t = \sqrt{\bar{\alpha}_t} \cdot  x_{0} + \sqrt{1 - \bar{\alpha}_t} \cdot  \varepsilon
$$

，其中 $\varepsilon  \sim \mathcal{N}(0, I)$

其中：

$$
\alpha _t = 1 - \beta _t
$$

，

$$
\bar{\alpha}_t = \prod _{s=1}^{t} \alpha _s
$$

（累积乘积）

笔记对应的代码：
```python
def q_sample(self, x_start, t, noise):
    sqrt_alpha_bar = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus  = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
    return sqrt_alpha_bar * x_start + sqrt_one_minus * noise
```

### 4.3 反向过程（去噪）与后验

**后验分布**（给定 x₀ 时的反向条件分布，可解析求出）：
$
q(x_{t-1}\mid x_t,x_0)=\mathcal N\left(x_{t-1};\tilde\mu_t,\tilde\sigma_t^2 I\right)
$
其中
$
\tilde\mu_t=\frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{1-\bar\alpha_t}\,x_0+\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\,x_t
$

### 4.4 训练目标（简化为预测噪声）

DDPM 最关键的简化 —— **不做复杂的变分下界优化，直接回归噪声**：
$
\mathcal L=\mathbb E_{t,x_0,\varepsilon}\left[\left\|\varepsilon-\varepsilon_\theta(x_t,t)\right\|^2\right]
$

笔记对应的代码：
```python
def p_losses(self, x_start, model_kwargs={}):
    t = torch.randint(0, nts, (b,), device=x_start.device).long()
    noise = torch.randn_like(x_start)
    x_t = self.q_sample(x_start, t, noise)
    model_pred = self.model(x_t, t, model_kwargs=model_kwargs)
    loss = F.mse_loss(model_pred, noise, reduction="none")
    loss = (loss * loss_weight).mean()
```

> **这就是扩散模型最优雅的地方**：训练一个网络，给它加噪后的图 x_t 和时间步 t，让它预测当初加进去的噪声 ε。抽掉预测的噪声，就前进了一步。

### 4.5 噪声预测网络的架构：U-Net

笔记原文：*"噪声预测网络使用 U-Net 结构"*

| 组件 | 作用（笔记原文） |
|---|---|
| **下采样路径** | 逐步缩小空间分辨率，增加通道数 |
| **上采样路径** | 逐步恢复分辨率，通过**跳连接（skip connections）**融合高分辨率特征 |
| **时间嵌入** | 使用**正弦位置编码**将时间步 t 嵌入为向量，通过 **FiLM 调制（scale + shift）**注入每个 ResNet Block |
| **Classifier-Free Guidance** | 训练时以概率 p 随机丢弃条件信息，推理时通过加权有条件和无条件预测的差值来增强条件控制 |

> **跨课程联系**：U-Net 本身在 **CV 课程 L10 p21-22** 有完整讲解（Ronneberger et al. 2015，医学图像分割）：
> *"The down-sampling is a typical convolutional network, which consists of repeated application of convolutions, each followed by a ReLU and a max pooling operation. During the down-sampling, the spatial information is reduced while the feature information is increased. The upsampling combines the feature and spatial information through a sequence of up-convolutions and concatenations with high-resolution features from the left."*
> **扩散模型的 U-Net 就是借用这个架构** —— 两门课在这里接上了。

### 4.6 噪声调度（Noise Schedule）

笔记实现了三种 β 调度策略：

| 调度方式 | 特点（笔记原文） |
|---|---|
| **Linear** | 原始 DDPM 论文提出，β 线性增长 |
| **Cosine** | 改善 SNR 分布，避免后期信息过早丢失 |
| **Sigmoid** | 适合高分辨率图像（> 64×64） |

> 对应文献：Linear 来自 DDPM (Ho et al. 2020)；Cosine 来自 *Improved Denoising Diffusion Probabilistic Models*（Nichol & Dhariwal, 2021）。

---

## 5. 三源相关的"前置知识"汇总（复习时需先掌握）

扩散模型虽然只有一处讲到，但它依赖的知识点散落在三门课里。以下是**在工作区中确实能找到**的前置内容：

| 前置知识点 | 出处 | 与扩散模型的关系 |
|---|---|---|
| **U-Net** | **CV L10 p21-22**（完整） | 扩散模型的噪声预测网络 |
| **跳连接 / skip connections** | CV L10（U-Net）、CV L7 p60（ResNet 残差） | U-Net 的核心 |
| **VAE 与重参数化** | **CV L11 p20-23**（完整，含 reparameterization 原文） | 扩散模型同属"近似密度 + 隐变量"路线；重参数化技巧同源 |
| **GAN** | **CV L11 p24-28**（完整）；**cs231n lecture_11**（详尽） | 生成模型的主要对手；课件原话 VAE "samples blurrier... compared to GANs" |
| **去噪自编码器 DAE** | **ML L10 p19-24** | 思想上最接近扩散模型的"加噪-去噪"训练范式 |
| **自编码器 AE** | ML L10（完整）；cs231n lecture_11 | VAE 的基础 |
| **马尔可夫链 / MDP** | ML L11（Markov property） | 扩散的前向/反向都是马尔可夫链 |
| **ELBO / 变分推断** | cs231n lecture_11（VAE 部分） | DDPM 的原始推导从变分下界出发 |
| **正弦位置编码** | cs231n `cs231n_lszy.pdf` 第 7.4.1 节 | 扩散用它编码时间步 t |
| **Classifier-Free Guidance** | cs231n `cs231n_lszy.pdf` 第 10.2 节 | 条件生成的关键技术 |
| **对比学习 / 自监督** | CV L11 p51-56；cs231n lecture_11 (SimCLR) | 同属"从数据自身学表示"的大范式 |

---

## 6. 考点研判

### 6.1 结论：扩散模型在本工作区**基本不是考点**

理由：
1. **ML 课程（DSAI4203）完全没提**，历年 4 份试卷（2023/2024×2/2025）**零涉及**。ML 课的生成模型止步于 autoencoder / denoising AE。
2. **CV 课程（COMP 4423）把它放在 Reading List**，与 Multimodal learning、Pretraining/large models 并列，属于"拓展阅读"。L1 里只是当 demo 展示。
3. **cs231n 官方作业与课件（2017–2020）**根本没有扩散模型；唯一来源 `cs231n_lszy.pdf` 是**第三方笔记**，不代表任何一门课的授课范围。

### 6.2 但仍值得准备的理由

- CV 的 L11 Reading List 是**官方点名**的，三个词（Multimodal / Pretraining / **Diffusion**）是"课程拓展"类题目的唯一线索。
- 若考"生成模型对比"类型的题（**VAE vs GAN vs Diffusion** 是标准送分对比题），答不上扩散模型会很吃亏。CV L11 已经讲了 VAE 和 GAN，出对比题的概率不低。

### 6.3 如果考，最可能的三种题型

**题型一：概念对比（概率最高）**
> "比较 VAE、GAN、Diffusion 三种生成模型的原理与优缺点。"

要点速答：
- **VAE**：显式建模近似密度，encoder 输出均值+方差，重参数化采样；缺点是**生成样本偏模糊**（课件原话）。
- **GAN**：**不建模显式密度**，生成器与判别器对抗训练；生成质量高但**训练不稳定**（模式崩塌）。
- **Diffusion**：前向加噪 + 反向去噪两条**马尔可夫链**；训练目标简单（回归噪声）；生成质量高、训练稳定，但**采样慢**（需迭代 T 步）。

**题型二：原理简答**
> "简述 DDPM 的前向过程与反向过程，并写出训练目标。"

要点：前向

$$
x_t = \sqrt{\bar{\alpha}_t}x_{0} + \sqrt{1-\bar{\alpha}_t}\varepsilon
$$

（一步到位）；反向学习 $p_\theta (x_{t-1}\mid x_t)$；训练目标

$$
\mathcal L = \mathbb E\!\left[\left\|\varepsilon-\varepsilon_\theta(x_t,t)\right\|^2\right]
$$

—— 即预测噪声。

**题型三：关联题（跨课程）**
> "扩散模型的 U-Net 与本课程 L10 讲的 U-Net 有何关系？"

要点：**同一个架构**。L10 的 U-Net 是为医学图像分割设计的（Ronneberger et al. 2015），其"下采样 + 上采样 + 跳连接"结构被扩散模型直接借用为噪声预测网络，区别是扩散的 U-Net 还需注入**时间步嵌入**（正弦位置编码 + FiLM 调制）。

---

## 7. 一页速查表

```
                    ┌─────────────────────────────────────────┐
                    │  Diffusion Model 在工作区的分布          │
                    └─────────────────────────────────────────┘

  DSAI4203 ML (19 份 PDF) ────────────►  0 处  ✗ 完全没有
       └─ 最接近：L10 denoising autoencoder（思想相关）

  COMP 4423 CV (11 讲) ───────────────►  2 处
       ├─ L1  p28-30  Stable Diffusion demo 网址 + 4 段 prompt   ☆ 仅名称
       ├─ L1  p37     AAAI 2024 自己的论文引用                    ☆ 仅名称
       └─ L11 p76     Reading List: "• Diffusion models"          ☆ 仅一个词
                      （与 Multimodal learning、Pretraining 并列）

  cs231n (24 份 slides + 作业) ───────►  1 处，但是唯一的实质来源
       ├─ 官方课件 2017-2020：0 处（早于 DDPM 发布，正常）
       │    └─ lecture_11 生成模型课：VAE + GAN（扩散的前置背景）
       ├─ assignment3.md（2020）：0 处（5 题含 GAN，无扩散）
       └─ cs231n_lszy.pdf 第 10 节：DDPM + U-Net + 噪声调度   ★ 唯一技术内容
            （第三方中文笔记，仅 2 页，无推导）

                    ┌─────────────────────────────────────────┐
                    │  结论：不是考点，但 Reading List 点名了   │
                    │  按"可能考对比题"准备即可                 │
                    └─────────────────────────────────────────┘
```

---

## 8. 文档索引

| 文件 | 关系 |
|---|---|
| `4423 CV\cs231n\slides\cs231n_lszy.pdf` | **唯一实质来源**（第 10 节，p26-27） |
| `4423 CV\L11-Lecture-Learning,paradigms-v4.5.pdf` | p76 Reading List |
| `4423 CV\L1-Lecture-Introduction-v5.1.pdf` | p28-30、p37 Stable Diffusion 展示 |
| `4423 CV\cs231n\slides\lecture_11.pdf` | 生成模型课（VAE/GAN，扩散的前置背景） |
| `4423 CV\cs231n\assignments\assignment3.md` | 官方作业说明（GAN，无扩散） |
| `_text_cs231n\cs231n_lszy.txt` | 上述笔记的纯文本提取（第 808-855 行为第 10 节） |
| `CV讲义精读.md` | 本工作区的 CV 讲义总览（L11 Reading List 已记录） |
| `讲义索引与考点对照.md` | ML 讲义与考点对照 |

**核验建议**：若需确认 `cs231n_lszy.pdf` 的权威性，可直接打开原文对照第 26-27 页；该笔记自称完成了三个 Assignment，但**与官方 2020 版 assignment3 的题目清单不符**（官方无 DDPM），推测对应的是更新版本的作业或作者自行扩展内容。**在考试情境下，应以 CV 课程 L11 的 Reading List 为准，而非这份笔记的详略程度。**
