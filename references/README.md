# 参考课件与引用来源

本目录收录若干**第三方教学材料**，版权归原作者/机构所有。
此处仅作个人学习参考，不代表本仓库作者拥有其版权。

## 引用一览

| 文件 | 标题（页数） | 作者 / 机构 | 日期 | 来源 |
|---|---|---|---|---|
| `lecture_14_Generative_Models_2_Diffusion.pdf` | CS231n Lecture 14: Generative Models (Part 2) — Diffusion（122） | Li Fei-Fei, Ehsan Adeli, Zane Durante, Justin Johnson, Serena Xiang（Stanford） | 2026-05-19 | [cs231n.stanford.edu](https://cs231n.stanford.edu/) |
| `lecture_13_Generative_Models_1_VAE_GAN.pdf` | CS231n Lecture 13: Generative Models (Part 1) — VAE / GAN（116） | 同上 | 2026-05-14 | [cs231n.stanford.edu](https://cs231n.stanford.edu/) |
| `lecture_11.pdf` | CS231n Lecture 11: Generative Models（130，2020 版） | Li Fei-Fei, Ranjay Krishna, Danfei Xu（Stanford） | 2020-05-14 | [cs231n.stanford.edu](https://cs231n.stanford.edu/) |
| `cs231n_lszy_sec10_diffusion_excerpt.pdf` | CS231n：从线性分类器到扩散模型 — **§10 节选，仅 2 页** | Lin | 2026-02-01 | 节选，**全文未分发**（见下） |
| `DDPM_tutorial_by_ZhangXin.pdf` | 【大白话01】一文理清 Diffusion Model 扩散模型（18） | 张鑫（bilibili UP 主「吃花椒的麦」） | 2024-07-01 | [BV1xih7ecEMb](https://www.bilibili.com/video/BV1xih7ecEMb) |
| `大连理工大学-DDPM.pdf` | Diffusion model 学习笔记（DDPM 原理，10） | 小红书用户 **😈**（小红书号 740869460，IP 辽宁） | 待补充 | [小红书笔记](https://www.xiaohongshu.com/discovery/item/68d69ee1000000001400b0e7) |

> 📌 **「大工」一份的出处**为小红书笔记（页面需登录，正文未能核实，依据使用者说明记录）。
> **`cs231n_lszy` 只上传了节选**：按使用者要求，不上传全文，仅保留与扩散模型相关的 §10
> （原文档共 27 页，此处 2 页）。若发现任何标注有误，欢迎提 issue 指正。

---

## 详细说明

### 1. Stanford CS231n — Lecture 14（122 页）· 主讲义

- **课程**：CS231n: Deep Learning for Computer Vision, Spring 2026
- **讲次**：Lecture 14 — Generative Models (Part 2)，聚焦扩散模型
- **署名**：首页标注 *"Fei-Fei, Adeli, Durante, Johnson, Xiang"*，日期 *May 19, 2026*
- **获取**：课程官网 <https://cs231n.stanford.edu/> 的 Slides 页面
- **引用格式**：
  > Fei-Fei, L., Adeli, E., Durante, Z., Johnson, J., & Xiang, S. (2026).
  > *Lecture 14: Generative Models (Part 2) — Diffusion* [Lecture slides].
  > CS231n, Stanford University.

### 2. Stanford CS231n — Lecture 13（116 页）· 前置讲义

- **讲次**：Lecture 13 — Generative Models (Part 1)，覆盖 **VAE / GAN**
- **日期**：2026-05-14（首页 *"May 14, 2026 Lecture 13"*）
- **作用**：理解扩散模型的前置背景（ELBO、变分推断、生成模型对比）
- **引用格式**：
  > Fei-Fei, L., Adeli, E., Durante, Z., Johnson, J., & Xiang, S. (2026).
  > *Lecture 13: Generative Models (Part 1)* [Lecture slides]. CS231n, Stanford University.

### 3. Stanford CS231n — Lecture 11（130 页）· 2020 版生成模型

- **讲次**：Lecture 11 — Generative Models（2020 年版），VAE / GAN 的经典讲法
- **署名**：首页 *"Fei-Fei Li, Ranjay Krishna, Danfei Xu, Lecture 11 - May 14, 2020"*
- **作用**：与 2026 版 L13 互为参照；早期版本在 ELBO 推导上更展开
- **引用格式**：
  > Li, F.-F., Krishna, R., & Xu, D. (2020). *Lecture 11: Generative Models*
  > [Lecture slides]. CS231n, Stanford University.

### 4. 《CS231n：从线性分类器到扩散模型》— §10 节选（2 页）

- ⚠️ **这是节选，不是全文**：原文档共 **27 页**，此处只保留 **§10「扩散模型：从噪声中生成图像」**
  （原第 26–27 页），因为本仓库的主题是扩散模型，其余章节（线性分类器、CNN、优化器等）
  与本仓库无关，**未分发**。
- **原文档**：《CS231n：从线性分类器到扩散模型》，作者 **Lin**，2026-02-01
- **性质**：第三方中文学习笔记，记录作者手写完成 CS231n 三个 Assignment 的过程
- **节选内容**：DDPM 理论框架、前向/反向过程、时间嵌入（正弦位置编码 + FiLM）、
  Classifier-Free Guidance、三种噪声调度对比
- **引用格式**：
  > Lin. (2026). *CS231n：从线性分类器到扩散模型*（§10 扩散模型部分）[笔记].

### 5. 张鑫 — Diffusion Model 教程（18 页）

- **视频**：*【大白话01】一文理清 Diffusion Model 扩散模型 | 原理图解+公式推导*
- **UP 主**：吃花椒的麦（UID 527467660）
- **链接**：<https://www.bilibili.com/video/BV1xih7ecEMb>
- **时长 / 投稿**：33:28 / 2024-07-01
- **PDF 元数据**：Author `张鑫`、Title `Diffusion Model`、创建于 `2024-07-18`
- ⚠️ **作者声明**：视频简介写明「视频中的 PPT 都是自己原创完成的」。
  **因此本仓库的转载未获授权，仅供个人学习；权利人若要求移除，我会立即删除。**
- **引用格式**：
  > 张鑫 (吃花椒的麦). (2024, 7月1日). *【大白话01】一文理清 Diffusion Model
  > 扩散模型 | 原理图解+公式推导* [视频]. bilibili.
  > https://www.bilibili.com/video/BV1xih7ecEMb

### 6. Diffusion model 学习笔记 — DDPM 原理（10 页）

- **来源**：小红书笔记《Diffusion model 学习笔记》
  <https://www.xiaohongshu.com/discovery/item/68d69ee1000000001400b0e7>
- **作者**：小红书用户 **😈**（小红书号 **740869460**，IP 属地 **辽宁**）
- ⚠️ **核实状态**：笔记正文需登录才能访问，**未能独立核实**；
  作者身份据其主页截图记录，出处依据使用者说明。
- **特点**：纯图片页，无可提取文本，PDF 元数据为空
- **旁证**：作者 IP 属地为**辽宁**，与"大连理工大学"（辽宁大连）这一课程来源相吻合
- **说明**：本仓库作者的[推导笔记](../docs/扩散模型课件小结_大工与ZhangXin.md)
  **修正了该讲义 §3.4 中的一处符号错误**（`+` 应为 `−`，已数值验证）
- **引用格式**：
  > 😈. 《Diffusion model 学习笔记》[图文]. 小红书.
  > https://www.xiaohongshu.com/discovery/item/68d69ee1000000001400b0e7

---

## 提到了但未收录的材料

以下材料在笔记中被引用，但因内容量极小或体积过大**未收录**，仅作引用记录：

| 材料 | 引用位置 | 未收录原因 |
|---|---|---|
| `L11-Lecture-Learning,paradigms-v4.5.pdf`（COMP 4423，77 页，作者元数据 `ypei`） | p76 Reading List 仅一行提及 diffusion | 全篇仅 1 处一个词，4.95 MB 不划算 |
| `L1-Lecture-Introduction-v5.1.pdf`（COMP 4423，108 页，作者元数据 `ypei`） | p28–30 两个 Stable Diffusion demo 网址；p37 引用一篇 AAAI 2024 论文 | 15.8 MB，仅 demo 截图 |
| `cs231n/slides/lecture_14.pdf`（2020 版，47 页） | — | **与扩散无关**：该讲为 AI 公平性/偏见（Timnit Gebru、Emily Denton） |

> 📄 以上三份材料的**相关原文摘录**已存档于
> [`citation_snippets.md`](citation_snippets.md) —— 引用可查证，但无需分发原件。

> 🔍 **一个容易踩的坑**：2020 版的 `lecture_14.pdf` **不是**扩散模型那一讲 ——
> 2026 版才有 L14（扩散）。旧版只有 L11（生成模型总论）。引用时别把两者搞混。

---

## 我对这些材料的二次整理

与原件不同，下面这份是我自己消化后重写的推导笔记（原创整理，含勘误）：

→ [`../docs/扩散模型课件小结_大工与ZhangXin.md`](../docs/扩散模型课件小结_大工与ZhangXin.md)

---

## 版权与移除

- 上述材料的**版权归原作者及所属机构所有**，本仓库不主张任何权利。
- 用途限于学习、研究，不作商业使用。
- 如果你是权利人且不希望此处分发，请提 issue 或通过
  [GitHub 主页](https://github.com/Minskot) 联系，我会**立即删除**。

## 建议优先阅读的公开资料

- **原始论文**：Ho, J., Jain, A., & Abbeel, P. (2020).
  *Denoising Diffusion Probabilistic Models*. NeurIPS 2020.
  [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)
- **综述**：Yang, L., et al. (2023). *Diffusion Models: A Comprehensive Survey of Methods and Applications*.
  [arXiv:2209.00796](https://arxiv.org/abs/2209.00796)
- **博客**：Sander Dieleman, *Perspectives on Diffusion*.
  [sander.ai](https://sander.ai/2023/07/20/perspectives.html)
