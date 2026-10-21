# 参考课件与引用来源

本目录收录三份**第三方教学材料**，版权归原作者/机构所有。
此处仅作个人学习参考，不代表本仓库作者拥有其版权。

## 引用一览

| 文件 | 标题 | 作者 / 机构 | 日期 | 来源 |
|---|---|---|---|---|
| `lecture_14_Generative_Models_2_Diffusion.pdf` | CS231n Lecture 14: Generative Models (Part 2) — Diffusion | Li Fei-Fei, Ehsan Adeli, Zane Durante, Justin Johnson, Serena Xiang（Stanford University） | 2026-05-19 | [cs231n.stanford.edu](https://cs231n.stanford.edu/) |
| `DDPM_tutorial_by_ZhangXin.pdf` | 【大白话01】一文理清 Diffusion Model 扩散模型 \| 原理图解+公式推导 | 张鑫（bilibili UP 主「吃花椒的麦」） | 2024-07-01 | [BV1xih7ecEMb](https://www.bilibili.com/video/BV1xih7ecEMb) |
| `大连理工大学-DDPM.pdf` | Diffusion model 学习笔记（DDPM 原理，课程讲义） | 小红书用户（昵称含 😈，身份待补充） | 待补充 | [小红书笔记](https://www.xiaohongshu.com/discovery/item/68d69ee1000000001400b0e7) |

> 📌 **三份材料的来源均已标注**，其中「大工」一份的原始出处为小红书笔记
> （该页面需登录访问，正文未能核实，引用信息依据使用者说明记录）。
> 若发现任何标注有误，欢迎提 issue 指正。

---

## 详细说明

### 1. Stanford CS231n — Lecture 14（122 页）

- **课程**：CS231n: Deep Learning for Computer Vision, Spring 2026
- **讲次**：Lecture 14 — Generative Models (Part 2)，聚焦扩散模型
- **署名**：首页标注 *"Fei-Fei, Adeli, Durante, Johnson, Xiang"*，日期 *May 19, 2026*
- **获取**：课程官网 <https://cs231n.stanford.edu/> 的 Slides 页面
- **引用格式**：
  > Fei-Fei, L., Adeli, E., Durante, Z., Johnson, J., & Xiang, S. (2026).
  > *Lecture 14: Generative Models (Part 2) — Diffusion* [Lecture slides].
  > CS231n: Deep Learning for Computer Vision, Stanford University.

### 2. 张鑫 — Diffusion Model 教程（18 页）

- **视频**：*【大白话01】一文理清 Diffusion Model 扩散模型 | 原理图解+公式推导*
- **UP 主**：吃花椒的麦（UID 527467660）
- **链接**：<https://www.bilibili.com/video/BV1xih7ecEMb>
- **时长 / 投稿**：33:28 / 2024-07-01
- **PDF 元数据**：Author `张鑫`、Title `Diffusion Model`、创建于 `2024-07-18`
- **内容**：从"什么是扩散模型"到数学推导的视觉化讲解（视频简介中的三个问题：
  *What is the diffusion model? / How to visually understand it? / How to derive it mathematically?*）
- ⚠️ **作者声明**：视频简介写明「视频中的 PPT 都是自己原创完成的」。
  **因此本仓库的转载未获授权，仅供个人学习；权利人若要求移除，我会立即删除。**
- **引用格式**：
  > 张鑫 (吃花椒的麦). (2024, 7月1日). *【大白话01】一文理清 Diffusion Model
  > 扩散模型 | 原理图解+公式推导* [视频]. bilibili.
  > https://www.bilibili.com/video/BV1xih7ecEMb

### 3. Diffusion model 学习笔记 — DDPM 原理（10 页）

- **来源**：小红书笔记《Diffusion model 学习笔记》
  <https://www.xiaohongshu.com/discovery/item/68d69ee1000000001400b0e7>
- ⚠️ **核实状态**：该页面需登录才能访问，**正文与作者身份未能独立核实**；
  出处依据使用者说明记录，引用时请自行确认。
- **内容**：DDPM 的数学推导（后验均值"配方"路线）
- **特点**：纯图片页，无可提取文本，PDF 元数据为空
- **作者**：小红书用户（分享标题显示昵称含 😈），具体身份待补充
- **说明**：本仓库作者的[推导笔记](../docs/扩散模型课件小结_大工与ZhangXin.md)
  **修正了该讲义 §3.4 中的一处符号错误**（`+` 应为 `−`，已数值验证）
- **引用格式**：
  > 《Diffusion model 学习笔记》[图文]. 小红书.
  > https://www.xiaohongshu.com/discovery/item/68d69ee1000000001400b0e7

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
