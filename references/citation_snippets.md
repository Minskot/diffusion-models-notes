# 引用原句存档（未收录材料的可查证摘录）

本文件用于记录那些**被引用但未上传原件**的材料中，与扩散模型相关的原文摘录。

这样做的原因：原文件体积大（合计约 21 MB）且内容量极小，不适合放入仓库；
但引用仍需**可查证**，故在此保留精确页码与原句。摘录仅为引用目的，版权归原作者。

---

## 1. COMP 4423 · `L11-Lecture-Learning,paradigms-v4.5.pdf`

- **页数**：77 页 ｜ **体积**：4.95 MB ｜ **作者元数据**：`ypei` ｜ **创建日期**：2026-03-30
- **相关位置**：第 **76** 页（倒数第 2 页）

**第 76 页原文（Reading List）**：

```
Reading List
• Multimodal learning
• Pretraining and large models
• Diffusion models
```

> 说明：全篇 77 页中，扩散模型**仅在此 Reading List 出现一次**（作为一个待阅读主题词），
> 正文没有展开讲解。因此本仓库只做引用记录，未收录原件。

---

## 2. COMP 4423 · `L1-Lecture-Introduction-v5.1.pdf`

- **页数**：108 页 ｜ **体积**：15.8 MB ｜ **作者元数据**：`ypei` ｜ **创建日期**：2026-01-13
- **相关位置**：第 **28–30** 页、第 **37** 页

**第 28 / 29 页原文**（Stable Diffusion 在线演示）：

```
https://stablediffusionweb.com/

interior design, frank lloyd wright house cave with forest canopy,
dark wood, streaks of light, light fog, living room :: bubbletech
– test –ar 9:16
```

**第 37 页原文**（作者自己的论文引用）：

```
Compositional Inversion for Stable Diffusion Models,
The 38th Annual AAAI Conference on Artificial Intelligence (AAAI), 2024
```

> 说明：这两处只是**演示截图与论文引用**，没有技术讲解。15.8 MB 的体积
> 与内容价值不成比例，故未收录。

---

## 3. Stanford CS231n · `lecture_14.pdf`（2020 版）

- **页数**：47 页 ｜ **体积**：5.87 MB
- **首页署名**：`Timnit Gebru`、`Emily Denton`

> ⚠️ **易错点**：这一讲的标题虽然也是 "lecture_14"，但内容是
> **AI 的公平性与偏见（fairness / bias）**，与扩散模型**无关**。
>
> 扩散模型只在 **2026 版 Lecture 14** 中讲授（已收录为
> [`lecture_14_Generative_Models_2_Diffusion.pdf`](lecture_14_Generative_Models_2_Diffusion.pdf)）。
> 2020 版课程里，与扩散沾边的只有 Lecture 11（生成模型总论，已收录）。

---

## 4. 统计：为什么可以放心不放原件

对 CV 资料包做全文检索（关键词 `diffusion | DDPM | denois | score match |
latent diffusion | stable diffusion`），命中分布如下：

| 文件 | 命中次数 | 处理方式 |
|---|---|---|
| `cs231n_lszy.txt`（中文笔记） | 8 | ✅ 已收录 §10 节选（2 页） |
| `L1-Lecture-Introduction-v5.1.txt` | 3 | 📄 本文件摘录 |
| `L11-Lecture-Learning,paradigms-v4.5.txt` | 1 | 📄 本文件摘录 |
| 其余 21 份 cs231n 讲义 | **0** | 无扩散内容 |
| 其余 10 份 CV 讲义 | **0** | 无扩散内容 |

> 结论：整个 CV 资料包中真正包含**实质**扩散内容的只有 `cs231n_lszy.pdf` 的 §10，
> 该部分已按使用者要求以**节选**形式收录。其余均为零散提及，摘录即已足够。

---

*摘录仅用于学术引用与溯源；如权利人要求，可立即删除。*
