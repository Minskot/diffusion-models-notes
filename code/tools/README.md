# 文档处理工具

整理本仓库时写的小工具，**通用可复用**，不依赖本仓库的其它代码。

依赖：`pip install pymupdf`（三个工具中前两个需要；`repo_linkcheck.py` 只用标准库）

| 工具 | 用途 |
|---|---|
| [`pdf_excerpt.py`](pdf_excerpt.py) | 从 PDF 提取**页范围或章节节选** —— 引用第三方材料但不宜分发全文时用 |
| [`pdf_recon.py`](pdf_recon.py) | 盘点 PDF **元数据与关键词命中** —— 整理引用信息、判断哪份文件真讲了某主题 |
| [`repo_linkcheck.py`](repo_linkcheck.py) | 校验 Markdown 的**相对链接与图片引用**是否可解析（可作 CI 检查） |

---

## 典型用法

### 1. 只截取某章节（本仓库的 `references/` 就是这么来的）

```bash
# 按章节锚点：找到「页首以该文本开头」的那一页，截到文末
python pdf_excerpt.py 原文.pdf 节选.pdf --anchor "10 扩散模型" --to-end \
    --title "原文标题 §10 节选" --author "作者名"

# 按页码（1-based，含首含尾）
python pdf_excerpt.py 原文.pdf 节选.pdf --start 26 --end 27

# 先预览会截哪几页，不写盘
python pdf_excerpt.py 原文.pdf out.pdf --start 26 --end 27 --dry-run
```

> 💡 **为什么用「页首匹配」而不是「全文包含」**：
> 目录页里也会出现章节标题字样，用「包含」会误把目录页当成起点
> （我在整理 `cs231n_lszy.pdf` 时就踩过这个坑，误抽出 25 页）。

### 2. 快速盘点一批课件的元数据

```bash
# 列出所有 PDF 的作者/日期/首页文本
python pdf_recon.py "D:/course/slides"

# 只关心哪些文件真讲了 diffusion，并给出命中页码
python pdf_recon.py "D:/course/slides" -k diffusion -k DDPM -k denoising --only-hits
```

元数据里的 `author` / `creationDate` 字段往往能直接填进引用表，
省去手工翻首页的功夫。

### 3. 发布前自检链接

```bash
python repo_linkcheck.py --root ..
# RESULT: PASS   -> 退出码 0
#                 -> 退出码 1（存在失效链接，可接入 CI）
```

会跳过 `http(s)://`、`mailto:`、锚点 `#...` 等外部或页内链接，
只校验指向仓库内文件的相对路径。

---

## 备注

这些工具是在整理本仓库过程中逐步写出来的，**没有做完整的单元测试**，
但都在真实数据上跑通过（包括本仓库 `references/` 下 6 份 PDF）。
如果用在别的仓库，注意 `pdf_recon.py` 默认只扫描每个 PDF 的前 300 页。
