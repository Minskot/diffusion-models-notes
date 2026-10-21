#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""盘点一批 PDF 的元数据、页数，并检索关键词命中，用于整理引用信息。

典型用途：手上一堆课程课件，需要快速判断
  1) 哪些文件真的包含某个主题（如 diffusion）
  2) 每个文件的作者/日期元数据（写引用要用）

用法：
    # 扫描整个目录（递归），列出全部 PDF 及其元数据
    python pdf_recon.py "D:/course/slides"

    # 只看含关键词的文件，并输出命中页
    python pdf_recon.py "D:/course/slides" --keyword diffusion --keyword DDPM

    # 只看部分文件
    python pdf_recon.py a.pdf b.pdf --keyword diffusion

依赖：pip install pymupdf
"""
import argparse
import os
import sys

try:
    import pymupdf
except ImportError:
    import fitz as pymupdf

MAX_PAGES_TO_SCAN = 300  # 防止超长文档拖慢


def iter_pdfs(paths):
    for p in paths:
        if os.path.isdir(p):
            for root, _, files in os.walk(p):
                for f in sorted(files):
                    if f.lower().endswith(".pdf"):
                        yield os.path.join(root, f)
        elif p.lower().endswith(".pdf") and os.path.exists(p):
            yield p


def scan(path, keywords):
    """返回 (元数据字符串, 关键词命中 {kw: [页码]})"""
    doc = pymupdf.open(path)
    md = doc.metadata or {}
    hits = {k.lower(): [] for k in keywords}
    n = min(doc.page_count, MAX_PAGES_TO_SCAN)
    if keywords:
        for i in range(n):
            text = doc[i].get_text()
            low = text.lower()
            for k in hits:
                if k in low:
                    hits[k].append(i + 1)
    first = " ".join(doc[0].get_text().split())[:150] if doc.page_count else ""
    doc.close()
    return md, hits, first, n


def main():
    ap = argparse.ArgumentParser(description="盘点 PDF 元数据与关键词命中")
    ap.add_argument("paths", nargs="+", help="PDF 文件或目录")
    ap.add_argument("--keyword", "-k", action="append", default=[],
                    help="关键词（可多次指定）")
    ap.add_argument("--only-hits", action="store_true",
                    help="只显示有命中的文件")
    args = ap.parse_args()

    files = list(iter_pdfs(args.paths))
    if not files:
        sys.exit("没有找到 PDF")

    n_hit = 0
    for p in files:
        try:
            md, hits, first, scanned = scan(p, args.keyword)
        except Exception as e:
            print(f"[ERROR] {p}: {type(e).__name__}: {e}")
            continue

        total_hits = sum(len(v) for v in hits.values())
        if args.only_hits and total_hits == 0:
            continue
        if total_hits:
            n_hit += 1

        print("=" * 78)
        print(os.path.basename(p))
        print(f"  {os.path.getsize(p) / 1e6:.2f} MB")
        for k in ("title", "author", "creator", "creationDate"):
            v = (md.get(k) or "").strip()
            if v:
                print(f"  {k}: {v[:80]}")
        if first:
            print(f"  首页: {first}")
        if args.keyword:
            for kw, pages in hits.items():
                if pages:
                    shown = pages[:12]
                    more = f" ...(+{len(pages) - 12})" if len(pages) > 12 else ""
                    print(f"  [{kw}] 命中 {len(pages)} 页: {shown}{more}")
                else:
                    print(f"  [{kw}] 0 命中")

    print("=" * 78)
    print(f"共 {len(files)} 个 PDF；含关键词命中的 {n_hit} 个"
          + (f"（已扫描前 {MAX_PAGES_TO_SCAN} 页）" if args.keyword else ""))


if __name__ == "__main__":
    main()
