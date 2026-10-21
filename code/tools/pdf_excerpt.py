#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 PDF 中提取指定页范围或某个章节的节选。

用途：需要引用第三方材料但不宜分发全文时，只保留相关章节。

用法：
    # 按页码（1-based，含首含尾）
    python pdf_excerpt.py src.pdf out.pdf --start 26 --end 27

    # 按章节锚点：找到「以该文本开头」的页作为起点，默认取到文末
    python pdf_excerpt.py src.pdf out.pdf --anchor "10 扩散模型" --to-end

    # 指定结束页
    python pdf_excerpt.py src.pdf out.pdf --anchor "10 扩散模型" --end 27

    # 加元数据（便于溯源）
    python pdf_excerpt.py src.pdf out.pdf --start 26 --end 27 \
        --title "原文标题 §10 节选" --author "作者名"

依赖：pip install pymupdf
"""
import argparse
import os
import sys

try:
    import pymupdf
except ImportError:  # 兼容旧包名
    import fitz as pymupdf


def find_anchor(doc, anchor):
    """返回第一个「正文以 anchor 开头」的页索引（0-based），找不到返回 None。

    只匹配页首可以避免命中目录页 —— 目录页里也会有章节标题字样。
    """
    for i in range(doc.page_count):
        text = " ".join(doc[i].get_text().split())
        if text.startswith(anchor):
            return i
    # 退化：退回到「包含」匹配，但要求不在目录页（前 20 字符不含「目录」）
    for i in range(doc.page_count):
        text = " ".join(doc[i].get_text().split())
        if anchor in text and "目录" not in text[:20]:
            return i
    return None


def main():
    ap = argparse.ArgumentParser(description="提取 PDF 页范围/章节节选")
    ap.add_argument("src", help="源 PDF")
    ap.add_argument("out", help="输出 PDF")
    ap.add_argument("--start", type=int, help="起始页（1-based）")
    ap.add_argument("--end", type=int, help="结束页（1-based，含）")
    ap.add_argument("--anchor", help="章节锚点文本（匹配页首）")
    ap.add_argument("--to-end", action="store_true", help="从起点取到文末")
    ap.add_argument("--title")
    ap.add_argument("--author")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(args.src):
        sys.exit(f"源文件不存在: {args.src}")

    doc = pymupdf.open(args.src)
    print(f"源文件: {doc.page_count} 页, {os.path.getsize(args.src) / 1e6:.2f} MB")

    if args.anchor:
        idx = find_anchor(doc, args.anchor)
        if idx is None:
            sys.exit(f"未找到锚点: {args.anchor!r}")
        start = idx
        end = doc.page_count - 1 if (args.to_end or not args.end) else args.end - 1
    else:
        if not args.start:
            sys.exit("需要 --start 或 --anchor")
        start = args.start - 1
        end = (args.end - 1) if args.end else doc.page_count - 1

    if start < 0 or end >= doc.page_count or start > end:
        sys.exit(f"页范围非法: {start + 1}..{end + 1}（共 {doc.page_count} 页）")

    count = end - start + 1
    print(f"提取 p{start + 1}..p{end + 1}（共 {count} 页，占原文 {count / doc.page_count:.0%}）")

    for i in range(start, end + 1):
        head = " ".join(doc[i].get_text().split())[:90]
        print(f"  p{i + 1}: {head or '(图片页/无文字)'}")

    if args.dry_run:
        print("(dry-run，未写盘)")
        return

    out = pymupdf.open()
    out.insert_pdf(doc, from_page=start, to_page=end)
    meta = {k: v for k, v in (out.metadata or {}).items() if v}
    if args.title:
        meta["title"] = args.title
    if args.author:
        meta["author"] = args.author
    meta["subject"] = f"Excerpt p{start + 1}-p{end + 1} of {os.path.basename(args.src)}"
    out.set_metadata(meta)
    out.save(args.out, garbage=4, deflate=True)
    out.close()

    print(f"已保存: {args.out}  ({os.path.getsize(args.out) / 1e6:.2f} MB)")
    doc.close()


if __name__ == "__main__":
    main()
