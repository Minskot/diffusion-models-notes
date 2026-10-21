#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""校验仓库内所有 Markdown 的相对链接与图片引用是否可解析。

用于发布前自检：任何失效的相对链接（例如重命名/移动文件后）
都会被列出来，避免读者点到 404。

用法：
    python repo_linkcheck.py                 # 默认当前目录
    python repo_linkcheck.py --root ../      # 指定仓库根

退出码：全部有效为 0，存在失效为 1（可直接用于 CI）。
"""
import argparse
import os
import re
import sys
import urllib.parse

SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv"}

# Markdown 行内链接 / 图片: [text](target)  与  ![alt](target)
MD_LINK = re.compile(r"\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
HTML_ATTR = re.compile(r'(?:src|href)="([^"]+)"')


def is_external(target):
    return target.startswith(("http://", "https://", "mailto:", "tel:", "#", "data:"))


def check(root):
    missing_link = []
    missing_img = []
    n_link = n_img = 0

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            if not fn.lower().endswith(".md"):
                continue
            path = os.path.join(dirpath, fn)
            try:
                text = open(path, encoding="utf-8").read()
            except (UnicodeDecodeError, OSError):
                continue

            for m in MD_LINK.finditer(text):
                target = m.group(1).strip()
                if is_external(target):
                    continue
                clean = urllib.parse.unquote(target.split("#")[0].split("?")[0])
                if not clean:
                    continue
                n_link += 1
                resolved = os.path.normpath(os.path.join(dirpath, clean))
                if not os.path.exists(resolved):
                    missing_link.append((os.path.relpath(path, root), target))

            for m in HTML_ATTR.finditer(text):
                target = m.group(1).strip()
                if is_external(target):
                    continue
                clean = urllib.parse.unquote(target.split("#")[0].split("?")[0])
                if not clean:
                    continue
                n_img += 1
                resolved = os.path.normpath(os.path.join(dirpath, clean))
                if not os.path.exists(resolved):
                    missing_img.append((os.path.relpath(path, root), target))

    return n_link, missing_link, n_img, missing_img


def main():
    ap = argparse.ArgumentParser(description="校验 Markdown 相对链接")
    ap.add_argument("--root", default=".", help="仓库根目录（默认当前目录）")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        sys.exit(f"不是目录: {root}")

    n_link, bad_link, n_img, bad_img = check(root)

    print(f"扫描根目录: {root}")
    print(f"相对链接 {n_link} 个，失效 {len(bad_link)} 个")
    for f, t in bad_link:
        print(f"  [FAIL] {f}  ->  {t}")
    print(f"图片/资源引用 {n_img} 个，失效 {len(bad_img)} 个")
    for f, t in bad_img:
        print(f"  [FAIL] {f}  ->  {t}")

    ok = not bad_link and not bad_img
    print("RESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
