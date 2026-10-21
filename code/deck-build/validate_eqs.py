"""校验 latex_spec.DISPLAY 中每条公式能否被 matplotlib mathtext 解析。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from latex_spec import DISPLAY

plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["mathtext.default"] = "regular"

bad = 0
for key, spec in DISPLAY.items():
    for i, ln in enumerate(spec.split("\n")):
        fig = plt.figure(figsize=(0.01, 0.01))
        try:
            fig.text(0, 0, ln, fontsize=20)
            fig.canvas.draw()
        except Exception as e:
            bad += 1
            msg = str(e).strip().split("\n")[-1][:120]
            print(f"[FAIL] {key} (line {i}): {msg}")
        finally:
            plt.close(fig)

print(f"\n检查 {len(DISPLAY)} 条公式，失败 {bad} 条")
