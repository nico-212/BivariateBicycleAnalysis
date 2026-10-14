#!/usr/bin/env python3
"""Extract the data diamonds of Fig. 2A of Bravyi et al. (Nature 627, 2024) from the
vector graphics of the arXiv PDF, calibrated against the axis tick lines.

Usage: python diamonds.py BB_paper.pdf [page_index=5]

Assumptions (verify if you reuse this on another PDF):
  * Figure 2A is the left panel on the given 0-based page.
  * Y ticks are 1e-1 ... 1e-13 (log axis), x ticks 1e-4, 1e-3, 1e-2 (log axis).
  * Data diamonds are small black closed paths (3.02 x 5.03 pt).
  * At each p, the codes appear in order of decreasing pL: [[72]], [[90]], [[108]],
    [[144]], [[288]] (legend order); BB72 = largest, BB144 = 4th largest.
    The values at p = 0.001 reproduce Table 1 of the paper (7e-5, 2e-7).
"""
import json
import sys
from collections import defaultdict

import numpy as np
import pdfplumber

pdf_path = sys.argv[1] if len(sys.argv) > 1 else "BB_paper.pdf"
page_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 5
pg = pdfplumber.open(pdf_path).pages[page_idx]

# --- axis calibration from the tick lines of the left panel (page coordinates in pt)
yticks = {1e-1: 138.157, 1e-3: 159.603, 1e-5: 181.05, 1e-7: 202.496,
          1e-9: 223.943, 1e-11: 245.389, 1e-13: 266.835}
xticks = {1e-4: 115.015, 1e-3: 192.516, 1e-2: 270.017}
ay, by = np.polyfit(list(yticks.values()), np.log10(list(yticks.keys())), 1)
ax, bx = np.polyfit(list(xticks.values()), np.log10(list(xticks.keys())), 1)
print(f"y: {-1/ay:.3f} pt/decade, x: {1/ax:.3f} pt/decade")

# --- diamonds: small black filled closed paths inside the left panel
groups = defaultdict(list)
for o in pg.curves:
    w, h = o["x1"] - o["x0"], o["bottom"] - o["top"]
    if (105 < o["x0"] and o["x1"] < 275 and 135 < o["top"] and o["bottom"] < 270
            and abs(w - 3.02) < 0.05 and abs(h - 5.03) < 0.05
            and o.get("non_stroking_color") == 0):
        cx, cy = (o["x0"] + o["x1"]) / 2, (o["top"] + o["bottom"]) / 2
        groups[round(cx, 1)].append((10 ** (ax * cx + bx), 10 ** (ay * cy + by)))

out = {"72": {}, "144": {}}
for cx in sorted(groups):
    vals = sorted(groups[cx], key=lambda t: -t[1])
    p = float("%.2g" % vals[0][0])
    if p not in (0.001, 0.002, 0.003, 0.004, 0.005, 0.006):
        continue
    out["72"][p] = vals[0][1]
    if len(vals) >= 4:
        out["144"][p] = vals[3][1]

for code in ("72", "144"):
    for p, v in sorted(out[code].items()):
        print(f"BB{code}  p={p:<6} pL={v:.4e}")
json.dump(out, open("diamonds_exact.json", "w"), indent=1)
