"""Draw every picture the final deck needs, in the deck's existing style.

Reads docs/final_results.json and the WEKA logs in docs/weka_runs/final/ (made by
notebooks/final_weka_results.py). Writes PNGs to docs/figures/final/.
Image pixel sizes follow the picture frames already in the deck, so nothing is stretched.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "notebooks"))
import weka_runner as W  # noqa: E402

OUT = ROOT / "docs" / "figures" / "final"
OUT.mkdir(parents=True, exist_ok=True)
R = json.loads((ROOT / "docs" / "final_results.json").read_text(encoding="utf-8"))
LOGS = ROOT / "docs" / "weka_runs" / "final"

plt.rcParams["font.family"] = "Segoe UI"
BLUE = "#2A78D6"
GREY_BAR = "#A3A9B1"
LABEL = "#555555"
TEXT = "#1A1A1A"
BAND = {"C0": "#A83A3A", "C1": "#DC713D", "C2": "#4C91D8", "C3": "#285B9F"}
GREEN = "#1BAF7A"
ORANGE = "#DC713D"

FRIENDLY = {
    "desired_department_match": "Got desired department", "attendance": "Class attendance",
    "weekly_study_time": "Self-study per week", "study_style": "Study style",
    "topic_clarity": "Topic clear after class", "sleep_duration": "Sleep per night",
    "stress_frequency": "Stress blocked studying", "weekly_responsibilities": "Outside commitments (tuition/job/club)",
    "distraction_frequency": "Phone / social media took study time", "study_environment": "Study-friendly living place",
    "support_level": "Family / friend support", "routine_manageability": "Semester routine manageable",
}


def fig_px(w, h, dpi=100):
    return plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=fig.dpi, facecolor="white")
    plt.close(fig)
    print("wrote", name)


# --------------------------------------------------------------------------- horizontal count bars
def hbars(name, labels, counts, total, w, h, label_size=26, value_size=26, left=0.25, wrap=None, bar_h=0.62):
    fig = fig_px(w, h)
    ax = fig.add_axes([left, 0.06, 0.98 - left - 0.22, 0.9])
    n = len(labels)
    ys = list(range(n))[::-1]
    mx = max(counts) if counts else 1
    for y, lab, c in zip(ys, labels, counts):
        ax.barh(y, c, color=BLUE, height=bar_h)
        pct = 100 * c / total if total else 0
        bold = "bold" if c == mx else "normal"
        ax.text(c + mx * 0.02, y, f"{c}  ({pct:.0f}%)", va="center", ha="left", fontsize=value_size,
                fontweight=bold, color=TEXT)
    ax.set_yticks(ys)
    ax.set_yticklabels([wrap(l) if wrap else l for l in labels], fontsize=label_size, color=LABEL)
    ax.set_xlim(0, mx * 1.02)
    ax.set_ylim(-0.6, n - 0.4)
    for s in ["top", "right", "bottom"]:
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#BBBBBB")
    ax.tick_params(axis="y", length=0, pad=8)
    ax.set_xticks([])
    ax.set_clip_on(False)
    save(fig, name)


def wrap2(text, width=16):
    words, lines, cur = text.split(), [], ""
    for wd in words:
        if len(cur) + len(wd) + 1 > width and cur:
            lines.append(cur)
            cur = wd
        else:
            cur = (cur + " " + wd).strip()
    lines.append(cur)
    return "\n".join(lines)


def answers(item):
    rows = R["answers"][item]
    return [r["value"] for r in rows], [r["n"] for r in rows]


N = R["final_rows"]

# slide 13 — universities
uni = sorted(R["universities"], key=lambda r: -r["n"])
hbars("rc_university", [u["value"] for u in uni], [u["n"] for u in uni], N, 1190, 832, label_size=30, value_size=30, left=0.14)

# slide 15 — CGPA groups (vertical bars)
fig = fig_px(1192, 832)
ax = fig.add_axes([0.06, 0.15, 0.9, 0.8])
for i, b in enumerate(R["cgpa_bands"]):
    ax.bar(i, b["n"], color=BAND[b["band"]], width=0.6)
    ax.text(i, b["n"] + 6, f"{b['n']}\n{b['pct']:.1f}%", ha="center", va="bottom", fontsize=30, color=TEXT, linespacing=1.1)
ax.set_xticks(range(4))
ax.set_xticklabels(["C0\n<3.20", "C1\n3.20–3.49", "C2\n3.50–3.74", "C3\n3.75+"], fontsize=30, color=LABEL)
ax.set_ylim(0, max(b["n"] for b in R["cgpa_bands"]) * 1.35)
for s in ["top", "right", "left"]:
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color("#BBBBBB")
ax.set_yticks([])
ax.tick_params(axis="x", length=0, pad=8)
save(fig, "rc_cgpa")

# slides 16-20 — answer distributions
lab, cnt = answers("attendance")
hbars("rc_attendance", lab, cnt, N, 1197, 832, label_size=30, value_size=30, left=0.28)
lab, cnt = answers("weekly_study_time")
hbars("rc_weekly_study_time", lab, cnt, N, 1194, 832, label_size=30, value_size=30, left=0.31)
for item, w in [("stress_frequency", 1196), ("sleep_duration", 1194), ("distraction_frequency", 1189),
                ("weekly_responsibilities", 1194)]:
    lab, cnt = answers(item)
    hbars(f"rc_{item}", lab, cnt, N, w, 832, label_size=30, value_size=30, left=0.30)
for item in ["desired_department_match", "support_level", "study_environment"]:
    lab, cnt = answers(item)
    hbars(f"rc_{item}_small", lab, cnt, N, 850, 812, label_size=26, value_size=26, left=0.36,
          wrap=lambda t: wrap2(t, 15))


# --------------------------------------------------------------------------- slide 21 — association strength
def association_chart():
    a = R["association_cgpa"]
    items = sorted(a, key=lambda k: -a[k]["rho"])
    fig = fig_px(1798, 1116)
    ax = fig.add_axes([0.42, 0.1, 0.53, 0.86])
    ys = list(range(len(items)))[::-1]
    lim = max(abs(a[k]["rho"]) for k in items) * 1.45
    for y, k in zip(ys, items):
        r, rel = a[k]["rho"], a[k]["reliable"]
        ax.barh(y, r, color=BLUE if rel else GREY_BAR, height=0.5)
        txt = f"{r:+.2f}"
        ax.text(r + (lim * 0.03 if r >= 0 else -lim * 0.03), y, txt, va="center",
                ha="left" if r >= 0 else "right", fontsize=30, fontweight="bold" if rel else "normal", color=TEXT)
        ax.text(-lim * 1.02, y, FRIENDLY[k], va="center", ha="right", fontsize=30, color=LABEL)
    ax.axvline(0, color="#999999", lw=1.5)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-0.7, len(items) - 0.3)
    ax.axis("off")
    ax.text(-lim * 0.1, -1.0, "← lower CGPA group", ha="right", va="center", fontsize=28, color=LABEL)
    ax.text(lim * 0.1, -1.0, "higher CGPA group →", ha="left", va="center", fontsize=28, color=LABEL)
    save(fig, "association_strength")


association_chart()


# --------------------------------------------------------------------------- slide 22 — SGPA vs CGPA patterns
def sgpa_cgpa_chart():
    c, s = R["association_cgpa"], R["association_sgpa"]
    items = sorted(c, key=lambda k: -c[k]["rho"])
    fig = fig_px(1851, 1233)
    ax = fig.add_axes([0.45, 0.08, 0.52, 0.8])
    ys = list(range(len(items)))[::-1]
    lim = max(max(abs(c[k]["rho"]), abs(s[k]["rho"])) for k in items) * 1.25
    for y, k in zip(ys, items):
        ax.barh(y + 0.18, s[k]["rho"], color=GREEN, height=0.32)
        ax.barh(y - 0.18, c[k]["rho"], color=BLUE, height=0.32)
        both = c[k]["reliable"] and s[k]["reliable"] and (c[k]["rho"] > 0) == (s[k]["rho"] > 0)
        name = FRIENDLY[k] + ("  ·  Similar direction" if both else "")
        ax.text(-lim * 1.05, y, name, ha="right", va="center", fontsize=28, color=LABEL)
    ax.axvline(0, color="#999999", lw=1.5)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-1.0, len(items) - 0.2)
    ax.axis("off")
    ax.text(-lim * 0.1, -1.4, "← goes with LOWER group", ha="right", va="center", fontsize=26, color=LABEL)
    ax.text(lim * 0.1, -1.4, "goes with HIGHER group →", ha="left", va="center", fontsize=26, color=LABEL)
    lg = fig.add_axes([0.45, 0.9, 0.52, 0.06])
    lg.axis("off")
    lg.add_patch(Rectangle((0.02, 0.35), 0.06, 0.3, color=GREEN, transform=lg.transAxes))
    lg.text(0.1, 0.5, "Recent SGPA group", va="center", fontsize=28, transform=lg.transAxes)
    lg.add_patch(Rectangle((0.52, 0.35), 0.06, 0.3, color=BLUE, transform=lg.transAxes))
    lg.text(0.6, 0.5, "CGPA group", va="center", fontsize=28, transform=lg.transAxes)
    save(fig, "sgpa_vs_cgpa_pattern")


sgpa_cgpa_chart()


# --------------------------------------------------------------------------- J48 tree drawings
def parse_dot(text):
    nodes, edges = {}, []
    for m in re.finditer(r'(N\d+)\s*\[label="([^"]*)"([^\]]*)\]', text):
        nodes[m.group(1)] = {"label": m.group(2), "leaf": "shape=box" in m.group(3)}
    for m in re.finditer(r'(N\d+)->(N\d+)\s*\[label="([^"]*)"\]', text):
        edges.append((m.group(1), m.group(2), m.group(3)))
    return nodes, edges


def draw_tree(dot_text, name, w, h, node_fs=22, leaf_fs=20, edge_fs=20, title=None):
    nodes, edges = parse_dot(dot_text)
    kids = {}
    for a, b, lab in edges:
        kids.setdefault(a, []).append((b, lab))
    root = "N0"
    pos, depth = {}, {}
    counter = [0]

    def place(n, d):
        depth[n] = d
        ch = kids.get(n, [])
        if not ch:
            pos[n] = counter[0]
            counter[0] += 1
        else:
            for c, _ in ch:
                place(c, d + 1)
            pos[n] = sum(pos[c] for c, _ in ch) / len(ch)

    place(root, 0)
    nleaf, maxd = max(counter[0], 1), max(depth.values())
    fig = fig_px(w, h)
    ax = fig.add_axes([0.01, 0.01, 0.98, 0.98])
    ax.set_xlim(-0.6, nleaf - 0.4)
    ax.set_ylim(-(maxd + 0.6), 0.6 if title is None else 1.0)
    ax.axis("off")
    ax.add_patch(Rectangle((-0.58, -(maxd + 0.58)), nleaf - 0.04, maxd + (1.16 if title is None else 1.56),
                           fill=False, ec="#C9D6E3", lw=1.2))
    for a, b, lab in edges:
        x1, y1, x2, y2 = pos[a], -depth[a], pos[b], -depth[b]
        ax.plot([x1, x2], [y1 - 0.12, y2 + 0.12], color="#555555", lw=1.2, zorder=1)
        ax.text((x1 + x2) / 2, (y1 + y2) / 2, lab, ha="center", va="center", fontsize=edge_fs,
                color=TEXT, bbox=dict(fc="white", ec="none", pad=0.5), zorder=3)
    sx = (nleaf - 0.2) / nleaf
    for n, d in nodes.items():
        x, y = pos[n], -depth[n]
        if d["leaf"]:
            ax.text(x, y, d["label"], ha="center", va="center", fontsize=leaf_fs, color=TEXT, zorder=4,
                    bbox=dict(boxstyle="square,pad=0.35", fc="#F2F2F2", ec="#B8B8B8", lw=1))
        else:
            ax.add_patch(Ellipse((x, y), width=min(1.9, 0.95 * sx * max(1.0, len(d["label"]) / 11)), height=0.32,
                                 fc="#E6ECF0", ec="none", zorder=2))
            ax.text(x, y, d["label"], ha="center", va="center", fontsize=node_fs, color=TEXT, zorder=4)
    if title:
        ax.text(-0.5, 0.75, title, ha="left", va="center", fontsize=16, color="#777777")
    save(fig, name)


java = [str(W.JAVA), "--add-opens", "java.base/java.lang=ALL-UNNAMED", "-cp", str(W.WEKA_JAR)]
for t, fs in [("main_cgpa", (26, 24, 24)), ("sgpa", (20, 18, 18)), ("questionnaire_cgpa", (16, 13.5, 14))]:
    dot = (LOGS / f"{t}_J48_tree_full.dot").read_text(encoding="utf-8")
    draw_tree(dot, f"tree_{t}", 1800, 950, *fs)
draw_tree((LOGS / "sgpa_J48_tree_full.dot").read_text(encoding="utf-8"), "tree_sgpa_large", 2327, 1794, 24, 22, 22)
# default J48 (-M 2) main tree, only to show it is unreadable
dot_m2 = subprocess.run(java + ["weka.classifiers.trees.J48", "-C", "0.25", "-M", "2", "-t",
                                str(ROOT / "cleaned-dataset" / "ours" / "final" / "final_920.arff"), "-g"],
                        capture_output=True, text=True).stdout
(LOGS / "main_cgpa_J48_M2_tree_full.dot").write_text(dot_m2, encoding="utf-8")
draw_tree(dot_m2, "tree_main_cgpa_default_M2", 800, 560, 2.5, 2.2, 2.0)


# --------------------------------------------------------------------------- plain confusion matrix image
def cm_image(cm, name, w=1249, h=1152, title=None):
    fig = fig_px(w, h)
    ax = fig.add_axes([0.16, 0.04, 0.8, 0.8])
    for i in range(4):
        for j in range(4):
            ax.add_patch(Rectangle((j + 0.03, 3 - i + 0.03), 0.94, 0.94, fc="#DBEAF7" if i == j else "#F2F2F2", ec="none"))
            ax.text(j + 0.5, 3 - i + 0.5, str(cm[i][j]), ha="center", va="center", fontsize=40,
                    fontweight="bold" if i == j else "normal", color=TEXT)
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 4)
    ax.axis("off")
    for k in range(4):
        ax.text(k + 0.5, 4.12, f"C{k}", ha="center", va="bottom", fontsize=34, color=TEXT)
        ax.text(-0.12, 3 - k + 0.5, f"C{k}", ha="right", va="center", fontsize=34, color=TEXT)
    ax.text(2, 4.45, "Predicted", ha="center", va="bottom", fontsize=30, color="#666666")
    ax.text(-0.55, 2, "Actual", ha="center", va="center", rotation=90, fontsize=30, color="#666666")
    if title:
        fig.text(0.5, 0.965, title, ha="center", va="center", fontsize=30, color=TEXT)
    save(fig, name)


for t in ["sgpa", "questionnaire_cgpa"]:
    cm_image(R["weka"][t]["split80"]["J48"]["confusion_matrix"], f"cm_split80_J48_{t}")


# --------------------------------------------------------------------------- WEKA output text panels
def weka_text_image(log_name, name, w, h, fs=13):
    text = (LOGS / f"{log_name}.txt").read_text(encoding="utf-8")
    lines = [l for l in text.splitlines() if not l.startswith("#")]
    while lines and not lines[0].strip():
        lines.pop(0)
    keep = [l.rstrip() for l in lines if "Time taken" not in l]
    fig = fig_px(w, h)
    fig.patch.set_facecolor("#F2F2F2")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("#F2F2F2")
    ax.axis("off")
    ax.add_patch(Rectangle((0.004, 0.004), 0.992, 0.992, fill=False, ec="#BDBDBD", lw=1.5, transform=ax.transAxes))
    ax.text(0.02, 0.985, "WEKA 3.8.7 output — " + log_name.replace("_", " "), transform=ax.transAxes,
            ha="left", va="top", fontsize=fs, color="#666666", family="Segoe UI")
    ax.text(0.02, 0.95, "\n".join(keep), transform=ax.transAxes, ha="left", va="top", fontsize=fs,
            family="Consolas", color="#111111", linespacing=1.15)
    fig.savefig(OUT / f"{name}.png", dpi=fig.dpi, facecolor="#F2F2F2")
    plt.close(fig)
    print("wrote", name)


weka_text_image("main_cgpa_RandomForest_cv5", "weka_output_main_rf_cv5", 1052, 880, 12.5)
weka_text_image("main_cgpa_SMO_cv5", "weka_output_main_smo_cv5", 1144, 700, 12.5)
weka_text_image("questionnaire_cgpa_RandomForest_cv5", "weka_output_quest_rf_cv5", 1052, 880, 12.5)


# --------------------------------------------------------------------------- new data-check slides
G = R["gap_analysis"]


def gap_distribution():
    fig = fig_px(1192, 832)
    ax = fig.add_axes([0.08, 0.16, 0.9, 0.78])
    dist = G["gap_distribution_all"]
    counts = [dist.get(str(k), dist.get(k, 0)) for k in range(4)]
    total = sum(counts)
    colors = [BLUE, BLUE, ORANGE, ORANGE]
    for i, c in enumerate(counts):
        ax.bar(i, c, color=colors[i], width=0.6)
        ax.text(i, c + total * 0.01, f"{c}\n{100 * c / total:.0f}%", ha="center", va="bottom", fontsize=30,
                color=TEXT, linespacing=1.1)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["same band", "1 band apart", "2 bands apart", "3 bands apart"], fontsize=28, color=LABEL)
    ax.set_ylim(0, max(counts) * 1.3)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#BBBBBB")
    ax.set_yticks([])
    ax.tick_params(axis="x", length=0, pad=8)
    save(fig, "gap_distribution_1102")


gap_distribution()


def share_bars(name, labels, far, n, w, h, left=0.28, highlight=None):
    fig = fig_px(w, h)
    ax = fig.add_axes([left, 0.05, 0.98 - left - 0.25, 0.9])
    ys = list(range(len(labels)))[::-1]
    for y, lab, f, t in zip(ys, labels, far, n):
        pct = 100 * f / t if t else 0
        col = ORANGE if (highlight and lab in highlight) else BLUE
        ax.barh(y, pct, color=col, height=0.6)
        ax.text(pct + 1.5, y, f"{pct:.0f}%  ({f} of {t})", va="center", ha="left", fontsize=26, color=TEXT)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=28, color=LABEL)
    ax.set_xlim(0, 45)
    ax.set_ylim(-0.6, len(labels) - 0.4)
    for s in ["top", "right", "bottom"]:
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#BBBBBB")
    ax.set_xticks([])
    ax.tick_params(axis="y", length=0, pad=8)
    save(fig, name)


uni_rows = sorted(G["far_gap_by_university"], key=lambda r: -r["n"])
share_bars("far_gap_by_university", [r["university"] for r in uni_rows], [r["far"] for r in uni_rows],
           [r["n"] for r in uni_rows], 1196, 832, left=0.16)
sem_rows = G["far_gap_by_kuet_semester"]
share_bars("far_gap_by_kuet_semester", [f"Semester {int(r['semester'])}" for r in sem_rows], [r["far"] for r in sem_rows],
           [r["n"] for r in sem_rows], 1194, 832, left=0.24, highlight={"Semester 7"})


def by_date():
    rows = G["kuet_sem7_by_date"]
    fig = fig_px(1190, 832)
    ax = fig.add_axes([0.08, 0.17, 0.9, 0.78])
    for i, r in enumerate(rows):
        ok = r["n"] - r["far"]
        ax.bar(i, ok, color="#BCD5F2", width=0.7)
        ax.bar(i, r["far"], bottom=ok, color=ORANGE, width=0.7)
        ax.text(i, r["n"] + 5, str(r["n"]), ha="center", va="bottom", fontsize=22, color=TEXT)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([r["date"][5:].replace("-", "/") for r in rows], fontsize=22, color=LABEL, rotation=0)
    ax.set_ylim(0, max(r["n"] for r in rows) * 1.15)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#BBBBBB")
    ax.set_yticks([])
    ax.tick_params(axis="x", length=0, pad=6)
    ax.add_patch(Rectangle((0.02, 0.9), 0.03, 0.04, color="#BCD5F2", transform=ax.transAxes))
    ax.text(0.06, 0.92, "SGPA and CGPA within 1 band", transform=ax.transAxes, va="center", fontsize=22)
    ax.add_patch(Rectangle((0.02, 0.82), 0.03, 0.04, color=ORANGE, transform=ax.transAxes))
    ax.text(0.06, 0.84, "2 or 3 bands apart", transform=ax.transAxes, va="center", fontsize=22)
    ax.text(0.5, -0.13, "submission date (month/day), KUET semester 7", transform=ax.transAxes, ha="center",
            fontsize=22, color=LABEL)
    save(fig, "kuet_sem7_by_date")


by_date()


def grid_image(table, name, w, h):
    """SGPA band (rows) x CGPA band (columns) table drawn like the deck's confusion matrices."""
    fig = fig_px(w, h)
    ax = fig.add_axes([0.2, 0.04, 0.76, 0.78])
    for i in range(4):
        for j in range(4):
            ax.add_patch(Rectangle((j + 0.03, 3 - i + 0.03), 0.94, 0.94, fc="#DBEAF7" if abs(i - j) <= 1 else "#F2F2F2", ec="none"))
            ax.text(j + 0.5, 3 - i + 0.5, str(table[i][j]), ha="center", va="center", fontsize=34, color=TEXT)
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 4)
    ax.axis("off")
    for k in range(4):
        ax.text(k + 0.5, 4.1, f"C{k}", ha="center", va="bottom", fontsize=30, color=TEXT)
        ax.text(-0.12, 3 - k + 0.5, f"C{k}", ha="right", va="center", fontsize=30, color=TEXT)
    ax.text(2, 4.45, "CGPA band", ha="center", va="bottom", fontsize=28, color="#666666")
    ax.text(-0.62, 2, "Recent SGPA band", ha="center", va="center", rotation=90, fontsize=28, color="#666666")
    save(fig, name)


grid_image(G["sgpa_x_cgpa_aug67"], "sgpa_x_cgpa_aug67", 1196, 832)
grid_image(G["sgpa_x_cgpa_rest"], "sgpa_x_cgpa_rest", 1196, 832)
print("done")
