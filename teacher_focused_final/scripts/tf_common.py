"""Shared constants and plotting helpers for the teacher-focused analysis.

Used by `notebooks/03_teacher_focused_analysis.ipynb` and by `build_deck.py`, so every chart,
colour and label in the deck comes from one place.

Colour choices were checked with the dataviz palette validator (light mode, white surface):
the four CGPA-class colours pass lightness band, chroma, colour-blind separation, normal-vision
separation and 3:1 contrast. Low classes are warm, high classes are cool.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.patches import Rectangle

# ---------------------------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------------------------
PROJ = Path(__file__).resolve().parents[2]
TF = PROJ / "teacher_focused_final"
ANALYSIS = TF / "analysis"
SPLITS = TF / "splits"
ASSETS = TF / "assets"
PRES = TF / "presentation"
ASSET_DIRS = {k: ASSETS / k for k in [
    "forms", "dataset_screenshots", "response_charts", "framework", "trees", "rules",
    "confusion_matrices", "related_work", "comparison_charts", "patterns"]}

RAW_MULTI = PROJ / "raw-data" / "University Student Performance Analysis Form (Responses) - Form Responses 1.csv"
RAW_KUET = PROJ / "raw-data" / "Student Performance Analysis Form (Responses) - Form Responses 1.csv"
MERGED_FULL = PROJ / "cleaned-dataset" / "ours" / "merged" / "merged_full_clean.csv"
REF_IUBAT = PROJ / "reference-data" / "IUBAT_Student_Performance_Dataset.csv"
REF_CLASSIFIED = PROJ / "reference-data" / "classified_dataset.csv"
WEKA_ZIP = PROJ / "Weka Result.zip"
DOCS = PROJ / "docs"
REPORT_DOCX = PROJ.parent / "Student_Academic_Performance_Analysis_Final_Report_Editable.docx"

FORM_URL_MULTI = ("https://docs.google.com/forms/d/e/"
                  "1FAIpQLSeDiQ5GpAk5FfdYyCw2d5d7Yma7K5Qjcqv5S6viXyaRlKyiwQ/viewform")
IUBAT_URL = "https://data.mendeley.com/datasets/ns87rtkv58/2"

for _d in [ANALYSIS, SPLITS, PRES, *ASSET_DIRS.values()]:
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------------------------
# Classes, colours, ink
# ---------------------------------------------------------------------------------------------
RANDOM_STATE = 42          # same seed as notebooks/pipeline_v2_source.py
TEST_SIZE = 0.20

CLASS_ORDER = ["C0_below_3_20", "C1_3_20_to_3_49", "C2_3_50_to_3_74", "C3_3_75_plus"]
CLASSES = ["C0", "C1", "C2", "C3"]
CLASS_RANGE = {"C0": "below 3.20", "C1": "3.20 to 3.49", "C2": "3.50 to 3.74", "C3": "3.75 and above"}
CLASS_RANGE_SHORT = {"C0": "<3.20", "C1": "3.20–3.49", "C2": "3.50–3.74", "C3": "3.75+"}
CLASS_COLOR = {"C0": "#a83a3a", "C1": "#e0703e", "C2": "#4b8fdc", "C3": "#2659a3"}
HIGH = ["C2", "C3"]
LOW = ["C0", "C1"]

INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#ffffff"
ACCENT = "#2a78d6"
NEUTRAL = "#a3a9b1"
GOOD = "#0ca30c"        # status colour: "correct" (always paired with a text label)
SERIOUS = "#ec835a"     # status colour: highlighted mistake (always paired with a text label)

# ---------------------------------------------------------------------------------------------
# Questionnaire scales — copied from notebooks/pipeline_v2_source.py (ORDINAL_MAPS, §5.9)
# ---------------------------------------------------------------------------------------------
ORDINAL_MAPS = {
    "admission_satisfaction":   ["Very dissatisfied", "Dissatisfied", "Neutral", "Satisfied", "Very satisfied"],
    "desired_department_match": ["Not at all what I wanted", "Mostly different", "Somewhat different",
                                 "Close to what I wanted", "Yes, exactly what I wanted"],
    "result_satisfaction":      ["Very dissatisfied", "Dissatisfied", "Neutral", "Satisfied", "Very satisfied"],
    "attendance":               ["Less than 40%", "40-59%", "60-74%", "75-89%", "90% or more"],
    "weekly_study_time":        ["Less than 3 hrs", "3-6 hrs", "7-10 hrs", "11-15 hrs", "More than 15 hrs"],
    "study_style":              ["Occasionally when necessary", "Mostly just before exams",
                                 "Regularly but not every day", "Mostly consistent routine",
                                 "Consistent throughout the semester"],
    "topic_clarity":            ["Almost nothing", "A little", "About half", "Most of it", "Almost everything"],
    "sleep_duration":           ["Less than 5 hrs", "5-6 hrs", "6-7 hrs", "7-8 hrs", "More than 8 hrs"],
    "stress_frequency":         ["Never", "Rarely", "Sometimes", "Often", "Almost always"],
    "weekly_responsibilities":  ["None", "1-5 hrs", "6-10 hrs", "11-15 hrs", "More than 15 hrs"],
    "distraction_frequency":    ["Never", "Rarely", "Sometimes", "Often", "Almost every day"],
    "study_environment":        ["Very distracting", "Somewhat distracting", "Neither good nor bad",
                                 "Mostly suitable", "Excellent for focused study"],
    "support_level":            ["No support", "Very little support", "Some support",
                                 "Good support", "Strong support"],
    "routine_manageability":    ["Completely unmanageable", "Difficult to manage", "Average",
                                 "Mostly manageable", "Very healthy and balanced"],
    "career_expectation":       ["Not at all", "Slightly", "Somewhat", "Mostly", "Completely"],
}
RECENT_SGPA_OPTIONS = ["below 3.20", "3.20-3.49", "3.50-3.74", "3.75+"]

# The 12 human-readable questionnaire features used by every explanatory tree.
TREE_FEATURES = [
    "attendance", "weekly_study_time", "study_style", "topic_clarity", "stress_frequency",
    "sleep_duration", "distraction_frequency", "weekly_responsibilities",
    "desired_department_match", "support_level", "study_environment", "routine_manageability",
]

FEATURE_LABEL = {
    "attendance": "Class attendance",
    "weekly_study_time": "Self-study per week",
    "study_style": "Study style",
    "topic_clarity": "Topic clear after class",
    "stress_frequency": "Stress blocked studying",
    "sleep_duration": "Sleep per night",
    "distraction_frequency": "Phone / social media took study time",
    "weekly_responsibilities": "Outside commitments (tuition/job/club)",
    "desired_department_match": "Got desired department",
    "support_level": "Family / friend support",
    "study_environment": "Study-friendly living place",
    "routine_manageability": "Semester routine manageable",
    "recent_sgpa": "Recent semester SGPA",
    "admission_satisfaction": "Satisfied with admission",
    "career_expectation": "Course leads to dream career",
    "result_satisfaction": "Satisfied with last result",
}

MORE_WORD = {
    "attendance": ("or more", "or less"), "weekly_study_time": ("or more", "or less"),
    "study_style": ("or more regular", "or less regular"), "topic_clarity": ("or more", "or less"),
    "stress_frequency": ("or more often", "or less often"), "sleep_duration": ("or more", "or less"),
    "distraction_frequency": ("or more often", "or less often"),
    "weekly_responsibilities": ("or more", "or less"),
    "desired_department_match": ("or closer", "or further"), "support_level": ("or more", "or less"),
    "study_environment": ("or better", "or worse"), "routine_manageability": ("or better", "or worse"),
    "recent_sgpa": ("or higher", "or lower"),
}


def options_for(feature: str) -> list[str]:
    return RECENT_SGPA_OPTIONS if feature == "recent_sgpa" else ORDINAL_MAPS[feature]


def code_col(feature: str) -> str:
    return "recent_sgpa_code" if feature == "recent_sgpa" else f"{feature}_code"


def load_clean() -> pd.DataFrame:
    """The official 1,108-row merged dataset with short class columns added."""
    # keep_default_na=False: the answer "None" (no outside responsibilities) must stay a category
    df = pd.read_csv(MERGED_FULL, keep_default_na=False, na_values=[""])
    df["cgpa_class"] = df["cgpa_band"].str[:2]
    df["sgpa_class"] = df["recent_sgpa_band"].str[:2]
    df["recent_sgpa_imputed"] = df["recent_sgpa_imputed"].astype(str).str.lower().eq("true")
    return df


def pct(x: float, digits: int = 1) -> str:
    return f"{100 * x:.{digits}f}%"


# ---------------------------------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------------------------------
def set_style() -> None:
    plt.rcParams.update({
        "font.family": ["Segoe UI", "DejaVu Sans"],
        "font.size": 16,
        "text.color": INK,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK2,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "savefig.dpi": 200,
    })


def _save(fig, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    return path


def tint(hex_color: str, amount: float) -> tuple:
    """Mix a colour with white; amount=0 keeps it, amount=1 gives white."""
    r, g, b = to_rgb(hex_color)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)


def _wrap(s: str, width: int) -> str:
    return textwrap.fill(str(s), width)


# ---------------------------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------------------------
def option_bar_chart(labels, counts, path, size=(6.2, 4.4), fs=16, wrap=20, colors=None):
    """Horizontal bars in questionnaire order (first option on top), each labelled count + %."""
    counts = [int(c) for c in counts]
    total = sum(counts)
    n = len(labels)
    fig, ax = plt.subplots(figsize=size)
    y = np.arange(n)[::-1]
    ax.barh(y, counts, height=0.64, color=colors or [ACCENT] * n, edgecolor=SURFACE, linewidth=2)
    mx = max(counts)
    for yi, c in zip(y, counts):
        ax.text(c + mx * 0.02, yi, f"{c:,}  ({100 * c / total:.0f}%)", va="center", ha="left",
                fontsize=fs - 1, color=INK, fontweight="bold" if c == mx else "normal")
    ax.set_yticks(y)
    ax.set_yticklabels([_wrap(l, wrap) for l in labels], fontsize=fs)
    ax.set_xlim(0, mx * 1.45)
    ax.set_xticks([])
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    return _save(fig, path)


def class_column_chart(counts: dict, path, size=(6.2, 4.4), fs=16, title_word="students"):
    """Columns for C0..C3 in class colours with count and % on each cap."""
    vals = [int(counts.get(c, 0)) for c in CLASSES]
    total = sum(vals)
    fig, ax = plt.subplots(figsize=size)
    x = np.arange(4)
    ax.bar(x, vals, width=0.62, color=[CLASS_COLOR[c] for c in CLASSES], edgecolor=SURFACE, linewidth=2)
    for xi, v in zip(x, vals):
        ax.text(xi, v + max(vals) * 0.02, f"{v:,}\n{100 * v / total:.1f}%", ha="center", va="bottom",
                fontsize=fs - 1, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n{CLASS_RANGE_SHORT[c]}" for c in CLASSES], fontsize=fs)
    ax.set_ylim(0, max(vals) * 1.3)
    ax.set_yticks([])
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="x", length=0)
    fig.tight_layout()
    return _save(fig, path)


def class_mix_chart(table: pd.DataFrame, path, size=(11.8, 5.0), fs=15, wrap=24, legend=True,
                    min_label=8.0):
    """100%-stacked horizontal bars: CGPA-class mix of each row. `table` holds counts, cols C0..C3."""
    table = table.reindex(columns=CLASSES).fillna(0)
    shares = table.div(table.sum(axis=1), axis=0) * 100
    n = len(shares)
    fig, ax = plt.subplots(figsize=size)
    y = np.arange(n)[::-1]
    left = np.zeros(n)
    for c in CLASSES:
        w = shares[c].values
        ax.barh(y, w, left=left, height=0.62, color=CLASS_COLOR[c], edgecolor=SURFACE, linewidth=2,
                label=f"{c}  ({CLASS_RANGE_SHORT[c]})")
        for yi, l, wi in zip(y, left, w):
            if wi >= min_label:
                ax.text(l + wi / 2, yi, f"{wi:.0f}%", ha="center", va="center", color="white",
                        fontsize=fs - 1, fontweight="bold")
        left += w
    ax.set_yticks(y)
    ax.set_yticklabels([_wrap(g, wrap) for g in shares.index], fontsize=fs)
    for yi, nn in zip(y, table.sum(axis=1).values):
        ax.text(101.5, yi, f"n = {int(nn):,}", va="center", ha="left", fontsize=fs - 2, color=INK2)
    ax.set_xlim(0, 100)
    ax.set_xticks([])
    ax.spines[["top", "right", "bottom", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    if legend:
        ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False,
                  fontsize=fs - 1, handlelength=1.1, columnspacing=1.4)
    fig.tight_layout()
    return _save(fig, path)


def confusion_plot(cm, path, highlight=None, normalize=False, size=(6.6, 6.0), fs=19, labels=None):
    """Rows = actual class, columns = predicted class. Diagonal outlined green (correct);
    `highlight=(row, col)` outlines one example mistake in orange."""
    cm = np.asarray(cm, dtype=float)
    labels = labels or CLASSES
    data = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1) * 100 if normalize else cm
    vmax = max(data.max(), 1)
    cmap = LinearSegmentedColormap.from_list("blues", ["#f3f8fe", "#86b6ef", "#184f95"])
    fig, ax = plt.subplots(figsize=size)
    ax.imshow(data, cmap=cmap, vmin=0, vmax=vmax)
    k = len(labels)
    for i in range(k):
        for j in range(k):
            v = data[i, j]
            txt = f"{v:.0f}%" if normalize else f"{int(v)}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=fs,
                    color="white" if v > 0.55 * vmax else INK, fontweight="bold" if i == j else "normal")
    for m in np.arange(-0.5, k, 1):
        ax.axhline(m, color=SURFACE, lw=3)
        ax.axvline(m, color=SURFACE, lw=3)
    for i in range(k):
        ax.add_patch(Rectangle((i - 0.46, i - 0.46), 0.92, 0.92, fill=False, ec=GOOD, lw=3.5))
    if highlight is not None:
        r, c = highlight
        ax.add_patch(Rectangle((c - 0.46, r - 0.46), 0.92, 0.92, fill=False, ec=SERIOUS, lw=4.5))
    ticks = [f"{c}\n{CLASS_RANGE_SHORT.get(c, '')}".strip() for c in labels]
    ax.set_xticks(range(k))
    ax.set_xticklabels(ticks, fontsize=fs - 4)
    ax.set_yticks(range(k))
    ax.set_yticklabels(ticks, fontsize=fs - 4)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.set_xlabel("PREDICTED group (model's answer)  →", fontsize=fs - 4, color=INK, labelpad=10)
    ax.set_ylabel("ACTUAL group (student's real answer)  →", fontsize=fs - 4, color=INK, labelpad=10)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    return _save(fig, path)


def train_test_chart(df: pd.DataFrame, path, size=(11.8, 5.0), fs=16):
    """Grouped columns per experiment: training accuracy (grey) and unseen-test accuracy (blue),
    with the 'always guess the most common group' test baseline as a thin line."""
    n = len(df)
    fig, ax = plt.subplots(figsize=size)
    x = np.arange(n)
    w = 0.34
    tr = df["train_accuracy"].values * 100
    te = df["test_accuracy"].values * 100
    ax.bar(x - w / 2 - 0.01, tr, width=w, color=NEUTRAL, edgecolor=SURFACE, lw=2, label="Training students")
    ax.bar(x + w / 2 + 0.01, te, width=w, color=ACCENT, edgecolor=SURFACE, lw=2, label="Unseen test students")
    base = df["baseline_test_accuracy"].values * 100 if "baseline_test_accuracy" in df else np.zeros(n)
    for xi, bl in zip(x, base):
        if bl:
            ax.plot([xi - 0.4, xi + 0.4], [bl, bl], color=INK, lw=1.8, zorder=3)
    if base.any():
        ax.plot([], [], color=INK, lw=1.8, label="Always guessing the most common group")
    for xi, a, b, bl in zip(x, tr, te, base):
        ya = a + 1 if abs(a - bl) > 3 or a > bl else bl + 1
        yb = b + 1 if abs(b - bl) > 3 or b > bl + 3 else bl + 1
        ax.text(xi - w / 2, max(ya, a + 1), f"{a:.1f}%", ha="center", va="bottom", fontsize=fs - 1, color=INK2)
        ax.text(xi + w / 2, max(yb, b + 1), f"{b:.1f}%", ha="center", va="bottom", fontsize=fs, color=INK,
                fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([_wrap(s, 26) for s in df["experiment_label"]], fontsize=fs)
    ax.set_ylim(0, max(tr.max(), te.max()) * 1.25)
    ax.set_yticks([])
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="x", length=0)
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=fs - 2)
    fig.tight_layout()
    return _save(fig, path)


# ---------------------------------------------------------------------------------------------
# Decision trees: readable drawing and rule extraction
# ---------------------------------------------------------------------------------------------
def _node_class_counts(tree, X, y, classes):
    ind = tree.decision_path(X)
    y = np.asarray(y)
    return {c: np.asarray(ind[y == c].sum(axis=0)).ravel() for c in classes}


def _question(feature: str, threshold: float) -> tuple[str, str]:
    k = int(np.floor(threshold))
    opts = options_for(feature)
    word = "" if k + 1 == len(opts) - 1 else f" {MORE_WORD[feature][0]}"
    return FEATURE_LABEL[feature], f"“{opts[k + 1]}”{word}?"


def draw_tree(tree, features, X, y, path, classes=None, title=None, fs=15, target_word="CGPA"):
    """Readable top-down drawing. Internal node = plain question; right branch = Yes.
    Leaf = students reaching it (training set), most common group and a mini class-mix bar."""
    classes = classes or CLASSES
    t = tree.tree_
    counts = _node_class_counts(tree, X, y, classes)
    pos, n_leaves = {}, [0]

    def walk(node, depth):
        if t.children_left[node] == -1:
            x = n_leaves[0]
            n_leaves[0] += 1
            pos[node] = (x, depth)
            return x
        xl = walk(t.children_left[node], depth + 1)
        xr = walk(t.children_right[node], depth + 1)
        pos[node] = ((xl + xr) / 2, depth)
        return pos[node][0]

    walk(0, 0)
    max_depth = max(d for _, d in pos.values())
    width = max(11.0, n_leaves[0] * 2.05)
    height = 1.0 + (max_depth + 1) * 2.05
    fig, ax = plt.subplots(figsize=(width, height))
    ys = lambda d: -d * 2.0

    for node, (x, d) in pos.items():
        if t.children_left[node] != -1:
            for child, word in [(t.children_left[node], "No"), (t.children_right[node], "Yes")]:
                cx, cd = pos[child]
                ax.plot([x, cx], [ys(d) - 0.45, ys(cd) + 0.55], color=AXIS, lw=1.6, zorder=1)
                ax.text((x + cx) / 2, (ys(d) - 0.45 + ys(cd) + 0.55) / 2, word, ha="center", va="center",
                        fontsize=fs - 1, color=INK2, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.18", fc=SURFACE, ec="none"), zorder=3)

    for node, (x, d) in pos.items():
        n = int(sum(counts[c][node] for c in classes))
        if t.children_left[node] != -1:
            label, q = _question(features[t.feature[node]], t.threshold[node])
            head = f"All {n} training students\n" if node == 0 else f"{n} students\n"
            ax.text(x, ys(d), f"{head}{_wrap(label, 26)}\n{_wrap(q, 26)}", ha="center", va="center",
                    fontsize=fs, color=INK, linespacing=1.25, zorder=4,
                    bbox=dict(boxstyle="round,pad=0.5", fc="#f6f7f9", ec=AXIS, lw=1.2))
        else:
            dist = np.array([counts[c][node] for c in classes], dtype=float)
            top = classes[int(np.argmax(t.value[node].ravel()))]
            share = dist[classes.index(top)] / max(dist.sum(), 1)
            ax.text(x, ys(d) + 0.15, f"{n} students\nmostly {top}\n({share * 100:.0f}% of them)",
                    ha="center", va="center", fontsize=fs, color=INK, linespacing=1.2, zorder=4,
                    bbox=dict(boxstyle="round,pad=0.45", fc=tint(CLASS_COLOR[top], 0.82),
                              ec=CLASS_COLOR[top], lw=2.2))
            bar_w = 0.66
            left = x - bar_w / 2
            for c, v in zip(classes, dist / max(dist.sum(), 1)):
                ax.add_patch(Rectangle((left, ys(d) - 0.62), bar_w * v, 0.16, facecolor=CLASS_COLOR[c],
                                       edgecolor=SURFACE, lw=1, zorder=4))
                left += bar_w * v

    xs = [p[0] for p in pos.values()]
    ax.set_xlim(min(xs) - 1.1, max(xs) + 1.1)
    ax.set_ylim(ys(max_depth) - 1.3, 0.95)
    ax.axis("off")
    handles = [Rectangle((0, 0), 1, 1, color=CLASS_COLOR[c]) for c in classes]
    ax.legend(handles, [f"{c} {target_word} {CLASS_RANGE_SHORT[c]}" for c in classes], ncol=4,
              loc="upper center", bbox_to_anchor=(0.5, -0.01), frameon=False, fontsize=fs)
    if title:
        ax.set_title(title, fontsize=fs + 4, color=INK, loc="left")
    fig.tight_layout()
    return _save(fig, path)


def describe_range(feature: str, lo: int, hi: int) -> str:
    opts = options_for(feature)
    top = len(opts) - 1
    more, less = MORE_WORD[feature]
    if lo == hi:
        val = f"“{opts[lo]}”"
    elif lo == 0 and hi < top:
        val = f"“{opts[hi]}” {less}"
    elif hi == top and lo > 0:
        val = f"“{opts[lo]}” {more}"
    else:
        val = f"“{opts[lo]}” to “{opts[hi]}”"
    return f"{FEATURE_LABEL[feature]}: {val}"


def extract_rules(tree, features, X_train, y_train, X_test, y_test, exp_id, classes=None):
    """One row per leaf: readable IF-conditions, training support/mix/purity and test support/mix."""
    classes = classes or CLASSES
    t = tree.tree_
    y_train, y_test = np.asarray(y_train), np.asarray(y_test)
    leaf_tr, leaf_te = tree.apply(X_train), tree.apply(X_test)
    rows = []

    def walk(node, bounds, masks):
        if t.children_left[node] == -1:
            ytr, yte = y_train[leaf_tr == node], y_test[leaf_te == node]
            pred = tree.classes_[int(np.argmax(t.value[node].ravel()))]
            conds = [describe_range(f, lo, hi) for f, (lo, hi) in bounds.items()]
            row = dict(experiment=exp_id, leaf_id=int(node), predicted_class=pred,
                       direction="higher" if pred in HIGH else "lower",
                       conditions=" AND ".join(conds), n_conditions=len(conds),
                       rule_text="IF " + " AND ".join(conds) + f"  →  mostly {pred}",
                       bounds=repr({f: b for f, b in bounds.items()}),
                       n_train=len(ytr), n_test=len(yte))
            for c in classes:
                row[f"train_{c}"] = int((ytr == c).sum())
                row[f"test_{c}"] = int((yte == c).sum())
            row["train_purity"] = row[f"train_{pred}"] / max(len(ytr), 1)
            row["test_share_predicted"] = row[f"test_{pred}"] / max(len(yte), 1) if len(yte) else np.nan
            rows.append(row)
            return
        f = features[t.feature[node]]
        k = int(np.floor(t.threshold[node]))
        lo, hi = bounds.get(f, (0, len(options_for(f)) - 1))
        left_b, right_b = dict(bounds), dict(bounds)
        left_b[f], right_b[f] = (lo, min(hi, k)), (max(lo, k + 1), hi)
        walk(t.children_left[node], left_b, masks)
        walk(t.children_right[node], right_b, masks)

    walk(0, {}, None)
    out = pd.DataFrame(rows)
    out.insert(1, "rule_id", [f"{exp_id}-R{i + 1}" for i in range(len(out))])
    return out


def rule_mask(df: pd.DataFrame, bounds: dict) -> pd.Series:
    """Boolean mask of students whose answers satisfy a rule's per-feature code ranges."""
    m = pd.Series(True, index=df.index)
    for f, (lo, hi) in bounds.items():
        m &= df[code_col(f)].between(lo, hi)
    return m
