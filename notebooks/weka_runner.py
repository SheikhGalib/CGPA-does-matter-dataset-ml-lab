"""Drive WEKA 3.8.7 from Python so every reported model number comes from a real WEKA log.

Why this exists
---------------
The course asks for WEKA. Clicking through the Explorer 150 times is not reproducible, so the
anomaly notebook writes ARFF files, calls WEKA's own command-line classifiers, saves each raw
WEKA output to ``docs/weka_runs/<name>.txt`` and parses the confusion matrix out of it. A teammate
can open any log, read the exact command on its first line, and repeat the run in the Explorer.

Only the *evaluation bookkeeping* (quadratic kappa, far-error rate) is computed in Python, and it
is computed from WEKA's own confusion matrix — never from a Python model.
"""
from __future__ import annotations

import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WEKA_DIR = Path("C:/Program Files/Weka-3-8-7")
WEKA_JAR = WEKA_DIR / "weka.jar"
JAVA = next(WEKA_DIR.glob("jre/*/bin/java.exe"), None)
RUN_DIR = ROOT / "docs" / "weka_runs"
ARFF_DIR = ROOT / "cleaned-dataset" / "ours" / "anomaly"

CLASSES = ["C0", "C1", "C2", "C3"]


# ----------------------------------------------------------------------------- ARFF

def write_arff(frame: pd.DataFrame, path: Path, relation: str, nominal: dict[str, list[str]],
               class_col: str = "cgpa_band") -> Path:
    """Write `frame` as ARFF. Columns in `nominal` get that exact value list (order preserved);
    every other column is numeric. The class column is moved last, which WEKA expects."""
    cols = [c for c in frame.columns if c != class_col] + [class_col]
    lines = [f"@relation {relation}", ""]
    for c in cols:
        if c in nominal:
            lines.append("@attribute {} {{{}}}".format(c, ",".join(nominal[c])))
        else:
            lines.append(f"@attribute {c} numeric")
    lines += ["", "@data"]
    data = frame[cols].astype(str).to_numpy()
    lines += [",".join(row) for row in data]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# ----------------------------------------------------------------------------- result

@dataclass
class WekaResult:
    name: str
    cm: np.ndarray                      # rows = actual, columns = predicted
    log: str
    command: str
    extra: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(self.cm.sum())

    @property
    def accuracy(self) -> float:
        return float(np.trace(self.cm) / self.cm.sum())

    def _kappa(self, weights=None) -> float:
        cm = self.cm.astype(float)
        k = cm.shape[0]
        i, j = np.indices((k, k))
        w = np.ones((k, k)) - np.eye(k) if weights is None else ((i - j) ** 2) / (k - 1) ** 2
        expected = np.outer(cm.sum(1), cm.sum(0)) / cm.sum()
        return float(1 - (w * cm).sum() / (w * expected).sum())

    @property
    def kappa(self) -> float:
        return self._kappa()

    @property
    def qwk(self) -> float:
        """Quadratic weighted kappa: the standard agreement metric for ORDERED classes.
        Predicting C0 for a C3 student costs 9x as much as a neighbouring-band error."""
        return self._kappa("quadratic")

    @property
    def far_rate(self) -> float:
        """Share of ALL predictions that land two or more bands away from the truth."""
        i, j = np.indices(self.cm.shape)
        return float(self.cm[np.abs(i - j) >= 2].sum() / self.cm.sum())

    @property
    def within_one(self) -> float:
        i, j = np.indices(self.cm.shape)
        return float(self.cm[np.abs(i - j) <= 1].sum() / self.cm.sum())

    @property
    def mae_bands(self) -> float:
        i, j = np.indices(self.cm.shape)
        return float((self.cm * np.abs(i - j)).sum() / self.cm.sum())

    @property
    def macro_f1(self) -> float:
        cm = self.cm.astype(float)
        tp = np.diag(cm)
        prec = np.divide(tp, cm.sum(0), out=np.zeros_like(tp), where=cm.sum(0) > 0)
        rec = np.divide(tp, cm.sum(1), out=np.zeros_like(tp), where=cm.sum(1) > 0)
        f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
        return float(f1.mean())

    def row(self, **extra) -> dict:
        return {"run": self.name, "n": self.n, "accuracy": self.accuracy, "kappa": self.kappa,
                "qwk": self.qwk, "far_rate": self.far_rate, "within_1": self.within_one,
                "mae_bands": self.mae_bands, "macro_f1": self.macro_f1, **self.extra, **extra}


# ----------------------------------------------------------------------------- running

_CM_ROW = re.compile(r"^\s*((?:\d+\s+)+)\|\s+\w+\s+=\s+(\S+)\s*$")


def _prediction_rows(text: str) -> list[tuple[int, int]]:
    """(actual, predicted) class indices from a WEKA CSV prediction block, if one is present."""
    if "=== Predictions under" not in text and "=== Predictions on" not in text:
        return []
    body = re.split(r"=== Predictions (?:under|on) [^=]*===", text, maxsplit=1)[1]
    pairs = []
    for line in body.strip().splitlines()[1:]:
        parts = line.strip().split(",")
        if len(parts) < 5 or not parts[0].isdigit():
            break
        pairs.append((int(parts[1].split(":")[0]) - 1, int(parts[2].split(":")[0]) - 1))
    return pairs


def parse_confusion(text: str) -> np.ndarray:
    """Return the LAST confusion matrix in a WEKA log (the evaluation one).

    When WEKA was asked to print per-instance predictions it may omit the statistics block; the
    matrix is then rebuilt from WEKA's own printed predictions (still WEKA's numbers, just tallied).
    """
    blocks = text.split("=== Confusion Matrix ===")
    if len(blocks) >= 2:
        rows = [list(map(int, m.group(1).split()))
                for line in blocks[-1].splitlines() if (m := _CM_ROW.match(line))]
        return np.array(rows)
    pairs = _prediction_rows(text)
    if pairs:
        cm = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
        for a, p in pairs:
            cm[a, p] += 1
        return cm
    raise ValueError("no confusion matrix or predictions in WEKA output:\n" + text[-1500:])


def parse_tree(text: str) -> dict:
    """Pull the root split, leaf count and size out of a J48 model printout."""
    body = text.split("------------------", 1)[-1] if "J48 pruned tree" in text else ""
    first = next((l.strip() for l in body.splitlines() if l.strip()), "")
    leaves = re.search(r"Number of Leaves\s*:\s*(\d+)", text)
    size = re.search(r"Size of the tree\s*:\s*(\d+)", text)
    root = re.split(r"\s*(<=|>|=|:)\s*", first)[0] if first else None
    return {"root": root, "first_rule": first,
            "leaves": int(leaves.group(1)) if leaves else None,
            "size": int(size.group(1)) if size else None}


def _pretty(cmd: list[str]) -> str:
    out = []
    for a in cmd:
        a = a.replace(str(ROOT).replace("\\", "/"), "<repo>").replace(str(ROOT), "<repo>")
        out.append(f'"{a}"' if " " in a else a)
    return " ".join(out)


def run(name: str, scheme: list[str], train: Path, *, folds: int | None = 10, seed: int = 1,
        test: Path | None = None, split: float | None = None, base: list[str] | None = None,
        keep_model: bool = False, cache: bool = True, general: list[str] | None = None,
        **extra) -> WekaResult:
    """Run one WEKA evaluation.

    scheme : classifier class + its own options, e.g. ["weka.classifiers.trees.J48", "-M", "40"]
    folds  : k for stratified k-fold CV (ignored when `test` or `split` is given)
    test   : a supplied test set ARFF (Explorer: "Supplied test set")
    split  : percentage for a random train/test split (Explorer: "Percentage split")
    base   : options for the base classifier of a meta scheme (placed after `--`)
    """
    if JAVA is None:
        raise FileNotFoundError(f"WEKA's bundled java.exe not found under {WEKA_DIR}")
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [str(JAVA), "--add-opens", "java.base/java.lang=ALL-UNNAMED", "-cp", str(WEKA_JAR)]
    cmd += scheme + ["-t", str(train)]
    if test is not None:
        cmd += ["-T", str(test)]
    elif split is not None:
        cmd += ["-split-percentage", str(split), "-s", str(seed)]
    else:
        cmd += ["-x", str(folds), "-s", str(seed)]
    if not keep_model:
        cmd += ["-o"]
    cmd += ["-v"] + (general or [])
    if base:
        cmd += ["--"] + base

    pretty = _pretty(cmd[4:])            # drop the java/jar prefix — the Explorer does not need it
    log_path = RUN_DIR / f"{name}.txt"            # a name like "bulk/perm_07" lands in a subfolder
    log_path.parent.mkdir(parents=True, exist_ok=True)
    header = f"# WEKA 3.8.7 command:\n# java -cp weka.jar {pretty}\n\n"
    if cache and log_path.exists():
        text = log_path.read_text(encoding="utf-8")
        if text.startswith(header):
            return WekaResult(name, parse_confusion(text), text, pretty,
                              {**extra, **(parse_tree(text) if keep_model else {})})

    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    out = proc.stdout
    try:
        cm = parse_confusion(out)
    except ValueError:
        raise RuntimeError(f"WEKA run {name!r} failed:\n{pretty}\n{proc.stderr[-2000:]}\n{out[-1500:]}") from None
    text = header + out
    log_path.write_text(text, encoding="utf-8")
    return WekaResult(name, cm, text, pretty, {**extra, **(parse_tree(text) if keep_model else {})})


def run_predictions(name: str, scheme: list[str], train: Path, *, folds: int = 10, seed: int = 1,
                    base: list[str] | None = None, id_attribute: int = 1) -> pd.DataFrame:
    """Per-instance cross-validation predictions straight from WEKA.

    `train` must carry a row-ID attribute (position `id_attribute`) that the scheme removes before
    learning (e.g. FilteredClassifier + Remove). WEKA prints that ID next to every prediction,
    which is how a prediction is traced back to the student it belongs to.
    Explorer equivalent: More options -> Output predictions -> CSV, "attributes" = the ID column.
    """
    res = run(name, scheme, train, folds=folds, seed=seed, base=base,
              general=["-classifications",
                       f"weka.classifiers.evaluation.output.prediction.CSV -p {id_attribute}"])
    body = res.log.split("=== Predictions under cross-validation ===", 1)[1]
    rows = []
    for line in body.strip().splitlines()[1:]:
        parts = line.strip().split(",")
        if len(parts) < 6 or not parts[0].isdigit():
            break
        rows.append({"actual": int(parts[1].split(":")[0]) - 1,
                     "predicted": int(parts[2].split(":")[0]) - 1,
                     "confidence": float(parts[4]),
                     "row_id": int(float(parts[-1]))})
    out = pd.DataFrame(rows)
    assert len(out) == res.n, f"parsed {len(out)} predictions, WEKA evaluated {res.n}"
    return out


def run_many(jobs: list[dict], workers: int = 8) -> list[WekaResult]:
    """Run a list of `run(**job)` calls in parallel. Each is its own JVM, so threads suffice."""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda j: run(**j), jobs))


# ----------------------------------------------------------------------------- common schemes

J48 = ["weka.classifiers.trees.J48", "-C", "0.25", "-M", "40"]
RF = ["weka.classifiers.trees.RandomForest", "-P", "100", "-I", "100", "-num-slots", "1",
      "-K", "0", "-M", "1.0", "-V", "0.001", "-S", "1"]
SMO = ["weka.classifiers.functions.SMO"]
LOGISTIC = ["weka.classifiers.functions.Logistic", "-R", "1.0E-8", "-M", "-1"]
NAIVE_BAYES = ["weka.classifiers.bayes.NaiveBayes"]
ZEROR = ["weka.classifiers.rules.ZeroR"]
IBK = ["weka.classifiers.lazy.IBk"]

STANDARD_MODELS = {"ZeroR": ZEROR, "J48": J48, "RandomForest": RF, "SMO": SMO,
                   "Logistic": LOGISTIC, "NaiveBayes": NAIVE_BAYES}


def cost_matrix(kind: str = "linear") -> str:
    """WEKA cost-matrix string. linear: |i-j|; quadratic: (i-j)^2."""
    k = len(CLASSES)
    rows = []
    for i in range(k):
        vals = [abs(i - j) if kind == "linear" else (i - j) ** 2 for j in range(k)]
        rows.append(" ".join(f"{float(v):.1f}" for v in vals))
    return "[" + "; ".join(rows) + "]"
