"""Convert a `# %%` percent-format Python script into a Jupyter notebook.

Usage:  python py2nb.py <source.py> <target.ipynb>

Cells are delimited by lines starting with `# %%`. A `# %% [markdown]` cell has
its leading `# ` comment prefix stripped so the text renders as Markdown.
"""
import json
import sys
from pathlib import Path


def split_cells(text):
    cells, current, kind = [], [], "code"
    for line in text.splitlines():
        if line.startswith("# %%"):
            if current:
                cells.append((kind, current))
            kind = "markdown" if "[markdown]" in line else "code"
            current = []
        else:
            current.append(line)
    if current:
        cells.append((kind, current))
    return cells


def build(src: Path):
    cells = []
    for kind, lines in split_cells(src.read_text(encoding="utf-8")):
        if kind == "markdown":
            body = [l[2:] if l.startswith("# ") else ("" if l.strip() == "#" else l)
                    for l in lines]
        else:
            body = lines
        while body and not body[0].strip():
            body.pop(0)
        while body and not body[-1].strip():
            body.pop()
        if not body:
            continue
        source = [l + "\n" for l in body[:-1]] + [body[-1]]
        cell = {"cell_type": kind, "metadata": {}, "source": source}
        if kind == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        cells.append(cell)
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "venv", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


if __name__ == "__main__":
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    dst.write_text(json.dumps(build(src), indent=1), encoding="utf-8")
    print(f"{src.name} -> {dst.name}: {len(build(src)['cells'])} cells")
