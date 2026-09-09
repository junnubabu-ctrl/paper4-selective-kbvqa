#!/usr/bin/env python
from __future__ import annotations
from pathlib import Path as _Path
import sys as _sys
_ROOT=_Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in _sys.path: _sys.path.insert(0, str(_ROOT / "src"))

import argparse
import csv
import json
import math
from pathlib import Path


def _fmt(v):
    if isinstance(v, float):
        return "NA" if math.isnan(v) else f"{v:.4f}"
    return str(v)


def _summary_row(path: Path, payload: dict) -> dict:
    cal = payload.get("calibration", {})
    sel = payload.get("selective", {})
    return {
        "run": path.stem,
        "dataset": payload.get("dataset", ""),
        "n": payload.get("n_evaluated", ""),
        "accuracy_percent": payload.get("soft_accuracy_percent", ""),
        "full_credit_accuracy_percent": payload.get("full_credit_accuracy_percent", ""),
        "ece": cal.get("ece", ""),
        "brier": cal.get("brier", ""),
        "aurc": cal.get("aurc", ""),
        "coverage": sel.get("coverage", ""),
        "selective_accuracy": sel.get("selective_accuracy", ""),
        "selective_risk": sel.get("selective_risk", ""),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate manuscript tables/plots strictly from recorded evaluation JSON files")
    ap.add_argument("--metrics", nargs="+", required=True, help="Evaluation JSON files from scripts/evaluate.py")
    ap.add_argument("--tables-dir", default="tables")
    ap.add_argument("--figures-dir", default="figures")
    args = ap.parse_args()

    metrics_paths = [Path(x) for x in args.metrics]
    payloads = [(p, json.loads(p.read_text(encoding="utf-8"))) for p in metrics_paths]
    rows = [_summary_row(p, x) for p, x in payloads]
    tables = Path(args.tables_dir); figures = Path(args.figures_dir)
    tables.mkdir(parents=True, exist_ok=True); figures.mkdir(parents=True, exist_ok=True)

    cols = list(rows[0].keys())
    with (tables / "main_results.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)

    md = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    md += ["| " + " | ".join(_fmt(r[c]) for c in cols) + " |" for r in rows]
    (tables / "main_results.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    latex = ["\\begin{tabular}{" + "l" * len(cols) + "}", "\\hline", " & ".join(cols) + r" \\", "\\hline"]
    latex += [" & ".join(_fmt(r[c]) for c in cols) + r" \\" for r in rows]
    latex += ["\\hline", "\\end{tabular}"]
    (tables / "main_results.tex").write_text("\n".join(latex) + "\n", encoding="utf-8")

    # Plot only when real per-question confidence/correctness records are present.
    import matplotlib.pyplot as plt
    from paper4_kbvqa.evaluation.metrics import risk_coverage_curve

    plotted = False
    plt.figure()
    for path, payload in payloads:
        pq = [x for x in payload.get("per_question", []) if x.get("confidence") is not None]
        if len(pq) < 2:
            continue
        conf = [float(x["confidence"]) for x in pq]
        corr = [int(x["correct_full_credit"]) for x in pq]
        coverage, risk = risk_coverage_curve(conf, corr)
        plt.plot(coverage, risk, label=path.stem)
        plotted = True
    if plotted:
        plt.xlabel("Coverage")
        plt.ylabel("Selective risk")
        plt.legend()
        plt.tight_layout()
        plt.savefig(figures / "risk_coverage.pdf")
        plt.savefig(figures / "risk_coverage.png", dpi=600)
    plt.close()

    print(f"Generated tables from {len(rows)} recorded evaluation file(s).")
    if plotted:
        print("Generated risk-coverage PDF and 600-dpi PNG from per-question records.")
    else:
        print("No risk-coverage figure generated: insufficient per-question confidence records.")


if __name__ == "__main__":
    main()
