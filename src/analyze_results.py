import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def bootstrap_ci(values, n_boot=1000, alpha=0.05, seed=42):
    rng = np.random.default_rng(seed)
    values = np.array(values)
    if len(values) == 0:
        return (0.0, 0.0)
    means = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        means.append(sample.mean())
    lower = np.percentile(means, 100 * (alpha / 2))
    upper = np.percentile(means, 100 * (1 - alpha / 2))
    return (float(lower), float(upper))


def plot_risk_coverage(df: pd.DataFrame, dataset: str, out_dir: Path) -> None:
    plt.figure(figsize=(6, 4))
    for method in sorted(df["method"].unique()):
        sub = df[df["method"] == method].copy()
        sub = sub.sort_values("coverage")
        plt.plot(sub["coverage"], sub["risk"], marker="o", label=method)
    plt.xlabel("Coverage")
    plt.ylabel("Risk (1 - accuracy on answered)")
    plt.title(f"Risk-Coverage: {dataset}")
    plt.legend()
    out_path = out_dir / f"risk_coverage_{dataset}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results/abstention_experiment")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    outputs_path = results_dir / "outputs.jsonl"
    analysis_path = results_dir / "analysis.json"
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with outputs_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    df = pd.DataFrame(rows)

    summary = {}
    for dataset in sorted(df["dataset"].unique()):
        summary[dataset] = {}
        df_dataset = df[df["dataset"] == dataset]
        for method in sorted(df_dataset["method"].unique()):
            df_method = df_dataset[df_dataset["method"] == method]
            for threshold in sorted(df_method["threshold"].astype(str).unique()):
                df_thr = df_method[df_method["threshold"].astype(str) == threshold]
                answered = df_thr[~df_thr["abstained"]]
                accuracy_answered = answered["correct"].mean() if len(answered) else 0.0
                risk = 1.0 - accuracy_answered if len(answered) else 1.0
                coverage = len(answered) / len(df_thr) if len(df_thr) else 0.0
                ci_low, ci_high = bootstrap_ci(answered["correct"].astype(float).values)
                summary[dataset].setdefault(method, {})[threshold] = {
                    "coverage": coverage,
                    "risk": risk,
                    "accuracy_answered": accuracy_answered,
                    "accuracy_ci": [ci_low, ci_high],
                    "n": len(df_thr),
                }

    with analysis_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    plot_rows = []
    for dataset, methods in summary.items():
        for method, thresholds in methods.items():
            for threshold, metrics in thresholds.items():
                plot_rows.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "threshold": threshold,
                        "coverage": metrics["coverage"],
                        "risk": metrics["risk"],
                    }
                )
    plot_df = pd.DataFrame(plot_rows)
    for dataset in plot_df["dataset"].unique():
        plot_risk_coverage(plot_df[plot_df["dataset"] == dataset], dataset, plots_dir)


if __name__ == "__main__":
    main()
