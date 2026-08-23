"""Generate new v2-accurate figures for the Experimental Evaluation section.

All existing figures/*.pdf are v1-era (stale mechanism, stale numbers) and are
deliberately not reused. These four figures are built directly from completed
v2 run CSVs (round_metrics) or from the exact 3-seed aggregates already
verified against those CSVs earlier in this revision (Core Comparison /
Component Ablation tables in Full paper.tex).
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RUNS = Path(__file__).resolve().parent / "runs"
FIGS = Path(__file__).resolve().parent / "figures"
FIGS.mkdir(exist_ok=True)

SEEDS = [7, 11, 19]
METHOD_ORDER = ["fedavg", "fltrust", "flame", "rfa", "agentlock"]
METHOD_LABEL = {
    "fedavg": "FedAvg", "fltrust": "FLTrust", "flame": "FLAME",
    "rfa": "RFA", "agentlock": "AgenticFedLock",
}
METHOD_COLOR = {
    "fedavg": "#888888", "fltrust": "#1f77b4", "flame": "#ff7f0e",
    "rfa": "#2ca02c", "agentlock": "#d62728",
}


def read_round_metrics(path: Path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def series_by_round(dataset: str, method: str, seeds, field: str, attack: str = "modelreplacement"):
    """Return dict round -> list of values across seeds."""
    per_round: dict[int, list[float]] = {}
    for seed in seeds:
        if attack == "modelreplacement":
            d = RUNS / f"{dataset}_seed{seed}_{method}"
        else:
            d = RUNS / f"{dataset}_adaptivemimic_seed{seed}_{method}"
        f = d / f"{method}_round_metrics.csv"
        if not f.exists():
            continue
        for row in read_round_metrics(f):
            r = int(row["round"])
            per_round.setdefault(r, []).append(float(row[field]))
    return per_round


def mean_std_curve(per_round: dict[int, list[float]]):
    rounds = sorted(per_round.keys())
    means = np.array([np.mean(per_round[r]) for r in rounds])
    stds = np.array([np.std(per_round[r]) for r in rounds])
    return np.array(rounds), means, stds


# ---------------------------------------------------------------------------
# Figure 1: Training/attack dynamics curves (EMNIST + CIFAR-10)
# ---------------------------------------------------------------------------
def _training_dynamics(attack: str, attack_label: str, out_name: str):
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.4), sharex="col")
    datasets = ["emnist", "cifar10"]
    titles = ["EMNIST Balanced", "CIFAR-10"]

    for col, (dataset, title) in enumerate(zip(datasets, titles)):
        for method in METHOD_ORDER:
            per_round_asr = series_by_round(dataset, method, SEEDS, "asr", attack=attack)
            if not per_round_asr:
                continue
            rounds, mean_asr, std_asr = mean_std_curve(per_round_asr)
            mean_pct, std_pct = mean_asr * 100, std_asr * 100
            lo = np.clip(mean_pct - std_pct, 0, 100)
            hi = np.clip(mean_pct + std_pct, 0, 100)
            axes[0, col].fill_between(rounds, lo, hi, color=METHOD_COLOR[method], alpha=0.15, linewidth=0,
                                       zorder=4 if method == "agentlock" else 2)
            axes[0, col].plot(
                rounds, mean_pct, label=METHOD_LABEL[method],
                color=METHOD_COLOR[method],
                linewidth=2.2 if method == "agentlock" else 1.4,
                zorder=5 if method == "agentlock" else 3,
            )

            per_round_clean = series_by_round(dataset, method, SEEDS, "clean_accuracy", attack=attack)
            rounds_c, mean_clean, std_clean = mean_std_curve(per_round_clean)
            mean_c_pct, std_c_pct = mean_clean * 100, std_clean * 100
            lo_c = np.clip(mean_c_pct - std_c_pct, 0, 100)
            hi_c = np.clip(mean_c_pct + std_c_pct, 0, 100)
            axes[1, col].fill_between(rounds_c, lo_c, hi_c, color=METHOD_COLOR[method], alpha=0.15, linewidth=0,
                                       zorder=4 if method == "agentlock" else 2)
            axes[1, col].plot(
                rounds_c, mean_c_pct, label=METHOD_LABEL[method],
                color=METHOD_COLOR[method],
                linewidth=2.2 if method == "agentlock" else 1.4,
                zorder=5 if method == "agentlock" else 3,
            )

        axes[0, col].set_title(f"{title} ({attack_label})", fontsize=11)
        axes[0, col].set_ylabel("Attack success rate (%)")
        axes[0, col].set_ylim(-3, 103)
        axes[0, col].grid(alpha=0.25)

        axes[1, col].set_xlabel("Communication round")
        axes[1, col].set_ylabel("Clean accuracy (%)")
        axes[1, col].grid(alpha=0.25)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Training and attack dynamics ({attack_label}), 3-seed mean", fontsize=12)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    out = FIGS / out_name
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def fig_training_dynamics():
    _training_dynamics("modelreplacement", "model replacement", "v2_training_dynamics.pdf")


def fig_adaptive_mimic_dynamics():
    _training_dynamics("adaptivemimic", "adaptive mimic", "v2_adaptive_mimic_dynamics.pdf")


# ---------------------------------------------------------------------------
# Figure 2: Core comparison bar chart (model replacement, ASR by dataset x method)
# ---------------------------------------------------------------------------
CORE_MR = {
    "MNIST":        {"fedavg": (99.93, 0.12), "fltrust": (0.98, 0.08), "flame": (0.80, 0.09), "rfa": (60.69, 38.34), "agentlock": (1.13, 0.31)},
    "Fashion-MNIST": {"fedavg": (99.19, 1.03), "fltrust": (7.39, 4.56), "flame": (10.76, 9.32), "rfa": (19.57, 11.46), "agentlock": (8.61, 4.91)},
    "EMNIST":       {"fedavg": (99.99, 0.01), "fltrust": (91.51, 2.81), "flame": (98.90, 0.70), "rfa": (99.84, 0.07), "agentlock": (15.61, 6.24)},
    "CIFAR-10":     {"fedavg": (95.42, 1.40), "fltrust": (90.33, 0.05), "flame": (92.84, 1.49), "rfa": (91.26, 3.34), "agentlock": (26.07, 17.20)},
}


CORE_AM = {
    "MNIST":        {"fedavg": (47.85, 37.01), "fltrust": (1.06, 0.12), "flame": (1.16, 0.29), "rfa": (37.42, 28.97), "agentlock": (3.47, 1.22)},
    "Fashion-MNIST": {"fedavg": (15.29, 9.86), "fltrust": (7.52, 4.56), "flame": (5.29, 7.57), "rfa": (10.63, 5.90), "agentlock": (5.04, 6.53)},
    "EMNIST":       {"fedavg": (99.45, 0.09), "fltrust": (93.62, 1.04), "flame": (99.64, 0.14), "rfa": (99.31, 0.12), "agentlock": (97.12, 0.39)},
    "CIFAR-10":     {"fedavg": (87.73, 4.74), "fltrust": (87.40, 1.96), "flame": (91.06, 2.32), "rfa": (87.43, 4.53), "agentlock": (90.40, 1.00)},
}


def _bar_chart(data, title, out_name):
    datasets = list(data.keys())
    x = np.arange(len(datasets))
    width = 0.16

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    for i, method in enumerate(METHOD_ORDER):
        means = [data[d][method][0] for d in datasets]
        stds = [data[d][method][1] for d in datasets]
        offset = (i - (len(METHOD_ORDER) - 1) / 2) * width
        ax.bar(x + offset, means, width, yerr=stds, capsize=2.5,
               label=METHOD_LABEL[method], color=METHOD_COLOR[method])

    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("Attack success rate (%)")
    ax.set_title(title)
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    out = FIGS / out_name
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def fig_core_comparison_bars():
    _bar_chart(CORE_MR, "Core comparison: model replacement, 3-seed mean $\\pm$ std", "v2_core_comparison_bars.pdf")


def fig_adaptive_mimic_bars():
    _bar_chart(CORE_AM, "Core comparison: adaptive mimic, 3-seed mean $\\pm$ std", "v2_adaptive_mimic_bars.pdf")


# ---------------------------------------------------------------------------
# Figure 3: Component-ablation bar chart
# ---------------------------------------------------------------------------
COMPONENT_ABLATION = [
    ("FedAvg\n(undefended)", 95.42, 1.40, "#888888"),
    ("Fixed masking\nonly", 95.38, 1.45, "#9ecae1"),
    ("Trust-weighted\naggr. only", 95.35, 1.55, "#9ecae1"),
    ("Aggressive\nmasking only", 95.28, 1.98, "#9ecae1"),
    ("Bandit\nselection only", 95.29, 0.62, "#9ecae1"),
    ("Allocator +\nJudge-Repair", 35.25, 3.98, "#fdae6b"),
    ("AgenticFedLock\n(full)", 26.07, 17.20, "#d62728"),
]


def fig_component_ablation_bars():
    labels = [c[0] for c in COMPONENT_ABLATION]
    means = [c[1] for c in COMPONENT_ABLATION]
    stds = [c[2] for c in COMPONENT_ABLATION]
    colors = [c[3] for c in COMPONENT_ABLATION]

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, capsize=3, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("Attack success rate (%)")
    ax.set_title("Component and role-separation ablation, CIFAR-10 model replacement")
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    out = FIGS / "v2_component_ablation_bars.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


# ---------------------------------------------------------------------------
# Figure 4: Repair-firing trace, unknown vs exact defender knowledge (seed 7)
# ---------------------------------------------------------------------------
def fig_repair_firing_trace():
    fig, ax = plt.subplots(figsize=(7.5, 3.8))

    unknown_dir = RUNS / "cifar10_seed7_agentlock"
    exact_dir = RUNS / "cifar10_seed7_agentlock_exactdefender"

    for d, label, color in [
        (unknown_dir, "Unknown defender knowledge (standard)", "#d62728"),
        (exact_dir, "Exact defender knowledge (oracle)", "#1f77b4"),
    ]:
        f = d / "agentlock_round_metrics.csv"
        rows = read_round_metrics(f)
        rounds = [int(r["round"]) for r in rows]
        repair_count = [int(r["repair_count"]) for r in rows]
        ax.plot(rounds, repair_count, label=label, color=color, linewidth=1.6, marker=".", markersize=3)

    ax.set_xlabel("Communication round")
    ax.set_ylabel("Repair count that round")
    ax.set_title("Repair fires in the large majority of rounds under both defender-knowledge settings\n(CIFAR-10 model replacement, seed 7)", fontsize=10.5)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=8.5)
    fig.tight_layout()
    out = FIGS / "v2_repair_firing_trace.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


# ---------------------------------------------------------------------------
# Figure 5: Malicious-client attribution quality (suspicion-score AUC)
# ---------------------------------------------------------------------------
def _pairwise_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels]
    neg = scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float((diff > 0).mean() + 0.5 * (diff == 0).mean())


def _load_decisions(dataset: str, attack: str, seeds=SEEDS):
    import pandas as pd
    frames = []
    for seed in seeds:
        if attack == "modelreplacement":
            p = RUNS / f"{dataset}_seed{seed}_agentlock" / "agentlock_client_decisions.csv"
        else:
            p = RUNS / f"{dataset}_adaptivemimic_seed{seed}_agentlock" / "agentlock_client_decisions.csv"
        if p.exists():
            frames.append(pd.read_csv(p))
    if not frames:
        return None
    import pandas as pd
    return pd.concat(frames, ignore_index=True)


def fig_attribution_auc():
    import pandas as pd

    datasets = ["mnist", "fashion", "emnist", "cifar10"]
    dataset_labels = ["MNIST", "Fashion-MNIST", "EMNIST", "CIFAR-10"]

    mr_auc, am_auc = [], []
    for dataset in datasets:
        df_mr = _load_decisions(dataset, "modelreplacement")
        late_mr = df_mr[df_mr["round"] >= 15]
        mr_auc.append(_pairwise_auc(late_mr["suspicion_post"].to_numpy(), late_mr["is_malicious"].to_numpy().astype(bool)))

        df_am = _load_decisions(dataset, "adaptivemimic")
        late_am = df_am[df_am["round"] >= 15]
        am_auc.append(_pairwise_auc(late_am["suspicion_post"].to_numpy(), late_am["is_malicious"].to_numpy().astype(bool)))

    pooled = []
    for dataset in datasets:
        d = _load_decisions(dataset, "modelreplacement")
        pooled.append(d)
    pooled_df = pd.concat(pooled, ignore_index=True)
    buckets = [(1, 5), (6, 10), (11, 20), (21, 50), (51, 100)]
    bucket_mid = []
    bucket_auc = []
    for lo, hi in buckets:
        b = pooled_df[(pooled_df["round"] >= lo) & (pooled_df["round"] <= hi)]
        bucket_mid.append((lo + hi) / 2)
        bucket_auc.append(_pairwise_auc(b["suspicion_post"].to_numpy(), b["is_malicious"].to_numpy().astype(bool)))

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

    x = np.arange(len(datasets))
    width = 0.32
    axes[0].bar(x - width / 2, mr_auc, width, label="Model replacement", color="#d62728")
    axes[0].bar(x + width / 2, am_auc, width, label="Adaptive mimic", color="#1f77b4")
    axes[0].axhline(0.5, color="black", linestyle="--", linewidth=1, alpha=0.6)
    axes[0].text(len(datasets) - 0.5, 0.52, "chance", fontsize=8, ha="right", va="bottom", alpha=0.7)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(dataset_labels, fontsize=9)
    axes[0].set_ylabel("Suspicion-score AUC\n(malicious vs. benign, round $\\geq$15)")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("(a) Attribution quality by attack type", fontsize=10.5)
    axes[0].legend(loc="lower left", fontsize=8.5, frameon=False)
    axes[0].grid(alpha=0.25, axis="y")

    axes[1].plot(bucket_mid, bucket_auc, marker="o", color="#d62728", linewidth=2)
    axes[1].axhline(0.5, color="black", linestyle="--", linewidth=1, alpha=0.6)
    axes[1].set_xlabel("Communication round")
    axes[1].set_ylabel("Suspicion-score AUC")
    axes[1].set_ylim(0.45, 1.05)
    axes[1].set_title("(b) Detection latency, model replacement\n(pooled across 4 datasets, 3 seeds)", fontsize=10.5)
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    out = FIGS / "v2_attribution_auc.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")
    print("mr_auc", dict(zip(dataset_labels, mr_auc)))
    print("am_auc", dict(zip(dataset_labels, am_auc)))
    print("bucket_auc", list(zip(buckets, bucket_auc)))


if __name__ == "__main__":
    fig_training_dynamics()
    fig_adaptive_mimic_dynamics()
    fig_core_comparison_bars()
    fig_adaptive_mimic_bars()
    fig_component_ablation_bars()
    fig_repair_firing_trace()
    fig_attribution_auc()
