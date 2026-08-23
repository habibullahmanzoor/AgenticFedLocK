"""Benign control (no attackers), FedAvg vs AgenticFedLock, all datasets.

v1's "Stage I: Benign Control" established that the defense doesn't cost
accuracy/utility when there's no attack to defend against. Neither fedavg
nor agentlock has an existing malicious_fraction=0.0 run (the Stage 1+2
sweep and every ablation battery this session used malicious_fraction=0.2
throughout), so both need fresh runs. ASR is undefined with no attackers and
is not reported.

CIFAR-10 was covered first (6 runs, already complete). This adds MNIST,
Fashion-MNIST, and EMNIST Balanced (18 more runs); FEMNIST is excluded from
this expansion since it dominates wall-clock cost for the remaining benefit
(~64 min/run at reduced scale vs a few minutes/run for the others).
Per-dataset hyperparameters are the same protocol values used by
`run_stage1_stage2.py` (Table tab:exp-setup in the paper), just with
malicious_fraction forced to 0.0.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = Path(__file__).resolve().parent / "runs"
PYTHON = sys.executable
SEEDS = [7, 11, 19]
STRATEGIES = ["fedavg", "agentlock"]

COMMON_ARGS = [
    "--device", "cuda",
    "--dirichlet-alpha", "0.4",
    "--malicious-fraction", "0.0",
    "--poison-fraction", "0.3",
    "--attack-mode", "model_replacement",  # moot with 0 malicious clients
    "--data-fraction", "0.5",
    "--defender-knowledge", "unknown",
    "--rarity-information-mode", "probe_feedback",
    "--min-client-size", "18",
    "--test-fraction", "0.25",
    "--skip-baseline",
]

DATASETS: dict[str, dict] = {
    "mnist": {
        "dataset": "mnist_local",
        "args": [
            "--num-clients", "20", "--rounds", "30", "--client-fraction", "0.5",
            "--local-epochs", "1", "--batch-size", "32", "--lr", "0.05",
            "--lr-schedule", "cosine", "--lr-min", "0.005", "--optimizer", "sgd",
            "--model-family", "cnn", "--channel-dims", "16,32", "--cnn-hidden-dim", "64",
            "--cnn-variant", "basic", "--backdoor-target", "0", "--rare-labels", "4,5",
            "--probe-fraction", "0.12",
        ],
    },
    "fashion": {
        "dataset": "fashion_mnist_local",
        "args": [
            "--num-clients", "20", "--rounds", "30", "--client-fraction", "0.5",
            "--local-epochs", "1", "--batch-size", "32", "--lr", "0.035",
            "--lr-schedule", "cosine", "--lr-min", "0.0035", "--optimizer", "sgd",
            "--model-family", "cnn", "--channel-dims", "16,32", "--cnn-hidden-dim", "96",
            "--cnn-variant", "basic", "--backdoor-target", "2", "--rare-labels", "0,1",
            "--probe-fraction", "0.12",
        ],
    },
    "emnist": {
        "dataset": "emnist_balanced_local",
        "args": [
            "--num-clients", "40", "--rounds", "40", "--client-fraction", "0.35",
            "--local-epochs", "1", "--batch-size", "32", "--lr", "0.001",
            "--lr-schedule", "cosine", "--lr-min", "0.0001", "--optimizer", "adam",
            "--model-family", "cnn", "--channel-dims", "48,96", "--cnn-hidden-dim", "256",
            "--cnn-variant", "basic", "--backdoor-target", "2", "--rare-labels", "0,1",
            "--probe-fraction", "0.08",
        ],
    },
    "cifar10": {
        "dataset": "cifar10_local",
        "args": [
            "--num-clients", "30", "--rounds", "100", "--client-fraction", "0.33",
            "--local-epochs", "3", "--batch-size", "64", "--lr", "0.001",
            "--lr-schedule", "cosine", "--lr-min", "0.00005", "--optimizer", "adam",
            "--model-family", "cnn", "--channel-dims", "32,64", "--cnn-hidden-dim", "128",
            "--cnn-variant", "residual", "--image-augmentation",
            "--backdoor-target", "2", "--rare-labels", "0,1", "--probe-fraction", "0.08",
            "--fedprox-mu", "0.01",
        ],
    },
}

# Fast-to-slow so quick feedback lands first; cifar10 already done, kept last.
DATASET_ORDER = ["mnist", "fashion", "emnist", "cifar10"]


def run_one(dataset_key: str, strategy: str, seed: int) -> None:
    out_dir = RUNS_DIR / f"{dataset_key}_seed{seed}_benign_{strategy}"
    summary_path = out_dir / f"{strategy}_summary.json"
    if summary_path.exists():
        print(f"[skip] {dataset_key} strategy={strategy} seed={seed} (already done)", flush=True)
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    spec = DATASETS[dataset_key]
    cmd = [
        PYTHON, str(REPO_ROOT / "run_digits_experiment.py"),
        "--strategy", strategy,
        "--dataset", spec["dataset"],
        "--seed", str(seed),
        "--results-dir", str(out_dir),
        *COMMON_ARGS,
        *spec["args"],
    ]
    if strategy == "agentlock":
        cmd += [
            "--repair-signal-mode", "subspace_consensus", "--trust-consensus-fraction", "0.5",
            "--root-anchor-weight", "0.7",
        ]

    print(f"[start] {dataset_key} strategy={strategy} seed={seed}", flush=True)
    started = time.time()
    with open(out_dir / "stdout.log", "w", encoding="utf-8") as stdout_f, \
         open(out_dir / "stderr.log", "w", encoding="utf-8") as stderr_f:
        result = subprocess.run(cmd, cwd=str(REPO_ROOT), stdout=stdout_f, stderr=stderr_f)
    elapsed = time.time() - started

    if result.returncode != 0:
        print(f"[FAIL] {dataset_key} strategy={strategy} seed={seed} exit={result.returncode} "
              f"elapsed={elapsed:.0f}s -- see {out_dir}/stderr.log", flush=True)
        return

    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        print(f"[done] {dataset_key} strategy={strategy} seed={seed} elapsed={elapsed:.0f}s "
              f"clean={summary.get('final_clean_accuracy')} rare={summary.get('final_rare_accuracy')} "
              f"bytes={summary.get('mean_byte_communication_ratio')}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[done, no summary readback] {dataset_key} strategy={strategy} seed={seed} elapsed={elapsed:.0f}s ({exc})", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--datasets", nargs="+", default=DATASET_ORDER, choices=list(DATASETS))
    args = parser.parse_args()
    seeds = args.seeds
    datasets = args.datasets

    total = len(seeds) * len(datasets) * len(STRATEGIES)
    done = 0
    for seed in seeds:
        for dataset_key in datasets:
            for strategy in STRATEGIES:
                done += 1
                print(f"=== [{done}/{total}] (seeds={seeds}, datasets={datasets}) ===", flush=True)
                run_one(dataset_key, strategy, seed)
    print(f"=== benign control complete (seeds={seeds}, datasets={datasets}) ===", flush=True)


if __name__ == "__main__":
    main()
