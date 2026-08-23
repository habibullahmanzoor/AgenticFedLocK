"""Stage 1 + Stage 2 sweep driver: 4 datasets x 3 seeds x 5-method suite.

Stage 1 (master plan): MNIST + Fashion-MNIST, 3 seeds, full 5-method suite.
Stage 2 (master plan): EMNIST Balanced + CIFAR-10, 3 seeds, full 5-method suite.
Run together (seed-major order) rather than gated stage-by-stage, per the
decision to pull Stage 2 forward given CIFAR-10's Stage 0 result.

Per-dataset hyperparameters are the already-validated ones from the recent
five-dataset baseline extension runs (`revision 1/e17-e19_*`) and the Stage 0
CIFAR-10 gate run, with data_fraction overridden to 0.5 per the restart
decision (50% data on all datasets, for faster iteration).

Resumable: skips any (dataset, seed, strategy) whose summary file already
exists in its target folder, so it can be re-launched after an interruption
or a partial manual run without recomputing finished work.
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
STRATEGIES = ["fedavg", "fltrust", "flame", "rfa", "agentlock"]
SEEDS = [7, 11, 19]

# Folder-name suffix per attack mode. model_replacement maps to "" (no suffix)
# to stay compatible with runs/ already produced by earlier invocations of this
# script; any other attack mode gets its own suffix so it can run against a
# disjoint set of output folders (e.g. in parallel with a model_replacement
# sweep already in progress, sharing the same GPU without file collisions).
ATTACK_FOLDER_SUFFIX = {
    "model_replacement": "",
    "adaptive_mimic": "_adaptivemimic",
}

COMMON_ARGS_BASE = [
    "--device", "cuda",
    "--dirichlet-alpha", "0.4",
    "--malicious-fraction", "0.2",
    "--poison-fraction", "0.3",
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
        ],
    },
}

# Fast-to-slow within each seed pass so quick feedback (Stage 1) lands first.
DATASET_ORDER = ["mnist", "fashion", "emnist", "cifar10"]


def run_one(dataset_key: str, seed: int, strategy: str, attack_mode: str) -> None:
    suffix = ATTACK_FOLDER_SUFFIX[attack_mode]
    out_dir = RUNS_DIR / f"{dataset_key}{suffix}_seed{seed}_{strategy}"
    summary_path = out_dir / f"{strategy}_summary.json"
    if summary_path.exists():
        print(f"[skip] {dataset_key} seed={seed} strategy={strategy} attack={attack_mode} (already done)", flush=True)
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    spec = DATASETS[dataset_key]
    cmd = [
        PYTHON, str(REPO_ROOT / "run_digits_experiment.py"),
        "--strategy", strategy,
        "--dataset", spec["dataset"],
        "--seed", str(seed),
        "--results-dir", str(out_dir),
        "--attack-mode", attack_mode,
        *COMMON_ARGS_BASE,
        *spec["args"],
    ]
    if strategy == "agentlock":
        cmd += [
            "--repair-signal-mode", "subspace_consensus", "--trust-consensus-fraction", "0.5",
            "--root-anchor-weight", "0.7",
        ]

    print(f"[start] {dataset_key} seed={seed} strategy={strategy} attack={attack_mode}", flush=True)
    started = time.time()
    with open(out_dir / "stdout.log", "w", encoding="utf-8") as stdout_f, \
         open(out_dir / "stderr.log", "w", encoding="utf-8") as stderr_f:
        result = subprocess.run(cmd, cwd=str(REPO_ROOT), stdout=stdout_f, stderr=stderr_f)
    elapsed = time.time() - started

    if result.returncode != 0:
        print(f"[FAIL] {dataset_key} seed={seed} strategy={strategy} attack={attack_mode} "
              f"exit={result.returncode} elapsed={elapsed:.0f}s -- see {out_dir}/stderr.log", flush=True)
        return

    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        asr = summary.get("final_asr")
        clean = summary.get("final_clean_accuracy")
        print(f"[done] {dataset_key} seed={seed} strategy={strategy} attack={attack_mode} "
              f"elapsed={elapsed:.0f}s ASR={asr} clean={clean}", flush=True)
    except Exception as exc:  # noqa: BLE001 -- best-effort progress line only
        print(f"[done, no summary readback] {dataset_key} seed={seed} strategy={strategy} attack={attack_mode} "
              f"elapsed={elapsed:.0f}s ({exc})", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack-mode", choices=sorted(ATTACK_FOLDER_SUFFIX), default="model_replacement")
    args = parser.parse_args()

    total = len(SEEDS) * len(DATASET_ORDER) * len(STRATEGIES)
    done = 0
    for seed in SEEDS:
        for dataset_key in DATASET_ORDER:
            for strategy in STRATEGIES:
                done += 1
                print(f"=== [{done}/{total}] ({args.attack_mode}) ===", flush=True)
                run_one(dataset_key, seed, strategy, args.attack_mode)
    print(f"=== stage1+2 sweep complete ({args.attack_mode}) ===", flush=True)


if __name__ == "__main__":
    main()
