"""Component + role-separation ablation, CIFAR-10 model replacement.

Answers reviewer Major Concern 1 (novelty beyond engineering integration):
does the full controller actually need all its parts, or would a simpler
subset do as well? Five ablation strategies vs the full `agentlock`
controller, all under the exact same v2 protocol used in the Stage 1+2 sweep
(50% data, unknown defender, 3 seeds).

Run on CIFAR-10 model-replacement rather than v1's choice of CIFAR-10
adaptive-mimic: that cell is a "nobody wins" regime for every method
(~87-91% ASR across the board, agentlock included, per the Stage 1+2 sweep),
so there is nothing for a component ablation to meaningfully decompose there.
Model-replacement is where the full controller actually achieves something
(26% vs baselines' 90%+), which is where "which component contributes what"
is an answerable question.

Component ablation: static_subspace, trust_only, communication_only vs the
already-completed fedavg/agentlock numbers from the Stage 1+2 sweep.
Role-separation ablation: selection_only, allocation_repair_only vs the same
agentlock reference.

Only `agentlock` and `allocation_repair_only` have enable_repair=True in
their StrategySpec (see agentlock/simulation.py:816-862), so only those two
get --repair-signal-mode subspace_consensus and --root-anchor-weight 0.7;
the others never touch repair and stay at config defaults.

Resumable: skips any (strategy, seed) whose summary already exists.
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

# Ablation strategies not already covered by the Stage 1+2 sweep (fedavg and
# agentlock for this exact cell already exist in runs/cifar10_seed{N}_{...}).
STRATEGIES = ["static_subspace", "trust_only", "communication_only", "selection_only", "allocation_repair_only"]
REPAIR_ENABLED_STRATEGIES = {"allocation_repair_only"}  # plus "agentlock", already done

CIFAR10_ARGS = [
    "--num-clients", "30", "--rounds", "100", "--client-fraction", "0.33",
    "--local-epochs", "3", "--batch-size", "64", "--lr", "0.001",
    "--lr-schedule", "cosine", "--lr-min", "0.00005", "--optimizer", "adam",
    "--model-family", "cnn", "--channel-dims", "32,64", "--cnn-hidden-dim", "128",
    "--cnn-variant", "residual", "--image-augmentation",
    "--backdoor-target", "2", "--rare-labels", "0,1", "--probe-fraction", "0.08",
]

COMMON_ARGS = [
    "--device", "cuda",
    "--dirichlet-alpha", "0.4",
    "--malicious-fraction", "0.2",
    "--poison-fraction", "0.3",
    "--attack-mode", "model_replacement",
    "--data-fraction", "0.5",
    "--defender-knowledge", "unknown",
    "--rarity-information-mode", "probe_feedback",
    "--min-client-size", "18",
    "--test-fraction", "0.25",
    "--fedprox-mu", "0.01",
    "--skip-baseline",
]


def run_one(strategy: str, seed: int) -> None:
    out_dir = RUNS_DIR / f"cifar10_seed{seed}_{strategy}"
    summary_path = out_dir / f"{strategy}_summary.json"
    if summary_path.exists():
        print(f"[skip] strategy={strategy} seed={seed} (already done)", flush=True)
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        PYTHON, str(REPO_ROOT / "run_digits_experiment.py"),
        "--strategy", strategy,
        "--dataset", "cifar10_local",
        "--seed", str(seed),
        "--results-dir", str(out_dir),
        *COMMON_ARGS,
        *CIFAR10_ARGS,
    ]
    if strategy in REPAIR_ENABLED_STRATEGIES:
        cmd += [
            "--repair-signal-mode", "subspace_consensus", "--trust-consensus-fraction", "0.5",
            "--root-anchor-weight", "0.7",
        ]

    print(f"[start] strategy={strategy} seed={seed}", flush=True)
    started = time.time()
    with open(out_dir / "stdout.log", "w", encoding="utf-8") as stdout_f, \
         open(out_dir / "stderr.log", "w", encoding="utf-8") as stderr_f:
        result = subprocess.run(cmd, cwd=str(REPO_ROOT), stdout=stdout_f, stderr=stderr_f)
    elapsed = time.time() - started

    if result.returncode != 0:
        print(f"[FAIL] strategy={strategy} seed={seed} exit={result.returncode} "
              f"elapsed={elapsed:.0f}s -- see {out_dir}/stderr.log", flush=True)
        return

    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        print(f"[done] strategy={strategy} seed={seed} elapsed={elapsed:.0f}s "
              f"ASR={summary.get('final_asr')} clean={summary.get('final_clean_accuracy')} "
              f"rare={summary.get('final_rare_accuracy')} bytes={summary.get('mean_byte_communication_ratio')}",
              flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[done, no summary readback] strategy={strategy} seed={seed} "
              f"elapsed={elapsed:.0f}s ({exc})", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=SEEDS,
        help="Restrict this process to a subset of seeds, so multiple instances "
             "can run different seeds concurrently without colliding (each "
             "(strategy, seed) combo is still individually skip-if-exists safe).",
    )
    args = parser.parse_args()
    seeds = args.seeds

    total = len(seeds) * len(STRATEGIES)
    done = 0
    for seed in seeds:
        for strategy in STRATEGIES:
            done += 1
            print(f"=== [{done}/{total}] (seeds={seeds}) ===", flush=True)
            run_one(strategy, seed)
    print(f"=== ablation sweep complete (seeds={seeds}) ===", flush=True)


if __name__ == "__main__":
    main()
