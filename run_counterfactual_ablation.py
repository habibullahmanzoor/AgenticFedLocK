"""Simulator-oracle counterfactual ablation, CIFAR-10 adaptive mimic.

Answers Major Concern 4 (`commetns of reviewers.txt`): "The paper claims to
evaluate alternative tiers for selected clients and even unselected clients
without additional communication... The authors must clarify whether these
counterfactual rewards are actually computed, approximated, or simulated,
and whether the extra computation/communication is counted."

v1's response (`answers to reviewers comments.tex`, Major Concern 4)
distinguished the deployable default (realised feedback from selected
clients only, counterfactual_mode=none) from a separately-labelled
simulator-oracle upper bound that centrally retrains simulated clients and
costs extra local-training-equivalents. Rerun under v2 for the same
distinction.

IMPORTANT, discovered after the first (broken) run of this script: the
counterfactual mechanism (`agentlock/simulation.py`, `counterfactual_enabled`
/ `selection_counterfactual_enabled`) only activates when
`config.attack_mode in {"badnets", "adaptive_mimic"}`. The original version
of this script ran under `--attack-mode model_replacement`, under which the
mechanism never engages at all (`mean_selection_counterfactual_gap`,
`selection_counterfactual_win_rate` were exactly 0.0 in all 3 seeds, and
`total_local_training_jobs` was byte-identical to the `none` baseline) --
those 3 runs measured a no-op, not the mechanism, and are not used in any
reported table. This corrected version uses `--attack-mode adaptive_mimic`,
one of the two modes the mechanism actually supports.

`counterfactual_mode=none` is the standing default and already exists as
`cifar10_adaptivemimic_seed{N}_agentlock` (attack_mode=adaptive_mimic,
counterfactual_mode=none, confirmed) -- reused, not rerun. Only
simulator_oracle needs fresh runs, written to
`cifar10_adaptivemimic_seed{N}_counterfactual_oracle`.
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
    "--attack-mode", "adaptive_mimic",
    "--data-fraction", "0.5",
    "--defender-knowledge", "unknown",
    "--rarity-information-mode", "probe_feedback",
    "--min-client-size", "18",
    "--test-fraction", "0.25",
    "--fedprox-mu", "0.01",
    "--skip-baseline",
    "--strategy", "agentlock",
    "--repair-signal-mode", "subspace_consensus",
    "--trust-consensus-fraction", "0.5",
    "--root-anchor-weight", "0.7",
    "--counterfactual-mode", "simulator_oracle",
]


def run_one(seed: int) -> None:
    out_dir = RUNS_DIR / f"cifar10_adaptivemimic_seed{seed}_counterfactual_oracle"
    summary_path = out_dir / "agentlock_summary.json"
    if summary_path.exists():
        print(f"[skip] seed={seed} (already done)", flush=True)
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        PYTHON, str(REPO_ROOT / "run_digits_experiment.py"),
        "--dataset", "cifar10_local",
        "--seed", str(seed),
        "--results-dir", str(out_dir),
        *COMMON_ARGS,
        *CIFAR10_ARGS,
    ]

    print(f"[start] seed={seed}", flush=True)
    started = time.time()
    with open(out_dir / "stdout.log", "w", encoding="utf-8") as stdout_f, \
         open(out_dir / "stderr.log", "w", encoding="utf-8") as stderr_f:
        result = subprocess.run(cmd, cwd=str(REPO_ROOT), stdout=stdout_f, stderr=stderr_f)
    elapsed = time.time() - started

    if result.returncode != 0:
        print(f"[FAIL] seed={seed} exit={result.returncode} elapsed={elapsed:.0f}s "
              f"-- see {out_dir}/stderr.log", flush=True)
        return

    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        print(f"[done] seed={seed} elapsed={elapsed:.0f}s "
              f"ASR={summary.get('final_asr')} clean={summary.get('final_clean_accuracy')} "
              f"rare={summary.get('final_rare_accuracy')} bytes={summary.get('mean_byte_communication_ratio')} "
              f"local_jobs={summary.get('total_local_training_jobs')} probes={summary.get('total_probe_evaluations')}",
              flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[done, no summary readback] seed={seed} elapsed={elapsed:.0f}s ({exc})", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = parser.parse_args()
    for seed in args.seeds:
        run_one(seed)
    print(f"=== counterfactual-oracle ablation complete (seeds={args.seeds}) ===", flush=True)


if __name__ == "__main__":
    main()
