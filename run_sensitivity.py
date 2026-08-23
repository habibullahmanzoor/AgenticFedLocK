"""Sensitivity study for v2's hand-designed repair/anchor coefficients.

Answers reviewer Comment 5 (Major Weaknesses list in `commetns of reviewers.txt`):
"Many components rely on hand-designed coefficients and thresholds... The
paper needs systematic sensitivity analysis to show that the performance is
not the result of carefully tuned heuristics."

v1's response (`answers to reviewers comments.tex`, Comment 5) ran a locked,
one-factor-at-a-time study for repair_risk_threshold (theta in {0.15, 0.22,
0.30}) and repair_shared_suppression (sigma in {0.60, 0.90, 0.98}), on
CIFAR-10 adaptive-mimic, 3 seeds. v2 needs the same two factors retested
under its own mechanism (the underlying risk signal's distribution changed
completely, so v1's numbers don't transfer), PLUS root_anchor_weight, which
is a new coefficient introduced today that never existed in v1 and hasn't
had this scrutiny yet.

Run on CIFAR-10 model replacement, not v1's choice of adaptive mimic -- same
reasoning as the component ablation (02_component_ablation_results.md):
adaptive mimic on CIFAR-10 is a "nobody wins" regime for every method
(~87-91% ASR regardless of tuning), so there is nothing for a sensitivity
study to meaningfully show there. Model replacement is where the mechanism
achieves something, so sensitivity to its coefficients is an answerable
question.

Default point (repair_risk_threshold=0.22, repair_shared_suppression=0.9,
root_anchor_weight=0.7) is already covered by the Stage 1+2 sweep's
`cifar10_seed{N}_agentlock` results -- reused directly, not rerun.

One factor changed at a time from the locked default; the other two stay at
default. Resumable: skips any (factor, value, seed) whose summary exists.
Supports --seeds for splitting across parallel processes, same as run_ablation.py.
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

DEFAULTS = {
    "repair_risk_threshold": 0.22,
    "repair_shared_suppression": 0.9,
    "root_anchor_weight": 0.7,
}

# One factor at a time; the other two stay at DEFAULTS. Default point itself
# is not re-run here (already exists as cifar10_seed{N}_agentlock).
FACTOR_VALUES = {
    "repair_risk_threshold": [0.15, 0.30],
    "repair_shared_suppression": [0.60, 0.98],
    "root_anchor_weight": [0.5, 0.9],
}

CLI_FLAG = {
    "repair_risk_threshold": "--repair-risk-threshold",
    "repair_shared_suppression": "--repair-shared-suppression",
    "root_anchor_weight": "--root-anchor-weight",
}

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
    "--strategy", "agentlock",
    "--repair-signal-mode", "subspace_consensus",
    "--trust-consensus-fraction", "0.5",
]


def run_one(factor: str, value: float, seed: int) -> None:
    tag = f"{factor}{value}".replace(".", "")
    out_dir = RUNS_DIR / f"cifar10_seed{seed}_sensitivity_{tag}"
    summary_path = out_dir / "agentlock_summary.json"
    if summary_path.exists():
        print(f"[skip] factor={factor} value={value} seed={seed} (already done)", flush=True)
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    settings = dict(DEFAULTS)
    settings[factor] = value
    cmd = [
        PYTHON, str(REPO_ROOT / "run_digits_experiment.py"),
        "--dataset", "cifar10_local",
        "--seed", str(seed),
        "--results-dir", str(out_dir),
        *COMMON_ARGS,
        *CIFAR10_ARGS,
        CLI_FLAG["repair_risk_threshold"], str(settings["repair_risk_threshold"]),
        CLI_FLAG["repair_shared_suppression"], str(settings["repair_shared_suppression"]),
        CLI_FLAG["root_anchor_weight"], str(settings["root_anchor_weight"]),
    ]

    print(f"[start] factor={factor} value={value} seed={seed}", flush=True)
    started = time.time()
    with open(out_dir / "stdout.log", "w", encoding="utf-8") as stdout_f, \
         open(out_dir / "stderr.log", "w", encoding="utf-8") as stderr_f:
        result = subprocess.run(cmd, cwd=str(REPO_ROOT), stdout=stdout_f, stderr=stderr_f)
    elapsed = time.time() - started

    if result.returncode != 0:
        print(f"[FAIL] factor={factor} value={value} seed={seed} exit={result.returncode} "
              f"elapsed={elapsed:.0f}s -- see {out_dir}/stderr.log", flush=True)
        return

    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        print(f"[done] factor={factor} value={value} seed={seed} elapsed={elapsed:.0f}s "
              f"ASR={summary.get('final_asr')} clean={summary.get('final_clean_accuracy')} "
              f"rare={summary.get('final_rare_accuracy')} bytes={summary.get('mean_byte_communication_ratio')}",
              flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[done, no summary readback] factor={factor} value={value} seed={seed} "
              f"elapsed={elapsed:.0f}s ({exc})", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = parser.parse_args()
    seeds = args.seeds

    jobs = [(factor, value) for factor, values in FACTOR_VALUES.items() for value in values]
    total = len(seeds) * len(jobs)
    done = 0
    for seed in seeds:
        for factor, value in jobs:
            done += 1
            print(f"=== [{done}/{total}] (seeds={seeds}) ===", flush=True)
            run_one(factor, value, seed)
    print(f"=== sensitivity sweep complete (seeds={seeds}) ===", flush=True)


if __name__ == "__main__":
    main()
