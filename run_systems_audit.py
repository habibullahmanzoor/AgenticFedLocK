"""Systems-cost audit: wall-clock, memory, communication beyond mask ratio.

Answers reviewer Comment 6 (`commetns of reviewers.txt`): "The framework
requires per-client probe evaluation, risk scoring, repair, bandit updates,
and counterfactual evaluation. These may introduce nontrivial server-side
computation. The paper should report runtime, memory cost, and communication
cost beyond the mask-based communication ratio."

v1's response (`answers to reviewers comments.tex`, Comment 6) explicitly
warned against "using timing from concurrently launched quality experiments"
and ran a separate sequential systems audit instead. This script deliberately
runs one method at a time, sequentially, with nothing else sharing the GPU --
launch it alone. Every summary.json already records elapsed_seconds,
peak_process_rss_bytes, peak_cuda_allocated_bytes, peak_cuda_reserved_bytes,
controller_state_bytes, total_local_training_jobs, total_probe_evaluations,
total_server_root_training_jobs, and the full byte-ratio breakdown, so no new
instrumentation is needed -- just clean, uncontended runs to measure it from.

CIFAR-10 model replacement, 50% data, unknown defender, seed 7, one run per
method (fedavg, fltrust, flame, rfa, agentlock). Single seed: this is a
systems/cost comparison, not an ASR comparison, and wall-clock variance
across seeds is not the object of study here.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = Path(__file__).resolve().parent / "runs"
PYTHON = sys.executable
SEED = 7
STRATEGIES = ["fedavg", "fltrust", "flame", "rfa", "agentlock"]

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

SYSTEMS_FIELDS = [
    "elapsed_seconds", "peak_process_rss_bytes", "peak_cuda_allocated_bytes",
    "peak_cuda_reserved_bytes", "controller_state_bytes",
    "total_local_training_jobs", "total_probe_evaluations",
    "total_server_root_training_jobs", "mean_byte_communication_ratio",
    "total_protocol_bytes",
]


def run_one(strategy: str) -> None:
    out_dir = RUNS_DIR / f"cifar10_seed{SEED}_systemsaudit_{strategy}"
    summary_path = out_dir / f"{strategy}_summary.json"
    if summary_path.exists():
        print(f"[skip] strategy={strategy} (already done)", flush=True)
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        PYTHON, str(REPO_ROOT / "run_digits_experiment.py"),
        "--strategy", strategy,
        "--dataset", "cifar10_local",
        "--seed", str(SEED),
        "--results-dir", str(out_dir),
        *COMMON_ARGS,
        *CIFAR10_ARGS,
    ]
    if strategy == "agentlock":
        cmd += [
            "--repair-signal-mode", "subspace_consensus", "--trust-consensus-fraction", "0.5",
            "--root-anchor-weight", "0.7",
        ]

    print(f"[start] strategy={strategy}", flush=True)
    started = time.time()
    with open(out_dir / "stdout.log", "w", encoding="utf-8") as stdout_f, \
         open(out_dir / "stderr.log", "w", encoding="utf-8") as stderr_f:
        result = subprocess.run(cmd, cwd=str(REPO_ROOT), stdout=stdout_f, stderr=stderr_f)
    elapsed = time.time() - started

    if result.returncode != 0:
        print(f"[FAIL] strategy={strategy} exit={result.returncode} elapsed={elapsed:.0f}s "
              f"-- see {out_dir}/stderr.log", flush=True)
        return

    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        fields = {k: summary.get(k) for k in SYSTEMS_FIELDS}
        print(f"[done] strategy={strategy} wrapper_elapsed={elapsed:.0f}s {fields}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[done, no summary readback] strategy={strategy} elapsed={elapsed:.0f}s ({exc})", flush=True)


def main() -> None:
    for strategy in STRATEGIES:
        run_one(strategy)
    print("=== systems audit complete ===", flush=True)


if __name__ == "__main__":
    main()
