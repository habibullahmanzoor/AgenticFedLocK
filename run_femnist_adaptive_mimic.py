"""FEMNIST vs full 5-method suite, reduced scale, adaptive mimic.

Follow-up to `run_femnist.py`, which covered model replacement only and
explicitly flagged this as "a separate, equally-sized follow-up if wanted
later, not bundled in here." That deferral meant this revision's FEMNIST
Generalization subsection tested only one attack, while the original
submission's FEMNIST evaluation (500 clients, 100 selected/round, FedAvg-only)
tested both. This closes that gap under the current mechanism and the full
five-method baseline set, matching `run_femnist.py`'s protocol exactly except
for the attack mode.

Same reduced scale as `run_femnist.py` (100 clients, 20 selected/round,
100 rounds, natural LEAF per-writer partitions) for the same cost reasons
documented there; same target/rare-label convention
(backdoor_target=2, rare_labels=0,1).
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
STRATEGIES = ["fedavg", "fltrust", "flame", "rfa", "agentlock"]

FEMNIST_ARGS = [
    "--num-clients", "100", "--client-fraction", "0.2", "--rounds", "100",
    "--local-epochs", "3", "--batch-size", "64", "--lr", "0.001",
    "--lr-schedule", "cosine", "--optimizer", "adam",
    "--model-family", "cnn", "--channel-dims", "64,128", "--cnn-hidden-dim", "512",
    "--cnn-variant", "residual",
    "--backdoor-target", "2", "--rare-labels", "0,1", "--probe-fraction", "0.08",
    "--min-client-size", "20",
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
    "--test-fraction", "0.25",
    "--fedprox-mu", "0.01",
    "--skip-baseline",
]


def run_one(strategy: str, seed: int) -> None:
    out_dir = RUNS_DIR / f"femnist_adaptivemimic_seed{seed}_{strategy}"
    summary_path = out_dir / f"{strategy}_summary.json"
    if summary_path.exists():
        print(f"[skip] strategy={strategy} seed={seed} (already done)", flush=True)
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        PYTHON, str(REPO_ROOT / "run_digits_experiment.py"),
        "--strategy", strategy,
        "--dataset", "femnist_local",
        "--seed", str(seed),
        "--results-dir", str(out_dir),
        *COMMON_ARGS,
        *FEMNIST_ARGS,
    ]
    if strategy == "agentlock":
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
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = parser.parse_args()
    seeds = args.seeds

    total = len(seeds) * len(STRATEGIES)
    done = 0
    for seed in seeds:
        for strategy in STRATEGIES:
            done += 1
            print(f"=== [{done}/{total}] (seeds={seeds}) ===", flush=True)
            run_one(strategy, seed)
    print(f"=== femnist adaptive-mimic sweep complete (seeds={seeds}) ===", flush=True)


if __name__ == "__main__":
    main()
