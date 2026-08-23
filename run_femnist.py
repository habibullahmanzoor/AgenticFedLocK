"""FEMNIST vs full 5-method suite, reduced scale, model replacement.

Not reviewer-demanded by name (checked: Comment 1's baseline-breadth
complaint refers to "the five-dataset generalization benchmark" generically,
never names FEMNIST; Comment 3 names FEMNIST but about training convergence,
already fixed in v1's benign-convergence validation). This closes a gap the
authors flagged themselves in `baseline_numbers_to_beat.md`: "FEMNIST's win
is untested against real baselines and must not be trusted until FLTrust/
FLAME are run there."

Reduced scale vs v1's FEMNIST protocol (500 clients, 100 selected/round):
100 clients, same 0.2 participation (20 selected/round). Empirically measured
(3-round probe, this session): agentlock at full scale (500/100) = 226 min;
at this reduced scale (100/20) = ~64 min projected over 100 rounds -- a 3.5x
cut. Still genuine natural (LEAF) per-writer partitions, just fewer of the
available writers sampled -- comparable in spirit to the 50%-data-fraction
choice already applied everywhere else in this revision, not a synthetic
shortcut.

Rounds kept at 100 (unchanged from v1) deliberately: reviewer Comment 3
specifically flagged FEMNIST for undertraining, already fixed by locking
rounds at 100 in v1's benign-convergence work. Cutting rounds here would
reopen that exact complaint. Target/rare-label separation corrected from
v1's config (which used backdoor_target=0, rare_labels=[] / auto-resolved --
the same confound found and fixed for Fashion-MNIST/EMNIST earlier in this
revision) to backdoor_target=2, rare_labels=0,1, matching the convention used
for every other dataset in the Stage 1+2 sweep.

Attack: model replacement only (not adaptive mimic). Matches the cost
estimate given before launch; adaptive mimic on FEMNIST (62-class, most
heterogeneous dataset in the suite) would likely land in the same
"nobody-wins" regime already seen for EMNIST/CIFAR-10 adaptive mimic, and is
a separate, equally-sized follow-up if wanted later, not bundled in here.
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
    "--attack-mode", "model_replacement",
    "--data-fraction", "0.5",
    "--defender-knowledge", "unknown",
    "--rarity-information-mode", "probe_feedback",
    "--test-fraction", "0.25",
    "--fedprox-mu", "0.01",
    "--skip-baseline",
]


def run_one(strategy: str, seed: int) -> None:
    out_dir = RUNS_DIR / f"femnist_seed{seed}_{strategy}"
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
    print(f"=== femnist sweep complete (seeds={seeds}) ===", flush=True)


if __name__ == "__main__":
    main()
