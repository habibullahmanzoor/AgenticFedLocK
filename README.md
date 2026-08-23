# AgenticFedLock

Code accompanying **"AgenticFedLock: Trust-Conditioned Multi-Role Agentic Control for Robust and Communication-Efficient Federated Learning"**, submitted to *AI Open* (Elsevier).

AgenticFedLock is a federated-learning framework in which a server-side controller — four coordinated roles (**Scout**, **Selection Planner**, **Subspace Allocator**, **Judge-Repair**) — assigns each client a trust-conditioned trainable parameter subspace each round, jointly targeting backdoor robustness, communication efficiency, and rare-class utility. The Judge-Repair role's backdoor-risk signal is trigger-agnostic by construction (subspace-localized root-cosine deviation + trusted-consensus deviation + norm-inflation detection), so it requires no knowledge of the attacker's trigger pattern or target class.

## What's in this repository

- **`agentlock/`** — the core framework: config (`config.py`), models (`model.py`), dataset loading (`data.py`), the LinUCB bandit (`bandit.py`), the four-role controller and attack implementations (`controller.py`), and the round-by-round simulation loop (`simulation.py`).
- **`run_benchmark.py`** / **`run_digits_experiment.py`** — general-purpose multi-seed benchmark runner and single-run entry point. Every other `run_*.py` script below is a thin driver around these.
- **`prepare_datasets.py`** — downloads/prepares the local dataset layout the code expects.
- **`generate_figures.py`** — regenerates the paper's figures from run output folders.
- One `run_*.py` script per paper result battery (below).

This is the code that produced the numbers reported in the paper (i.e. the mechanism described in the manuscript's Judge-Repair section: SLRT + CAR + norm-inflation, fused with per-signal recentering, plus the norm-confidence-gated root-anchor reweighting).

## Setup

```bash
pip install -r requirements.txt
python prepare_datasets.py --datasets mnist fashion_mnist emnist_balanced femnist cifar10
```

Requires Python 3.10+. A GPU is not required but speeds up CIFAR-10/FEMNIST substantially.

## Reproducing the paper's results

Every script below writes its output (summary JSON, per-round CSV metrics, per-client decision logs) to a results folder under `outputs/` (or wherever `--results-dir` points). Each script's own docstring explains the exact experimental question it answers and which paper subsection it supports; the table below is a map.

| Script | Paper section | What it runs |
|---|---|---|
| `run_stage1_stage2.py` | §7.3 Core Comparison | 4 datasets × 2 attacks × 5 methods (FedAvg, FLTrust, FLAME, RFA, AgenticFedLock) × 3 seeds |
| `run_femnist.py` | §7.4 FEMNIST Generalization | FEMNIST, model replacement, reduced scale, 5-method suite |
| `run_femnist_adaptive_mimic.py` | §7.4 FEMNIST Generalization | FEMNIST, adaptive mimic, same reduced scale |
| `run_benign_control.py` | §7.2 Benign Control | malicious fraction = 0, FedAvg vs AgenticFedLock, 4 datasets |
| `run_ablation.py` | §7.6 Component/Role-Separation Ablation | strips the controller to each mechanism in isolation |
| `run_defender_knowledge.py` | §7.7 Defender-Knowledge Ablation | unknown vs. exact (oracle) trigger/target knowledge |
| `run_sensitivity.py` | §7.8 Sensitivity Study | one-at-a-time perturbation of the three tuned coefficients |
| `run_slrt_localization_ablation.py` | §7.9 SLRT Localization Ablation | per-region root-cosine vs. whole-model cosine |
| `run_risk_fusion_ablation.py` | §7.10 Risk-Fusion-Order Ablation | recenter-then-fuse vs. fuse-then-recenter |
| `run_systems_audit.py` | §7.11 Systems-Cost Audit | sequential single-method runtime/memory/communication audit |
| `run_rarity_ablation.py` | §7.12 Rarity-Information and Spoofing Ablation | ground-truth / probe-feedback / disabled / spoofed rarity signal |
| `run_counterfactual_ablation.py` | §7.13 Simulator-Oracle Counterfactual Ablation | deployable default vs. optional simulator-oracle mode |

Most scripts run standalone with the exact locked protocol used in the paper:

```bash
python run_stage1_stage2.py --attack-mode model_replacement
python run_ablation.py
python run_sensitivity.py
```

`--seeds` (default `7 11 19`, matching the paper) can be overridden on every script that accepts it, e.g. `python run_ablation.py --seeds 7`. `run_systems_audit.py` should be launched alone (nothing else on the GPU) since it measures wall-clock and memory.

For the general-purpose benchmark runner directly:

```bash
python run_benchmark.py \
  --datasets mnist_local fashion_mnist_local emnist_balanced_local cifar10_local \
  --strategies fedavg fltrust flame rfa agenticfedlock \
  --attack-mode model_replacement --seeds 7 11 19 \
  --results-dir outputs/core_comparison
```

Run `python run_benchmark.py --help` for the full list of supported strategies, attack modes, and datasets.

## Reproducibility notes

Running this code with the same seeds and protocol will reproduce the same **qualitative** findings (AgenticFedLock dominating on EMNIST Balanced/CIFAR-10 under model replacement, competitive with FLTrust/FLAME on MNIST/Fashion-MNIST, the shared weakness against adaptive mimic on EMNIST/CIFAR-10, and the ~60–67% communication savings), but it will not reproduce the paper's tables to the exact decimal, for reasons the paper itself discloses (Sec. 7.3.2, "Seed-to-seed variance"):

- **GPU/cuDNN nondeterminism.** By default, this code does not force deterministic cuDNN kernels, so even *re-running the same seed on the same machine* shifts results — the paper measures this shift at "several ASR points" on EMNIST. Several headline cells already have wide spread across just 3 seeds in the reported tables themselves (e.g. CIFAR-10 model-replacement AgenticFedLock: 26.07% ± 17.20% ASR, from per-seed values of 42.4% / 8.1% / 27.8%); a fresh run can land anywhere in a similarly wide band.
- Set the environment variable `AGENTICFEDLOCK_DETERMINISTIC=1` before running to enable `torch.backends.cudnn.deterministic`. This makes repeated runs on *your* machine consistent with each other; it was **not** set for the paper's own runs, so it does not make a re-run match the published numbers either — it only removes one additional source of variance on top of the seed-level one above.
- Secondary factors: PyTorch/CUDA/driver version differences across machines, and `prepare_datasets.py` pulling from external mirrors (NIST, a GitHub media mirror, a University of Toronto mirror) that are stable but outside this project's control.
- Python/NumPy/scikit-learn-level randomness (client partitioning, non-IID splits, data subsampling) is fully seeded via `set_global_seed()` in `agentlock/data.py` and does not vary between runs.

## Repository scope

This repository contains the framework code and experiment drivers only. It does **not** include raw dataset files, run output logs, or the manuscript sources — those are excluded by `.gitignore` (`data/`, `outputs/`) or simply not part of the code release. `prepare_datasets.py` reconstructs the expected `data/` layout locally.

## Citation

If you use this code, please cite:

```bibtex
@article{manzoor2026agenticfedlock,
  title   = {AgenticFedLock: Trust-Conditioned Multi-Role Agentic Control for Robust and Communication-Efficient Federated Learning},
  author  = {Manzoor, Habib Ullah and Manzoor, Sanaullah and Arshad, Kamran and Assaleh, Khaled and Imran, Muhammad and Zoha, Ahmed},
  journal = {AI Open},
  year    = {2026},
  note    = {Under review}
}
```

## License

MIT — see [LICENSE](LICENSE).
