from __future__ import annotations

from collections import OrderedDict, defaultdict
from dataclasses import dataclass, replace
import csv
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from agentlock.config import SimulationConfig
from agentlock.controller import AgenticFedLockController, AgentLockController, ClientAction, MaskBundle, SelectionMetadata
from agentlock.data import TaskData, set_global_seed
from agentlock.model import add_delta_in_place, average_deltas, build_model, clone_state_dict, flatten_state_dict, subtract_state_dicts

try:
    import psutil
except ImportError:  # pragma: no cover - psutil is optional for experiment portability.
    psutil = None


def _model_features(features: torch.Tensor, device: str) -> torch.Tensor:
    """Move a minibatch to the model and normalize compact uint8 image inputs."""
    was_integer = not features.is_floating_point()
    features = features.to(device=device, dtype=torch.float32)
    return features / 255.0 if was_integer else features


def _resolve_execution_device(requested_device: str) -> str:
    if requested_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but the installed PyTorch build cannot access a CUDA device.")
    return requested_device


def _apply_trigger(features: torch.Tensor, task_data: TaskData) -> torch.Tensor:
    poisoned = features.clone()
    if task_data.trigger_kind == "image_patch":
        patch = poisoned[:, :, -2:, -2:]
        if task_data.image_trigger_variant == "solid_patch":
            patch.fill_(task_data.trigger_value)
        elif task_data.image_trigger_variant == "checkerboard":
            patch[..., 0, 0] = task_data.trigger_value
            patch[..., 0, 1] = 0.0
            patch[..., 1, 0] = 0.0
            patch[..., 1, 1] = task_data.trigger_value
        elif task_data.image_trigger_variant == "blended":
            # Low-amplitude stamped patch: retain 75% of the original local
            # pixels and blend in 25% of the trigger value.
            patch.mul_(0.75).add_(0.25 * task_data.trigger_value)
        else:  # TaskData is normally built from validated SimulationConfig.
            raise ValueError(f"Unknown image trigger variant: {task_data.image_trigger_variant!r}")
        return poisoned
    flat = poisoned.view(poisoned.size(0), -1)
    flat[:, list(task_data.trigger_indices)] = task_data.trigger_value
    return flat.view_as(poisoned)


def _evaluate_model(model: nn.Module, loader: DataLoader, device: str) -> tuple[float, float]:
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    model.eval()
    with torch.no_grad():
        for features, targets in loader:
            features = _model_features(features, device)
            targets = targets.to(device)
            logits = model(features)
            loss = criterion(logits, targets)
            total_loss += loss.item() * targets.size(0)
            total_correct += (logits.argmax(dim=1) == targets).sum().item()
            total_samples += targets.size(0)
    return total_loss / max(1, total_samples), total_correct / max(1, total_samples)


def _evaluate_backdoor_asr(
    model: nn.Module,
    loader: DataLoader,
    task_data: TaskData,
    device: str,
) -> float:
    total = 0
    targeted_hits = 0
    model.eval()
    with torch.no_grad():
        for features, targets in loader:
            keep_mask = targets != task_data.backdoor_target
            if not keep_mask.any():
                continue
            features = _apply_trigger(_model_features(features[keep_mask], device), task_data)
            logits = model(features)
            predictions = logits.argmax(dim=1).cpu()
            targeted_hits += (predictions == task_data.backdoor_target).sum().item()
            total += predictions.size(0)
    return targeted_hits / max(1, total)


def _evaluate_rare_accuracy(
    model: nn.Module,
    loader: DataLoader,
    rare_labels: tuple[int, ...],
    device: str,
) -> float:
    label_tensor = torch.tensor(rare_labels)
    total = 0
    correct = 0
    model.eval()
    with torch.no_grad():
        for features, targets in loader:
            keep_mask = torch.isin(targets, label_tensor)
            if not keep_mask.any():
                continue
            filtered_features = _model_features(features[keep_mask], device)
            filtered_targets = targets[keep_mask].to(device)
            logits = model(filtered_features)
            correct += (logits.argmax(dim=1) == filtered_targets).sum().item()
            total += filtered_targets.size(0)
    return correct / max(1, total)


def _evaluate_probe_bundle(
    model: nn.Module,
    loader: DataLoader,
    task_data: TaskData,
    device: str,
    include_backdoor_signal: bool = True,
) -> dict[str, float]:
    loss, accuracy = _evaluate_model(model, loader, device)
    rare_accuracy = _evaluate_rare_accuracy(model, loader, task_data.rare_labels, device)
    backdoor_asr = _evaluate_backdoor_asr(model, loader, task_data, device) if include_backdoor_signal else 0.0
    return {
        "loss": loss,
        "accuracy": accuracy,
        "rare_accuracy": rare_accuracy,
        "backdoor_asr": backdoor_asr,
    }


def _state_to_device(state_dict: OrderedDict[str, torch.Tensor], device: str) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict((name, tensor.to(device)) for name, tensor in state_dict.items())


def _scheduled_learning_rate(config: SimulationConfig, round_id: int) -> float:
    if config.lr_schedule == "constant" or config.rounds <= 1:
        return config.lr
    progress = (round_id - 1) / max(1, config.rounds - 1)
    return config.lr_min + (config.lr - config.lr_min) * 0.5 * (1.0 + math.cos(math.pi * progress))


def _build_optimizer(model: nn.Module, config: SimulationConfig, round_id: int = 1) -> torch.optim.Optimizer:
    learning_rate = _scheduled_learning_rate(config, round_id)
    if config.local_optimizer.lower() == "adam":
        return torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=config.weight_decay)
    return torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=config.sgd_momentum,
        weight_decay=config.weight_decay,
    )


def _augment_cifar_training_batch(features: torch.Tensor, config: SimulationConfig) -> torch.Tensor:
    """Apply training-only random crop and horizontal flip on the execution device."""
    if not config.image_augmentation or not config.dataset_name.startswith("cifar10"):
        return features
    padded = torch.nn.functional.pad(features, (4, 4, 4, 4), mode="reflect")
    crops = []
    for index in range(features.size(0)):
        top = int(torch.randint(0, 9, (1,), device=features.device).item())
        left = int(torch.randint(0, 9, (1,), device=features.device).item())
        crops.append(padded[index:index + 1, :, top:top + 32, left:left + 32])
    augmented = torch.cat(crops, dim=0)
    flip_mask = torch.rand(features.size(0), device=features.device) < 0.5
    augmented[flip_mask] = torch.flip(augmented[flip_mask], dims=(3,))
    return augmented


def _masked_local_train(
    global_state: OrderedDict[str, torch.Tensor],
    client_loader: DataLoader,
    task_data: TaskData,
    config: SimulationConfig,
    device: str,
    parameter_masks: dict[str, torch.Tensor],
    is_malicious: bool,
    poison_fraction: float | None = None,
    prox_mu: float = 0.0,
    round_id: int = 1,
) -> OrderedDict[str, torch.Tensor]:
    model = build_model(task_data.model_spec).to(device)
    model.load_state_dict(_state_to_device(global_state, device))
    optimizer = _build_optimizer(model, config, round_id)
    criterion = nn.CrossEntropyLoss()
    mask_cache = {name: mask.to(device) for name, mask in parameter_masks.items()}
    poison_fraction = config.poison_fraction if poison_fraction is None else poison_fraction
    global_parameter_cache = None
    if prox_mu > 0.0:
        global_parameter_cache = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
        }

    model.train()
    for _ in range(config.local_epochs):
        for features, targets in client_loader:
            features = _model_features(features, device)
            features = _augment_cifar_training_batch(features, config)
            targets = targets.to(device)
            if is_malicious:
                poison_count = max(1, int(features.size(0) * poison_fraction))
                poison_mask = torch.zeros(features.size(0), dtype=torch.bool, device=device)
                poison_mask[:poison_count] = True
                features = features.clone()
                targets = targets.clone()
                features[poison_mask] = _apply_trigger(features[poison_mask], task_data)
                targets[poison_mask] = task_data.backdoor_target
            optimizer.zero_grad()
            loss = criterion(model(features), targets)
            if prox_mu > 0.0 and global_parameter_cache is not None:
                prox_penalty = torch.tensor(0.0, device=device)
                for name, parameter in model.named_parameters():
                    prox_penalty = prox_penalty + torch.sum((parameter - global_parameter_cache[name]) ** 2)
                loss = loss + 0.5 * prox_mu * prox_penalty
            loss.backward()
            for name, parameter in model.named_parameters():
                parameter.grad.mul_(mask_cache[name])
            optimizer.step()

    # Aggregation state is intentionally CPU-resident. Construct this mapping
    # directly from the CUDA model rather than modifying a cloned state in
    # place, which can leave mixed-device tensors in some module state dicts.
    local_state = OrderedDict(
        (name, tensor.detach().to(device="cpu").clone())
        for name, tensor in model.state_dict().items()
    )
    cpu_global_state = OrderedDict(
        (name, tensor.detach().to(device="cpu"))
        for name, tensor in global_state.items()
    )
    delta = subtract_state_dicts(local_state, cpu_global_state)
    for name, mask in parameter_masks.items():
        delta[name].mul_(mask.to(device=delta[name].device, dtype=delta[name].dtype))
    return OrderedDict((name, tensor.detach().to(device="cpu")) for name, tensor in delta.items())


def _evaluate_delta_on_probe(
    global_state: OrderedDict[str, torch.Tensor],
    delta: OrderedDict[str, torch.Tensor],
    probe_loader: DataLoader,
    task_data: TaskData,
    config: SimulationConfig,
    device: str,
    include_backdoor_signal: bool = True,
) -> dict[str, float]:
    temp_state = clone_state_dict(global_state)
    add_delta_in_place(temp_state, delta, scale=1.0)
    temp_model = build_model(task_data.model_spec).to(device)
    temp_model.load_state_dict(_state_to_device(temp_state, device))
    return _evaluate_probe_bundle(
        temp_model,
        probe_loader,
        task_data,
        device,
        include_backdoor_signal=include_backdoor_signal,
    )


def _local_training_job_cost(config: SimulationConfig, is_malicious: bool) -> int:
    """Count actual local optimization jobs, including adaptive-mimic clean/poison retraining."""
    return 2 if is_malicious and config.attack_mode == "adaptive_mimic" else 1


def _build_loaders(task_data: TaskData, config: SimulationConfig) -> tuple[DataLoader, DataLoader]:
    test_loader = DataLoader(task_data.test_dataset, batch_size=256, shuffle=False)
    probe_loader = DataLoader(task_data.probe_dataset, batch_size=256, shuffle=False)
    return probe_loader, test_loader


def _client_loader(task_data: TaskData, client_id: int, config: SimulationConfig, round_id: int) -> DataLoader:
    subset = Subset(task_data.train_dataset, task_data.client_indices[client_id])
    generator = torch.Generator().manual_seed(config.seed + 10000 * round_id + client_id)
    return DataLoader(subset, batch_size=config.batch_size, shuffle=True, generator=generator)


def _full_masks(reference_state: OrderedDict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: torch.ones_like(tensor, dtype=torch.float32) for name, tensor in reference_state.items()}


def _sample_clients(config: SimulationConfig, client_ids: list[int], round_id: int) -> list[int]:
    rng = np.random.default_rng(config.seed + 100 + round_id)
    count = max(2, int(len(client_ids) * config.client_fraction))
    return rng.choice(client_ids, size=count, replace=False).tolist()


def _save_csv_rows(path: Path, rows: list[dict[str, float | int | str | bool]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _save_results(
    results_dir: Path,
    strategy_name: str,
    rows: list[dict[str, float | int | str]],
    summary: dict[str, float | int | str],
) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    _save_csv_rows(results_dir / f"{strategy_name}_round_metrics.csv", rows)
    json_path = results_dir / f"{strategy_name}_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _scale_delta(delta: OrderedDict[str, torch.Tensor], scale: float) -> None:
    for tensor in delta.values():
        tensor.mul_(scale)


def _suppress_shared_component(
    delta: OrderedDict[str, torch.Tensor],
    shared_mask: dict[str, torch.Tensor],
    suppression: float,
) -> None:
    for name, tensor in delta.items():
        tensor.sub_(tensor * shared_mask[name] * suppression)


def _suppress_target_row_component(
    delta: OrderedDict[str, torch.Tensor],
    classifier_weight_name: str,
    classifier_bias_name: str,
    target_label: int,
    suppression: float,
) -> None:
    weight = delta.get(classifier_weight_name)
    if weight is not None and 0 <= target_label < weight.size(0):
        weight[target_label].mul_(1.0 - suppression)
    bias = delta.get(classifier_bias_name)
    if bias is not None and 0 <= target_label < bias.size(0):
        bias[target_label].mul_(1.0 - suppression)


def _probe_effects(global_probe_metrics: dict[str, float], candidate_metrics: dict[str, float]) -> dict[str, float]:
    clean_gain = float(torch.sigmoid(torch.tensor(9.0 * (global_probe_metrics["loss"] - candidate_metrics["loss"]))).item())
    rare_gain = float(torch.sigmoid(torch.tensor(8.0 * (candidate_metrics["rare_accuracy"] - global_probe_metrics["rare_accuracy"]))).item())
    backdoor_increase = max(0.0, candidate_metrics["backdoor_asr"] - global_probe_metrics["backdoor_asr"])
    backdoor_risk = float(np.clip(0.14 * candidate_metrics["backdoor_asr"] + 0.95 * backdoor_increase, 0.0, 1.0))
    return {
        "clean_gain": clean_gain,
        "rare_gain": rare_gain,
        "backdoor_risk": backdoor_risk,
        "probe_loss": candidate_metrics["loss"],
        "probe_accuracy": candidate_metrics["accuracy"],
        "probe_rare_accuracy": candidate_metrics["rare_accuracy"],
        "probe_backdoor_asr": candidate_metrics["backdoor_asr"],
    }


def _decision_reward(
    clean_gain: float,
    rare_gain: float,
    backdoor_risk: float,
    communication_ratio: float,
    overlap_ratio: float,
) -> float:
    centered_clean = clean_gain - 0.5
    centered_rare = rare_gain - 0.5
    overlap_penalty = backdoor_risk * overlap_ratio
    return float(
        0.95 * centered_clean
        + 0.85 * centered_rare
        - 1.15 * backdoor_risk
        - 0.04 * communication_ratio
        - 0.08 * overlap_penalty
    )


def _selection_reward(clean_gain: float, rare_gain: float, backdoor_risk: float) -> float:
    centered_clean = clean_gain - 0.5
    centered_rare = rare_gain - 0.5
    return float(1.0 * centered_clean + 0.95 * centered_rare - 1.0 * backdoor_risk)


def _selection_feedback_reward(
    clean_gain: float,
    rare_gain: float,
    backdoor_risk: float,
    rare_credit: float,
    stability: float,
    deception: float,
) -> float:
    base_reward = _selection_reward(clean_gain, rare_gain, backdoor_risk)
    protected_rare_bonus = rare_credit * max(0.0, rare_gain - 0.5)
    stable_rare_bonus = rare_credit * stability * max(0.0, rare_gain - 0.45)
    rare_alignment_bonus = max(0.0, rare_gain - clean_gain)
    deceptive_rare_penalty = rare_credit * deception * backdoor_risk
    return float(
        base_reward
        + 0.18 * protected_rare_bonus
        + 0.1 * stable_rare_bonus
        + 0.08 * rare_alignment_bonus
        - 0.08 * deceptive_rare_penalty
    )


def _mix_delayed_reward(local_reward: float, round_reward: float, delayed_reward_mix: float, adjustment: float = 0.0) -> float:
    return float((1.0 - delayed_reward_mix) * local_reward + delayed_reward_mix * round_reward + adjustment)


def _adaptive_repair_factors(config: SimulationConfig, global_probe_metrics: dict[str, float]) -> tuple[float, float, float, float, float]:
    if not config.adaptive_repair_enabled:
        return 1.0, config.repair_risk_threshold, 1.0, 1.0, 1.0
    pressure = float(
        np.clip(
            global_probe_metrics["backdoor_asr"] / max(config.adaptive_repair_reference_asr, 1e-6),
            config.adaptive_repair_floor,
            config.adaptive_repair_ceiling,
        )
    )
    risk_threshold = float(np.clip(config.repair_risk_threshold + 0.18 * (1.0 - pressure), 0.0, 0.995))
    shared_scale = float(np.clip(0.3 + 0.7 * pressure, 0.25, 1.0))
    weight_scale = float(np.clip(0.35 + 0.65 * pressure, 0.25, 1.0))
    target_scale = float(np.clip(0.25 + 0.75 * pressure, 0.2, 1.0))
    return pressure, risk_threshold, shared_scale, weight_scale, target_scale


def _flatten_masked_delta(delta: OrderedDict[str, torch.Tensor], mask: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([(delta[name] * mask[name]).reshape(-1) for name in delta]).float()


def _group_projected_norm(
    deltas: list[OrderedDict[str, torch.Tensor]],
    weights: list[float],
    mask: dict[str, torch.Tensor],
    indices: list[int],
) -> float:
    if not indices:
        return 0.0
    accumulator = None
    for index in indices:
        vector = _flatten_masked_delta(deltas[index], mask) * weights[index]
        accumulator = vector if accumulator is None else accumulator + vector
    return float(accumulator.norm().item()) if accumulator is not None else 0.0


def _cosine_similarity(vector_a: torch.Tensor, vector_b: torch.Tensor) -> float:
    norm_a = float(vector_a.norm().item())
    norm_b = float(vector_b.norm().item())
    if norm_a <= 1e-12 or norm_b <= 1e-12:
        return 0.0
    return float(torch.dot(vector_a, vector_b).item() / (norm_a * norm_b))


def _trigger_agnostic_risk(
    client_deltas: list[OrderedDict[str, torch.Tensor]],
    selected: list[int],
    controller: "AgentLockController",
    shared_core_mask: dict[str, torch.Tensor],
    classifier_mask: dict[str, torch.Tensor] | None,
    root_delta: OrderedDict[str, torch.Tensor] | None,
    trust_consensus_fraction: float,
    slrt_localization_mode: str = "per_region",
    risk_fusion_order: str = "recenter_then_fuse",
) -> tuple[dict[int, float], float]:
    """v2 trigger-agnostic composite risk (Mechanisms A+B+C+E).

    Fuses subspace-localized root-cosine deviation (SLRT, mirrors FLTrust but
    computed per security-relevant region rather than over the whole update, so
    a concentrated backdoor is not diluted by thousands of unaffected benign
    parameters), trusted-consensus deviation on the shared core (CAR, mirrors
    FLAME/RFA-style outlier detection but anchored to the controller's own
    persistent trust rather than a fresh per-round vote), norm-inflation
    detection (Mechanism E, catches magnitude-scaling attacks like model
    replacement that SLRT/CAR's purely angular comparisons can miss), and the
    controller's persistent suspicion memory. Requires no knowledge of the
    attack trigger, so it fires under unknown-defender knowledge where the
    legacy triggered-probe signal cannot be computed at all.

    Also returns a per-round norm-confidence scalar (the strongest raw
    norm-inflation evidence seen this round, before recentering) so callers
    can tell how much of the detection burden Mechanism E is actually
    carrying this round -- see `_apply_root_anchor_blend`'s use of it to
    avoid diluting an already-confident norm-based detection with a noisier
    global cosine signal, while still leaning on that cosine signal when
    norm-inflation finds nothing (e.g. a norm-matched attack).
    """
    if not selected:
        return {}, 0.0
    shared_vectors = {
        client_id: _flatten_masked_delta(client_deltas[index], shared_core_mask)
        for index, client_id in enumerate(selected)
    }
    classifier_vectors = None
    if classifier_mask is not None:
        classifier_vectors = {
            client_id: _flatten_masked_delta(client_deltas[index], classifier_mask)
            for index, client_id in enumerate(selected)
        }

    import os as _os_diag
    if _os_diag.environ.get("AGENTLOCK_DEBUG_COLLUSION"):
        mal_ids = {int(x) for x in _os_diag.environ["AGENTLOCK_DEBUG_COLLUSION"].split(",") if x.strip()}
        for region_name, vecs in [("shared", shared_vectors), ("clf", classifier_vectors)]:
            if vecs is None:
                continue
            peak = {}
            for cid in selected:
                sims = [
                    _cosine_similarity(vecs[cid], vecs[other])
                    for other in selected if other != cid
                ]
                peak[cid] = max(sims) if sims else 0.0
            mal = [peak[c] for c in selected if c in mal_ids]
            ben = [peak[c] for c in selected if c not in mal_ids]
            mm = sum(mal) / len(mal) if mal else 0.0
            bm = sum(ben) / len(ben) if ben else 0.0
            print(f"[collusion-debug] region={region_name} peak-pairwise-cos "
                  f"malicious_mean={mm:.4f} benign_mean={bm:.4f} gap={mm-bm:+.4f}")

    # Mechanism A: SLRT -- per-region root-cosine deviation, worst region wins.
    # Ablation branch (`slrt_localization_mode="whole_model"`): a single cosine
    # deviation over the full unmasked delta, matching plain FLTrust-style
    # comparison, to test whether region-localization actually matters or
    # whether a whole-model comparison would have done just as well.
    slrt_risk: dict[int, float] = {client_id: 0.0 for client_id in selected}
    if root_delta is not None:
        if slrt_localization_mode == "whole_model":
            root_full = flatten_state_dict(root_delta)
            for index, client_id in enumerate(selected):
                full_vec = flatten_state_dict(client_deltas[index])
                slrt_risk[client_id] = 1.0 - max(0.0, _cosine_similarity(full_vec, root_full))
        else:
            root_shared = _flatten_masked_delta(root_delta, shared_core_mask)
            root_classifier = _flatten_masked_delta(root_delta, classifier_mask) if classifier_mask is not None else None
            for client_id in selected:
                region_risks = [1.0 - max(0.0, _cosine_similarity(shared_vectors[client_id], root_shared))]
                if root_classifier is not None and classifier_vectors is not None:
                    region_risks.append(
                        1.0 - max(0.0, _cosine_similarity(classifier_vectors[client_id], root_classifier))
                    )
                slrt_risk[client_id] = max(region_risks)

    # Mechanism B: CAR -- deviation from the high-trust selected clients' shared-core consensus.
    trust_values = {client_id: controller.effective_trust(client_id) for client_id in selected}
    ranked_by_trust = sorted(selected, key=lambda client_id: trust_values[client_id], reverse=True)
    anchor_count = max(1, int(round(len(selected) * trust_consensus_fraction)))
    anchor_clients = ranked_by_trust[:anchor_count]
    anchor_stack = torch.stack([shared_vectors[client_id] for client_id in anchor_clients], dim=0)
    consensus_vector = torch.median(anchor_stack, dim=0).values
    car_risk = {
        client_id: 1.0 - max(0.0, _cosine_similarity(shared_vectors[client_id], consensus_vector))
        for client_id in selected
    }

    # Mechanism E: norm-inflation detection. SLRT/CAR are purely angular (cosine)
    # comparisons and are blind to magnitude by construction. Model replacement's
    # defining behavior is scaling the ENTIRE malicious update by a large factor
    # to dominate aggregation -- a direction-only detector can miss this if the
    # scaled update still points in a roughly plausible direction. This is
    # computed on the FULL (unmasked) delta, not the shared-core-restricted
    # vector: unlike a backdoor's angular signature (which genuinely
    # concentrates in shared/classifier regions, so SLRT/CAR benefit from
    # localization), model-replacement scaling is applied uniformly across the
    # whole update, so restricting to a partial-overlap template's shared-core
    # subset would dilute exactly the signal being measured here.
    full_norms = {
        client_id: float(flatten_state_dict(client_deltas[index]).float().norm().item())
        for index, client_id in enumerate(selected)
    }
    median_norm = float(np.median(list(full_norms.values())))
    norm_risk: dict[int, float] = {}
    for client_id in selected:
        norm_ratio = full_norms[client_id] / max(1e-8, median_norm)
        norm_risk[client_id] = float(np.clip((norm_ratio - 1.0) / 2.0, 0.0, 1.0))

    # Per-round relative recalibration -- applied to EACH structural signal
    # independently, before fusion. SLRT/CAR are cosine-based deviations that
    # never reach zero even for benign clients (ordinary SGD noise keeps
    # cosine similarity below 1.0); on some datasets (e.g. CIFAR-10 with a
    # residual CNN and short local training under non-IID partitioning) this
    # noise floor can sit near-saturation for EVERY client, benign or not,
    # because individual update directions are naturally far from the root
    # direction regardless of malice. Recentering the already-FUSED score
    # against its own round median (the earlier design) let that saturated,
    # non-discriminative signal define what counts as "normal" and swamp a
    # genuinely clean signal (norm-inflation, which stays near 0 for benign
    # clients and near 1 for malicious ones every round). Recentering each
    # signal against its OWN round median before taking the max fixes this:
    # a signal that is uniformly saturated collapses to ~0 for everyone once
    # centered on its own median, while a signal that is genuinely clean and
    # bimodal (like norm-inflation) is left essentially untouched.
    def _recenter(risk: dict[int, float]) -> dict[int, float]:
        # Floor the denominator so amplification is bounded (at most 1/floor).
        # Without this, a client whose raw signal happens to hit exactly the
        # ceiling (e.g. SLRT's cosine-to-root clamped to <=0, which ordinary
        # noise can trigger even for benign clients on simple tasks) always
        # recenters to exactly 1.0 regardless of the median -- (1-m)/(1-m) -- even
        # when several peers are nearly as close to that same ceiling and the
        # gap is not actually meaningful. Requiring a genuinely large absolute
        # gap from the median before saturating fixes false positives that were
        # measured at ~40% among benign MNIST clients (vs 100% true-positive
        # rate on malicious ones) before this fix.
        median = float(np.median(list(risk.values()))) if risk else 0.0
        denom = max(0.5, 1.0 - median)
        return {
            client_id: float(np.clip((value - median) / denom, 0.0, 1.0))
            for client_id, value in risk.items()
        }

    slrt_recentered = _recenter(slrt_risk)
    car_recentered = _recenter(car_risk)
    norm_recentered = _recenter(norm_risk)

    # Ablation branch (`risk_fusion_order="fuse_then_recenter"`): take the max
    # of the RAW (non-recentered) structural signals first, and recenter only
    # the fused result -- the alternative this design was chosen over, kept
    # here so the choice can be tested directly rather than only argued for.
    if risk_fusion_order == "fuse_then_recenter":
        fused_raw = {
            client_id: max(slrt_risk[client_id], car_risk[client_id], norm_risk[client_id])
            for client_id in selected
        }
        structural_risk_map = _recenter(fused_raw)
    else:
        structural_risk_map = {
            client_id: max(slrt_recentered[client_id], car_recentered[client_id], norm_recentered[client_id])
            for client_id in selected
        }

    # Mechanism C: fuse with persistent suspicion memory (the controller's cross-round state).
    # Structural signals (SLRT, CAR, norm-inflation) are combined by MAX, not a
    # weighted average: an attack need only be anomalous on ONE axis to be
    # caught (e.g., model replacement may be norm-inflated with a plausible
    # direction, or direction-deviant without inflated norm), so averaging
    # would dilute a signal that is strong on only one axis below the
    # detection threshold. Suspicion (persistent, cross-round) is blended in
    # afterward as a smaller, additive adjustment.
    composite: dict[int, float] = {}
    for client_id in selected:
        suspicion = float(controller.suspicion_scores.get(client_id, 0.0))
        structural_risk = structural_risk_map[client_id]
        composite[client_id] = float(np.clip(0.85 * structural_risk + 0.15 * suspicion, 0.0, 1.0))

    norm_confidence = max(norm_risk.values()) if norm_risk else 0.0

    import os
    if os.environ.get("AGENTLOCK_DEBUG_RISK"):
        print(f"[risk-debug] median_norm={median_norm:.4f} norm_confidence={norm_confidence:.4f}")
        for client_id in selected:
            print(
                f"[risk-debug] client={client_id} full_norm={full_norms[client_id]:.4f} "
                f"slrt={slrt_risk[client_id]:.4f}->{slrt_recentered[client_id]:.4f} "
                f"car={car_risk[client_id]:.4f}->{car_recentered[client_id]:.4f} "
                f"norm={norm_risk[client_id]:.4f}->{norm_recentered[client_id]:.4f} "
                f"composite={composite[client_id]:.4f}"
            )
    return composite, norm_confidence


def _pairwise_mask_overlap(mask_a: dict[str, torch.Tensor], mask_b: dict[str, torch.Tensor]) -> float:
    overlap = 0.0
    active_a = 0.0
    active_b = 0.0
    for name in mask_a:
        a = mask_a[name]
        b = mask_b[name]
        overlap += float((a * b).sum().item())
        active_a += float(a.sum().item())
        active_b += float(b.sum().item())
    return overlap / max(1.0, min(active_a, active_b))


def _overlap_state_features(
    selected: list[int],
    actions: dict[int, ClientAction],
    controller: AgenticFedLockController,
) -> dict[int, dict[str, float]]:
    features: dict[int, dict[str, float]] = {}
    for client_id in selected:
        action = actions[client_id]
        stable_weight = 0.0
        stable_overlap = 0.0
        suspicious_weight = 0.0
        suspicious_overlap = 0.0
        for peer_id in selected:
            if peer_id == client_id:
                continue
            overlap = _pairwise_mask_overlap(action.mask_bundle.masks, actions[peer_id].mask_bundle.masks)
            peer_trust = max(0.0, controller.effective_trust(peer_id) - 0.3)
            peer_suspicion = max(0.0, controller.suspicion_scores[peer_id] - 0.18)
            if peer_trust > 0:
                stable_weight += peer_trust
                stable_overlap += overlap * peer_trust
            if peer_suspicion > 0:
                suspicious_weight += peer_suspicion
                suspicious_overlap += overlap * peer_suspicion
        stable_score = stable_overlap / max(1e-6, stable_weight) if stable_weight > 0 else 0.0
        suspicious_score = suspicious_overlap / max(1e-6, suspicious_weight) if suspicious_weight > 0 else 0.0
        shared_exposure = action.mask_bundle.overlap_ratio
        features[client_id] = {
            "shared_exposure": float(np.clip(shared_exposure, 0.0, 1.0)),
            "stable_overlap": float(np.clip(stable_score, 0.0, 1.0)),
            "suspicious_overlap": float(np.clip(suspicious_score, 0.0, 1.0)),
        }
    return features


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _select_counterfactual_clients(actions: dict[int, ClientAction], limit: int) -> list[int]:
    if limit <= 0 or not actions:
        return []
    ranked_groups = [
        sorted(actions.values(), key=lambda action: action.suspicion, reverse=True),
        sorted(actions.values(), key=lambda action: action.rarity, reverse=True),
        sorted(actions.values(), key=lambda action: action.trust, reverse=True),
    ]
    selected: list[int] = []
    for group in ranked_groups:
        for action in group:
            if action.client_id not in selected:
                selected.append(action.client_id)
                break
        if len(selected) >= limit:
            return selected[:limit]
    return selected[:limit]


def _select_selection_counterfactual_clients(
    metadata_map: dict[int, SelectionMetadata],
    selected: list[int],
    limit: int,
) -> list[int]:
    if limit <= 0 or not metadata_map:
        return []
    selected_set = set(selected)
    unselected = [metadata for client_id, metadata in metadata_map.items() if client_id not in selected_set]
    if not unselected:
        return []
    ranked_groups = [
        sorted(unselected, key=lambda metadata: metadata.combined_score, reverse=True),
        sorted(unselected, key=lambda metadata: metadata.heuristic_score, reverse=True),
        sorted(unselected, key=lambda metadata: metadata.bandit_score, reverse=True),
    ]
    chosen: list[int] = []
    for group in ranked_groups:
        for metadata in group:
            if metadata.client_id not in chosen:
                chosen.append(metadata.client_id)
                break
        if len(chosen) >= limit:
            return chosen[:limit]
    for metadata in ranked_groups[0]:
        if metadata.client_id not in chosen:
            chosen.append(metadata.client_id)
        if len(chosen) >= limit:
            break
    return chosen[:limit]


def _selection_counterfactual_summary(
    rows: list[dict[str, float | int | str | bool]],
) -> tuple[float, float]:
    if not rows:
        return 0.0, 0.0
    grouped: dict[int, list[dict[str, float | int | str | bool]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["round"])].append(row)
    win_count = 0
    gaps: list[float] = []
    for round_rows in grouped.values():
        selected_rows = [row for row in round_rows if bool(row["was_selected"])]
        alternate_rows = [row for row in round_rows if not bool(row["was_selected"])]
        if not selected_rows or not alternate_rows:
            continue
        selected_mean = _mean([float(row["selection_reward_score"]) for row in selected_rows])
        alternate_mean = _mean([float(row["selection_reward_score"]) for row in alternate_rows])
        if selected_mean >= alternate_mean - 1e-9:
            win_count += 1
        gaps.append(selected_mean - alternate_mean)
    if not gaps:
        return 0.0, 0.0
    return win_count / len(gaps), _mean(gaps)


def _template_counts(actions: dict[int, ClientAction]) -> dict[str, int]:
    counts = {template: 0 for template in AgenticFedLockController.TEMPLATE_NAMES}
    for action in actions.values():
        counts[action.template] += 1
    return counts


def _policy_mode_counts(actions: dict[int, ClientAction]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for action in actions.values():
        counts[action.policy_mode] += 1
    return counts


def _counterfactual_summary(counterfactual_rows: list[dict[str, float | int | str | bool]]) -> tuple[float, float]:
    if not counterfactual_rows:
        return 0.0, 0.0
    grouped: dict[tuple[int, int], list[dict[str, float | int | str | bool]]] = defaultdict(list)
    for row in counterfactual_rows:
        grouped[(int(row["round"]), int(row["client_id"]))].append(row)
    optimal_hits = 0
    gaps: list[float] = []
    for rows in grouped.values():
        best_reward = max(float(row["reward_score"]) for row in rows)
        chosen_row = next(row for row in rows if bool(row["policy_selected"]))
        chosen_reward = float(chosen_row["reward_score"])
        if chosen_reward >= best_reward - 1e-9:
            optimal_hits += 1
        gaps.append(best_reward - chosen_reward)
    return optimal_hits / max(1, len(grouped)), _mean(gaps)


@dataclass(frozen=True)
class StrategySpec:
    name: str
    controller_mode: str
    selection_mode: str
    mask_mode: str
    aggregation_mode: str
    trust_weighting: bool = False
    enable_repair: bool = False
    enable_counterfactual: bool = False
    update_controller_scores: bool = False
    update_bandits: bool = False
    prox_mu: float = 0.0


def _resolve_strategy_spec(strategy_name: str, config: SimulationConfig) -> StrategySpec:
    specs = {
        "fedavg": StrategySpec("fedavg", "none", "random", "full", "weighted_mean"),
        "agentlock": StrategySpec(
            "agentlock",
            "agentic",
            "controller",
            "dynamic",
            "weighted_mean",
            trust_weighting=True,
            enable_repair=True,
            enable_counterfactual=True,
            update_controller_scores=True,
            update_bandits=True,
        ),
        "static_subspace": StrategySpec("static_subspace", "static", "random", "static_subspace", "weighted_mean"),
        "communication_only": StrategySpec("communication_only", "static", "random", "communication_only", "weighted_mean"),
        "trust_only": StrategySpec(
            "trust_only",
            "trust",
            "random",
            "full",
            "weighted_mean",
            trust_weighting=True,
            update_controller_scores=True,
        ),
        "selection_only": StrategySpec(
            "selection_only",
            "agentic",
            "controller",
            "selection_only",
            "weighted_mean",
            update_controller_scores=True,
            update_bandits=True,
        ),
        "allocation_repair_only": StrategySpec(
            "allocation_repair_only",
            "agentic",
            "random",
            "dynamic",
            "weighted_mean",
            trust_weighting=True,
            enable_repair=True,
            enable_counterfactual=True,
            update_controller_scores=True,
            update_bandits=True,
        ),
        "fedprox": StrategySpec("fedprox", "none", "random", "full", "weighted_mean", prox_mu=config.fedprox_mu),
        "trimmed_mean": StrategySpec("trimmed_mean", "none", "random", "full", "trimmed_mean"),
        "krum": StrategySpec("krum", "none", "random", "full", "krum"),
        "foolsgold": StrategySpec("foolsgold", "none", "random", "full", "foolsgold"),
        "fltrust": StrategySpec("fltrust", "none", "random", "full", "fltrust"),
        "flame": StrategySpec("flame", "none", "random", "full", "flame"),
        "rfa": StrategySpec("rfa", "none", "random", "full", "rfa"),
        "bulyan": StrategySpec("bulyan", "none", "random", "full", "bulyan"),
    }
    if strategy_name not in specs:
        raise ValueError(f"Unsupported strategy: {strategy_name}")
    return specs[strategy_name]


def _build_controller(task_data: TaskData, config: SimulationConfig, spec: StrategySpec) -> AgentLockController | None:
    if spec.controller_mode == "none":
        return None
    return AgenticFedLockController(
        task_data.client_profiles,
        model_spec=task_data.model_spec,
        seed=config.seed,
        bandit_alpha=config.bandit_alpha,
        bandit_mix=config.bandit_mix,
        bandit_warmup_rounds=config.bandit_warmup_rounds,
        diversity_strength=config.diversity_strength,
        quota_guard_enabled=config.quota_guard_enabled,
        selection_alpha=config.selection_alpha,
        selection_mix=config.selection_mix,
        selection_warmup_rounds=config.selection_warmup_rounds,
        rarity_feedback_enabled=config.rarity_information_mode != "none",
    )


def _normalize_weights(weights: list[float]) -> list[float]:
    clipped = [max(0.0, float(weight)) for weight in weights]
    total = sum(clipped)
    if total <= 0:
        return [1.0 / max(1, len(clipped)) for _ in clipped]
    return [weight / total for weight in clipped]


def _blend_deltas(
    primary: OrderedDict[str, torch.Tensor],
    secondary: OrderedDict[str, torch.Tensor],
    primary_weight: float,
) -> OrderedDict[str, torch.Tensor]:
    primary_weight = float(np.clip(primary_weight, 0.0, 1.0))
    secondary_weight = 1.0 - primary_weight
    return OrderedDict((name, primary[name] * primary_weight + secondary[name] * secondary_weight) for name in primary)


def _state_from_flat_vector(
    reference_state: OrderedDict[str, torch.Tensor],
    flat_vector: torch.Tensor,
) -> OrderedDict[str, torch.Tensor]:
    rebuilt = OrderedDict()
    offset = 0
    for name, tensor in reference_state.items():
        count = tensor.numel()
        rebuilt[name] = flat_vector[offset:offset + count].view_as(tensor).detach().cpu().clone()
        offset += count
    return rebuilt


def _aggregate_trimmed_mean(
    deltas: list[OrderedDict[str, torch.Tensor]],
    trim_ratio: float,
) -> tuple[OrderedDict[str, torch.Tensor], list[float]]:
    if len(deltas) == 1:
        return clone_state_dict(deltas[0]), [1.0]
    stacked = torch.stack([flatten_state_dict(delta).float() for delta in deltas], dim=0)
    trim_count = min(int(len(deltas) * trim_ratio), max(0, (len(deltas) - 1) // 2))
    if trim_count > 0 and stacked.size(0) > 2 * trim_count:
        trimmed = torch.sort(stacked, dim=0).values[trim_count: stacked.size(0) - trim_count]
    else:
        trimmed = stacked
    return _state_from_flat_vector(deltas[0], trimmed.mean(dim=0)), _normalize_weights([1.0] * len(deltas))


def _root_anchor_trust_scores(
    deltas: list[OrderedDict[str, torch.Tensor]],
    root_delta: OrderedDict[str, torch.Tensor],
) -> list[float]:
    """FLTrust's ReLU-cosine-to-root trust score, computed per client delta."""
    root_vector = flatten_state_dict(root_delta).float()
    root_norm = float(root_vector.norm().item())
    if root_norm <= 1e-12:
        return [0.0] * len(deltas)
    scores: list[float] = []
    for delta in deltas:
        vector = flatten_state_dict(delta).float()
        vector_norm = float(vector.norm().item())
        if vector_norm <= 1e-12:
            scores.append(0.0)
            continue
        cosine = float(torch.dot(vector, root_vector).item() / (vector_norm * root_norm))
        scores.append(max(0.0, cosine))
    return scores


def _norm_confidence_gate(norm_confidence: float, low: float = 0.6, high: float = 0.85) -> float:
    """Smoothstep suppression factor for the root-anchor blend: 1.0 = engage
    the anchor at full configured strength (no norm-inflation evidence this
    round -- e.g. a norm-matched attack like adaptive mimic, which defeats
    Mechanism E by construction), 0.0 = fully suppress it (strong
    norm-inflation evidence already caught by Mechanism E -- e.g. model
    replacement -- so a noisier global cosine signal would only dilute an
    already-correct, already-targeted suppression).

    The [0.6, 0.85] window is calibrated against measured per-round traces,
    not guessed: EMNIST model-replacement's norm_confidence sat at exactly
    1.0 in 39/40 rounds (one dip to 0.58); MNIST adaptive-mimic's never
    exceeded ~0.66 across 12 rounds (median 0.39, min 0.13). Keeping the
    transition band above adaptive-mimic's whole observed range means it
    engages the anchor near full strength every round, rather than the ~40%
    a plain linear (1 - norm_confidence) scaling left on the table.
    """
    if norm_confidence <= low:
        return 1.0
    if norm_confidence >= high:
        return 0.0
    t = (norm_confidence - low) / (high - low)
    return 1.0 - (3.0 * t * t - 2.0 * t * t * t)


def _apply_root_anchor_blend(
    client_deltas: list[OrderedDict[str, torch.Tensor]],
    agg_weights: list[float],
    root_delta: OrderedDict[str, torch.Tensor] | None,
    blend: float,
) -> list[float]:
    """Layer FLTrust-style root-cosine trust onto agentlock's own aggregation
    weights, as an additional independent defense on top of masking/repair
    (rather than instead of it). blend=0.0 is an exact no-op -- the containment
    and repair mechanisms already ran on client_deltas before this is called, so
    this only affects the final weighting of already-cleaned updates."""
    if blend <= 0.0 or root_delta is None:
        return agg_weights
    anchor_scores = _root_anchor_trust_scores(client_deltas, root_delta)
    base_weights = _normalize_weights(agg_weights)
    anchor_weights = _normalize_weights(anchor_scores)
    blend = float(np.clip(blend, 0.0, 1.0))
    return [
        (1.0 - blend) * base + blend * anchor
        for base, anchor in zip(base_weights, anchor_weights)
    ]


def _aggregate_fltrust(
    deltas: list[OrderedDict[str, torch.Tensor]],
    root_delta: OrderedDict[str, torch.Tensor],
) -> tuple[OrderedDict[str, torch.Tensor], list[float]]:
    """FLTrust-style ReLU cosine trust with root-norm normalization."""
    root_vector = flatten_state_dict(root_delta).float()
    root_norm = float(root_vector.norm().item())
    if root_norm <= 1e-12:
        return clone_state_dict(root_delta), [0.0] * len(deltas)

    normalized_vectors: list[torch.Tensor] = []
    trust_scores: list[float] = []
    for delta in deltas:
        vector = flatten_state_dict(delta).float()
        vector_norm = float(vector.norm().item())
        if vector_norm <= 1e-12:
            normalized_vectors.append(torch.zeros_like(root_vector))
            trust_scores.append(0.0)
            continue
        cosine = float(torch.dot(vector, root_vector).item() / (vector_norm * root_norm))
        trust_scores.append(max(0.0, cosine))
        normalized_vectors.append(vector * (root_norm / vector_norm))

    trust_total = sum(trust_scores)
    if trust_total <= 1e-12:
        return clone_state_dict(root_delta), [0.0] * len(deltas)
    weights = [score / trust_total for score in trust_scores]
    aggregate_vector = sum(weight * vector for weight, vector in zip(weights, normalized_vectors))
    return _state_from_flat_vector(deltas[0], aggregate_vector), weights


def _aggregate_geometric_median(
    deltas: list[OrderedDict[str, torch.Tensor]],
    sample_weights: list[float],
    max_iterations: int,
    tolerance: float,
) -> tuple[OrderedDict[str, torch.Tensor], list[float]]:
    """Weighted Weiszfeld geometric median used by robust federated aggregation."""
    if len(deltas) == 1:
        return clone_state_dict(deltas[0]), [1.0]
    points = torch.stack([flatten_state_dict(delta).float() for delta in deltas], dim=0)
    base_weights = torch.tensor(_normalize_weights(sample_weights), dtype=points.dtype)
    estimate = torch.sum(points * base_weights[:, None], dim=0)
    effective_weights = base_weights
    for _ in range(max_iterations):
        distances = torch.linalg.vector_norm(points - estimate[None, :], dim=1)
        coincident = distances <= tolerance
        if bool(coincident.any()):
            chosen = int(torch.argmax(coincident.to(torch.int64)).item())
            effective_weights = torch.zeros_like(base_weights)
            effective_weights[chosen] = 1.0
            estimate = points[chosen]
            break
        effective_weights = base_weights / torch.clamp(distances, min=tolerance)
        effective_weights = effective_weights / effective_weights.sum()
        updated = torch.sum(points * effective_weights[:, None], dim=0)
        if float(torch.linalg.vector_norm(updated - estimate).item()) <= tolerance:
            estimate = updated
            break
        estimate = updated
    return _state_from_flat_vector(deltas[0], estimate), [float(value) for value in effective_weights.tolist()]


def _aggregate_flame(
    deltas: list[OrderedDict[str, torch.Tensor]],
    min_cluster_size: int = 0,
    noise_multiplier: float = 0.0,
    seed: int | None = None,
) -> tuple[OrderedDict[str, torch.Tensor], list[float]]:
    """FLAME-style cosine clustering, median norm clipping, and optional Gaussian noise."""
    if len(deltas) == 1:
        return clone_state_dict(deltas[0]), [1.0]

    points = torch.stack([flatten_state_dict(delta).float() for delta in deltas], dim=0)
    norms = torch.linalg.vector_norm(points, dim=1)
    normalized = torch.nn.functional.normalize(points, dim=1)
    cosine_distance = torch.clamp(1.0 - normalized @ normalized.T, min=0.0, max=2.0).cpu().numpy()
    np.fill_diagonal(cosine_distance, 0.0)

    requested_cluster_size = min_cluster_size if min_cluster_size > 0 else (len(deltas) // 2 + 1)
    requested_cluster_size = max(2, min(len(deltas), requested_cluster_size))
    try:
        from sklearn.cluster import HDBSCAN

        labels = HDBSCAN(
            metric="precomputed",
            min_cluster_size=requested_cluster_size,
            min_samples=1,
            allow_single_cluster=True,
        ).fit_predict(cosine_distance)
    except Exception:
        labels = np.zeros(len(deltas), dtype=int)

    non_noise_labels = [label for label in sorted(set(labels.tolist())) if label >= 0]
    if non_noise_labels:
        cluster_label = max(non_noise_labels, key=lambda label: int(np.sum(labels == label)))
        selected_indices = [index for index, label in enumerate(labels.tolist()) if label == cluster_label]
    else:
        selected_indices = list(range(len(deltas)))
    if not selected_indices:
        selected_indices = list(range(len(deltas)))

    clip_norm = float(torch.median(norms).item())
    selected_vectors = []
    for index in selected_indices:
        vector = points[index].clone()
        vector_norm = float(norms[index].item())
        if clip_norm > 0.0 and vector_norm > clip_norm:
            vector *= clip_norm / max(vector_norm, 1e-12)
        selected_vectors.append(vector)
    aggregate_vector = torch.stack(selected_vectors, dim=0).mean(dim=0)

    if noise_multiplier > 0.0 and clip_norm > 0.0:
        generator = torch.Generator(device=aggregate_vector.device)
        if seed is not None:
            generator.manual_seed(int(seed))
        noise = torch.normal(
            mean=0.0,
            std=float(noise_multiplier * clip_norm),
            size=aggregate_vector.shape,
            generator=generator,
            dtype=aggregate_vector.dtype,
            device=aggregate_vector.device,
        )
        aggregate_vector = aggregate_vector + noise

    weights = [0.0] * len(deltas)
    selected_weight = 1.0 / len(selected_indices)
    for index in selected_indices:
        weights[index] = selected_weight
    return _state_from_flat_vector(deltas[0], aggregate_vector), weights


def _aggregate_krum(
    deltas: list[OrderedDict[str, torch.Tensor]],
    malicious_fraction: float,
) -> tuple[OrderedDict[str, torch.Tensor], list[float]]:
    if len(deltas) == 1:
        return clone_state_dict(deltas[0]), [1.0]
    if len(deltas) < 4:
        return average_deltas(deltas, [1.0] * len(deltas)), _normalize_weights([1.0] * len(deltas))
    stacked = torch.stack([flatten_state_dict(delta).float() for delta in deltas], dim=0)
    client_count = stacked.size(0)
    byzantine = min(max(0, int(round(client_count * malicious_fraction))), max(0, client_count - 3))
    neighbor_count = max(1, min(client_count - 1, client_count - byzantine - 2))
    distances = torch.cdist(stacked, stacked, p=2).pow(2)
    scores: list[float] = []
    for index in range(client_count):
        row = torch.cat([distances[index, :index], distances[index, index + 1:]])
        nearest = torch.topk(row, k=neighbor_count, largest=False).values
        scores.append(float(nearest.sum().item()))
    ranking = np.argsort(scores)
    candidate_count = max(1, min(client_count, client_count - byzantine - 2))
    chosen = ranking[:candidate_count].tolist()
    weights = [1.0 if index in chosen else 0.0 for index in range(client_count)]
    return average_deltas([deltas[index] for index in chosen], [1.0] * len(chosen)), _normalize_weights(weights)


def _aggregate_bulyan(
    deltas: list[OrderedDict[str, torch.Tensor]],
    malicious_fraction: float,
) -> tuple[OrderedDict[str, torch.Tensor], list[float]]:
    """Bulyan with iterative Krum selection and coordinate-wise trimmed averaging.

    Bulyan requires at least ``4f + 3`` submitted updates for a Byzantine budget
    ``f``.  When that condition is not met, the implementation deliberately falls
    back to coordinate-wise trimmed mean rather than claiming a Bulyan guarantee.
    """
    if len(deltas) == 1:
        return clone_state_dict(deltas[0]), [1.0]
    points = torch.stack([flatten_state_dict(delta).float() for delta in deltas], dim=0)
    client_count = points.size(0)
    byzantine = _bulyan_byzantine_count(client_count, malicious_fraction)
    if byzantine == 0:
        return average_deltas(deltas, [1.0] * len(deltas)), _normalize_weights([1.0] * len(deltas))
    if client_count < 4 * byzantine + 3:
        return _aggregate_trimmed_mean(deltas, byzantine / client_count)

    candidate_count = client_count - 2 * byzantine
    remaining = list(range(client_count))
    selected_indices: list[int] = []
    while len(selected_indices) < candidate_count:
        remaining_points = points[remaining]
        distances = torch.cdist(remaining_points, remaining_points, p=2).pow(2)
        neighbor_count = max(1, len(remaining) - byzantine - 2)
        scores: list[float] = []
        for local_index in range(len(remaining)):
            row = torch.cat([distances[local_index, :local_index], distances[local_index, local_index + 1:]])
            scores.append(float(torch.topk(row, k=min(neighbor_count, row.numel()), largest=False).values.sum().item()))
        chosen_local_index = int(np.argmin(scores))
        selected_indices.append(remaining.pop(chosen_local_index))

    candidates = points[selected_indices]
    sorted_candidates = torch.sort(candidates, dim=0).values
    trimmed = sorted_candidates[byzantine:candidate_count - byzantine]
    aggregate_vector = trimmed.mean(dim=0) if trimmed.size(0) else candidates.mean(dim=0)
    weights = [1.0 if index in selected_indices else 0.0 for index in range(client_count)]
    return _state_from_flat_vector(deltas[0], aggregate_vector), _normalize_weights(weights)


def _bulyan_byzantine_count(client_count: int, malicious_fraction: float) -> int:
    """Conservative integer Byzantine budget used for Bulyan feasibility checks."""
    return max(0, int(math.ceil(client_count * malicious_fraction - 1e-12)))


def _bulyan_cohort_is_valid(client_count: int, malicious_fraction: float) -> bool:
    byzantine = _bulyan_byzantine_count(client_count, malicious_fraction)
    return byzantine == 0 or client_count >= 4 * byzantine + 3


def _controller_control_state_bytes(controller: AgentLockController | None) -> int:
    """Logical persistent controller-state footprint, excluding shared task/model data."""
    if controller is None:
        return 0
    client_state_names = (
        "base_rarity", "trust_scores", "suspicion_scores", "rare_scores",
        "selection_reward_memory", "shared_exposure_scores", "stable_overlap_scores",
        "suspicious_overlap_scores", "contamination_memory", "deception_scores",
        "stability_scores", "previous_cosine_scores", "previous_norm_scores",
        "previous_clean_gains", "previous_rare_gains", "previous_backdoor_risks",
    )
    client_state_bytes = sum(len(getattr(controller, name, {})) * 8 for name in client_state_names)
    bandit_bytes = 0
    for bandit_name in ("bandit", "selection_bandit"):
        bandit = getattr(controller, bandit_name, None)
        if bandit is None:
            continue
        bandit_bytes += sum(matrix.nbytes for matrix in bandit._matrices.values())
        bandit_bytes += sum(vector.nbytes for vector in bandit._vectors.values())
        bandit_bytes += len(bandit._counts) * 8
    mask_bytes = sum(tensor.numel() * tensor.element_size() for tensor in controller.shared_core_masks().values())
    return int(client_state_bytes + bandit_bytes + mask_bytes)


def _aggregate_foolsgold(
    deltas: list[OrderedDict[str, torch.Tensor]],
    selected: list[int],
    sample_weights: list[float],
    history: dict[int, torch.Tensor],
) -> tuple[OrderedDict[str, torch.Tensor], list[float]]:
    if len(deltas) == 1:
        history[selected[0]] = flatten_state_dict(deltas[0]).float()
        return clone_state_dict(deltas[0]), [1.0]
    history_vectors: list[torch.Tensor] = []
    for client_id, delta in zip(selected, deltas):
        update_vector = flatten_state_dict(delta).float()
        history[client_id] = 0.8 * history[client_id] + update_vector if client_id in history else update_vector
        history_vectors.append(history[client_id])
    stacked = torch.stack(history_vectors, dim=0)
    normalized = torch.nn.functional.normalize(stacked, dim=1)
    similarity = torch.clamp(normalized @ normalized.T, min=0.0, max=1.0)
    similarity.fill_diagonal_(0.0)
    max_similarity = similarity.max(dim=1).values
    pardoned = similarity.clone()
    for left in range(similarity.size(0)):
        for right in range(similarity.size(1)):
            if left == right:
                continue
            if max_similarity[left] < max_similarity[right] and max_similarity[right] > 0:
                pardoned[left, right] *= max_similarity[left] / max_similarity[right]
    foolsgold_scores = torch.clamp(1.0 - pardoned.max(dim=1).values, min=0.0)
    if float(foolsgold_scores.max().item()) > 0:
        foolsgold_scores = foolsgold_scores / foolsgold_scores.max()
    weights = [float(foolsgold_scores[index].item() * sample_weights[index]) for index in range(len(deltas))]
    if sum(weights) <= 0:
        weights = list(sample_weights)
    return average_deltas(deltas, weights), _normalize_weights(weights)


def _static_template_assignments(task_data: TaskData, mode: str) -> dict[int, str]:
    assignments: dict[int, str] = {}
    for client_id, profile in task_data.client_profiles.items():
        rarity = profile.rarity_score
        bandwidth = profile.bandwidth_score
        if mode == "communication_only":
            if bandwidth >= 0.82:
                template = "shared_large"
            elif bandwidth >= 0.55:
                template = "shared_small"
            else:
                template = "quarantine"
        else:
            if rarity >= 0.72 and bandwidth >= 0.45:
                template = "hybrid_rare"
            elif bandwidth < 0.45:
                template = "shared_small"
            else:
                template = "shared_large"
        assignments[client_id] = template
    return assignments


def _static_action(
    controller: AgenticFedLockController,
    client_id: int,
    round_id: int,
    template: str,
    selection_policy_mode: str,
) -> ClientAction:
    action = controller.action_for_template(client_id, round_id, template)
    action.policy_mode = "static"
    action.heuristic_template = template
    action.bandit_template = template
    action.selection_policy_mode = selection_policy_mode
    action.selection_heuristic_score = 0.0
    action.selection_bandit_score = 0.0
    action.selection_combined_score = 0.0
    return action


def _full_overlap_state_features(
    selected: list[int],
    controller: AgenticFedLockController,
) -> dict[int, dict[str, float]]:
    features: dict[int, dict[str, float]] = {}
    for client_id in selected:
        peer_ids = [peer_id for peer_id in selected if peer_id != client_id]
        if peer_ids:
            stable_overlap = float(np.clip(np.mean([controller.effective_trust(peer_id) for peer_id in peer_ids]), 0.0, 1.0))
            suspicious_overlap = float(np.clip(1.35 * np.mean([controller.suspicion_scores[peer_id] for peer_id in peer_ids]), 0.0, 1.0))
        else:
            stable_overlap = 0.0
            suspicious_overlap = 0.0
        features[client_id] = {
            "shared_exposure": 1.0,
            "stable_overlap": stable_overlap,
            "suspicious_overlap": suspicious_overlap,
        }
    return features


def _local_attack_train(
    global_state: OrderedDict[str, torch.Tensor],
    client_loader: DataLoader,
    task_data: TaskData,
    config: SimulationConfig,
    device: str,
    parameter_masks: dict[str, torch.Tensor],
    is_malicious: bool,
    prox_mu: float,
    round_id: int,
) -> OrderedDict[str, torch.Tensor]:
    if not is_malicious:
        return _masked_local_train(
            global_state=global_state,
            client_loader=client_loader,
            task_data=task_data,
            config=config,
            device=device,
            parameter_masks=parameter_masks,
            is_malicious=False,
            prox_mu=prox_mu,
            round_id=round_id,
        )
    if config.attack_mode == "adaptive_mimic":
        clean_delta = _masked_local_train(
            global_state=global_state,
            client_loader=client_loader,
            task_data=task_data,
            config=config,
            device=device,
            parameter_masks=parameter_masks,
            is_malicious=False,
            prox_mu=prox_mu,
            round_id=round_id,
        )
        poison_delta = _masked_local_train(
            global_state=global_state,
            client_loader=client_loader,
            task_data=task_data,
            config=config,
            device=device,
            parameter_masks=parameter_masks,
            is_malicious=True,
            poison_fraction=max(0.08, 0.85 * config.poison_fraction),
            prox_mu=prox_mu,
            round_id=round_id,
        )
        blended = _blend_deltas(poison_delta, clean_delta, config.adaptive_attack_blend)
        clean_norm = float(flatten_state_dict(clean_delta).norm().item())
        blended_norm = float(flatten_state_dict(blended).norm().item())
        target_norm = clean_norm * (1.02 + 0.08 * config.adaptive_attack_blend)
        if blended_norm > 0 and target_norm > 0:
            _scale_delta(blended, target_norm / blended_norm)
        return blended
    poison_fraction = config.poison_fraction
    if config.attack_mode == "distributed_backdoor":
        poison_fraction = max(0.08, 0.4 * config.poison_fraction)
    return _masked_local_train(
        global_state=global_state,
        client_loader=client_loader,
        task_data=task_data,
        config=config,
        device=device,
        parameter_masks=parameter_masks,
        is_malicious=True,
        poison_fraction=poison_fraction,
        prox_mu=prox_mu,
        round_id=round_id,
    )


def _postprocess_attack_deltas(
    client_deltas: list[OrderedDict[str, torch.Tensor]],
    selected: list[int],
    task_data: TaskData,
    agg_weights: list[float],
    config: SimulationConfig,
) -> None:
    malicious_indices = [index for index, client_id in enumerate(selected) if task_data.client_profiles[client_id].is_malicious]
    if not malicious_indices:
        return
    if config.attack_mode == "model_replacement":
        total_weight = sum(agg_weights)
        malicious_weight = sum(agg_weights[index] for index in malicious_indices)
        if malicious_weight > 0:
            scale = max(1.0, config.model_replacement_scale * total_weight / malicious_weight)
            for index in malicious_indices:
                _scale_delta(client_deltas[index], scale)
    elif config.attack_mode == "colluding":
        coalition_delta = average_deltas([client_deltas[index] for index in malicious_indices], [1.0] * len(malicious_indices))
        for index in malicious_indices:
            client_deltas[index] = _blend_deltas(coalition_delta, client_deltas[index], config.collusion_strength)
    elif config.attack_mode == "distributed_backdoor":
        coalition_delta = average_deltas([client_deltas[index] for index in malicious_indices], [1.0] * len(malicious_indices))
        for index in malicious_indices:
            client_deltas[index] = _blend_deltas(coalition_delta, client_deltas[index], 0.65)
            _scale_delta(client_deltas[index], config.distributed_attack_scale)


def _aggregate_for_strategy(
    spec: StrategySpec,
    client_deltas: list[OrderedDict[str, torch.Tensor]],
    agg_weights: list[float],
    selected: list[int],
    foolsgold_history: dict[int, torch.Tensor],
    config: SimulationConfig,
    root_delta: OrderedDict[str, torch.Tensor] | None = None,
    round_id: int = 0,
    norm_confidence: float = 0.0,
) -> tuple[OrderedDict[str, torch.Tensor], list[float]]:
    if spec.aggregation_mode == "weighted_mean":
        # Scale the configured blend by how UNconfident Mechanism E (norm-inflation)
        # is this round. When norm_confidence is high (a client is clearly
        # norm-inflated -- the model-replacement signature), repair has already
        # handled it via suppression; blending in a noisier global root-cosine
        # signal on top only dilutes that correct, already-targeted suppression
        # (measured: it made CIFAR-10/EMNIST model-replacement worse, not
        # better). When norm_confidence is low (nobody looks norm-inflated --
        # e.g. a norm-matched attack like adaptive mimic, which defeats
        # Mechanism E by design), lean on the root-anchor signal, since it's
        # the only lever left that catches this attack family.
        #
        # Gate is a smoothstep, not linear, calibrated against measured
        # per-round norm_confidence traces: EMNIST model-replacement sits at
        # exactly 1.0 in 39/40 rounds (one dip to 0.58); MNIST adaptive-mimic
        # never exceeds ~0.66 across 12 rounds (median 0.39, min 0.13) --
        # there is no genuine norm-inflation evidence there since the attack
        # norm-matches by construction, so its whole observed range should
        # engage the anchor near full strength, not the ~40% a linear
        # 1-norm_confidence scaling left on the table.
        gate = _norm_confidence_gate(norm_confidence)
        effective_blend = config.root_anchor_weight * gate
        effective_weights = _apply_root_anchor_blend(
            client_deltas, agg_weights, root_delta, effective_blend,
        )
        return average_deltas(client_deltas, effective_weights), _normalize_weights(effective_weights)
    if spec.aggregation_mode == "trimmed_mean":
        return _aggregate_trimmed_mean(client_deltas, config.trim_ratio)
    if spec.aggregation_mode == "krum":
        return _aggregate_krum(client_deltas, config.malicious_fraction)
    if spec.aggregation_mode == "foolsgold":
        return _aggregate_foolsgold(client_deltas, selected, agg_weights, foolsgold_history)
    if spec.aggregation_mode == "fltrust":
        if root_delta is None:
            raise ValueError("FLTrust requires a server root update")
        return _aggregate_fltrust(client_deltas, root_delta)
    if spec.aggregation_mode == "rfa":
        return _aggregate_geometric_median(
            client_deltas,
            agg_weights,
            max_iterations=config.rfa_max_iterations,
            tolerance=config.rfa_tolerance,
        )
    if spec.aggregation_mode == "flame":
        return _aggregate_flame(
            client_deltas,
            min_cluster_size=config.flame_min_cluster_size,
            noise_multiplier=config.flame_noise_multiplier,
            seed=config.seed * 1_000_003 + round_id,
        )
    if spec.aggregation_mode == "bulyan":
        byzantine_fraction = (
            config.malicious_fraction
            if config.bulyan_byzantine_fraction < 0.0
            else config.bulyan_byzantine_fraction
        )
        return _aggregate_bulyan(client_deltas, byzantine_fraction)
    raise ValueError(f"Unsupported aggregation mode: {spec.aggregation_mode}")


def run_strategy(
    strategy_name: str,
    task_data: TaskData,
    config: SimulationConfig,
) -> dict[str, float | int | str]:
    set_global_seed(config.seed)
    device = _resolve_execution_device(config.device)
    spec = _resolve_strategy_spec(strategy_name, config)
    controller = _build_controller(task_data, config, spec)
    model = build_model(task_data.model_spec).to(device)
    # The canonical federated state, masks, and aggregation deltas stay on CPU.
    # Individual models are copied to the execution device only for training or
    # evaluation, keeping all aggregation-side diagnostics device-consistent.
    global_state = OrderedDict(
        (name, tensor.detach().to(device="cpu").clone())
        for name, tensor in model.state_dict().items()
    )
    probe_loader, test_loader = _build_loaders(task_data, config)
    client_ids = sorted(task_data.client_profiles.keys())
    full_mask = _full_masks(global_state)
    full_param_count = sum(tensor.numel() for tensor in global_state.values())
    full_model_bytes = sum(tensor.numel() * tensor.element_size() for tensor in global_state.values())
    value_bytes_per_parameter = full_model_bytes / max(1, full_param_count)
    dense_mask_metadata_bytes_per_client = math.ceil(full_param_count / 8)
    shared_core_mask = controller.shared_core_masks() if controller else full_mask
    classifier_mask: dict[str, torch.Tensor] | None = None
    if config.repair_signal_mode == "subspace_consensus":
        classifier_names = {task_data.model_spec.classifier_weight_name, task_data.model_spec.classifier_bias_name}
        if classifier_names & set(global_state.keys()):
            classifier_mask = {
                name: (torch.ones_like(tensor) if name in classifier_names else torch.zeros_like(tensor))
                for name, tensor in global_state.items()
            }
    static_templates = (
        _static_template_assignments(task_data, spec.mask_mode)
        if controller and spec.mask_mode in {"static_subspace", "communication_only"}
        else {}
    )
    foolsgold_history: dict[int, torch.Tensor] = {}
    simulator_oracle_enabled = config.counterfactual_mode == "simulator_oracle"
    counterfactual_enabled = (
        spec.enable_counterfactual
        and simulator_oracle_enabled
        and config.attack_mode in {"badnets", "adaptive_mimic"}
    )
    defense_has_trigger_knowledge = config.defender_knowledge == "exact"

    round_rows: list[dict[str, float | int | str]] = []
    decision_rows: list[dict[str, float | int | str | bool]] = []
    counterfactual_rows: list[dict[str, float | int | str | bool]] = []
    selection_counterfactual_rows: list[dict[str, float | int | str | bool]] = []
    diagnostic_rows: list[dict[str, float | int | str]] = []
    total_local_training_jobs = 0
    total_server_root_training_jobs = 0
    total_probe_evaluations = 0
    total_stage_seconds: dict[str, float] = defaultdict(float)
    total_uplink_bytes = 0
    total_downlink_model_bytes = 0
    total_mask_metadata_bytes = 0
    total_full_payload_bytes = 0
    controller_state_bytes = _controller_control_state_bytes(controller)
    process = psutil.Process() if psutil is not None else None
    initial_process_rss_bytes = int(process.memory_info().rss) if process is not None else 0
    peak_process_rss_bytes = initial_process_rss_bytes
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start_time = time.time()

    for round_id in range(1, config.rounds + 1):
        stage_start = time.perf_counter()
        if controller and spec.selection_mode == "controller":
            selected = controller.select_clients(client_ids, config.client_fraction, round_id)
        else:
            selected = _sample_clients(config, client_ids, round_id)
        total_stage_seconds["selection"] += time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        global_model = build_model(task_data.model_spec).to(device)
        global_model.load_state_dict(_state_to_device(global_state, device))
        global_probe_metrics = _evaluate_probe_bundle(
            global_model,
            probe_loader,
            task_data,
            device,
            include_backdoor_signal=defense_has_trigger_knowledge,
        )
        total_probe_evaluations += 1
        root_delta: OrderedDict[str, torch.Tensor] | None = None
        needs_root_delta = spec.aggregation_mode == "fltrust" or (
            config.repair_signal_mode == "subspace_consensus" and spec.enable_repair
        )
        if needs_root_delta:
            root_delta = _masked_local_train(
                global_state=global_state,
                client_loader=probe_loader,
                task_data=task_data,
                config=config,
                device=device,
                parameter_masks=full_mask,
                is_malicious=False,
                prox_mu=0.0,
                round_id=round_id,
            )
            total_server_root_training_jobs += 1
        total_stage_seconds["global_probe_root"] += time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        if controller and spec.mask_mode in {"dynamic", "selection_only"}:
            actions = controller.choose_actions(selected, round_id)
            if spec.mask_mode == "selection_only":
                full_bundle = MaskBundle(full_mask, full_param_count, 1.0)
                actions = {
                    client_id: replace(
                        action,
                        mask_bundle=full_bundle,
                        policy_mode="selection_only",
                    )
                    for client_id, action in actions.items()
                }
        elif controller and spec.mask_mode in {"static_subspace", "communication_only"}:
            actions = {
                client_id: _static_action(controller, client_id, round_id, static_templates[client_id], "random_static")
                for client_id in selected
            }
        else:
            actions = {}

        selection_pool_metadata = controller.selection_metadata_snapshot() if controller and spec.selection_mode == "controller" else {}
        selection_counterfactual_enabled = (
            bool(selection_pool_metadata)
            and simulator_oracle_enabled
            and config.attack_mode in {"badnets", "adaptive_mimic"}
        )
        selection_counterfactual_clients = _select_selection_counterfactual_clients(
            selection_pool_metadata,
            selected,
            config.selection_counterfactual_candidates_per_round,
        ) if selection_counterfactual_enabled else []
        counterfactual_clients = _select_counterfactual_clients(actions, config.counterfactual_clients_per_round) if counterfactual_enabled else []
        total_stage_seconds["allocation"] += time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        client_deltas: list[OrderedDict[str, torch.Tensor]] = []
        agg_weights: list[float] = []
        telemetry: dict[int, dict[str, float]] = {}
        chosen_effects: dict[int, dict[str, float]] = {}
        bandit_feedback_rows: list[dict[str, object]] = []
        selection_feedback_rows: list[dict[str, object]] = []
        active_params_total = 0
        overlap_total = 0.0
        repair_pressure = 1.0
        repair_threshold = config.repair_risk_threshold
        repair_count = 0
        repair_suppression_total = 0.0
        selected_selection_rewards: list[float] = []
        alternate_selection_rewards: list[float] = []

        for client_id in selected:
            profile = task_data.client_profiles[client_id]
            loader = _client_loader(task_data, client_id, config, round_id)
            action = actions.get(client_id)
            parameter_masks = action.mask_bundle.masks if action else full_mask
            if action:
                active_params_total += action.mask_bundle.active_params
                overlap_total += action.mask_bundle.overlap_ratio
            else:
                active_params_total += full_param_count
                overlap_total += 1.0 if controller else 0.0

            agg_weight = float(profile.sample_count)
            if spec.trust_weighting and action is not None:
                agg_weight *= float(action.aggregation_weight)
            delta = _local_attack_train(
                global_state=global_state,
                client_loader=loader,
                task_data=task_data,
                config=config,
                device=device,
                parameter_masks=parameter_masks,
                is_malicious=profile.is_malicious,
                prox_mu=spec.prox_mu,
                round_id=round_id,
            )
            total_local_training_jobs += _local_training_job_cost(config, profile.is_malicious)
            client_deltas.append(delta)
            agg_weights.append(agg_weight)

        _postprocess_attack_deltas(client_deltas, selected, task_data, agg_weights, config)
        pre_repair_client_deltas = [clone_state_dict(delta) for delta in client_deltas]
        pre_repair_agg_weights = list(agg_weights)
        total_stage_seconds["local_training"] += time.perf_counter() - stage_start

        trigger_agnostic_risks: dict[int, float] = {}
        norm_confidence = 0.0
        if config.repair_signal_mode == "subspace_consensus" and controller and spec.enable_repair and selected:
            trigger_agnostic_risks, norm_confidence = _trigger_agnostic_risk(
                client_deltas,
                selected,
                controller,
                shared_core_mask,
                classifier_mask,
                root_delta,
                config.trust_consensus_fraction,
                slrt_localization_mode=config.slrt_localization_mode,
                risk_fusion_order=config.risk_fusion_order,
            )

        stage_start = time.perf_counter()
        if controller and actions:
            for index, client_id in enumerate(selected):
                profile = task_data.client_profiles[client_id]
                action = actions[client_id]
                delta = client_deltas[index]
                candidate_metrics = _evaluate_delta_on_probe(
                    global_state,
                    delta,
                    probe_loader,
                    task_data,
                    config,
                    device,
                    include_backdoor_signal=defense_has_trigger_knowledge,
                )
                total_probe_evaluations += 1
                effects = _probe_effects(global_probe_metrics, candidate_metrics)
                if client_id in trigger_agnostic_risks:
                    effects = dict(effects)
                    effects["backdoor_risk"] = trigger_agnostic_risks[client_id]
                chosen_effects[client_id] = effects
                if spec.update_controller_scores:
                    telemetry[client_id] = {
                        "clean_gain": effects["clean_gain"],
                        "rare_gain": effects["rare_gain"],
                        "backdoor_risk": effects["backdoor_risk"],
                        "norm_ratio": 1.0,
                        "cosine": 0.0,
                    }
                if spec.update_bandits:
                    chosen_comm_ratio = action.mask_bundle.active_params / full_param_count
                    chosen_reward = _decision_reward(
                        effects["clean_gain"],
                        effects["rare_gain"],
                        effects["backdoor_risk"],
                        chosen_comm_ratio,
                        action.mask_bundle.overlap_ratio,
                    )
                    if counterfactual_enabled and client_id in counterfactual_clients:
                        for template in controller.template_names():
                            if template == action.template:
                                cf_action = action
                                cf_effects = effects
                            else:
                                cf_action = controller.action_for_template(client_id, round_id, template)
                                cf_loader = _client_loader(task_data, client_id, config, round_id)
                                cf_delta = _local_attack_train(
                                    global_state=global_state,
                                    client_loader=cf_loader,
                                    task_data=task_data,
                                    config=config,
                                    device=device,
                                    parameter_masks=cf_action.mask_bundle.masks,
                                    is_malicious=profile.is_malicious,
                                    prox_mu=spec.prox_mu,
                                    round_id=round_id,
                                )
                                total_local_training_jobs += _local_training_job_cost(config, profile.is_malicious)
                                cf_metrics = _evaluate_delta_on_probe(
                                    global_state,
                                    cf_delta,
                                    probe_loader,
                                    task_data,
                                    config,
                                    device,
                                    include_backdoor_signal=defense_has_trigger_knowledge,
                                )
                                total_probe_evaluations += 1
                                cf_effects = _probe_effects(global_probe_metrics, cf_metrics)
                            cf_comm_ratio = cf_action.mask_bundle.active_params / full_param_count
                            cf_reward = _decision_reward(
                                cf_effects["clean_gain"],
                                cf_effects["rare_gain"],
                                cf_effects["backdoor_risk"],
                                cf_comm_ratio,
                                cf_action.mask_bundle.overlap_ratio,
                            )
                            counterfactual_rows.append(
                                {
                                    "round": round_id,
                                    "client_id": client_id,
                                    "is_malicious": profile.is_malicious,
                                    "policy_selected": template == action.template,
                                    "selected_policy_mode": action.policy_mode,
                                    "selection_policy_mode": action.selection_policy_mode,
                                    "evaluated_policy_mode": cf_action.policy_mode,
                                    "chosen_template": action.template,
                                    "evaluated_template": template,
                                    "heuristic_template": cf_action.heuristic_template,
                                    "bandit_template": cf_action.bandit_template,
                                    "trust": round(cf_action.trust, 6),
                                    "suspicion": round(cf_action.suspicion, 6),
                                    "rarity": round(cf_action.rarity, 6),
                                    "shared_exposure": round(cf_action.shared_exposure, 6),
                                    "stable_overlap": round(cf_action.stable_overlap, 6),
                                    "suspicious_overlap": round(cf_action.suspicious_overlap, 6),
                                    "contamination_memory": round(cf_action.contamination_memory, 6),
                                    "communication_ratio": round(cf_comm_ratio, 6),
                                    "overlap_ratio": round(cf_action.mask_bundle.overlap_ratio, 6),
                                    "heuristic_score": round(cf_action.heuristic_score, 6),
                                    "bandit_score": round(cf_action.bandit_score, 6),
                                    "combined_score": round(cf_action.combined_score, 6),
                                    "clean_gain": round(cf_effects["clean_gain"], 6),
                                    "rare_gain": round(cf_effects["rare_gain"], 6),
                                    "backdoor_risk": round(cf_effects["backdoor_risk"], 6),
                                    "reward_score": round(cf_reward, 6),
                                }
                            )
                            bandit_feedback_rows.append(
                                {
                                    "client_id": client_id,
                                    "template": template,
                                    "context_vector": cf_action.context_vector,
                                    "local_reward": cf_reward,
                                }
                            )
                    else:
                        bandit_feedback_rows.append(
                            {
                                "client_id": client_id,
                                "template": action.template,
                                "context_vector": action.context_vector,
                                "local_reward": chosen_reward,
                            }
                        )
                    if spec.selection_mode == "controller":
                        selection_reward_value = _selection_feedback_reward(
                            effects["clean_gain"],
                            effects["rare_gain"],
                            effects["backdoor_risk"],
                            controller.effective_rarity(client_id),
                            controller.stability_scores[client_id],
                            controller.deception_scores[client_id],
                        )
                        selected_selection_rewards.append(selection_reward_value)
                        selection_feedback_rows.append(
                            {
                                "client_id": client_id,
                                "local_reward": selection_reward_value,
                                "context_vector": selection_pool_metadata[client_id].context_vector,
                                "was_selected": True,
                            }
                        )
                        if simulator_oracle_enabled:
                            selection_counterfactual_rows.append(
                                {
                                    "round": round_id,
                                    "client_id": client_id,
                                    "was_selected": True,
                                    "selection_policy_mode": action.selection_policy_mode,
                                    "selection_heuristic_score": round(action.selection_heuristic_score, 6),
                                    "selection_bandit_score": round(action.selection_bandit_score, 6),
                                    "selection_combined_score": round(action.selection_combined_score, 6),
                                    "selection_reward_memory": round(action.selection_reward_memory, 6),
                                    "template": action.template,
                                    "communication_ratio": round(chosen_comm_ratio, 6),
                                    "selection_reward_score": round(selection_reward_value, 6),
                                    "clean_gain": round(effects["clean_gain"], 6),
                                    "rare_gain": round(effects["rare_gain"], 6),
                                    "backdoor_risk": round(effects["backdoor_risk"], 6),
                                    "trust": round(action.trust, 6),
                                    "suspicion": round(action.suspicion, 6),
                                    "rarity": round(action.rarity, 6),
                                    "deception_score": round(controller.deception_scores[client_id], 6),
                                    "stability_score": round(controller.stability_scores[client_id], 6),
                                }
                            )

        if selection_counterfactual_clients:
            for alt_client_id in selection_counterfactual_clients:
                alt_metadata = selection_pool_metadata[alt_client_id]
                alt_action = controller.preview_action(alt_client_id, round_id)
                alt_loader = _client_loader(task_data, alt_client_id, config, round_id)
                alt_delta = _local_attack_train(
                    global_state=global_state,
                    client_loader=alt_loader,
                    task_data=task_data,
                    config=config,
                    device=device,
                    parameter_masks=alt_action.mask_bundle.masks,
                    is_malicious=task_data.client_profiles[alt_client_id].is_malicious,
                    prox_mu=spec.prox_mu,
                    round_id=round_id,
                )
                alt_is_malicious = task_data.client_profiles[alt_client_id].is_malicious
                total_local_training_jobs += _local_training_job_cost(config, alt_is_malicious)
                alt_metrics = _evaluate_delta_on_probe(
                    global_state,
                    alt_delta,
                    probe_loader,
                    task_data,
                    config,
                    device,
                    include_backdoor_signal=defense_has_trigger_knowledge,
                )
                total_probe_evaluations += 1
                alt_effects = _probe_effects(global_probe_metrics, alt_metrics)
                alt_selection_reward = _selection_feedback_reward(
                    alt_effects["clean_gain"],
                    alt_effects["rare_gain"],
                    alt_effects["backdoor_risk"],
                    controller.effective_rarity(alt_client_id),
                    controller.stability_scores[alt_client_id],
                    controller.deception_scores[alt_client_id],
                )
                alternate_selection_rewards.append(alt_selection_reward)
                selection_feedback_rows.append(
                    {
                        "client_id": alt_client_id,
                        "local_reward": alt_selection_reward,
                        "context_vector": alt_metadata.context_vector,
                        "was_selected": False,
                    }
                )
                selection_counterfactual_rows.append(
                    {
                        "round": round_id,
                        "client_id": alt_client_id,
                        "was_selected": False,
                        "selection_policy_mode": alt_metadata.policy_mode,
                        "selection_heuristic_score": round(alt_metadata.heuristic_score, 6),
                        "selection_bandit_score": round(alt_metadata.bandit_score, 6),
                        "selection_combined_score": round(alt_metadata.combined_score, 6),
                        "selection_reward_memory": round(alt_metadata.reward_memory, 6),
                        "template": alt_action.template,
                        "communication_ratio": round(alt_action.mask_bundle.active_params / full_param_count, 6),
                        "selection_reward_score": round(alt_selection_reward, 6),
                        "clean_gain": round(alt_effects["clean_gain"], 6),
                        "rare_gain": round(alt_effects["rare_gain"], 6),
                        "backdoor_risk": round(alt_effects["backdoor_risk"], 6),
                        "trust": round(alt_action.trust, 6),
                        "suspicion": round(alt_action.suspicion, 6),
                        "rarity": round(alt_action.rarity, 6),
                        "deception_score": round(controller.deception_scores[alt_client_id], 6),
                        "stability_score": round(controller.stability_scores[alt_client_id], 6),
                    }
                )

        if controller and spec.update_controller_scores:
            flat_updates = [flatten_state_dict(delta) for delta in client_deltas]
            stacked = torch.stack(flat_updates)
            centroid = torch.median(stacked, dim=0).values
            norms = stacked.norm(dim=1)
            median_norm = torch.median(norms)
            for client_id, update_vector, update_norm in zip(selected, flat_updates, norms):
                if client_id not in telemetry:
                    telemetry[client_id] = {
                        "clean_gain": 0.5,
                        "rare_gain": 0.5,
                        "backdoor_risk": 0.0,
                        "norm_ratio": 1.0,
                        "cosine": 0.0,
                    }
                cosine = torch.nn.functional.cosine_similarity(update_vector, centroid, dim=0).item()
                norm_ratio = torch.exp(-torch.abs(update_norm - median_norm) / (median_norm + 1e-6)).item()
                telemetry[client_id]["cosine"] = float(cosine)
                telemetry[client_id]["norm_ratio"] = float(norm_ratio)
            overlap_state = _overlap_state_features(selected, actions, controller) if actions else _full_overlap_state_features(selected, controller)
            for client_id in selected:
                telemetry[client_id].update(overlap_state[client_id])
                telemetry[client_id]["contamination_proxy"] = float(
                    np.clip(
                        0.55 * telemetry[client_id]["backdoor_risk"] * telemetry[client_id]["shared_exposure"]
                        + 0.25 * telemetry[client_id]["suspicious_overlap"]
                        + 0.2 * controller.contamination_memory[client_id],
                        0.0,
                        1.0,
                    )
                )
            controller.update_scores(telemetry)
            if spec.enable_repair and actions:
                repair_pressure, repair_threshold, repair_shared_scale, repair_weight_scale, repair_target_scale = _adaptive_repair_factors(config, global_probe_metrics)
                for index, client_id in enumerate(selected):
                    effective_trust = controller.effective_trust(client_id)
                    rare_credit = controller.effective_rarity(client_id)
                    current_norm = float(norms[index].item())
                    target_norm = float(median_norm.item() * (1.12 + 0.28 * effective_trust + 0.12 * rare_credit))
                    if target_norm > 0 and current_norm > target_norm:
                        _scale_delta(client_deltas[index], target_norm / current_norm)
                    effects = chosen_effects.get(client_id, {"backdoor_risk": 0.0})
                    risk_score = float(effects["backdoor_risk"])
                    contamination_proxy = float(telemetry[client_id]["contamination_proxy"])
                    if risk_score >= repair_threshold:
                        suppression = float(
                            np.clip(
                                config.repair_shared_suppression * repair_shared_scale * (0.7 * risk_score + 0.3 * contamination_proxy),
                                0.0,
                                0.98,
                            )
                        )
                        _suppress_shared_component(client_deltas[index], shared_core_mask, suppression)
                        repair_count += 1
                        repair_suppression_total += suppression
                        target_row_suppression = float(
                            np.clip(
                                config.repair_target_row_suppression * repair_target_scale * (0.75 * risk_score + 0.25 * contamination_proxy),
                                0.0,
                                0.995,
                            )
                        ) if defense_has_trigger_knowledge else 0.0
                        if target_row_suppression > 0.0:
                            _suppress_target_row_component(
                                client_deltas[index],
                                task_data.model_spec.classifier_weight_name,
                                task_data.model_spec.classifier_bias_name,
                                task_data.backdoor_target,
                                target_row_suppression,
                            )
                        agg_weights[index] *= max(0.08, 1.0 - config.repair_weight_penalty * repair_weight_scale * risk_score)
                    agg_weights[index] *= 0.85 + 0.15 * (0.75 * effective_trust + 0.25 * rare_credit)
            elif spec.trust_weighting:
                for index, client_id in enumerate(selected):
                    effective_trust = controller.effective_trust(client_id)
                    rare_credit = controller.effective_rarity(client_id)
                    agg_weights[index] *= 0.85 + 0.15 * (0.75 * effective_trust + 0.25 * rare_credit)
        else:
            norms = torch.stack([flatten_state_dict(delta).norm() for delta in client_deltas])
        total_stage_seconds["probe_feedback_repair"] += time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        aggregated_delta, normalized_weights = _aggregate_for_strategy(
            spec,
            client_deltas,
            agg_weights,
            selected,
            foolsgold_history,
            config,
            root_delta=root_delta,
            round_id=round_id,
            norm_confidence=norm_confidence,
        )
        bulyan_fraction = (
            config.malicious_fraction
            if config.bulyan_byzantine_fraction < 0.0
            else config.bulyan_byzantine_fraction
        )
        bulyan_cohort_valid = (
            spec.aggregation_mode != "bulyan"
            or _bulyan_cohort_is_valid(len(selected), bulyan_fraction)
        )
        effective_aggregation_mode = (
            "trimmed_mean_fallback"
            if spec.aggregation_mode == "bulyan" and not bulyan_cohort_valid
            else spec.aggregation_mode
        )
        malicious_indices = [index for index, client_id in enumerate(selected) if task_data.client_profiles[client_id].is_malicious]
        benign_indices = [index for index, client_id in enumerate(selected) if not task_data.client_profiles[client_id].is_malicious]
        pre_repair_weights = _normalize_weights(pre_repair_agg_weights)
        pre_repair_malicious_shared_norm = _group_projected_norm(pre_repair_client_deltas, pre_repair_weights, shared_core_mask, malicious_indices)
        pre_repair_benign_shared_norm = _group_projected_norm(pre_repair_client_deltas, pre_repair_weights, shared_core_mask, benign_indices)
        pre_repair_malicious_total_norm = _group_projected_norm(pre_repair_client_deltas, pre_repair_weights, full_mask, malicious_indices)
        pre_repair_benign_total_norm = _group_projected_norm(pre_repair_client_deltas, pre_repair_weights, full_mask, benign_indices)
        pre_repair_contamination_ratio = pre_repair_malicious_shared_norm / max(1e-8, pre_repair_benign_shared_norm)
        malicious_shared_norm = _group_projected_norm(client_deltas, normalized_weights, shared_core_mask, malicious_indices)
        benign_shared_norm = _group_projected_norm(client_deltas, normalized_weights, shared_core_mask, benign_indices)
        malicious_total_norm = _group_projected_norm(client_deltas, normalized_weights, full_mask, malicious_indices)
        benign_total_norm = _group_projected_norm(client_deltas, normalized_weights, full_mask, benign_indices)
        contamination_ratio = malicious_shared_norm / max(1e-8, benign_shared_norm)
        total_stage_seconds["aggregation"] += time.perf_counter() - stage_start

        if actions:
            bb_overlap: list[float] = []
            bm_overlap: list[float] = []
            mm_overlap: list[float] = []
            for left in range(len(selected)):
                for right in range(left + 1, len(selected)):
                    left_client = selected[left]
                    right_client = selected[right]
                    overlap = _pairwise_mask_overlap(actions[left_client].mask_bundle.masks, actions[right_client].mask_bundle.masks)
                    left_malicious = task_data.client_profiles[left_client].is_malicious
                    right_malicious = task_data.client_profiles[right_client].is_malicious
                    if left_malicious and right_malicious:
                        mm_overlap.append(overlap)
                    elif left_malicious or right_malicious:
                        bm_overlap.append(overlap)
                    else:
                        bb_overlap.append(overlap)
            counts = _template_counts(actions)
            policy_counts = _policy_mode_counts(actions)
            arm_counts = controller.bandit_counts() if controller and spec.update_bandits else {template: 0 for template in AgentLockController.TEMPLATE_NAMES}
            selection_bandit_count = controller.selection_bandit_count() if controller and spec.update_bandits else 0
            selection_bandit_selected = sum(action.selection_policy_mode == "bandit" for action in actions.values())
            selection_random_selected = sum(action.selection_policy_mode in {"random_warmup", "random_static"} for action in actions.values())
            selection_heuristic_selected = sum(action.selection_policy_mode == "heuristic_warmup" for action in actions.values())
            quota_adjusted_count = sum(action.quota_adjusted for action in actions.values())
        else:
            bb_overlap = [1.0]
            bm_overlap = [1.0]
            mm_overlap = [1.0]
            counts = {template: 0 for template in AgenticFedLockController.TEMPLATE_NAMES}
            policy_counts = {}
            arm_counts = {template: 0 for template in AgenticFedLockController.TEMPLATE_NAMES}
            selection_bandit_count = 0
            selection_bandit_selected = 0
            selection_random_selected = 0
            selection_heuristic_selected = 0
            quota_adjusted_count = 0

        for index, client_id in enumerate(selected):
            if client_id not in actions:
                continue
            action = actions[client_id]
            effects = chosen_effects.get(client_id, {"clean_gain": 0.0, "rare_gain": 0.0, "backdoor_risk": 0.0})
            communication_ratio = action.mask_bundle.active_params / full_param_count
            signal = telemetry.get(
                client_id,
                {
                    "cosine": 0.0,
                    "norm_ratio": 1.0,
                    "shared_exposure": action.mask_bundle.overlap_ratio,
                    "stable_overlap": 0.0,
                    "suspicious_overlap": 0.0,
                    "contamination_proxy": 0.0,
                },
            )
            decision_rows.append(
                {
                    "round": round_id,
                    "client_id": client_id,
                    "is_malicious": task_data.client_profiles[client_id].is_malicious,
                    "template": action.template,
                    "selection_policy_mode": action.selection_policy_mode,
                    "selection_heuristic_score": round(action.selection_heuristic_score, 6),
                    "selection_bandit_score": round(action.selection_bandit_score, 6),
                    "selection_combined_score": round(action.selection_combined_score, 6),
                    "selection_reward_memory": round(action.selection_reward_memory, 6),
                    "policy_mode": action.policy_mode,
                    "heuristic_template": action.heuristic_template,
                    "bandit_template": action.bandit_template,
                    "heuristic_score": round(action.heuristic_score, 6),
                    "bandit_score": round(action.bandit_score, 6),
                    "combined_score": round(action.combined_score, 6),
                    "diversity_bonus": round(action.diversity_bonus, 6),
                    "quota_adjusted": action.quota_adjusted,
                    "quota_reason": action.quota_reason,
                    "trust_pre": round(action.trust, 6),
                    "suspicion_pre": round(action.suspicion, 6),
                    "rarity_pre": round(action.rarity, 6),
                    "shared_exposure_pre": round(action.shared_exposure, 6),
                    "stable_overlap_pre": round(action.stable_overlap, 6),
                    "suspicious_overlap_pre": round(action.suspicious_overlap, 6),
                    "contamination_memory_pre": round(action.contamination_memory, 6),
                    "deception_score_pre": round(controller.deception_scores[client_id], 6) if controller else 0.0,
                    "stability_score_pre": round(controller.stability_scores[client_id], 6) if controller else 0.0,
                    "trust_post": round(controller.trust_scores[client_id], 6) if controller else round(action.trust, 6),
                    "suspicion_post": round(controller.suspicion_scores[client_id], 6) if controller else round(action.suspicion, 6),
                    "rarity_post": round(controller.rare_scores[client_id], 6) if controller else round(action.rarity, 6),
                    "shared_exposure_post": round(controller.shared_exposure_scores[client_id], 6) if controller else round(action.mask_bundle.overlap_ratio, 6),
                    "stable_overlap_post": round(controller.stable_overlap_scores[client_id], 6) if controller else 0.0,
                    "suspicious_overlap_post": round(controller.suspicious_overlap_scores[client_id], 6) if controller else 0.0,
                    "contamination_memory_post": round(controller.contamination_memory[client_id], 6) if controller else 0.0,
                    "deception_score_post": round(controller.deception_scores[client_id], 6) if controller else 0.0,
                    "stability_score_post": round(controller.stability_scores[client_id], 6) if controller else 0.0,
                    "bandwidth": round(action.bandwidth, 6),
                    "active_params": action.mask_bundle.active_params,
                    "communication_ratio": round(communication_ratio, 6),
                    "overlap_ratio": round(action.mask_bundle.overlap_ratio, 6),
                    "agg_weight_pre_repair": round(pre_repair_agg_weights[index], 6),
                    "agg_weight_raw": round(agg_weights[index], 6),
                    "agg_weight_normalized": round(normalized_weights[index], 6),
                    "update_norm": round(float(norms[index].item()), 6),
                    "clean_gain": round(effects["clean_gain"], 6),
                    "rare_gain": round(effects["rare_gain"], 6),
                    "backdoor_risk": round(effects["backdoor_risk"], 6),
                    "cosine": round(signal["cosine"], 6),
                    "norm_ratio": round(signal["norm_ratio"], 6),
                    "shared_exposure_signal": round(signal["shared_exposure"], 6),
                    "stable_overlap_signal": round(signal["stable_overlap"], 6),
                    "suspicious_overlap_signal": round(signal["suspicious_overlap"], 6),
                    "contamination_proxy": round(signal["contamination_proxy"], 6),
                    "reward_score": round(_decision_reward(effects["clean_gain"], effects["rare_gain"], effects["backdoor_risk"], communication_ratio, action.mask_bundle.overlap_ratio), 6),
                    "selection_reward_score": round(_selection_reward(effects["clean_gain"], effects["rare_gain"], effects["backdoor_risk"]), 6),
                }
            )

        stage_start = time.perf_counter()
        add_delta_in_place(global_state, aggregated_delta, scale=1.0)
        model.load_state_dict(_state_to_device(global_state, device))

        round_probe_metrics = _evaluate_probe_bundle(
            model,
            probe_loader,
            task_data,
            device,
            include_backdoor_signal=defense_has_trigger_knowledge,
        )
        total_probe_evaluations += 1
        clean_loss, clean_acc = _evaluate_model(model, test_loader, device)
        asr = _evaluate_backdoor_asr(model, test_loader, task_data, device)
        rare_acc = _evaluate_rare_accuracy(model, test_loader, task_data.rare_labels, device)
        total_stage_seconds["evaluation"] += time.perf_counter() - stage_start

        average_active_params = active_params_total / max(1, len(selected))
        communication_ratio = average_active_params / full_param_count
        uplink_bytes = int(round(active_params_total * value_bytes_per_parameter))
        downlink_model_bytes = int(round(active_params_total * value_bytes_per_parameter))
        # We cost an explicit dense 1-bit mask for every masked client. This is a
        # conservative, reproducible wire-format assumption rather than treating
        # templates as free metadata.
        mask_metadata_bytes = (
            dense_mask_metadata_bytes_per_client * len(selected)
            if actions and spec.mask_mode != "selection_only"
            else 0
        )
        protocol_bytes = uplink_bytes + downlink_model_bytes + mask_metadata_bytes
        full_payload_bytes = 2 * full_model_bytes * len(selected)
        byte_communication_ratio = protocol_bytes / max(1, full_payload_bytes)
        total_uplink_bytes += uplink_bytes
        total_downlink_model_bytes += downlink_model_bytes
        total_mask_metadata_bytes += mask_metadata_bytes
        total_full_payload_bytes += full_payload_bytes
        if process is not None:
            peak_process_rss_bytes = max(peak_process_rss_bytes, int(process.memory_info().rss))
        avg_trust = float(np.mean([controller.trust_scores[client_id] for client_id in selected])) if controller else 1.0
        avg_suspicion = float(np.mean([controller.suspicion_scores[client_id] for client_id in selected])) if controller else 0.0
        avg_rare_score = float(np.mean([controller.rare_scores[client_id] for client_id in selected])) if controller else 0.0
        avg_overlap = overlap_total / max(1, len(selected)) if actions else (1.0 if controller else 0.0)
        avg_shared_exposure = float(np.mean([controller.shared_exposure_scores[client_id] for client_id in selected])) if controller else 1.0
        avg_stable_overlap = float(np.mean([controller.stable_overlap_scores[client_id] for client_id in selected])) if controller else 1.0
        avg_suspicious_overlap = float(np.mean([controller.suspicious_overlap_scores[client_id] for client_id in selected])) if controller else 0.0
        avg_contamination_memory = float(np.mean([controller.contamination_memory[client_id] for client_id in selected])) if controller else 0.0
        avg_deception = float(np.mean([controller.deception_scores[client_id] for client_id in selected])) if controller else 0.0
        avg_stability = float(np.mean([controller.stability_scores[client_id] for client_id in selected])) if controller else 1.0
        round_allocator_reward = 0.0
        round_selection_reward = 0.0
        round_selection_counterfactual_gap = _mean(selected_selection_rewards) - _mean(alternate_selection_rewards) if alternate_selection_rewards else 0.0
        if controller and spec.selection_mode == "controller" and alternate_selection_rewards:
            controller.update_selection_gap_feedback(round_selection_counterfactual_gap)

        if controller and spec.update_bandits and actions:
            round_effects = _probe_effects(global_probe_metrics, round_probe_metrics)
            round_allocator_reward = _decision_reward(
                round_effects["clean_gain"],
                round_effects["rare_gain"],
                round_effects["backdoor_risk"],
                communication_ratio,
                avg_overlap,
            )
            selected_rare_credit = _mean([controller.effective_rarity(client_id) for client_id in selected]) if selected else 0.0
            selected_stability_credit = _mean([controller.stability_scores[client_id] for client_id in selected]) if selected else 1.0
            selected_deception_penalty = _mean([controller.deception_scores[client_id] for client_id in selected]) if selected else 0.0
            round_selection_reward = _selection_feedback_reward(
                round_effects["clean_gain"],
                round_effects["rare_gain"],
                round_effects["backdoor_risk"],
                selected_rare_credit,
                selected_stability_credit,
                selected_deception_penalty,
            )
            template_share = {template: counts[template] / max(1, len(selected)) for template in controller.template_names()}
            delayed_allocator_feedback: list[dict[str, object]] = []
            for row in bandit_feedback_rows:
                template = str(row["template"])
                share_gap = controller.template_target_share[template] - template_share[template]
                diversity_adjustment = float(np.clip(config.diversity_strength * share_gap, -0.1, 0.05))
                delayed_allocator_feedback.append(
                    {
                        "template": template,
                        "context_vector": row["context_vector"],
                        "reward": _mix_delayed_reward(float(row["local_reward"]), round_allocator_reward, config.delayed_reward_mix, diversity_adjustment),
                    }
                )
            delayed_selection_feedback: list[dict[str, object]] = []
            for row in selection_feedback_rows:
                delayed_reward = float(row["local_reward"])
                if bool(row.get("was_selected", True)):
                    delayed_reward = _mix_delayed_reward(delayed_reward, round_selection_reward, config.delayed_reward_mix)
                delayed_selection_feedback.append(
                    {
                        "client_id": row["client_id"],
                        "context_vector": row.get("context_vector"),
                        "reward": delayed_reward,
                    }
                )
            controller.update_bandit(delayed_allocator_feedback)
            controller.update_selection_bandit(delayed_selection_feedback)
            controller.update_round_statistics(actions)
            arm_counts = controller.bandit_counts()
            selection_bandit_count = controller.selection_bandit_count()
        diagnostic_rows.append(
            {
                "strategy": strategy_name,
                "attack_mode": config.attack_mode,
                "round": round_id,
                "selected_clients": len(selected),
                "selected_malicious": len(malicious_indices),
                "selected_benign": len(benign_indices),
                "effective_aggregation_mode": effective_aggregation_mode,
                "bulyan_cohort_valid": bulyan_cohort_valid,
                "pre_repair_malicious_shared_norm": round(pre_repair_malicious_shared_norm, 6),
                "pre_repair_benign_shared_norm": round(pre_repair_benign_shared_norm, 6),
                "pre_repair_malicious_total_norm": round(pre_repair_malicious_total_norm, 6),
                "pre_repair_benign_total_norm": round(pre_repair_benign_total_norm, 6),
                "pre_repair_contamination_ratio": round(pre_repair_contamination_ratio, 6),
                "malicious_shared_norm": round(malicious_shared_norm, 6),
                "benign_shared_norm": round(benign_shared_norm, 6),
                "malicious_total_norm": round(malicious_total_norm, 6),
                "benign_total_norm": round(benign_total_norm, 6),
                "contamination_ratio": round(contamination_ratio, 6),
                "post_repair_contamination_ratio": round(contamination_ratio, 6),
                "bb_overlap_mean": round(_mean(bb_overlap), 6),
                "bm_overlap_mean": round(_mean(bm_overlap), 6),
                "mm_overlap_mean": round(_mean(mm_overlap), 6),
                "shared_large_count": counts["shared_large"],
                "shared_small_count": counts["shared_small"],
                "hybrid_rare_count": counts["hybrid_rare"],
                "quarantine_count": counts["quarantine"],
                "policy_bandit_count": policy_counts.get("bandit", 0),
                "policy_guardrail_count": policy_counts.get("guardrail", 0),
                "policy_heuristic_warmup_count": policy_counts.get("heuristic_warmup", 0),
                "policy_static_count": policy_counts.get("static", 0),
                "bandit_shared_large_count": arm_counts["shared_large"],
                "bandit_shared_small_count": arm_counts["shared_small"],
                "bandit_hybrid_rare_count": arm_counts["hybrid_rare"],
                "bandit_quarantine_count": arm_counts["quarantine"],
                "selection_bandit_count": selection_bandit_count,
                "selection_bandit_selected": selection_bandit_selected,
                "selection_random_selected": selection_random_selected,
                "selection_heuristic_selected": selection_heuristic_selected,
                "quota_adjusted_count": quota_adjusted_count,
                "avg_shared_exposure": round(avg_shared_exposure, 6),
                "avg_stable_overlap": round(avg_stable_overlap, 6),
                "avg_suspicious_overlap": round(avg_suspicious_overlap, 6),
                "avg_contamination_memory": round(avg_contamination_memory, 6),
                "avg_deception": round(avg_deception, 6),
                "avg_stability": round(avg_stability, 6),
                "selection_counterfactual_gap": round(round_selection_counterfactual_gap, 6),
                "repair_pressure": round(repair_pressure, 6),
                "repair_threshold": round(repair_threshold, 6),
                "repair_count": repair_count,
                "avg_repair_suppression": round(repair_suppression_total / max(1, repair_count), 6) if repair_count else 0.0,
                "ema_shared_large": round(controller.template_usage_snapshot().get("shared_large", 0.0), 6) if controller else 0.0,
                "ema_shared_small": round(controller.template_usage_snapshot().get("shared_small", 0.0), 6) if controller else 0.0,
                "ema_hybrid_rare": round(controller.template_usage_snapshot().get("hybrid_rare", 0.0), 6) if controller else 0.0,
                "ema_quarantine": round(controller.template_usage_snapshot().get("quarantine", 0.0), 6) if controller else 0.0,
            }
        )

        round_rows.append(
            {
                "strategy": strategy_name,
                "attack_mode": config.attack_mode,
                "effective_aggregation_mode": effective_aggregation_mode,
                "bulyan_cohort_valid": bulyan_cohort_valid,
                "round": round_id,
                "clean_loss": round(clean_loss, 6),
                "clean_accuracy": round(clean_acc, 6),
                "asr": round(asr, 6),
                "rare_accuracy": round(rare_acc, 6),
                "avg_trust": round(avg_trust, 6),
                "avg_suspicion": round(avg_suspicion, 6),
                "avg_rare_score": round(avg_rare_score, 6),
                "avg_overlap": round(avg_overlap, 6),
                "avg_shared_exposure": round(avg_shared_exposure, 6),
                "avg_stable_overlap": round(avg_stable_overlap, 6),
                "avg_suspicious_overlap": round(avg_suspicious_overlap, 6),
                "avg_contamination_memory": round(avg_contamination_memory, 6),
                "avg_deception": round(avg_deception, 6),
                "avg_stability": round(avg_stability, 6),
                "selection_counterfactual_gap": round(round_selection_counterfactual_gap, 6),
                "repair_pressure": round(repair_pressure, 6),
                "repair_threshold": round(repair_threshold, 6),
                "repair_count": repair_count,
                "communication_ratio": round(communication_ratio, 6),
                "uplink_bytes": uplink_bytes,
                "downlink_model_bytes": downlink_model_bytes,
                "mask_metadata_bytes": mask_metadata_bytes,
                "protocol_bytes": protocol_bytes,
                "byte_communication_ratio": round(byte_communication_ratio, 6),
                "controller_state_bytes": controller_state_bytes,
                "round_allocator_reward": round(round_allocator_reward, 6),
                "round_selection_reward": round(round_selection_reward, 6),
                "selected_clients": len(selected),
                "bandit_min_count": min(arm_counts.values()) if actions and controller and spec.update_bandits else 0,
                "selection_bandit_count": selection_bandit_count if controller and spec.update_bandits else 0,
            }
        )

    summary = {
        "strategy": strategy_name,
        "dataset": config.dataset_name,
        "attack_mode": config.attack_mode,
        "rounds": config.rounds,
        "num_clients": config.num_clients,
        "malicious_fraction": config.malicious_fraction,
        "optimizer": config.local_optimizer,
        "device_requested": config.device,
        "device": device,
        "counterfactual_mode": config.counterfactual_mode,
        "defender_knowledge": config.defender_knowledge,
        "rarity_information_mode": config.rarity_information_mode,
        "repair_enabled_for_strategy": spec.enable_repair,
        "adaptive_repair_enabled": bool(config.adaptive_repair_enabled and spec.enable_repair),
        "bulyan_byzantine_fraction": (
            config.malicious_fraction if config.bulyan_byzantine_fraction < 0.0 else config.bulyan_byzantine_fraction
        ),
        "final_clean_accuracy": round_rows[-1]["clean_accuracy"],
        "final_asr": round_rows[-1]["asr"],
        "final_rare_accuracy": round_rows[-1]["rare_accuracy"],
        "mean_communication_ratio": round(float(np.mean([row["communication_ratio"] for row in round_rows])), 6),
        "full_model_bytes": full_model_bytes,
        "mask_metadata_encoding": "dense_1bit_per_parameter_per_masked_client",
        "total_uplink_bytes": total_uplink_bytes,
        "mean_uplink_bytes_per_round": round(total_uplink_bytes / max(1, config.rounds), 6),
        "total_downlink_model_bytes": total_downlink_model_bytes,
        "mean_downlink_model_bytes_per_round": round(total_downlink_model_bytes / max(1, config.rounds), 6),
        "total_mask_metadata_bytes": total_mask_metadata_bytes,
        "mean_mask_metadata_bytes_per_round": round(total_mask_metadata_bytes / max(1, config.rounds), 6),
        "total_protocol_bytes": total_uplink_bytes + total_downlink_model_bytes + total_mask_metadata_bytes,
        "mean_protocol_bytes_per_round": round((total_uplink_bytes + total_downlink_model_bytes + total_mask_metadata_bytes) / max(1, config.rounds), 6),
        "mean_byte_communication_ratio": round((total_uplink_bytes + total_downlink_model_bytes + total_mask_metadata_bytes) / max(1, total_full_payload_bytes), 6),
        "controller_state_bytes": controller_state_bytes,
        "initial_process_rss_bytes": initial_process_rss_bytes,
        "peak_process_rss_bytes": peak_process_rss_bytes,
        "peak_process_rss_delta_bytes": max(0, peak_process_rss_bytes - initial_process_rss_bytes),
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()) if device == "cuda" else 0,
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved()) if device == "cuda" else 0,
        "elapsed_seconds": round(time.time() - start_time, 2),
        "total_local_training_jobs": total_local_training_jobs,
        "mean_local_training_jobs_per_round": round(total_local_training_jobs / max(1, config.rounds), 6),
        "total_probe_evaluations": total_probe_evaluations,
        "mean_probe_evaluations_per_round": round(total_probe_evaluations / max(1, config.rounds), 6),
        "total_server_root_training_jobs": total_server_root_training_jobs,
        "mean_server_root_training_jobs_per_round": round(total_server_root_training_jobs / max(1, config.rounds), 6),
    }
    for stage_name in (
        "selection",
        "global_probe_root",
        "allocation",
        "local_training",
        "probe_feedback_repair",
        "aggregation",
        "evaluation",
    ):
        elapsed = total_stage_seconds.get(stage_name, 0.0)
        summary[f"{stage_name}_seconds"] = round(elapsed, 6)
        summary[f"mean_{stage_name}_seconds_per_round"] = round(elapsed / max(1, config.rounds), 6)
    if controller and spec.update_bandits:
        summary["bandit_arm_counts"] = controller.bandit_counts()
        summary["bandit_min_count"] = min(summary["bandit_arm_counts"].values())
        summary["selection_bandit_count"] = controller.selection_bandit_count()
        summary["template_usage_ema"] = {key: round(value, 6) for key, value in controller.template_usage_snapshot().items()}
        summary["mean_quota_adjusted_count"] = round(_mean([float(row["quota_adjusted_count"]) for row in diagnostic_rows]), 6) if diagnostic_rows else 0.0
    if diagnostic_rows:
        summary["mean_pre_repair_contamination_ratio"] = round(_mean([float(row["pre_repair_contamination_ratio"]) for row in diagnostic_rows]), 6)
        summary["mean_post_repair_contamination_ratio"] = round(_mean([float(row["post_repair_contamination_ratio"]) for row in diagnostic_rows]), 6)
        summary["mean_contamination_ratio"] = round(_mean([float(row["contamination_ratio"]) for row in diagnostic_rows]), 6)
        summary["mean_malicious_shared_norm"] = round(_mean([float(row["malicious_shared_norm"]) for row in diagnostic_rows]), 6)
        summary["mean_bm_overlap"] = round(_mean([float(row["bm_overlap_mean"]) for row in diagnostic_rows]), 6)
        summary["mean_shared_exposure"] = round(_mean([float(row["avg_shared_exposure"]) for row in diagnostic_rows]), 6)
        summary["mean_suspicious_overlap"] = round(_mean([float(row["avg_suspicious_overlap"]) for row in diagnostic_rows]), 6)
        summary["mean_contamination_memory"] = round(_mean([float(row["avg_contamination_memory"]) for row in diagnostic_rows]), 6)
        summary["mean_deception"] = round(_mean([float(row["avg_deception"]) for row in diagnostic_rows]), 6)
        summary["mean_stability"] = round(_mean([float(row["avg_stability"]) for row in diagnostic_rows]), 6)
        summary["mean_selection_counterfactual_gap"] = round(_mean([float(row["selection_counterfactual_gap"]) for row in diagnostic_rows]), 6)
        summary["mean_repair_pressure"] = round(_mean([float(row["repair_pressure"]) for row in diagnostic_rows]), 6)
        summary["mean_repair_count"] = round(_mean([float(row["repair_count"]) for row in diagnostic_rows]), 6)
        summary["mean_repair_suppression"] = round(_mean([float(row["avg_repair_suppression"]) for row in diagnostic_rows]), 6)
        if spec.aggregation_mode == "bulyan":
            summary["bulyan_valid_rounds"] = sum(bool(row["bulyan_cohort_valid"]) for row in diagnostic_rows)
            summary["bulyan_fallback_rounds"] = sum(not bool(row["bulyan_cohort_valid"]) for row in diagnostic_rows)
    if counterfactual_rows:
        optimal_rate, reward_gap = _counterfactual_summary(counterfactual_rows)
        summary["counterfactual_optimal_rate"] = round(optimal_rate, 6)
        summary["counterfactual_reward_gap"] = round(reward_gap, 6)
    if selection_counterfactual_rows:
        selection_win_rate, selection_reward_gap = _selection_counterfactual_summary(selection_counterfactual_rows)
        summary["selection_counterfactual_win_rate"] = round(selection_win_rate, 6)
        summary["selection_counterfactual_reward_gap"] = round(selection_reward_gap, 6)

    _save_results(config.results_dir, strategy_name, round_rows, summary)
    _save_csv_rows(config.results_dir / f"{strategy_name}_diagnostics.csv", diagnostic_rows)
    optional_outputs = (
        (config.results_dir / f"{strategy_name}_client_decisions.csv", decision_rows),
        (config.results_dir / f"{strategy_name}_counterfactuals.csv", counterfactual_rows),
        (config.results_dir / f"{strategy_name}_selection_counterfactuals.csv", selection_counterfactual_rows),
    )
    for output_path, output_rows in optional_outputs:
        if output_rows:
            _save_csv_rows(output_path, output_rows)
        else:
            output_path.unlink(missing_ok=True)
    return summary
