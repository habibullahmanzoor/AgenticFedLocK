from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class SimulationConfig:
    seed: int = 7
    device: str = "auto"
    dataset_name: str = "sklearn_digits"
    num_clients: int = 24
    rounds: int = 30
    client_fraction: float = 0.5
    local_epochs: int = 3
    batch_size: int = 32
    lr: float = 0.1
    lr_schedule: str = "constant"
    lr_min: float = 0.0
    local_optimizer: str = "sgd"
    sgd_momentum: float = 0.0
    weight_decay: float = 0.0
    bandit_alpha: float = 0.4
    bandit_mix: float = 0.45
    bandit_warmup_rounds: int = 6
    delayed_reward_mix: float = 0.35
    diversity_strength: float = 0.12
    quota_guard_enabled: bool = False
    repair_risk_threshold: float = 0.22
    repair_shared_suppression: float = 0.9
    repair_weight_penalty: float = 0.65
    repair_target_row_suppression: float = 0.0
    adaptive_repair_enabled: bool = True
    adaptive_repair_reference_asr: float = 0.15
    adaptive_repair_floor: float = 0.35
    adaptive_repair_ceiling: float = 1.0
    attack_mode: str = "badnets"
    fedprox_mu: float = 0.0
    trim_ratio: float = 0.1
    rfa_max_iterations: int = 50
    rfa_tolerance: float = 1e-6
    bulyan_byzantine_fraction: float = -1.0
    flame_min_cluster_size: int = 0
    flame_noise_multiplier: float = 0.001
    model_replacement_scale: float = 1.0
    adaptive_attack_blend: float = 0.7
    collusion_strength: float = 0.85
    distributed_attack_scale: float = 1.15
    selection_alpha: float = 0.05
    selection_mix: float = 0.9
    selection_warmup_rounds: int = 10
    model_family: str = 'auto'
    hidden_dims: tuple[int, int] = (128, 64)
    channel_dims: tuple[int, int] = (16, 32)
    cnn_hidden_dim: int = 64
    cnn_variant: str = "basic"
    image_augmentation: bool = False
    malicious_fraction: float = 0.2
    poison_fraction: float = 0.3
    backdoor_target: int = 0
    image_trigger_variant: str = "solid_patch"
    dirichlet_alpha: float = 0.4
    probe_fraction: float = 0.12
    test_fraction: float = 0.25
    data_fraction: float = 1.0
    rare_labels: tuple[int, ...] = (8, 9)
    min_client_size: int = 18
    bandwidth_levels: tuple[float, float, float] = (0.35, 0.6, 1.0)
    counterfactual_clients_per_round: int = 2
    selection_counterfactual_candidates_per_round: int = 2
    counterfactual_mode: str = "none"
    defender_knowledge: str = "exact"
    rarity_information_mode: str = "label_oracle"
    repair_signal_mode: str = "triggered_probe"
    slrt_localization_mode: str = "per_region"
    risk_fusion_order: str = "recenter_then_fuse"
    trust_consensus_fraction: float = 0.5
    root_anchor_weight: float = 0.0
    data_root: Path = Path('data')
    results_dir: Path = Path("results")
    compare_baseline: bool = True

    def __post_init__(self) -> None:
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be one of 'auto', 'cpu', or 'cuda'")
        if not 0.0 < self.data_fraction <= 1.0:
            raise ValueError("data_fraction must be in (0, 1]")
        if self.cnn_variant not in {"basic", "residual"}:
            raise ValueError("cnn_variant must be 'basic' or 'residual'")
        if self.image_trigger_variant not in {"solid_patch", "checkerboard", "blended"}:
            raise ValueError(
                "image_trigger_variant must be one of 'solid_patch', 'checkerboard', or 'blended'"
            )
        if self.lr_schedule not in {"constant", "cosine"}:
            raise ValueError("lr_schedule must be 'constant' or 'cosine'")
        if self.lr_min < 0.0 or self.lr_min > self.lr:
            raise ValueError("lr_min must be non-negative and no greater than lr")
        valid_counterfactual_modes = {"none", "simulator_oracle"}
        if self.counterfactual_mode not in valid_counterfactual_modes:
            raise ValueError(
                f"counterfactual_mode must be one of {sorted(valid_counterfactual_modes)}, "
                f"got {self.counterfactual_mode!r}"
            )
        valid_defender_knowledge = {"exact", "unknown"}
        if self.defender_knowledge not in valid_defender_knowledge:
            raise ValueError(
                f"defender_knowledge must be one of {sorted(valid_defender_knowledge)}, "
                f"got {self.defender_knowledge!r}"
            )
        valid_rarity_information_modes = {"label_oracle", "probe_feedback", "none", "malicious_spoof"}
        if self.rarity_information_mode not in valid_rarity_information_modes:
            raise ValueError(
                f"rarity_information_mode must be one of {sorted(valid_rarity_information_modes)}, "
                f"got {self.rarity_information_mode!r}"
            )
        valid_repair_signal_modes = {"triggered_probe", "subspace_consensus"}
        if self.repair_signal_mode not in valid_repair_signal_modes:
            raise ValueError(
                f"repair_signal_mode must be one of {sorted(valid_repair_signal_modes)}, "
                f"got {self.repair_signal_mode!r}"
            )
        valid_slrt_localization_modes = {"per_region", "whole_model"}
        if self.slrt_localization_mode not in valid_slrt_localization_modes:
            raise ValueError(
                f"slrt_localization_mode must be one of {sorted(valid_slrt_localization_modes)}, "
                f"got {self.slrt_localization_mode!r}"
            )
        valid_risk_fusion_orders = {"recenter_then_fuse", "fuse_then_recenter"}
        if self.risk_fusion_order not in valid_risk_fusion_orders:
            raise ValueError(
                f"risk_fusion_order must be one of {sorted(valid_risk_fusion_orders)}, "
                f"got {self.risk_fusion_order!r}"
            )
        if not 0.0 < self.trust_consensus_fraction <= 1.0:
            raise ValueError("trust_consensus_fraction must be in (0, 1]")
        if not 0.0 <= self.root_anchor_weight <= 1.0:
            raise ValueError("root_anchor_weight must be in [0, 1]")
        if self.rfa_max_iterations < 1:
            raise ValueError("rfa_max_iterations must be at least 1")
        if self.rfa_tolerance <= 0:
            raise ValueError("rfa_tolerance must be positive")
        if self.bulyan_byzantine_fraction != -1.0 and not 0.0 <= self.bulyan_byzantine_fraction < 0.5:
            raise ValueError("bulyan_byzantine_fraction must be -1 (use malicious_fraction) or in [0, 0.5)")
        if self.flame_min_cluster_size < 0:
            raise ValueError("flame_min_cluster_size must be non-negative")
        if self.flame_noise_multiplier < 0:
            raise ValueError("flame_noise_multiplier must be non-negative")
