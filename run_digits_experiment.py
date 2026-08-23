from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from agentlock.config import SimulationConfig
from agentlock.data import available_datasets, build_task, set_global_seed
from agentlock.simulation import run_strategy


STRATEGY_CHOICES = [
    'both',
    'fedavg',
    'agentlock',
    'agenticfedlock',
    'static_subspace',
    'trust_only',
    'communication_only',
    'selection_only',
    'allocation_repair_only',
    'fedprox',
    'trimmed_mean',
    'krum',
    'foolsgold',
    'fltrust',
    'flame',
    'rfa',
    'bulyan',
]

ATTACK_CHOICES = [
    'badnets',
    'model_replacement',
    'distributed_backdoor',
    'adaptive_mimic',
    'colluding',
]


def _normalize_strategy(value: str) -> str:
    return 'agentlock' if value == 'agenticfedlock' else value


def _parse_hidden_dims(value: str) -> tuple[int, int]:
    parts = [part.strip() for part in value.split(',') if part.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError('hidden dims must be formatted like 128,64')
    return int(parts[0]), int(parts[1])


def _parse_channel_dims(value: str) -> tuple[int, int]:
    parts = [part.strip() for part in value.split(',') if part.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError('channel dims must be formatted like 16,32')
    return int(parts[0]), int(parts[1])


def _parse_label_tuple(value: str) -> tuple[int, ...]:
    cleaned = value.strip().lower()
    if cleaned in {'', 'auto', 'none'}:
        return ()
    return tuple(int(part.strip()) for part in value.split(',') if part.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run the AgenticFedLock experiment.')
    parser.add_argument('--dataset', choices=available_datasets(), default='sklearn_digits')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--data-root', type=Path, default=Path('data'))
    parser.add_argument('--rounds', type=int, default=30)
    parser.add_argument('--num-clients', type=int, default=24)
    parser.add_argument('--client-fraction', type=float, default=0.5)
    parser.add_argument('--local-epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--lr-schedule', choices=['constant', 'cosine'], default='constant')
    parser.add_argument('--lr-min', type=float, default=0.0)
    parser.add_argument('--optimizer', choices=['sgd', 'adam'], default='sgd')
    parser.add_argument('--momentum', type=float, default=0.0)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--model-family', choices=['auto', 'mlp', 'cnn'], default='auto')
    parser.add_argument('--hidden-dims', type=_parse_hidden_dims, default=(128, 64))
    parser.add_argument('--channel-dims', type=_parse_channel_dims, default=(16, 32))
    parser.add_argument('--cnn-hidden-dim', type=int, default=64)
    parser.add_argument('--cnn-variant', choices=['basic', 'residual'], default='basic')
    parser.add_argument('--image-augmentation', action='store_true')
    parser.add_argument('--malicious-fraction', type=float, default=0.2)
    parser.add_argument('--poison-fraction', type=float, default=0.3)
    parser.add_argument('--backdoor-target', type=int, default=0)
    parser.add_argument(
        '--image-trigger-variant',
        choices=['solid_patch', 'checkerboard', 'blended'],
        default='solid_patch',
        help='Image trigger used for poisoning/ASR: solid 2x2 patch, checkerboard 2x2, or low-amplitude blended patch.',
    )
    parser.add_argument('--rare-labels', type=_parse_label_tuple, default=())
    parser.add_argument('--dirichlet-alpha', type=float, default=0.4)
    parser.add_argument('--probe-fraction', type=float, default=0.12)
    parser.add_argument('--test-fraction', type=float, default=0.25)
    parser.add_argument(
        '--data-fraction',
        type=float,
        default=1.0,
        help='Fraction of raw training examples retained before the probe split; recorded in the run manifest.',
    )
    parser.add_argument('--min-client-size', type=int, default=18)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--counterfactual-clients', type=int, default=2)
    parser.add_argument('--selection-counterfactual-clients', type=int, default=2)
    parser.add_argument(
        '--counterfactual-mode',
        choices=['none', 'simulator_oracle'],
        default='none',
        help='Deployable default is none. simulator_oracle centrally retrains simulated clients and must be costed as extra work.',
    )
    parser.add_argument(
        '--defender-knowledge',
        choices=['exact', 'unknown'],
        default='exact',
        help='exact uses the configured trigger/target during defence; unknown hides them from probe scoring and target-row repair.',
    )
    parser.add_argument(
        '--rarity-information-mode',
        choices=['label_oracle', 'probe_feedback', 'none', 'malicious_spoof'],
        default='label_oracle',
        help='Controller rarity prior: legacy label oracle, neutral probe-feedback, no rarity signal, or malicious declaration spoofing.',
    )
    parser.add_argument(
        '--repair-signal-mode',
        choices=['triggered_probe', 'subspace_consensus'],
        default='triggered_probe',
        help='triggered_probe is the v1 legacy signal (requires trigger knowledge, inert under unknown-defender). '
             'subspace_consensus is the v2 trigger-agnostic composite (SLRT root-cosine + trusted-consensus deviation + suspicion memory).',
    )
    parser.add_argument(
        '--slrt-localization-mode',
        choices=['per_region', 'whole_model'],
        default='per_region',
        help='per_region computes SLRT separately on the shared-core and classifier regions and takes the worst '
             '(the default, v2 design). whole_model computes a single cosine deviation over the full unmasked '
             'delta, matching plain FLTrust-style comparison, for ablation against the localized default.',
    )
    parser.add_argument(
        '--risk-fusion-order',
        choices=['recenter_then_fuse', 'fuse_then_recenter'],
        default='recenter_then_fuse',
        help='recenter_then_fuse recenters each structural signal (SLRT, CAR, norm-inflation) against its own '
             'round median before max-fusion (the default, v2 design). fuse_then_recenter takes the max of the '
             'raw signals first and recenters only the fused result, for ablation against the default order.',
    )
    parser.add_argument('--trust-consensus-fraction', type=float, default=0.5,
                         help='Fraction of selected clients (highest effective trust) used as the CAR consensus anchor.')
    parser.add_argument('--root-anchor-weight', type=float, default=0.0,
                         help='Blend FLTrust-style root-cosine trust into the final aggregation weighting, on top of '
                              'masking/repair (0.0 = off/legacy, 1.0 = pure FLTrust-style weighting of the already-repaired deltas).')
    parser.add_argument('--bandit-alpha', type=float, default=0.4)
    parser.add_argument('--bandit-mix', type=float, default=0.45)
    parser.add_argument('--bandit-warmup-rounds', type=int, default=6)
    parser.add_argument('--delayed-reward-mix', type=float, default=0.35)
    parser.add_argument('--diversity-strength', type=float, default=0.12)
    parser.add_argument('--repair-risk-threshold', type=float, default=0.22)
    parser.add_argument('--repair-shared-suppression', type=float, default=0.9)
    parser.add_argument('--repair-weight-penalty', type=float, default=0.65)
    parser.add_argument('--repair-target-row-suppression', type=float, default=0.0)
    parser.add_argument('--disable-adaptive-repair', action='store_true')
    parser.add_argument('--adaptive-repair-reference-asr', type=float, default=0.15)
    parser.add_argument('--adaptive-repair-floor', type=float, default=0.35)
    parser.add_argument('--adaptive-repair-ceiling', type=float, default=1.0)
    parser.add_argument('--attack-mode', choices=ATTACK_CHOICES, default='badnets')
    parser.add_argument('--fedprox-mu', type=float, default=0.01)
    parser.add_argument('--trim-ratio', type=float, default=0.1)
    parser.add_argument('--rfa-max-iterations', type=int, default=50)
    parser.add_argument('--rfa-tolerance', type=float, default=1e-6)
    parser.add_argument(
        '--bulyan-byzantine-fraction',
        type=float,
        default=-1.0,
        help='Expected Byzantine fraction for Bulyan; -1 uses --malicious-fraction. Bulyan falls back when cohort-size conditions are invalid.',
    )
    parser.add_argument('--flame-min-cluster-size', type=int, default=0)
    parser.add_argument('--flame-noise-multiplier', type=float, default=0.001)
    parser.add_argument('--model-replacement-scale', type=float, default=1.0)
    parser.add_argument('--adaptive-attack-blend', type=float, default=0.7)
    parser.add_argument('--collusion-strength', type=float, default=0.85)
    parser.add_argument('--distributed-attack-scale', type=float, default=1.15)
    parser.add_argument('--selection-alpha', type=float, default=0.05)
    parser.add_argument('--selection-mix', type=float, default=0.9)
    parser.add_argument('--selection-warmup-rounds', type=int, default=10)
    parser.add_argument('--enable-quota-guard', action='store_true')
    parser.add_argument('--results-dir', type=Path, default=Path('results'))
    parser.add_argument('--strategy', choices=STRATEGY_CHOICES, default='both')
    parser.add_argument('--skip-baseline', action='store_true')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SimulationConfig(
        device=args.device,
        dataset_name=args.dataset,
        data_root=args.data_root,
        rounds=args.rounds,
        num_clients=args.num_clients,
        client_fraction=args.client_fraction,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lr_schedule=args.lr_schedule,
        lr_min=args.lr_min,
        local_optimizer=args.optimizer,
        sgd_momentum=args.momentum,
        weight_decay=args.weight_decay,
        model_family=args.model_family,
        hidden_dims=args.hidden_dims,
        channel_dims=args.channel_dims,
        cnn_hidden_dim=args.cnn_hidden_dim,
        cnn_variant=args.cnn_variant,
        image_augmentation=args.image_augmentation,
        malicious_fraction=args.malicious_fraction,
        poison_fraction=args.poison_fraction,
        backdoor_target=args.backdoor_target,
        image_trigger_variant=args.image_trigger_variant,
        rare_labels=args.rare_labels,
        dirichlet_alpha=args.dirichlet_alpha,
        probe_fraction=args.probe_fraction,
        test_fraction=args.test_fraction,
        data_fraction=args.data_fraction,
        min_client_size=args.min_client_size,
        seed=args.seed,
        counterfactual_clients_per_round=args.counterfactual_clients,
        selection_counterfactual_candidates_per_round=args.selection_counterfactual_clients,
        counterfactual_mode=args.counterfactual_mode,
        defender_knowledge=args.defender_knowledge,
        rarity_information_mode=args.rarity_information_mode,
        repair_signal_mode=args.repair_signal_mode,
        slrt_localization_mode=args.slrt_localization_mode,
        risk_fusion_order=args.risk_fusion_order,
        trust_consensus_fraction=args.trust_consensus_fraction,
        root_anchor_weight=args.root_anchor_weight,
        bandit_alpha=args.bandit_alpha,
        bandit_mix=args.bandit_mix,
        bandit_warmup_rounds=args.bandit_warmup_rounds,
        delayed_reward_mix=args.delayed_reward_mix,
        diversity_strength=args.diversity_strength,
        quota_guard_enabled=args.enable_quota_guard,
        repair_risk_threshold=args.repair_risk_threshold,
        repair_shared_suppression=args.repair_shared_suppression,
        repair_weight_penalty=args.repair_weight_penalty,
        repair_target_row_suppression=args.repair_target_row_suppression,
        adaptive_repair_enabled=not args.disable_adaptive_repair,
        adaptive_repair_reference_asr=args.adaptive_repair_reference_asr,
        adaptive_repair_floor=args.adaptive_repair_floor,
        adaptive_repair_ceiling=args.adaptive_repair_ceiling,
        attack_mode=args.attack_mode,
        fedprox_mu=args.fedprox_mu,
        trim_ratio=args.trim_ratio,
        rfa_max_iterations=args.rfa_max_iterations,
        rfa_tolerance=args.rfa_tolerance,
        bulyan_byzantine_fraction=args.bulyan_byzantine_fraction,
        flame_min_cluster_size=args.flame_min_cluster_size,
        flame_noise_multiplier=args.flame_noise_multiplier,
        model_replacement_scale=args.model_replacement_scale,
        adaptive_attack_blend=args.adaptive_attack_blend,
        collusion_strength=args.collusion_strength,
        distributed_attack_scale=args.distributed_attack_scale,
        selection_alpha=args.selection_alpha,
        selection_mix=args.selection_mix,
        selection_warmup_rounds=args.selection_warmup_rounds,
        results_dir=args.results_dir,
        compare_baseline=not args.skip_baseline,
    )
    set_global_seed(config.seed)
    task_data = build_task(config)

    config.results_dir.mkdir(parents=True, exist_ok=True)
    resolved_config = asdict(config)
    resolved_config['data_root'] = str(config.data_root.resolve())
    resolved_config['results_dir'] = str(config.results_dir.resolve())
    (config.results_dir / 'resolved_config.json').write_text(
        json.dumps(resolved_config, indent=2),
        encoding='utf-8',
    )
    task_manifest = {
        'dataset': task_data.dataset_name,
        'train_examples': len(task_data.train_dataset),
        'probe_examples': len(task_data.probe_dataset),
        'test_examples': len(task_data.test_dataset),
        'num_clients': len(task_data.client_profiles),
        'num_classes': task_data.num_classes,
        'rare_labels': list(task_data.rare_labels),
        'backdoor_target': task_data.backdoor_target,
        'trigger_kind': task_data.trigger_kind,
        'image_trigger_variant': task_data.image_trigger_variant,
        'malicious_client_ids': sorted(
            client_id
            for client_id, profile in task_data.client_profiles.items()
            if profile.is_malicious
        ),
        'model_spec': asdict(task_data.model_spec),
    }
    (config.results_dir / 'task_manifest.json').write_text(
        json.dumps(task_manifest, indent=2),
        encoding='utf-8',
    )

    summaries = []
    if args.strategy == 'both':
        if config.compare_baseline:
            summaries.append(run_strategy('fedavg', task_data, config))
        summaries.append(run_strategy('agentlock', task_data, config))
    else:
        summaries.append(run_strategy(_normalize_strategy(args.strategy), task_data, config))

    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
