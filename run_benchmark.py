from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from agentlock.data import available_datasets


DATASET_PROFILES: dict[str, dict[str, object]] = {
    'sklearn_digits': {
        'num_clients': 24,
        'client_fraction': 0.5,
        'rounds': 30,
        'local_epochs': 3,
        'batch_size': 32,
        'lr': 0.1,
        'optimizer': 'sgd',
        'momentum': 0.0,
        'weight_decay': 0.0,
        'dirichlet_alpha': 0.4,
        'probe_fraction': 0.12,
        'test_fraction': 0.25,
        'min_client_size': 18,
        'model_family': 'mlp',
        'hidden_dims': (128, 64),
        'channel_dims': (16, 32),
        'cnn_hidden_dim': 64,
        'rare_labels': (8, 9),
        'backdoor_target': 0,
        'counterfactual_clients_per_round': 2,
        'selection_counterfactual_clients_per_round': 2,
        'bandit_warmup_rounds': 6,
        'selection_warmup_rounds': 10,
        'bandit_mix': 0.65,
        'selection_mix': 0.95,
        'quota_guard_enabled': False,
        'adaptive_repair_enabled': False,
        'repair_risk_threshold': 1.0,
        'repair_shared_suppression': 0.0,
        'repair_weight_penalty': 0.0,
        'repair_target_row_suppression': 0.0,
    },
    'sklearn_wine': {
        'num_clients': 12,
        'client_fraction': 0.5,
        'rounds': 35,
        'local_epochs': 4,
        'batch_size': 16,
        'lr': 0.05,
        'optimizer': 'sgd',
        'momentum': 0.0,
        'weight_decay': 0.0,
        'dirichlet_alpha': 0.7,
        'probe_fraction': 0.12,
        'test_fraction': 0.25,
        'min_client_size': 6,
        'model_family': 'mlp',
        'hidden_dims': (64, 32),
        'channel_dims': (16, 32),
        'cnn_hidden_dim': 64,
        'rare_labels': (),
        'backdoor_target': 0,
        'counterfactual_clients_per_round': 1,
        'selection_counterfactual_clients_per_round': 2,
        'bandit_warmup_rounds': 6,
        'selection_warmup_rounds': 10,
        'bandit_mix': 0.6,
        'selection_mix': 0.95,
        'quota_guard_enabled': False,
        'adaptive_repair_enabled': False,
        'repair_risk_threshold': 0.25,
        'repair_shared_suppression': 0.9,
        'repair_weight_penalty': 0.6,
        'repair_target_row_suppression': 0.5,
    },
    'sklearn_breast_cancer': {
        'num_clients': 20,
        'client_fraction': 0.5,
        'rounds': 35,
        'local_epochs': 4,
        'batch_size': 16,
        'lr': 0.03,
        'optimizer': 'sgd',
        'momentum': 0.0,
        'weight_decay': 0.0,
        'dirichlet_alpha': 0.7,
        'probe_fraction': 0.12,
        'test_fraction': 0.25,
        'min_client_size': 10,
        'model_family': 'mlp',
        'hidden_dims': (96, 48),
        'channel_dims': (16, 32),
        'cnn_hidden_dim': 64,
        'rare_labels': (),
        'backdoor_target': 0,
        'counterfactual_clients_per_round': 1,
        'selection_counterfactual_clients_per_round': 2,
        'bandit_warmup_rounds': 6,
        'selection_warmup_rounds': 10,
        'bandit_mix': 0.45,
        'selection_mix': 0.9,
        'quota_guard_enabled': True,
        'adaptive_repair_enabled': False,
        'repair_risk_threshold': 0.12,
        'repair_shared_suppression': 1.0,
        'repair_weight_penalty': 0.95,
        'repair_target_row_suppression': 1.0,
    },
    'mnist_local': {
        'num_clients': 20,
        'client_fraction': 0.5,
        'rounds': 10,
        'local_epochs': 1,
        'batch_size': 64,
        'lr': 0.05,
        'optimizer': 'sgd',
        'momentum': 0.0,
        'weight_decay': 0.0,
        'dirichlet_alpha': 0.3,
        'probe_fraction': 0.12,
        'test_fraction': 0.25,
        'min_client_size': 25,
        'model_family': 'cnn',
        'hidden_dims': (128, 64),
        'channel_dims': (16, 32),
        'cnn_hidden_dim': 64,
        'rare_labels': (),
        'backdoor_target': 0,
        'counterfactual_clients_per_round': 2,
        'selection_counterfactual_clients_per_round': 2,
        'bandit_warmup_rounds': 4,
        'selection_warmup_rounds': 4,
        'bandit_mix': 0.55,
        'selection_mix': 0.92,
        'quota_guard_enabled': False,
        'adaptive_repair_enabled': False,
        'repair_risk_threshold': 0.18,
        'repair_shared_suppression': 0.95,
        'repair_weight_penalty': 0.75,
        'repair_target_row_suppression': 0.85,
    },
    'fashion_mnist_local': {
        'num_clients': 20,
        'client_fraction': 0.5,
        'rounds': 10,
        'local_epochs': 1,
        'batch_size': 64,
        'lr': 0.035,
        'optimizer': 'sgd',
        'momentum': 0.0,
        'weight_decay': 0.0,
        'dirichlet_alpha': 0.3,
        'probe_fraction': 0.12,
        'test_fraction': 0.25,
        'min_client_size': 25,
        'model_family': 'cnn',
        'hidden_dims': (128, 64),
        'channel_dims': (16, 32),
        'cnn_hidden_dim': 96,
        'rare_labels': (),
        'backdoor_target': 0,
        'counterfactual_clients_per_round': 2,
        'selection_counterfactual_clients_per_round': 2,
        'bandit_warmup_rounds': 4,
        'selection_warmup_rounds': 4,
        'bandit_mix': 0.5,
        'selection_mix': 0.94,
        'quota_guard_enabled': False,
        'adaptive_repair_enabled': False,
        'repair_risk_threshold': 0.12,
        'repair_shared_suppression': 1.0,
        'repair_weight_penalty': 0.95,
        'repair_target_row_suppression': 1.0,
    },
    'emnist_balanced_local': {
        'num_clients': 40,
        'client_fraction': 0.35,
        'rounds': 15,
        'local_epochs': 1,
        'batch_size': 64,
        'lr': 0.001,
        'optimizer': 'adam',
        'momentum': 0.0,
        'weight_decay': 0.0,
        'dirichlet_alpha': 0.3,
        'probe_fraction': 0.08,
        'test_fraction': 0.2,
        'min_client_size': 30,
        'model_family': 'cnn',
        'hidden_dims': (128, 64),
        'channel_dims': (48, 96),
        'cnn_hidden_dim': 256,
        'rare_labels': (),
        'backdoor_target': 0,
        'counterfactual_clients_per_round': 2,
        'selection_counterfactual_clients_per_round': 2,
        'bandit_warmup_rounds': 5,
        'selection_warmup_rounds': 5,
        'bandit_mix': 0.54,
        'selection_mix': 0.92,
        'quota_guard_enabled': False,
        'adaptive_repair_enabled': False,
        'repair_risk_threshold': 0.14,
        'repair_shared_suppression': 1.0,
        'repair_weight_penalty': 0.82,
        'repair_target_row_suppression': 0.9,
    },
    'femnist_local': {
        'num_clients': 500,
        'client_fraction': 0.2,
        'rounds': 15,
        'local_epochs': 2,
        'batch_size': 64,
        'lr': 0.001,
        'optimizer': 'adam',
        'momentum': 0.0,
        'weight_decay': 0.0,
        'dirichlet_alpha': 0.3,
        'probe_fraction': 0.08,
        'test_fraction': 0.2,
        'min_client_size': 20,
        'model_family': 'cnn',
        'hidden_dims': (128, 64),
        'channel_dims': (48, 96),
        'cnn_hidden_dim': 256,
        'rare_labels': (),
        'backdoor_target': 0,
        'counterfactual_clients_per_round': 2,
        'selection_counterfactual_clients_per_round': 2,
        'bandit_warmup_rounds': 5,
        'selection_warmup_rounds': 6,
        'bandit_mix': 0.52,
        'selection_mix': 0.92,
        'quota_guard_enabled': False,
        'adaptive_repair_enabled': False,
        'repair_risk_threshold': 0.12,
        'repair_shared_suppression': 1.0,
        'repair_weight_penalty': 0.95,
        'repair_target_row_suppression': 0.9,
    },
    'cifar10_local': {
        'num_clients': 30,
        'client_fraction': 0.33,
        'rounds': 20,
        'local_epochs': 2,
        'batch_size': 64,
        'lr': 0.001,
        'optimizer': 'adam',
        'momentum': 0.0,
        'weight_decay': 0.0005,
        'dirichlet_alpha': 0.25,
        'probe_fraction': 0.08,
        'test_fraction': 0.2,
        'min_client_size': 40,
        'model_family': 'cnn',
        'hidden_dims': (128, 64),
        'channel_dims': (48, 96),
        'cnn_hidden_dim': 256,
        'rare_labels': (),
        'backdoor_target': 0,
        'counterfactual_clients_per_round': 2,
        'selection_counterfactual_clients_per_round': 2,
        'bandit_warmup_rounds': 5,
        'selection_warmup_rounds': 6,
        'bandit_mix': 0.54,
        'selection_mix': 0.92,
        'quota_guard_enabled': False,
        'adaptive_repair_enabled': False,
        'repair_risk_threshold': 0.12,
        'repair_shared_suppression': 1.0,
        'repair_weight_penalty': 0.92,
        'repair_target_row_suppression': 0.95,
    },
}

STRATEGY_CHOICES = [
    'fedavg',
    'agentlock',
    'agenticfedlock',
    'static_subspace',
    'trust_only',
    'communication_only',
    'fedprox',
    'trimmed_mean',
    'krum',
    'foolsgold',
    'fltrust',
    'flame',
    'rfa',
    'bulyan',
]

METRICS = (
    'final_clean_accuracy',
    'final_asr',
    'final_rare_accuracy',
    'mean_communication_ratio',
    'mean_byte_communication_ratio',
    'mean_uplink_bytes_per_round',
    'mean_downlink_model_bytes_per_round',
    'mean_mask_metadata_bytes_per_round',
    'mean_protocol_bytes_per_round',
    'controller_state_bytes',
    'peak_process_rss_bytes',
    'peak_process_rss_delta_bytes',
    'peak_cuda_allocated_bytes',
    'peak_cuda_reserved_bytes',
    'mean_pre_repair_contamination_ratio',
    'mean_post_repair_contamination_ratio',
    'mean_contamination_ratio',
    'mean_shared_exposure',
    'mean_suspicious_overlap',
    'mean_contamination_memory',
    'mean_deception',
    'mean_stability',
    'mean_selection_counterfactual_gap',
    'counterfactual_optimal_rate',
    'selection_counterfactual_win_rate',
    'selection_counterfactual_reward_gap',
    'elapsed_seconds',
    'mean_local_training_jobs_per_round',
    'mean_probe_evaluations_per_round',
    'mean_server_root_training_jobs_per_round',
    'mean_selection_seconds_per_round',
    'mean_global_probe_root_seconds_per_round',
    'mean_allocation_seconds_per_round',
    'mean_local_training_seconds_per_round',
    'mean_probe_feedback_repair_seconds_per_round',
    'mean_aggregation_seconds_per_round',
    'mean_evaluation_seconds_per_round',
)


def _normalize_strategy(value: str) -> str:
    return 'agentlock' if value == 'agenticfedlock' else value


def _save_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _aggregate_rows(run_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    aggregate_rows: list[dict[str, object]] = []
    groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in run_rows:
        groups.setdefault((str(row['dataset']), str(row['strategy'])), []).append(row)
    for (dataset, strategy), rows in sorted(groups.items()):
        aggregate_row: dict[str, object] = {
            'dataset': dataset,
            'strategy': strategy,
            'seeds': len(rows),
        }
        attack_modes = sorted({str(row.get('attack_mode', 'badnets')) for row in rows})
        if len(attack_modes) == 1:
            aggregate_row['attack_mode'] = attack_modes[0]
        for metric in METRICS:
            values = [float(row[metric]) for row in rows if metric in row]
            if not values:
                continue
            aggregate_row[f'{metric}_mean'] = round(float(np.mean(values)), 6)
            aggregate_row[f'{metric}_std'] = round(float(np.std(values, ddof=1)), 6) if len(values) > 1 else 0.0
        aggregate_rows.append(aggregate_row)
    return aggregate_rows


def _tuple_arg(values: tuple[int, ...]) -> str:
    return ','.join(str(value) for value in values)


def _label_arg(values: tuple[int, ...]) -> str:
    return 'auto' if not values else ','.join(str(value) for value in values)


def _run_strategy_process(
    dataset: str,
    seed: int,
    strategy: str,
    profile: dict[str, object],
    args: argparse.Namespace,
) -> dict[str, object]:
    strategy = _normalize_strategy(strategy)
    seed_dir = args.results_dir / dataset / args.attack_mode / strategy / f'seed_{seed}'
    client_fraction = float(profile.get('client_fraction', args.client_fraction if args.client_fraction is not None else 0.5))
    bandit_mix = profile['bandit_mix'] if args.bandit_mix is None else args.bandit_mix
    selection_mix = profile['selection_mix'] if args.selection_mix is None else args.selection_mix
    command = [
        sys.executable,
        'run_digits_experiment.py',
        '--dataset', dataset,
        '--device', args.device,
        '--data-root', str(args.data_root),
        '--rounds', str(profile['rounds']),
        '--num-clients', str(profile['num_clients']),
        '--client-fraction', str(client_fraction),
        '--local-epochs', str(profile['local_epochs']),
        '--batch-size', str(profile['batch_size']),
        '--lr', str(profile['lr']),
        '--lr-schedule', args.lr_schedule,
        '--lr-min', str(args.lr_min),
        '--optimizer', str(profile.get('optimizer', 'sgd')),
        '--momentum', str(profile.get('momentum', 0.0)),
        '--weight-decay', str(profile.get('weight_decay', 0.0)),
        '--model-family', str(profile['model_family']),
        '--hidden-dims', _tuple_arg(tuple(profile['hidden_dims'])),
        '--channel-dims', _tuple_arg(tuple(profile['channel_dims'])),
        '--cnn-hidden-dim', str(profile['cnn_hidden_dim']),
        '--cnn-variant', str(profile.get('cnn_variant', 'basic')),
        '--malicious-fraction', str(args.malicious_fraction),
        '--poison-fraction', str(args.poison_fraction),
        '--backdoor-target', str(profile.get('backdoor_target', 0)),
        '--rare-labels', _label_arg(tuple(profile.get('rare_labels', ()))),
        '--dirichlet-alpha', str(profile['dirichlet_alpha']),
        '--probe-fraction', str(profile.get('probe_fraction', 0.12)),
        '--test-fraction', str(profile.get('test_fraction', 0.25)),
        '--data-fraction', str(args.data_fraction),
        '--min-client-size', str(profile['min_client_size']),
        '--seed', str(seed),
        '--counterfactual-clients', str(profile['counterfactual_clients_per_round']),
        '--selection-counterfactual-clients', str(profile.get('selection_counterfactual_clients_per_round', 2)),
        '--counterfactual-mode', args.counterfactual_mode,
        '--defender-knowledge', args.defender_knowledge,
        '--rarity-information-mode', args.rarity_information_mode,
        '--bandit-warmup-rounds', str(profile['bandit_warmup_rounds']),
        '--selection-warmup-rounds', str(profile['selection_warmup_rounds']),
        '--bandit-mix', str(bandit_mix),
        '--selection-mix', str(selection_mix),
        '--repair-risk-threshold', str(profile['repair_risk_threshold']),
        '--repair-shared-suppression', str(profile['repair_shared_suppression']),
        '--repair-weight-penalty', str(profile['repair_weight_penalty']),
        '--repair-target-row-suppression', str(profile['repair_target_row_suppression']),
        '--attack-mode', args.attack_mode,
        '--fedprox-mu', str(args.fedprox_mu),
        '--trim-ratio', str(args.trim_ratio),
        '--rfa-max-iterations', str(args.rfa_max_iterations),
        '--rfa-tolerance', str(args.rfa_tolerance),
        '--bulyan-byzantine-fraction', str(args.bulyan_byzantine_fraction),
        '--flame-min-cluster-size', str(args.flame_min_cluster_size),
        '--flame-noise-multiplier', str(args.flame_noise_multiplier),
        '--model-replacement-scale', str(args.model_replacement_scale),
        '--adaptive-attack-blend', str(args.adaptive_attack_blend),
        '--collusion-strength', str(args.collusion_strength),
        '--distributed-attack-scale', str(args.distributed_attack_scale),
        '--results-dir', str(seed_dir),
        '--strategy', strategy,
        '--skip-baseline',
    ]
    if bool(profile['quota_guard_enabled']):
        command.append('--enable-quota-guard')
    if args.image_augmentation:
        command.append('--image-augmentation')
    adaptive_repair_enabled = bool(profile.get('adaptive_repair_enabled', True))
    if args.adaptive_repair == 'on':
        adaptive_repair_enabled = True
    elif args.adaptive_repair == 'off':
        adaptive_repair_enabled = False
    if not adaptive_repair_enabled:
        command.append('--disable-adaptive-repair')
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f'Benchmark command failed for {dataset} seed {seed} strategy {strategy}\n'
            f'STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}'
        )
    summaries = json.loads(completed.stdout)
    if len(summaries) != 1:
        raise RuntimeError(f'Expected one summary for {strategy}, got: {completed.stdout}')
    return summaries[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run multi-seed AgenticFedLock benchmarks on local datasets.')
    parser.add_argument('--datasets', nargs='+', choices=available_datasets(), default=['sklearn_digits', 'sklearn_wine'])
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--strategies', nargs='+', choices=STRATEGY_CHOICES, default=['fedavg', 'agentlock'])
    parser.add_argument('--attack-mode', choices=['badnets', 'model_replacement', 'distributed_backdoor', 'adaptive_mimic', 'colluding'], default='badnets')
    parser.add_argument('--seeds', nargs='+', type=int, default=[7, 11, 19])
    parser.add_argument('--results-dir', type=Path, default=Path('benchmark_results'))
    parser.add_argument('--data-root', type=Path, default=Path('data'))
    parser.add_argument('--malicious-fraction', type=float, default=0.2)
    parser.add_argument('--poison-fraction', type=float, default=0.3)
    parser.add_argument('--client-fraction', type=float, default=None)
    parser.add_argument('--data-fraction', type=float, default=1.0)
    parser.add_argument('--image-augmentation', action='store_true')
    parser.add_argument('--lr-schedule', choices=['constant', 'cosine'], default='constant')
    parser.add_argument('--lr-min', type=float, default=0.0)
    parser.add_argument('--bandit-mix', type=float, default=None)
    parser.add_argument('--selection-mix', type=float, default=None)
    parser.add_argument('--fedprox-mu', type=float, default=0.01)
    parser.add_argument('--trim-ratio', type=float, default=0.1)
    parser.add_argument('--rfa-max-iterations', type=int, default=50)
    parser.add_argument('--rfa-tolerance', type=float, default=1e-6)
    parser.add_argument('--bulyan-byzantine-fraction', type=float, default=-1.0)
    parser.add_argument('--flame-min-cluster-size', type=int, default=0)
    parser.add_argument('--flame-noise-multiplier', type=float, default=0.001)
    parser.add_argument('--model-replacement-scale', type=float, default=1.0)
    parser.add_argument('--adaptive-attack-blend', type=float, default=0.7)
    parser.add_argument('--collusion-strength', type=float, default=0.85)
    parser.add_argument('--distributed-attack-scale', type=float, default=1.15)
    parser.add_argument('--counterfactual-mode', choices=['none', 'simulator_oracle'], default='none')
    parser.add_argument('--defender-knowledge', choices=['exact', 'unknown'], default='exact')
    parser.add_argument(
        '--rarity-information-mode',
        choices=['label_oracle', 'probe_feedback', 'none', 'malicious_spoof'],
        default='label_oracle',
    )
    parser.add_argument(
        '--adaptive-repair',
        choices=['profile', 'on', 'off'],
        default='profile',
        help='Use the dataset profile setting, force pressure-adaptive repair on, or force it off.',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    run_rows: list[dict[str, object]] = []

    for dataset in args.datasets:
        profile = DATASET_PROFILES[dataset]
        for seed in args.seeds:
            for strategy in args.strategies:
                summary = _run_strategy_process(dataset, seed, strategy, profile, args)
                row = {
                    'dataset': dataset,
                    'seed': seed,
                    'strategy': _normalize_strategy(strategy),
                    'attack_mode': args.attack_mode,
                    'device_requested': args.device,
                    'counterfactual_mode': args.counterfactual_mode,
                    'defender_knowledge': args.defender_knowledge,
                    'rarity_information_mode': args.rarity_information_mode,
                    'adaptive_repair': args.adaptive_repair,
                }
                row.update(summary)
                run_rows.append(row)

    aggregate_rows = _aggregate_rows(run_rows)
    _save_csv(args.results_dir / 'run_summaries.csv', run_rows)
    _save_csv(args.results_dir / 'aggregate_summary.csv', aggregate_rows)
    (args.results_dir / 'run_summaries.json').write_text(json.dumps(run_rows, indent=2), encoding='utf-8')
    (args.results_dir / 'aggregate_summary.json').write_text(json.dumps(aggregate_rows, indent=2), encoding='utf-8')
    print(json.dumps(aggregate_rows, indent=2))


if __name__ == '__main__':
    main()
