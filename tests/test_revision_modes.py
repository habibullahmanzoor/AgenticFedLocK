from __future__ import annotations

import unittest
from collections import OrderedDict
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from agentlock.config import SimulationConfig
from agentlock.data import (
    ClientProfile,
    TaskData,
    _apply_rarity_information_mode,
    _normalize_femnist_pt_images,
    _stratified_data_subset,
    _subsample_predefined_client_data,
)
from agentlock.model import build_model, build_model_spec
from agentlock.simulation import (
    _aggregate_bulyan,
    _aggregate_flame,
    _aggregate_fltrust,
    _aggregate_geometric_median,
    _evaluate_probe_bundle,
    _apply_trigger,
    _local_training_job_cost,
    _model_features,
    _resolve_execution_device,
    _scheduled_learning_rate,
    _trigger_agnostic_risk,
)
from run_benchmark import _aggregate_rows


class RevisionModeTests(unittest.TestCase):
    def test_revision_defaults_disable_simulator_counterfactuals(self) -> None:
        config = SimulationConfig()
        self.assertEqual(config.counterfactual_mode, "none")

    def test_invalid_revision_modes_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            SimulationConfig(counterfactual_mode="free_oracle")
        with self.assertRaises(ValueError):
            SimulationConfig(defender_knowledge="partial")
        with self.assertRaises(ValueError):
            SimulationConfig(rarity_information_mode="client_magic")
        with self.assertRaises(ValueError):
            SimulationConfig(data_fraction=0.0)
        with self.assertRaises(ValueError):
            SimulationConfig(image_trigger_variant="semantic_magic")

    def test_image_trigger_variants_are_distinct_and_reproducible(self) -> None:
        model_spec = build_model_spec(
            input_dim=64,
            num_classes=10,
            feature_shape=(1, 8, 8),
            model_family="mlp",
            hidden_dims=(8, 4),
        )
        dataset = TensorDataset(torch.zeros((1, 1, 8, 8)), torch.tensor([0]))

        def task(variant: str) -> TaskData:
            return TaskData(
                dataset_name="unit",
                train_dataset=dataset,
                probe_dataset=dataset,
                test_dataset=dataset,
                client_indices={0: [0]},
                client_profiles={},
                input_dim=64,
                feature_shape=(1, 8, 8),
                num_classes=10,
                rare_labels=(2, 3),
                backdoor_target=0,
                trigger_kind="image_patch",
                trigger_indices=(),
                trigger_value=1.0,
                model_spec=model_spec,
                image_trigger_variant=variant,
            )

        features = torch.full((1, 1, 8, 8), 0.2)
        solid = _apply_trigger(features, task("solid_patch"))
        checker = _apply_trigger(features, task("checkerboard"))
        blended = _apply_trigger(features, task("blended"))
        self.assertTrue(torch.allclose(solid[..., -2:, -2:], torch.ones((1, 1, 2, 2))))
        self.assertTrue(torch.allclose(checker[..., -2:, -2:], torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])))
        self.assertTrue(torch.allclose(blended[..., -2:, -2:], torch.full((1, 1, 2, 2), 0.4)))
        self.assertTrue(torch.allclose(features, torch.full((1, 1, 8, 8), 0.2)))

    def test_data_subset_is_reproducible_and_class_stratified(self) -> None:
        features = np.arange(80).reshape(40, 2)
        labels = np.repeat(np.arange(4), 10)
        selected_x, selected_y = _stratified_data_subset(features, labels, 0.5, 7)
        repeated_x, repeated_y = _stratified_data_subset(features, labels, 0.5, 7)
        self.assertEqual(selected_x.shape[0], 20)
        self.assertTrue(np.array_equal(selected_x, repeated_x))
        self.assertTrue(np.array_equal(selected_y, repeated_y))
        self.assertTrue(np.array_equal(np.bincount(selected_y), np.array([5, 5, 5, 5])))

    def test_predefined_client_subsetting_preserves_clients_and_remaps_indices(self) -> None:
        features = np.arange(24).reshape(12, 2)
        labels = np.arange(12) % 3
        _, selected_y, client_indices = _subsample_predefined_client_data(
            features,
            labels,
            {0: list(range(0, 6)), 1: list(range(6, 12))},
            0.5,
            7,
        )
        self.assertEqual(len(selected_y), 6)
        self.assertEqual(set(client_indices), {0, 1})
        self.assertEqual(sorted(index for values in client_indices.values() for index in values), list(range(6)))

    def test_residual_cnn_has_a_valid_forward_pass(self) -> None:
        model_spec = build_model_spec(
            input_dim=3 * 32 * 32,
            num_classes=10,
            feature_shape=(3, 32, 32),
            model_family="cnn",
            channel_dims=(16, 32),
            cnn_hidden_dim=32,
            cnn_variant="residual",
        )
        logits = build_model(model_spec)(torch.zeros((2, 3, 32, 32)))
        self.assertEqual(tuple(logits.shape), (2, 10))

    def test_cosine_learning_rate_reaches_configured_floor(self) -> None:
        config = SimulationConfig(rounds=5, lr=0.01, lr_schedule="cosine", lr_min=0.001)
        self.assertAlmostEqual(_scheduled_learning_rate(config, 1), 0.01)
        self.assertAlmostEqual(_scheduled_learning_rate(config, 5), 0.001)

    def test_rarity_information_ablation_modes_hide_or_spoof_the_label_prior(self) -> None:
        profiles = {
            0: ClientProfile(0, 10, [8, 2], [0, 1], 1.0, 0.2, False),
            1: ClientProfile(1, 10, [2, 8], [0, 1], 1.0, 0.8, True),
        }
        neutral = _apply_rarity_information_mode(profiles, "probe_feedback")
        self.assertEqual([profile.rarity_score for profile in neutral.values()], [0.5, 0.5])
        no_rarity = _apply_rarity_information_mode(profiles, "none")
        self.assertEqual([profile.rarity_score for profile in no_rarity.values()], [0.5, 0.5])
        spoofed = _apply_rarity_information_mode(profiles, "malicious_spoof")
        self.assertEqual(spoofed[0].rarity_score, 0.2)
        self.assertEqual(spoofed[1].rarity_score, 1.0)

    def test_compact_uint8_images_are_normalized_only_at_model_input(self) -> None:
        compact = torch.tensor([[[[0, 255]]]], dtype=torch.uint8)
        converted = _model_features(compact, "cpu")
        self.assertEqual(converted.dtype, torch.float32)
        self.assertTrue(torch.allclose(converted, torch.tensor([[[[0.0, 1.0]]]])))

    def test_femnist_images_are_normalized_to_compact_uint8(self) -> None:
        raw_pixels = [[0, 128, 255] + [0] * (28 * 28 - 3)]
        already_scaled = [[0.0, 1.0, 1.0] + [0.0] * (28 * 28 - 3)]

        from_raw = _normalize_femnist_pt_images(raw_pixels)
        from_scaled = _normalize_femnist_pt_images(already_scaled)

        self.assertEqual(from_raw.dtype, np.uint8)
        self.assertEqual(from_raw.shape, (1, 1, 28, 28))
        self.assertEqual(from_scaled.dtype, np.uint8)
        np.testing.assert_array_equal(from_raw[0, 0, 0, :3], [0, 128, 255])
        np.testing.assert_array_equal(from_scaled[0, 0, 0, :3], [0, 255, 255])

    def test_explicit_cuda_request_fails_without_a_cuda_build(self) -> None:
        if torch.cuda.is_available():
            self.assertEqual(_resolve_execution_device("cuda"), "cuda")
        else:
            with self.assertRaises(RuntimeError):
                _resolve_execution_device("cuda")

    def test_unknown_defender_does_not_invoke_triggered_probe(self) -> None:
        model_spec = build_model_spec(
            input_dim=64,
            num_classes=10,
            feature_shape=(1, 8, 8),
            model_family="mlp",
            hidden_dims=(8, 4),
        )
        dataset = TensorDataset(
            torch.zeros((4, 1, 8, 8), dtype=torch.float32),
            torch.tensor([0, 1, 2, 3], dtype=torch.long),
        )
        task = TaskData(
            dataset_name="unit",
            train_dataset=dataset,
            probe_dataset=dataset,
            test_dataset=dataset,
            client_indices={0: [0, 1, 2, 3]},
            client_profiles={},
            input_dim=64,
            feature_shape=(1, 8, 8),
            num_classes=10,
            rare_labels=(2, 3),
            backdoor_target=0,
            trigger_kind="image_patch",
            trigger_indices=(),
            trigger_value=1.0,
            model_spec=model_spec,
        )
        model = build_model(model_spec)
        loader = DataLoader(dataset, batch_size=4)
        with patch("agentlock.simulation._evaluate_backdoor_asr") as evaluate_asr:
            metrics = _evaluate_probe_bundle(
                model,
                loader,
                task,
                "cpu",
                include_backdoor_signal=False,
            )
        evaluate_asr.assert_not_called()
        self.assertEqual(metrics["backdoor_asr"], 0.0)

    def test_adaptive_mimic_cost_counts_two_malicious_training_jobs(self) -> None:
        config = SimulationConfig(attack_mode="adaptive_mimic")
        self.assertEqual(_local_training_job_cost(config, is_malicious=True), 2)
        self.assertEqual(_local_training_job_cost(config, is_malicious=False), 1)

    def test_aggregate_rows_uses_sample_standard_deviation(self) -> None:
        rows = [
            {"dataset": "d", "strategy": "s", "final_clean_accuracy": 1.0},
            {"dataset": "d", "strategy": "s", "final_clean_accuracy": 3.0},
            {"dataset": "d", "strategy": "s", "final_clean_accuracy": 5.0},
        ]
        aggregate = _aggregate_rows(rows)[0]
        self.assertAlmostEqual(
            aggregate["final_clean_accuracy_std"],
            float(np.std([1.0, 3.0, 5.0], ddof=1)),
            places=6,
        )

    def test_fltrust_rejects_opposing_update(self) -> None:
        root = OrderedDict(weight=torch.tensor([1.0, 0.0]))
        aligned = OrderedDict(weight=torch.tensor([2.0, 0.0]))
        opposed = OrderedDict(weight=torch.tensor([-10.0, 0.0]))
        aggregate, weights = _aggregate_fltrust([aligned, opposed], root)
        self.assertAlmostEqual(weights[0], 1.0)
        self.assertAlmostEqual(weights[1], 0.0)
        self.assertTrue(torch.allclose(aggregate["weight"], root["weight"]))

    def test_geometric_median_resists_single_outlier(self) -> None:
        deltas = [
            OrderedDict(weight=torch.tensor([0.0])),
            OrderedDict(weight=torch.tensor([0.1])),
            OrderedDict(weight=torch.tensor([100.0])),
        ]
        aggregate, weights = _aggregate_geometric_median(
            deltas,
            [1.0, 1.0, 1.0],
            max_iterations=100,
            tolerance=1e-7,
        )
        self.assertLess(float(aggregate["weight"].item()), 1.0)
        self.assertAlmostEqual(sum(weights), 1.0, places=6)

    def test_flame_clusters_and_clips_single_outlier(self) -> None:
        deltas = [
            OrderedDict(weight=torch.tensor([1.0, 0.0])),
            OrderedDict(weight=torch.tensor([1.2, 0.0])),
            OrderedDict(weight=torch.tensor([0.9, 0.0])),
            OrderedDict(weight=torch.tensor([-10.0, 0.0])),
        ]
        aggregate, weights = _aggregate_flame(
            deltas,
            min_cluster_size=3,
            noise_multiplier=0.0,
            seed=7,
        )
        self.assertAlmostEqual(weights[3], 0.0)
        self.assertGreater(sum(weights[:3]), 0.99)
        self.assertGreater(float(aggregate["weight"][0].item()), 0.8)
        self.assertLess(float(aggregate["weight"][0].item()), 1.3)

    def test_trigger_agnostic_risk_flags_opposing_client_without_any_trigger_knowledge(self) -> None:
        class _StubController:
            def __init__(self, trust: dict[int, float], suspicion: dict[int, float]) -> None:
                self._trust = trust
                self.suspicion_scores = suspicion

            def effective_trust(self, client_id: int) -> float:
                return self._trust[client_id]

        # Three benign clients aligned with the root direction, one malicious
        # client whose shared-core component points the opposite way. No
        # trigger/target information is used anywhere in this computation.
        shared_mask = {"weight": torch.tensor([1.0, 1.0])}
        deltas = [
            OrderedDict(weight=torch.tensor([1.0, 0.9])),
            OrderedDict(weight=torch.tensor([1.1, 1.0])),
            OrderedDict(weight=torch.tensor([0.9, 1.1])),
            OrderedDict(weight=torch.tensor([-1.0, -1.0])),
        ]
        selected = [0, 1, 2, 3]
        root_delta = OrderedDict(weight=torch.tensor([1.0, 1.0]))
        controller = _StubController(
            trust={0: 0.9, 1: 0.85, 2: 0.88, 3: 0.1},
            suspicion={0: 0.05, 1: 0.05, 2: 0.05, 3: 0.6},
        )

        risk, norm_confidence = _trigger_agnostic_risk(
            deltas, selected, controller, shared_mask, None, root_delta, trust_consensus_fraction=0.75,
        )

        self.assertLess(risk[0], 0.1)
        self.assertLess(risk[1], 0.1)
        self.assertLess(risk[2], 0.1)
        self.assertGreater(risk[3], 0.5)
        self.assertGreater(risk[3], risk[0] + 0.3)
        # Purely angular attack -- no magnitude evidence, so Mechanism E should
        # report low confidence.
        self.assertLess(norm_confidence, 0.3)

    def test_trigger_agnostic_risk_flags_norm_inflation_even_when_direction_is_plausible(self) -> None:
        class _StubController:
            def __init__(self, trust: dict[int, float], suspicion: dict[int, float]) -> None:
                self._trust = trust
                self.suspicion_scores = suspicion

            def effective_trust(self, client_id: int) -> float:
                return self._trust[client_id]

        # A model-replacement-style attacker: same direction as everyone else
        # (would evade a purely angular/cosine detector) but scaled to a much
        # larger magnitude to dominate aggregation. Mechanism E must catch this
        # even though SLRT/CAR alone would not.
        shared_mask = {"weight": torch.tensor([1.0, 1.0])}
        deltas = [
            OrderedDict(weight=torch.tensor([1.0, 1.0])),
            OrderedDict(weight=torch.tensor([1.05, 0.95])),
            OrderedDict(weight=torch.tensor([0.95, 1.05])),
            OrderedDict(weight=torch.tensor([6.0, 6.0])),  # same direction, 6x magnitude
        ]
        selected = [0, 1, 2, 3]
        root_delta = OrderedDict(weight=torch.tensor([1.0, 1.0]))
        controller = _StubController(
            trust={0: 0.9, 1: 0.85, 2: 0.88, 3: 0.5},
            suspicion={0: 0.05, 1: 0.05, 2: 0.05, 3: 0.05},
        )

        risk, norm_confidence = _trigger_agnostic_risk(
            deltas, selected, controller, shared_mask, None, root_delta, trust_consensus_fraction=0.75,
        )

        self.assertLess(risk[0], 0.3)
        self.assertGreater(risk[3], 0.5)
        # A 6x-magnitude outlier should register as high norm confidence.
        self.assertGreater(norm_confidence, 0.5)

    def test_bulyan_filters_a_single_extreme_outlier_when_cohort_is_valid(self) -> None:
        deltas = [
            OrderedDict(weight=torch.tensor([0.0])),
            OrderedDict(weight=torch.tensor([0.1])),
            OrderedDict(weight=torch.tensor([0.2])),
            OrderedDict(weight=torch.tensor([0.3])),
            OrderedDict(weight=torch.tensor([0.4])),
            OrderedDict(weight=torch.tensor([0.5])),
            OrderedDict(weight=torch.tensor([100.0])),
        ]
        aggregate, weights = _aggregate_bulyan(deltas, malicious_fraction=1.0 / 7.0)
        self.assertLess(float(aggregate["weight"].item()), 1.0)
        self.assertAlmostEqual(sum(weights), 1.0, places=6)
        self.assertEqual(weights[-1], 0.0)

    def test_bulyan_uses_guarded_fallback_for_an_invalid_cohort(self) -> None:
        deltas = [
            OrderedDict(weight=torch.tensor([0.0])),
            OrderedDict(weight=torch.tensor([0.1])),
            OrderedDict(weight=torch.tensor([0.2])),
            OrderedDict(weight=torch.tensor([100.0])),
        ]
        aggregate, weights = _aggregate_bulyan(deltas, malicious_fraction=0.25)
        self.assertLess(float(aggregate["weight"].item()), 1.0)
        self.assertAlmostEqual(sum(weights), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
