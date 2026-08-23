from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import torch

from agentlock.bandit import LinUCBBandit
from agentlock.data import ClientProfile
from agentlock.model import ModelSpec, build_model


class MaskBundle(NamedTuple):
    masks: dict[str, torch.Tensor]
    active_params: int
    overlap_ratio: float


@dataclass
class SelectionMetadata:
    client_id: int
    context_vector: tuple[float, ...]
    policy_mode: str
    heuristic_score: float
    bandit_score: float
    combined_score: float
    reward_memory: float


@dataclass
class ClientAction:
    client_id: int
    template: str
    trust: float
    rarity: float
    bandwidth: float
    suspicion: float
    shared_exposure: float
    stable_overlap: float
    suspicious_overlap: float
    contamination_memory: float
    aggregation_weight: float
    mask_bundle: MaskBundle
    context_vector: tuple[float, ...]
    policy_mode: str
    heuristic_template: str
    bandit_template: str
    heuristic_score: float
    bandit_score: float
    combined_score: float
    diversity_bonus: float
    quota_adjusted: bool
    quota_reason: str
    selection_policy_mode: str
    selection_heuristic_score: float
    selection_bandit_score: float
    selection_combined_score: float
    selection_reward_memory: float


class AgenticFedLockController:
    TEMPLATE_NAMES = ("shared_large", "shared_small", "hybrid_rare", "quarantine")

    def __init__(
        self,
        client_profiles: dict[int, ClientProfile],
        model_spec: ModelSpec,
        seed: int = 0,
        bandit_alpha: float = 0.4,
        bandit_mix: float = 0.45,
        bandit_warmup_rounds: int = 6,
        diversity_strength: float = 0.12,
        quota_guard_enabled: bool = False,
        selection_alpha: float = 0.05,
        selection_mix: float = 0.9,
        selection_warmup_rounds: int = 10,
        rarity_feedback_enabled: bool = True,
    ) -> None:
        self.client_profiles = client_profiles
        self.model_spec = model_spec
        self.input_dim = model_spec.input_dim
        self.feature_shape = model_spec.feature_shape
        self.num_classes = model_spec.num_classes
        self.model_family = model_spec.family
        self.hidden_one, self.hidden_two = model_spec.hidden_dims
        self.conv_one, self.conv_two = model_spec.channel_dims
        self.cnn_hidden = model_spec.cnn_hidden_dim
        self.conv_feature_height, self.conv_feature_width = model_spec.conv_feature_shape
        self.conv_feature_size = self.conv_feature_height * self.conv_feature_width
        self.base_rarity = {client_id: profile.rarity_score for client_id, profile in client_profiles.items()}
        self.trust_scores = {client_id: 0.84 for client_id in client_profiles}
        self.suspicion_scores = {client_id: 0.06 for client_id in client_profiles}
        self.rare_scores = dict(self.base_rarity)
        self.selection_reward_memory = {client_id: 0.0 for client_id in client_profiles}
        self.shared_exposure_scores = {client_id: 0.0 for client_id in client_profiles}
        self.stable_overlap_scores = {client_id: 0.12 for client_id in client_profiles}
        self.suspicious_overlap_scores = {client_id: 0.04 for client_id in client_profiles}
        self.contamination_memory = {client_id: 0.02 for client_id in client_profiles}
        self.deception_scores = {client_id: 0.04 for client_id in client_profiles}
        self.stability_scores = {client_id: 0.78 for client_id in client_profiles}
        self.previous_cosine_scores = {client_id: 0.5 for client_id in client_profiles}
        self.previous_norm_scores = {client_id: 1.0 for client_id in client_profiles}
        self.previous_clean_gains = {client_id: 0.5 for client_id in client_profiles}
        self.previous_rare_gains = {client_id: 0.5 for client_id in client_profiles}
        self.previous_backdoor_risks = {client_id: 0.0 for client_id in client_profiles}
        self.selection_gap_ema = 0.0
        self._rng = np.random.default_rng(seed)

        self.hybrid_private_h3 = 0
        self.quarantine_private_h3 = 0
        if self.model_family == 'mlp':
            shared_large_h1 = max(2, min(self.hidden_one, int(round(self.hidden_one * 0.8125))))
            shared_large_h2 = max(2, min(self.hidden_two, int(round(self.hidden_two * 0.8125))))
            shared_small_h1 = max(2, min(shared_large_h1, int(round(self.hidden_one * 0.625))))
            shared_small_h2 = max(2, min(shared_large_h2, int(round(self.hidden_two * 0.625))))
            self.hybrid_private_h1 = max(4, int(round(self.hidden_one * 0.15625)))
            self.hybrid_private_h2 = max(2, int(round(self.hidden_two * 0.15625)))
            self.quarantine_private_h1 = max(2, int(round(self.hidden_one * 0.078125)))
            self.quarantine_private_h2 = max(1, int(round(self.hidden_two * 0.078125)))
            self.shared_large = (list(range(0, shared_large_h1)), list(range(0, shared_large_h2)))
            self.shared_small = (list(range(0, shared_small_h1)), list(range(0, shared_small_h2)))
        else:
            shared_large_c1 = max(2, min(self.conv_one, int(round(self.conv_one * 0.8125))))
            shared_large_c2 = max(2, min(self.conv_two, int(round(self.conv_two * 0.8125))))
            shared_large_hidden = max(4, min(self.cnn_hidden, int(round(self.cnn_hidden * 0.8125))))
            shared_small_c1 = max(2, min(shared_large_c1, int(round(self.conv_one * 0.625))))
            shared_small_c2 = max(2, min(shared_large_c2, int(round(self.conv_two * 0.625))))
            shared_small_hidden = max(4, min(shared_large_hidden, int(round(self.cnn_hidden * 0.625))))
            self.hybrid_private_h1 = max(2, int(round(self.conv_one * 0.15625)))
            self.hybrid_private_h2 = max(2, int(round(self.conv_two * 0.15625)))
            self.hybrid_private_h3 = max(4, int(round(self.cnn_hidden * 0.15625)))
            self.quarantine_private_h1 = max(1, int(round(self.conv_one * 0.09375)))
            self.quarantine_private_h2 = max(1, int(round(self.conv_two * 0.09375)))
            self.quarantine_private_h3 = max(2, int(round(self.cnn_hidden * 0.09375)))
            self.shared_large = (
                list(range(0, shared_large_c1)),
                list(range(0, shared_large_c2)),
                list(range(0, shared_large_hidden)),
            )
            self.shared_small = (
                list(range(0, shared_small_c1)),
                list(range(0, shared_small_c2)),
                list(range(0, shared_small_hidden)),
            )

        self.bandit_mix = bandit_mix
        self.bandit_warmup_rounds = bandit_warmup_rounds
        self.diversity_strength = diversity_strength
        self.quota_guard_enabled = quota_guard_enabled
        self.bandit = LinUCBBandit(self.TEMPLATE_NAMES, feature_dim=16, alpha=bandit_alpha, ridge_lambda=1.0)
        self.selection_mix = selection_mix
        self.selection_warmup_rounds = selection_warmup_rounds
        self.rarity_feedback_enabled = rarity_feedback_enabled
        self.selection_bandit = LinUCBBandit(("select",), feature_dim=18, alpha=selection_alpha, ridge_lambda=1.0)
        self.template_target_share = {
            "shared_large": 0.45,
            "shared_small": 0.18,
            "hybrid_rare": 0.22,
            "quarantine": 0.15,
        }
        self.template_usage_ema = dict(self.template_target_share)
        self._last_selection_metadata: dict[int, SelectionMetadata] = {}
        self._last_selection_pool_metadata: dict[int, SelectionMetadata] = {}
        if self.model_family == 'mlp':
            self._shared_core_mask = self._create_mask_tensors(
                shared_h1=self.shared_large[0],
                shared_h2=self.shared_large[1],
                label_rows=list(range(self.num_classes)),
            )
        else:
            self._shared_core_mask = self._create_mask_tensors(
                shared_h1=self.shared_large[0],
                shared_h2=self.shared_large[1],
                shared_hidden=self.shared_large[2],
                label_rows=list(range(self.num_classes)),
            )

    def template_names(self) -> tuple[str, ...]:
        return self.TEMPLATE_NAMES

    def shared_core_masks(self) -> dict[str, torch.Tensor]:
        return {name: mask.clone() for name, mask in self._shared_core_mask.items()}

    def select_clients(self, client_ids: list[int], fraction: float, round_id: int) -> list[int]:
        count = max(2, int(len(client_ids) * fraction))
        selection_mode = self._selection_policy_mode(round_id)
        metadata = self._selection_metadata_map(client_ids, round_id, selection_mode)

        self._last_selection_pool_metadata = dict(metadata)
        if selection_mode == "random_warmup":
            selected = self._rng.choice(client_ids, size=count, replace=False).tolist()
            self._last_selection_metadata = {client_id: metadata[client_id] for client_id in selected}
            return selected

        rare_candidates = [
            client_id
            for client_id in client_ids
            if self.effective_rarity(client_id) > 0.68
            and (
                self.effective_trust(client_id) > 0.38
                or (
                    self.stability_scores[client_id] > 0.72
                    and self.deception_scores[client_id] < 0.42
                    and self.contamination_memory[client_id] < 0.34
                )
            )
        ]
        rare_candidates.sort(
            key=lambda client_id: self.effective_rarity(client_id)
            + 0.18 * self.effective_trust(client_id)
            + 0.08 * self.stability_scores[client_id]
            - 0.04 * self.deception_scores[client_id]
            + 0.05 * metadata[client_id].combined_score,
            reverse=True,
        )
        rare_quota = min(max(1, int(np.ceil(count * 0.3))), len(rare_candidates))
        selected = rare_candidates[:rare_quota]

        core_candidates = [client_id for client_id in client_ids if client_id not in selected]
        core_candidates.sort(
            key=lambda client_id: self.effective_trust(client_id)
            + 0.15 * self.client_profiles[client_id].bandwidth_score
            - 0.2 * self.suspicion_scores[client_id]
            + 0.05 * metadata[client_id].combined_score,
            reverse=True,
        )
        core_quota = min(max(2, count // 4), len(core_candidates))
        selected.extend(core_candidates[:core_quota])

        remaining = [client_id for client_id in client_ids if client_id not in selected]
        remaining_count = count - len(selected)
        if remaining_count <= 0:
            self._last_selection_metadata = {client_id: metadata[client_id] for client_id in selected[:count]}
            return selected[:count]

        weights = np.array([self._selection_weight(client_id, metadata[client_id]) for client_id in remaining], dtype=np.float64)
        if weights.sum() <= 0:
            sampled = self._rng.choice(remaining, size=remaining_count, replace=False).tolist()
        else:
            weights /= weights.sum()
            sampled = self._rng.choice(remaining, size=remaining_count, replace=False, p=weights).tolist()
        selected = selected + sampled
        self._last_selection_metadata = {client_id: metadata[client_id] for client_id in selected}
        return selected

    def choose_actions(self, client_ids: list[int], round_id: int) -> dict[int, ClientAction]:
        actions = {client_id: self._select_action(client_id, round_id) for client_id in client_ids}
        return self._apply_template_quota_guard(actions, round_id)

    def action_for_template(self, client_id: int, round_id: int, template: str) -> ClientAction:
        context_vector = self._context_vector(client_id, round_id)
        heuristic_scores = self._heuristic_scores(client_id, round_id)
        bandit_stats = self.bandit.scores(context_vector)
        bandit_optimistic = {name: bandit_stats[name]["optimistic"] for name in self.TEMPLATE_NAMES}
        combined_scores, diversity_bonus = self._combined_scores(heuristic_scores, bandit_optimistic)
        heuristic_template = max(heuristic_scores, key=heuristic_scores.get)
        bandit_template = max(bandit_optimistic, key=bandit_optimistic.get)
        policy_mode = "counterfactual"
        return self._build_action(
            client_id=client_id,
            template=template,
            round_id=round_id,
            context_vector=context_vector,
            heuristic_scores=heuristic_scores,
            bandit_optimistic=bandit_optimistic,
            combined_scores=combined_scores,
            policy_mode=policy_mode,
            heuristic_template=heuristic_template,
            bandit_template=bandit_template,
            diversity_bonus=diversity_bonus,
        )

    def update_scores(self, telemetry: dict[int, dict[str, float]]) -> None:
        for client_id, metrics in telemetry.items():
            cosine_score = float(np.clip((metrics["cosine"] + 1.0) / 2.0, 0.0, 1.0))
            norm_score = float(np.clip(metrics["norm_ratio"], 0.0, 1.0))
            clean_gain = float(np.clip(metrics["clean_gain"], 0.0, 1.0))
            rare_gain = float(np.clip(metrics["rare_gain"], 0.0, 1.0))
            backdoor_risk = float(np.clip(metrics["backdoor_risk"], 0.0, 1.0))
            shared_exposure = float(np.clip(metrics.get("shared_exposure", self.shared_exposure_scores[client_id]), 0.0, 1.0))
            stable_overlap = float(np.clip(metrics.get("stable_overlap", self.stable_overlap_scores[client_id]), 0.0, 1.0))
            suspicious_overlap = float(np.clip(metrics.get("suspicious_overlap", self.suspicious_overlap_scores[client_id]), 0.0, 1.0))
            contamination_proxy = float(np.clip(metrics.get("contamination_proxy", self.contamination_memory[client_id]), 0.0, 1.0))

            previous_trust = self.trust_scores[client_id]
            previous_suspicion = self.suspicion_scores[client_id]
            previous_rare = self.rare_scores[client_id]
            previous_shared_exposure = self.shared_exposure_scores[client_id]
            previous_stable_overlap = self.stable_overlap_scores[client_id]
            previous_suspicious_overlap = self.suspicious_overlap_scores[client_id]
            previous_contamination = self.contamination_memory[client_id]
            previous_deception = self.deception_scores[client_id]
            previous_stability = self.stability_scores[client_id]
            previous_cosine = self.previous_cosine_scores[client_id]
            previous_norm = self.previous_norm_scores[client_id]
            previous_clean = self.previous_clean_gains[client_id]
            previous_rare_gain = self.previous_rare_gains[client_id]
            previous_backdoor_risk = self.previous_backdoor_risks[client_id]

            risk_jump = max(0.0, backdoor_risk - previous_backdoor_risk)
            clean_drop = max(0.0, previous_clean - clean_gain)
            rare_drop = max(0.0, previous_rare_gain - rare_gain)
            cosine_drift = abs(cosine_score - previous_cosine)
            norm_drift = abs(norm_score - previous_norm)
            deception_observation = float(
                np.clip(
                    0.34 * risk_jump
                    + 0.18 * cosine_drift
                    + 0.12 * norm_drift
                    + 0.14 * clean_drop
                    + 0.1 * rare_drop
                    + 0.06 * suspicious_overlap
                    + 0.06 * previous_trust * min(1.0, risk_jump + cosine_drift),
                    0.0,
                    1.0,
                )
            )
            stability_observation = float(
                np.clip(
                    1.0
                    - (0.4 * cosine_drift + 0.24 * norm_drift + 0.18 * clean_drop + 0.18 * rare_drop),
                    0.0,
                    1.0,
                )
            )

            suspicion_observation = (
                0.3 * backdoor_risk
                + 0.1 * (1.0 - cosine_score)
                + 0.08 * (1.0 - clean_gain)
                + 0.05 * (1.0 - norm_score)
                + 0.05 * max(0.0, 0.55 - rare_gain)
                + 0.07 * suspicious_overlap
                + 0.08 * contamination_proxy
                + 0.03 * shared_exposure
                + 0.04 * deception_observation
                + 0.02 * (1.0 - stability_observation)
            )
            trust_observation = (
                0.28 * cosine_score
                + 0.12 * norm_score
                + 0.28 * clean_gain
                + 0.18 * rare_gain
                + 0.08 * stable_overlap
                + 0.06 * stability_observation
                - 0.08 * backdoor_risk
                - 0.04 * suspicious_overlap
                - 0.04 * contamination_proxy
                - 0.05 * deception_observation
            )
            rare_observation = (0.76 * rare_gain + 0.2 * clean_gain + 0.06 * stable_overlap) * (1.0 - 0.28 * backdoor_risk - 0.04 * deception_observation)

            self.deception_scores[client_id] = float(np.clip(0.7 * previous_deception + 0.3 * deception_observation, 0.0, 1.0))
            self.stability_scores[client_id] = float(np.clip(0.74 * previous_stability + 0.26 * stability_observation, 0.0, 1.0))
            self.suspicion_scores[client_id] = float(np.clip(0.76 * previous_suspicion + 0.24 * suspicion_observation, 0.02, 0.99))
            self.trust_scores[client_id] = float(np.clip(0.74 * previous_trust + 0.26 * trust_observation, 0.05, 0.99))
            updated_rare = float(np.clip(0.8 * previous_rare + 0.2 * rare_observation, 0.0, 1.0))
            self.rare_scores[client_id] = (
                max(self.base_rarity[client_id], updated_rare)
                if self.rarity_feedback_enabled
                else self.base_rarity[client_id]
            )
            self.shared_exposure_scores[client_id] = float(np.clip(0.74 * previous_shared_exposure + 0.26 * shared_exposure, 0.0, 1.0))
            self.stable_overlap_scores[client_id] = float(np.clip(0.72 * previous_stable_overlap + 0.28 * stable_overlap, 0.0, 1.0))
            self.suspicious_overlap_scores[client_id] = float(np.clip(0.7 * previous_suspicious_overlap + 0.3 * suspicious_overlap, 0.0, 1.0))
            self.contamination_memory[client_id] = float(np.clip(0.7 * previous_contamination + 0.24 * contamination_proxy + 0.04 * deception_observation, 0.0, 1.0))
            self.previous_cosine_scores[client_id] = cosine_score
            self.previous_norm_scores[client_id] = norm_score
            self.previous_clean_gains[client_id] = clean_gain
            self.previous_rare_gains[client_id] = rare_gain
            self.previous_backdoor_risks[client_id] = backdoor_risk

    def update_bandit(self, feedback_rows: list[dict[str, object]]) -> None:
        if not feedback_rows:
            return
        self.bandit.update_batch(feedback_rows)

    def update_selection_bandit(self, feedback_rows: list[dict[str, object]]) -> None:
        if not feedback_rows:
            return
        observed_clients: set[int] = set()
        selection_rows: list[dict[str, object]] = []
        for row in feedback_rows:
            client_id = int(row["client_id"])
            reward = float(row["reward"])
            observed_clients.add(client_id)
            context_vector = row.get("context_vector")
            if context_vector is None:
                selection_metadata = self._last_selection_metadata.get(client_id) or self._last_selection_pool_metadata.get(client_id)
                if selection_metadata is None:
                    continue
                context_vector = selection_metadata.context_vector
            selection_rows.append(
                {
                    "template": "select",
                    "context_vector": context_vector,
                    "reward": reward,
                }
            )
            previous_reward = self.selection_reward_memory[client_id]
            self.selection_reward_memory[client_id] = float(np.clip(0.76 * previous_reward + 0.24 * reward, -1.5, 1.5))
        for client_id in self.selection_reward_memory:
            if client_id not in observed_clients:
                self.selection_reward_memory[client_id] *= 0.98
        self.selection_bandit.update_batch(selection_rows)

    def effective_trust(self, client_id: int) -> float:
        adjusted = (
            self.trust_scores[client_id]
            - 0.18 * self.suspicion_scores[client_id]
            - 0.05 * self.deception_scores[client_id]
            + 0.05 * self.stability_scores[client_id]
        )
        return float(np.clip(adjusted, 0.05, 0.99))

    def effective_rarity(self, client_id: int) -> float:
        return float(np.clip(max(self.base_rarity[client_id], self.rare_scores[client_id]), 0.0, 1.0))

    def bandit_counts(self) -> dict[str, int]:
        return self.bandit.counts()

    def selection_bandit_count(self) -> int:
        return self.selection_bandit.count("select")

    def update_selection_gap_feedback(self, gap: float) -> None:
        self.selection_gap_ema = float(np.clip(0.72 * self.selection_gap_ema + 0.28 * gap, -0.4, 0.4))

    def selection_metadata_snapshot(self) -> dict[int, SelectionMetadata]:
        return dict(self._last_selection_pool_metadata)

    def preview_action(self, client_id: int, round_id: int) -> ClientAction:
        return self._select_action(client_id, round_id)

    def template_usage_snapshot(self) -> dict[str, float]:
        return dict(self.template_usage_ema)

    def update_round_statistics(self, actions: dict[int, ClientAction]) -> None:
        if not actions:
            return
        observed_share = {template: 0.0 for template in self.TEMPLATE_NAMES}
        total = float(len(actions))
        for action in actions.values():
            observed_share[action.template] += 1.0 / total
        for template in self.TEMPLATE_NAMES:
            previous = self.template_usage_ema[template]
            self.template_usage_ema[template] = float(0.88 * previous + 0.12 * observed_share[template])

    def _apply_template_quota_guard(self, actions: dict[int, ClientAction], round_id: int) -> dict[int, ClientAction]:
        if len(actions) < 4 or not self.quota_guard_enabled:
            return actions
        counts = self._action_template_counts(actions)
        targets = self._round_template_targets(actions)

        quota_active = counts["shared_large"] > targets["max"]["shared_large"] + 1 or self.template_usage_ema["shared_large"] > 0.72
        if not quota_active:
            return actions

        for template in ("quarantine", "hybrid_rare"):
            if targets["min"][template] <= 0 or counts[template] > 0:
                continue
            candidate = self._best_reassignment(actions, round_id, counts, targets, template)
            if candidate is None:
                continue
            client_id, reassigned = candidate
            counts[actions[client_id].template] -= 1
            counts[template] += 1
            actions[client_id] = reassigned

        while counts["shared_large"] > targets["max"]["shared_large"] + 1:
            candidate = self._best_shared_large_reassignment(actions, round_id, counts, targets)
            if candidate is None:
                break
            client_id, reassigned = candidate
            counts[actions[client_id].template] -= 1
            counts[reassigned.template] += 1
            actions[client_id] = reassigned

        return actions

    def _action_template_counts(self, actions: dict[int, ClientAction]) -> dict[str, int]:
        counts = {template: 0 for template in self.TEMPLATE_NAMES}
        for action in actions.values():
            counts[action.template] += 1
        return counts

    def _round_template_targets(self, actions: dict[int, ClientAction]) -> dict[str, dict[str, int]]:
        total = max(1, len(actions))
        trust_values = np.array([action.trust for action in actions.values()], dtype=np.float64)
        rarity_values = np.array([action.rarity for action in actions.values()], dtype=np.float64)
        suspicion_values = np.array([action.suspicion for action in actions.values()], dtype=np.float64)

        rare_pressure = float(np.mean((rarity_values > 0.72) & (trust_values > 0.4)))
        suspicious_pressure = float(np.mean((suspicion_values > 0.48) | (trust_values < 0.24)))
        stable_pressure = float(np.mean((trust_values > 0.68) & (suspicion_values < 0.22)))

        shares = dict(self.template_target_share)
        shares["hybrid_rare"] = float(np.clip(shares["hybrid_rare"] + 0.08 * rare_pressure, 0.16, 0.3))
        shares["quarantine"] = float(np.clip(shares["quarantine"] + 0.1 * suspicious_pressure, 0.12, 0.24))
        shares["shared_small"] = float(np.clip(0.1 + 0.03 * (1.0 - stable_pressure) + 0.015 * suspicious_pressure, 0.06, 0.16))
        shares["shared_large"] = max(0.24, 1.0 - shares["hybrid_rare"] - shares["quarantine"] - shares["shared_small"])
        share_sum = sum(shares.values())
        shares = {template: value / share_sum for template, value in shares.items()}

        min_counts = {
            "shared_large": 0,
            "shared_small": 0,
            "hybrid_rare": 1 if total >= 10 and rare_pressure > 0.26 else 0,
            "quarantine": 1 if total >= 8 and suspicious_pressure > 0.14 else 0,
        }
        max_shared_large = max(2, int(np.ceil((shares["shared_large"] + 0.12) * total)))
        remaining_floor = min_counts["shared_small"] + min_counts["hybrid_rare"] + min_counts["quarantine"]
        max_shared_large = min(max_shared_large, max(2, total - remaining_floor))
        max_counts = {
            "shared_large": max_shared_large,
            "shared_small": total,
            "hybrid_rare": total,
            "quarantine": total,
        }
        return {"min": min_counts, "max": max_counts}

    def _best_reassignment(
        self,
        actions: dict[int, ClientAction],
        round_id: int,
        counts: dict[str, int],
        targets: dict[str, dict[str, int]],
        target_template: str,
    ) -> tuple[int, ClientAction] | None:
        best_choice: tuple[float, int, ClientAction] | None = None
        for client_id, action in actions.items():
            if action.policy_mode == "guardrail" or action.template == target_template:
                continue
            if not self._template_feasible(action, target_template):
                continue
            if action.template == "quarantine" and target_template != "quarantine" and action.suspicion > 0.55:
                continue
            candidate = self.action_for_template(client_id, round_id, target_template)
            score_loss = self._quota_reassignment_loss(action, candidate, counts, targets, target_template)
            if best_choice is None or score_loss < best_choice[0]:
                candidate.policy_mode = action.policy_mode
                candidate.quota_adjusted = True
                candidate.quota_reason = f"quota_min_{target_template}"
                best_choice = (score_loss, client_id, candidate)
        if best_choice is None or best_choice[0] > 0.14:
            return None
        return best_choice[1], best_choice[2]

    def _best_shared_large_reassignment(
        self,
        actions: dict[int, ClientAction],
        round_id: int,
        counts: dict[str, int],
        targets: dict[str, dict[str, int]],
    ) -> tuple[int, ClientAction] | None:
        best_choice: tuple[float, int, ClientAction] | None = None
        for client_id, action in actions.items():
            if action.template != "shared_large" or action.policy_mode == "guardrail":
                continue
            target_templates = ["shared_small"]
            if action.rarity > 0.74 and self._template_feasible(action, "hybrid_rare"):
                target_templates.append("hybrid_rare")
            if action.suspicion > 0.58 or action.trust < 0.26:
                target_templates.append("quarantine")
            for target_template in target_templates:
                if not self._template_feasible(action, target_template):
                    continue
                candidate = self.action_for_template(client_id, round_id, target_template)
                score_loss = self._quota_reassignment_loss(action, candidate, counts, targets, target_template)
                if best_choice is None or score_loss < best_choice[0]:
                    candidate.policy_mode = action.policy_mode
                    candidate.quota_adjusted = True
                    candidate.quota_reason = f"quota_cap_shared_large_to_{target_template}"
                    best_choice = (score_loss, client_id, candidate)
        if best_choice is None or best_choice[0] > 0.18:
            return None
        return best_choice[1], best_choice[2]

    def _quota_reassignment_loss(
        self,
        current: ClientAction,
        candidate: ClientAction,
        counts: dict[str, int],
        targets: dict[str, dict[str, int]],
        target_template: str,
    ) -> float:
        current_score = self._policy_basis_score(current)
        candidate_score = self._policy_basis_score(candidate) + self._template_fit_bonus(candidate, target_template)
        if counts[target_template] < targets["min"][target_template]:
            candidate_score += 0.08
        if target_template == "quarantine" and candidate.suspicion < 0.45:
            candidate_score -= 0.15
        if target_template == "shared_small":
            candidate_score += 0.04 * candidate.trust - 0.015 * candidate.suspicion
        if target_template == "hybrid_rare" and candidate.rarity < 0.74:
            candidate_score -= 0.1
        return current_score - candidate_score

    def _policy_basis_score(self, action: ClientAction) -> float:
        if action.policy_mode == "heuristic_warmup":
            return float(action.heuristic_score)
        if action.policy_mode == "guardrail":
            return float(action.heuristic_score - 0.5)
        return float(action.combined_score)

    def _template_feasible(self, action: ClientAction, target_template: str) -> bool:
        if target_template == "hybrid_rare":
            return action.rarity > 0.58 and action.trust > 0.3
        if target_template == "shared_small":
            return action.trust > 0.18 and action.suspicion < 0.78
        if target_template == "quarantine":
            return action.suspicion > 0.42 or action.trust < 0.32
        if target_template == "shared_large":
            return action.trust > 0.38 and action.suspicion < 0.68
        return True

    def _template_fit_bonus(self, action: ClientAction, target_template: str) -> float:
        if target_template == "hybrid_rare":
            return float(0.08 * action.rarity + 0.03 * action.trust + 0.03 * action.stable_overlap - 0.04 * action.suspicion - 0.05 * action.contamination_memory)
        if target_template == "shared_small":
            return float(0.05 * (1.0 - action.bandwidth) + 0.03 * action.trust + 0.02 * action.shared_exposure - 0.04 * action.contamination_memory)
        if target_template == "quarantine":
            return float(0.12 * action.suspicion + 0.08 * action.contamination_memory + 0.05 * action.suspicious_overlap + 0.05 * (1.0 - action.trust))
        if target_template == "shared_large":
            return float(0.06 * action.trust + 0.03 * action.bandwidth + 0.03 * action.stable_overlap - 0.08 * action.suspicion - 0.08 * action.contamination_memory)
        return 0.0

    def _select_action(self, client_id: int, round_id: int) -> ClientAction:
        context_vector = self._context_vector(client_id, round_id)
        heuristic_scores = self._heuristic_scores(client_id, round_id)
        bandit_stats = self.bandit.scores(context_vector)
        bandit_optimistic = {name: bandit_stats[name]["optimistic"] for name in self.TEMPLATE_NAMES}
        combined_scores, diversity_bonus = self._combined_scores(heuristic_scores, bandit_optimistic)
        heuristic_template = max(heuristic_scores, key=heuristic_scores.get)
        bandit_template = max(bandit_optimistic, key=bandit_optimistic.get)
        guardrail_template = self._guardrail_template(client_id)

        if guardrail_template is not None:
            chosen_template = guardrail_template
            policy_mode = "guardrail"
        elif round_id <= self.bandit_warmup_rounds or self.bandit.min_count() < 2:
            chosen_template = heuristic_template
            policy_mode = "heuristic_warmup"
        else:
            chosen_template = max(combined_scores, key=combined_scores.get)
            policy_mode = "bandit"

        return self._build_action(
            client_id=client_id,
            template=chosen_template,
            round_id=round_id,
            context_vector=context_vector,
            heuristic_scores=heuristic_scores,
            bandit_optimistic=bandit_optimistic,
            combined_scores=combined_scores,
            policy_mode=policy_mode,
            heuristic_template=heuristic_template,
            bandit_template=bandit_template,
            diversity_bonus=diversity_bonus,
        )

    def _build_action(
        self,
        client_id: int,
        template: str,
        round_id: int,
        context_vector: np.ndarray,
        heuristic_scores: dict[str, float],
        bandit_optimistic: dict[str, float],
        combined_scores: dict[str, float],
        policy_mode: str,
        heuristic_template: str,
        bandit_template: str,
        diversity_bonus: dict[str, float],
        quota_adjusted: bool = False,
        quota_reason: str = "",
    ) -> ClientAction:
        profile = self.client_profiles[client_id]
        selection_metadata = self._last_selection_metadata.get(client_id)
        if selection_metadata is None:
            selection_mode = self._selection_policy_mode(round_id)
            selection_metadata = self._selection_metadata_map([client_id], round_id, selection_mode)[client_id]
        effective_trust = self.effective_trust(client_id)
        effective_rarity = self.effective_rarity(client_id)
        bandwidth = profile.bandwidth_score
        suspicion = self.suspicion_scores[client_id]
        shared_exposure = self.shared_exposure_scores[client_id]
        stable_overlap = self.stable_overlap_scores[client_id]
        suspicious_overlap = self.suspicious_overlap_scores[client_id]
        contamination_memory = self.contamination_memory[client_id]
        deception = self.deception_scores[client_id]
        stability = self.stability_scores[client_id]
        mask_bundle = self._mask_for_client(client_id, profile, template)
        base_weight = 0.6 * effective_trust + 0.21 * effective_rarity + 0.13 * bandwidth + 0.03 * stable_overlap + 0.03 * stability
        penalty = 0.2 * suspicion + 0.04 * contamination_memory + 0.01 * shared_exposure * max(suspicious_overlap, 0.05) + 0.08 * deception
        aggregation_weight = max(0.18, base_weight * (1.0 - penalty))
        if template == "quarantine":
            aggregation_weight = min(aggregation_weight, 0.2)
        return ClientAction(
            client_id=client_id,
            template=template,
            trust=effective_trust,
            rarity=effective_rarity,
            bandwidth=bandwidth,
            suspicion=suspicion,
            shared_exposure=shared_exposure,
            stable_overlap=stable_overlap,
            suspicious_overlap=suspicious_overlap,
            contamination_memory=contamination_memory,
            aggregation_weight=aggregation_weight,
            mask_bundle=mask_bundle,
            context_vector=tuple(float(value) for value in context_vector.tolist()),
            policy_mode=policy_mode,
            heuristic_template=heuristic_template,
            bandit_template=bandit_template,
            heuristic_score=float(heuristic_scores[template]),
            bandit_score=float(bandit_optimistic[template]),
            combined_score=float(combined_scores[template]),
            diversity_bonus=float(diversity_bonus[template]),
            quota_adjusted=quota_adjusted,
            quota_reason=quota_reason,
            selection_policy_mode=selection_metadata.policy_mode,
            selection_heuristic_score=selection_metadata.heuristic_score,
            selection_bandit_score=selection_metadata.bandit_score,
            selection_combined_score=selection_metadata.combined_score,
            selection_reward_memory=selection_metadata.reward_memory,
        )

    def _context_vector(self, client_id: int, round_id: int) -> np.ndarray:
        trust = self.effective_trust(client_id)
        rarity = self.effective_rarity(client_id)
        suspicion = self.suspicion_scores[client_id]
        bandwidth = self.client_profiles[client_id].bandwidth_score
        shared_exposure = self.shared_exposure_scores[client_id]
        stable_overlap = self.stable_overlap_scores[client_id]
        suspicious_overlap = self.suspicious_overlap_scores[client_id]
        contamination_memory = self.contamination_memory[client_id]
        deception = self.deception_scores[client_id]
        stability = self.stability_scores[client_id]
        round_feature = float(np.tanh(round_id / 10.0))
        return np.array(
            [
                1.0,
                trust,
                rarity,
                bandwidth,
                suspicion,
                shared_exposure,
                stable_overlap,
                suspicious_overlap,
                contamination_memory,
                deception,
                stability,
                trust * rarity,
                trust * (1.0 - suspicion),
                (1.0 - deception) * stability,
                suspicious_overlap + deception,
                round_feature,
            ],
            dtype=np.float64,
        )

    def _heuristic_scores(self, client_id: int, round_id: int) -> dict[str, float]:
        trust = self.effective_trust(client_id)
        rarity = self.effective_rarity(client_id)
        bandwidth = self.client_profiles[client_id].bandwidth_score
        suspicion = self.suspicion_scores[client_id]
        shared_exposure = self.shared_exposure_scores[client_id]
        stable_overlap = self.stable_overlap_scores[client_id]
        suspicious_overlap = self.suspicious_overlap_scores[client_id]
        contamination_memory = self.contamination_memory[client_id]
        deception = self.deception_scores[client_id]
        stability = self.stability_scores[client_id]
        scores = {
            "shared_large": 0.36 + 0.48 * trust + 0.18 * bandwidth + 0.1 * rarity + 0.03 * stable_overlap + 0.04 * stability - 0.34 * suspicion - 0.04 * suspicious_overlap - 0.03 * contamination_memory - 0.09 * deception,
            "shared_small": 0.44 + 0.3 * trust + 0.08 * bandwidth + 0.08 * rarity + 0.02 * shared_exposure + 0.02 * stability - 0.16 * suspicion - 0.02 * contamination_memory - 0.04 * deception,
            "hybrid_rare": 0.28 + 0.22 * trust + 0.54 * rarity + 0.08 * bandwidth + 0.04 * stable_overlap + 0.04 * stability - 0.22 * suspicion - 0.03 * contamination_memory - 0.05 * deception,
            "quarantine": 0.02 + 0.65 * suspicion + 0.08 * contamination_memory + 0.06 * suspicious_overlap + 0.4 * (1.0 - trust) + 0.1 * deception + 0.08 * (1.0 - stability) - 0.14 * rarity,
        }
        if round_id <= 4:
            scores["shared_large"] += 0.22
            scores["shared_small"] += 0.16
            scores["hybrid_rare"] += 0.08
            scores["quarantine"] -= 0.22
        if rarity > 0.75 and trust > 0.4 and contamination_memory < 0.4:
            scores["hybrid_rare"] += 0.12
        if suspicion > 0.75 or contamination_memory > 0.75:
            scores["quarantine"] += 0.14
            scores["shared_large"] -= 0.18
        return scores

    def _combined_scores(self, heuristic_scores: dict[str, float], bandit_optimistic: dict[str, float]) -> tuple[dict[str, float], dict[str, float]]:
        heuristic_scaled = self._normalize_scores(heuristic_scores)
        bandit_scaled = self._normalize_scores(bandit_optimistic)
        diversity_bonus = self._template_diversity_bonus()
        combined: dict[str, float] = {}
        for template in self.TEMPLATE_NAMES:
            combined[template] = self.bandit_mix * heuristic_scaled[template] + (1.0 - self.bandit_mix) * bandit_scaled[template] + diversity_bonus[template]
        return combined, diversity_bonus

    def _template_diversity_bonus(self) -> dict[str, float]:
        bonus: dict[str, float] = {}
        for template in self.TEMPLATE_NAMES:
            gap = self.template_target_share[template] - self.template_usage_ema[template]
            bonus[template] = float(np.clip(self.diversity_strength * gap, -0.08, 0.08))
        return bonus

    def _normalize_scores(self, scores: dict[object, float]) -> dict[object, float]:
        values = list(scores.values())
        min_value = min(values)
        max_value = max(values)
        scale = max(1e-6, max_value - min_value)
        return {key: (value - min_value) / scale for key, value in scores.items()}

    def _guardrail_template(self, client_id: int) -> str | None:
        trust = self.effective_trust(client_id)
        suspicion = self.suspicion_scores[client_id]
        contamination_memory = self.contamination_memory[client_id]
        deception = self.deception_scores[client_id]
        stability = self.stability_scores[client_id]
        if suspicion > 0.92 or trust < 0.08 or contamination_memory > 0.9 or deception > 0.9:
            return "quarantine"
        if deception > 0.78 and stability < 0.38:
            return "quarantine"
        return None

    def _selection_policy_mode(self, round_id: int) -> str:
        if round_id <= 2:
            return "random_warmup"
        if round_id <= self.selection_warmup_rounds or self.selection_bandit.count("select") < 12:
            return "heuristic_warmup"
        return "bandit"

    def _selection_metadata_map(
        self,
        client_ids: list[int],
        round_id: int,
        policy_mode: str,
    ) -> dict[int, SelectionMetadata]:
        metadata: dict[int, SelectionMetadata] = {}
        heuristic_scores: dict[int, float] = {}
        bandit_scores: dict[int, float] = {}

        for client_id in client_ids:
            context_vector = self._selection_context(client_id, round_id)
            heuristic_score = self._selection_heuristic_score(client_id, round_id)
            bandit_score = float(np.clip(self.selection_bandit.score("select", context_vector)["optimistic"], -1.5, 1.5))
            heuristic_scores[client_id] = heuristic_score
            bandit_scores[client_id] = bandit_score
            metadata[client_id] = SelectionMetadata(
                client_id=client_id,
                context_vector=tuple(float(value) for value in context_vector.tolist()),
                policy_mode=policy_mode,
                heuristic_score=heuristic_score,
                bandit_score=bandit_score,
                combined_score=heuristic_score,
                reward_memory=float(self.selection_reward_memory[client_id]),
            )

        heuristic_scaled = self._normalize_scores(heuristic_scores)
        bandit_scaled = self._normalize_scores(bandit_scores)
        adaptive_selection_mix = float(
            np.clip(
                self.selection_mix + 0.35 * max(0.0, -self.selection_gap_ema) - 0.08 * max(0.0, self.selection_gap_ema),
                0.55,
                0.98,
            )
        )
        for client_id in client_ids:
            combined_score = adaptive_selection_mix * heuristic_scaled[client_id] + (1.0 - adaptive_selection_mix) * bandit_scaled[client_id]
            if policy_mode == "random_warmup":
                combined_score = 0.5 * heuristic_scaled[client_id] + 0.5 * bandit_scaled[client_id]
            elif policy_mode == "heuristic_warmup":
                combined_score = heuristic_scaled[client_id]
            metadata[client_id].combined_score = float(combined_score)
        return metadata

    def _selection_context(self, client_id: int, round_id: int) -> np.ndarray:
        trust = self.effective_trust(client_id)
        rarity = self.effective_rarity(client_id)
        suspicion = self.suspicion_scores[client_id]
        bandwidth = self.client_profiles[client_id].bandwidth_score
        reward_memory = float(np.clip(self.selection_reward_memory[client_id], -1.5, 1.5))
        shared_exposure = self.shared_exposure_scores[client_id]
        stable_overlap = self.stable_overlap_scores[client_id]
        suspicious_overlap = self.suspicious_overlap_scores[client_id]
        contamination_memory = self.contamination_memory[client_id]
        deception = self.deception_scores[client_id]
        stability = self.stability_scores[client_id]
        round_feature = float(np.tanh(round_id / 10.0))
        return np.array(
            [
                1.0,
                trust,
                rarity,
                bandwidth,
                suspicion,
                reward_memory,
                shared_exposure,
                stable_overlap,
                suspicious_overlap,
                contamination_memory,
                deception,
                stability,
                trust * rarity,
                trust * (1.0 - suspicion),
                reward_memory * (1.0 - suspicion),
                stability * (1.0 - deception),
                deception * (1.0 - trust),
                round_feature,
            ],
            dtype=np.float64,
        )

    def _selection_heuristic_score(self, client_id: int, round_id: int) -> float:
        effective_trust = self.effective_trust(client_id)
        rare_bonus = self.effective_rarity(client_id)
        bandwidth = self.client_profiles[client_id].bandwidth_score
        suspicion = self.suspicion_scores[client_id]
        shared_exposure = self.shared_exposure_scores[client_id]
        stable_overlap = self.stable_overlap_scores[client_id]
        suspicious_overlap = self.suspicious_overlap_scores[client_id]
        contamination_memory = self.contamination_memory[client_id]
        deception = self.deception_scores[client_id]
        stability = self.stability_scores[client_id]
        rare_stable_protection = rare_bonus > 0.72 and stability > 0.72 and contamination_memory < 0.35
        deception_penalty = max(0.04, 0.1 - 0.04 * rare_bonus) * deception
        score = 0.3 + 0.46 * effective_trust + 0.19 * rare_bonus + 0.08 * bandwidth + 0.02 * stable_overlap + 0.04 * stability
        score -= 0.02 * suspicious_overlap + 0.02 * contamination_memory + 0.01 * shared_exposure * max(suspicion, 0.05) + deception_penalty + 0.03 * (1.0 - stability)
        if round_id <= 4:
            score += 0.06
        if deception > 0.78 or suspicion > 0.8 or contamination_memory > 0.8:
            score *= 0.58 if rare_stable_protection else 0.42
        elif deception > 0.58 or suspicion > 0.62 or contamination_memory > 0.6:
            score *= 0.86 if rare_stable_protection else 0.74
        if rare_bonus > 0.76 and effective_trust > 0.4 and contamination_memory < 0.32 and (deception < 0.42 or stability > 0.72):
            score += 0.07
        return float(max(score, 0.02))

    def _selection_weight(self, client_id: int, metadata: SelectionMetadata) -> float:
        rarity = self.effective_rarity(client_id)
        stability = self.stability_scores[client_id]
        deception = self.deception_scores[client_id]
        weight = metadata.heuristic_score
        weight *= 1.0 + 0.03 * metadata.combined_score
        weight *= 1.0 - 0.13 * deception + 0.05 * stability
        if rarity > 0.74 and stability > 0.7 and self.contamination_memory[client_id] < 0.34:
            weight *= 1.08
        return float(max(weight, 0.02))

    def _private_units(self, client_id: int, size_h1: int, size_h2: int, size_h3: int = 0):
        if self.model_family == 'mlp':
            h1_pool = list(range(len(self.shared_large[0]), self.hidden_one)) or list(range(self.hidden_one))
            h2_pool = list(range(len(self.shared_large[1]), self.hidden_two)) or list(range(self.hidden_two))
            start_h1 = (client_id * 11) % len(h1_pool)
            start_h2 = (client_id * 7) % len(h2_pool)
            h1_units = [h1_pool[(start_h1 + offset) % len(h1_pool)] for offset in range(min(size_h1, len(h1_pool)))]
            h2_units = [h2_pool[(start_h2 + offset) % len(h2_pool)] for offset in range(min(size_h2, len(h2_pool)))]
            return h1_units, h2_units
        h1_pool = list(range(len(self.shared_large[0]), self.conv_one)) or list(range(self.conv_one))
        h2_pool = list(range(len(self.shared_large[1]), self.conv_two)) or list(range(self.conv_two))
        h3_pool = list(range(len(self.shared_large[2]), self.cnn_hidden)) or list(range(self.cnn_hidden))
        start_h1 = (client_id * 11) % len(h1_pool)
        start_h2 = (client_id * 7) % len(h2_pool)
        start_h3 = (client_id * 5) % len(h3_pool)
        h1_units = [h1_pool[(start_h1 + offset) % len(h1_pool)] for offset in range(min(size_h1, len(h1_pool)))]
        h2_units = [h2_pool[(start_h2 + offset) % len(h2_pool)] for offset in range(min(size_h2, len(h2_pool)))]
        h3_units = [h3_pool[(start_h3 + offset) % len(h3_pool)] for offset in range(min(size_h3, len(h3_pool)))]
        return h1_units, h2_units, h3_units

    def _mask_for_client(self, client_id: int, profile: ClientProfile, template: str) -> MaskBundle:
        all_labels = list(range(self.num_classes))
        if self.model_family == 'mlp':
            if template == "shared_large":
                h1_units = self.shared_large[0]
                h2_units = self.shared_large[1]
                label_rows = all_labels
            elif template == "shared_small":
                h1_units = self.shared_small[0]
                h2_units = self.shared_small[1]
                label_rows = all_labels
            elif template == "hybrid_rare":
                private_h1, private_h2 = self._private_units(client_id, size_h1=self.hybrid_private_h1, size_h2=self.hybrid_private_h2)
                h1_units = sorted(set(self.shared_small[0] + private_h1))
                h2_units = sorted(set(self.shared_small[1] + private_h2))
                label_rows = all_labels
            elif template == "quarantine":
                h1_units, h2_units = self._private_units(client_id, size_h1=self.quarantine_private_h1, size_h2=self.quarantine_private_h2)
                label_rows = profile.observed_labels
            else:
                raise ValueError(f"Unknown template: {template}")
            return self._build_mask(shared_h1=h1_units, shared_h2=h2_units, label_rows=label_rows)

        if template == "shared_large":
            h1_units = self.shared_large[0]
            h2_units = self.shared_large[1]
            h3_units = self.shared_large[2]
            label_rows = all_labels
        elif template == "shared_small":
            h1_units = self.shared_small[0]
            h2_units = self.shared_small[1]
            h3_units = self.shared_small[2]
            label_rows = all_labels
        elif template == "hybrid_rare":
            private_h1, private_h2, private_h3 = self._private_units(
                client_id,
                size_h1=self.hybrid_private_h1,
                size_h2=self.hybrid_private_h2,
                size_h3=self.hybrid_private_h3,
            )
            h1_units = sorted(set(self.shared_small[0] + private_h1))
            h2_units = sorted(set(self.shared_small[1] + private_h2))
            h3_units = sorted(set(self.shared_small[2] + private_h3))
            label_rows = all_labels
        elif template == "quarantine":
            h1_units, h2_units, h3_units = self._private_units(
                client_id,
                size_h1=self.quarantine_private_h1,
                size_h2=self.quarantine_private_h2,
                size_h3=self.quarantine_private_h3,
            )
            label_rows = profile.observed_labels
        else:
            raise ValueError(f"Unknown template: {template}")
        return self._build_mask(shared_h1=h1_units, shared_h2=h2_units, shared_hidden=h3_units, label_rows=label_rows)

    def _build_mask(
        self,
        shared_h1: list[int],
        shared_h2: list[int],
        label_rows: list[int],
        shared_hidden: list[int] | None = None,
    ) -> MaskBundle:
        masks = self._create_mask_tensors(shared_h1=shared_h1, shared_h2=shared_h2, shared_hidden=shared_hidden, label_rows=label_rows)
        active_params = int(sum(mask.sum().item() for mask in masks.values()))
        shared_active = sum(mask.sum().item() for mask in self._shared_core_mask.values())
        overlap = sum((masks[name] * self._shared_core_mask[name]).sum().item() for name in masks)
        overlap_ratio = float(overlap / max(1.0, min(active_params, shared_active)))
        return MaskBundle(masks=masks, active_params=active_params, overlap_ratio=overlap_ratio)

    def _create_mask_tensors(
        self,
        shared_h1: list[int],
        shared_h2: list[int],
        label_rows: list[int],
        shared_hidden: list[int] | None = None,
    ) -> dict[str, torch.Tensor]:
        if self.model_family == 'mlp':
            masks = {
                "fc1.weight": torch.zeros((self.hidden_one, self.input_dim), dtype=torch.float32),
                "fc1.bias": torch.zeros(self.hidden_one, dtype=torch.float32),
                "fc2.weight": torch.zeros((self.hidden_two, self.hidden_one), dtype=torch.float32),
                "fc2.bias": torch.zeros(self.hidden_two, dtype=torch.float32),
                "fc3.weight": torch.zeros((self.num_classes, self.hidden_two), dtype=torch.float32),
                "fc3.bias": torch.zeros(self.num_classes, dtype=torch.float32),
            }
            if shared_h1:
                masks["fc1.weight"][shared_h1, :] = 1.0
                masks["fc1.bias"][shared_h1] = 1.0
            if shared_h2 and shared_h1:
                row_index = torch.tensor(shared_h2, dtype=torch.long).unsqueeze(1)
                col_index = torch.tensor(shared_h1, dtype=torch.long).unsqueeze(0)
                masks["fc2.weight"][row_index, col_index] = 1.0
                masks["fc2.bias"][shared_h2] = 1.0
            if shared_h2 and label_rows:
                output_rows = torch.tensor(label_rows, dtype=torch.long).unsqueeze(1)
                hidden_cols = torch.tensor(shared_h2, dtype=torch.long).unsqueeze(0)
                masks["fc3.weight"][output_rows, hidden_cols] = 1.0
            if label_rows:
                masks["fc3.bias"][label_rows] = 1.0
            return masks

        hidden_units = shared_hidden or []
        in_channels = int(self.feature_shape[0])
        masks = {
            "conv1.weight": torch.zeros((self.conv_one, in_channels, 3, 3), dtype=torch.float32),
            "conv1.bias": torch.zeros(self.conv_one, dtype=torch.float32),
            "conv2.weight": torch.zeros((self.conv_two, self.conv_one, 3, 3), dtype=torch.float32),
            "conv2.bias": torch.zeros(self.conv_two, dtype=torch.float32),
            "fc1.weight": torch.zeros((self.cnn_hidden, self.conv_two * self.conv_feature_size), dtype=torch.float32),
            "fc1.bias": torch.zeros(self.cnn_hidden, dtype=torch.float32),
            "fc2.weight": torch.zeros((self.num_classes, self.cnn_hidden), dtype=torch.float32),
            "fc2.bias": torch.zeros(self.num_classes, dtype=torch.float32),
        }
        if shared_h1:
            masks["conv1.weight"][shared_h1, :, :, :] = 1.0
            masks["conv1.bias"][shared_h1] = 1.0
        if shared_h2 and shared_h1:
            row_index = torch.tensor(shared_h2, dtype=torch.long).unsqueeze(1)
            col_index = torch.tensor(shared_h1, dtype=torch.long).unsqueeze(0)
            masks["conv2.weight"][row_index, col_index, :, :] = 1.0
            masks["conv2.bias"][shared_h2] = 1.0
        if hidden_units and shared_h2:
            hidden_index = torch.tensor(hidden_units, dtype=torch.long)
            masks["fc1.bias"][hidden_units] = 1.0
            for channel in shared_h2:
                start = channel * self.conv_feature_size
                end = start + self.conv_feature_size
                masks["fc1.weight"][hidden_index, start:end] = 1.0
        if hidden_units and label_rows:
            output_rows = torch.tensor(label_rows, dtype=torch.long).unsqueeze(1)
            hidden_cols = torch.tensor(hidden_units, dtype=torch.long).unsqueeze(0)
            masks["fc2.weight"][output_rows, hidden_cols] = 1.0
        if label_rows:
            masks["fc2.bias"][label_rows] = 1.0
        # Residual CNN layers are intentionally fully shared. This keeps every
        # trainable tensor defined while the original controlled layers retain
        # the AgenticFedLock mask structure.
        for name, tensor in build_model(self.model_spec).named_parameters():
            masks.setdefault(name, torch.ones_like(tensor, dtype=torch.float32))
        return masks


# Backward-compatible alias for older imports.
AgentLockController = AgenticFedLockController
