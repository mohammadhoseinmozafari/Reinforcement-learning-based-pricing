"""Day 4 tests for universal recurrent agents, replay, and shared runtime."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from env.pricing_contracts import AgentArchitecture, PricingObservationFeature
from evaluation.universal_pricing_evaluator import UniversalPricingEvaluator
from models.recurrent_sac_pricing import (
    OpponentEmbeddingRecurrentSACPricingAgent,
    RecurrentSACPricingAgent,
    RecurrentSACPricingAgentConfig,
)
from models.sac_pricing import SACPricingAgentConfig
from models.universal_pricing_agents import UniversalPricingAgentFactory
from models.universal_pricing_replay import UniversalPricingTransition
from models.universal_pricing_sequence_replay import (
    UniversalPricingEpisode,
    UniversalPricingSequenceReplayBuffer,
)
from train.universal_pricing_protocol import (
    ExperimentMatrix,
    TrainingBudgetConfig,
    load_universal_pricing_protocol,
)
from train.universal_pricing_trainer import UniversalPricingTrainer


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "config/protocols/universal_pricing_v1.yaml"


def make_observation(index: int, *, decision: bool = False) -> np.ndarray:
    value = np.zeros(18, dtype=np.float32)
    value[0] = np.float32(index / 200)
    value[PricingObservationFeature.OWN_REGIME.index] = -1.0
    value[PricingObservationFeature.REGIME_DECISION_ALLOWED.index] = (
        1.0 if decision else -1.0
    )
    return value


def make_episode(length: int = 40) -> UniversalPricingEpisode:
    transitions = []
    for index in range(length):
        transitions.append(
            UniversalPricingTransition(
                observation=make_observation(
                    index, decision=index % 10 == 0
                ),
                effective_action=np.asarray(
                    [1.0, 0.0, 0.1, -0.2, 0.3],
                    dtype=np.float32,
                ),
                reward=index / 100.0,
                next_observation=make_observation(
                    index + 1, decision=(index + 1) % 10 == 0
                ),
                done=index == length - 1,
                regime_decision_mask=float(index % 10 == 0),
                opponent_price_controls=np.asarray(
                    [0.2, -0.1, 0.4], dtype=np.float32
                ),
            )
        )
    return UniversalPricingEpisode(
        tuple(transitions),
        episode_index=0,
        consumer_seed=1,
        opponent_seed=2,
        opponent_family="uniform",
        opponent_policy_name="uniform_fixed",
    )


class UniversalSequenceReplayTests(unittest.TestCase):
    def make_replay(self, seed=123) -> UniversalPricingSequenceReplayBuffer:
        replay = UniversalPricingSequenceReplayBuffer(
            capacity_episodes=10,
            learning_sequence_length=16,
            burn_in_length=16,
            batch_size=4,
            replay_sampling_seed=seed,
        )
        replay.push_episode(make_episode())
        return replay

    def test_burn_in_loss_and_valid_masks(self) -> None:
        batch = self.make_replay().sample()
        self.assertEqual(batch.observations.shape, (4, 32, 18))
        for valid, loss in zip(batch.valid_masks, batch.loss_masks):
            self.assertLessEqual(int(loss.sum()), 16)
            self.assertTrue(np.all(loss <= valid))
            learning_indices = np.flatnonzero(loss[:, 0])
            if len(learning_indices):
                self.assertLessEqual(learning_indices[0], 16)
                self.assertTrue(
                    np.all(loss[: learning_indices[0]] == 0)
                )

    def test_sampling_and_restored_rng_are_exact(self) -> None:
        first = self.make_replay()
        repeated = self.make_replay()
        a = first.sample()
        b = repeated.sample()
        for field_name in a.__dataclass_fields__:
            np.testing.assert_array_equal(
                getattr(a, field_name), getattr(b, field_name)
            )
        state = first.state_dict()
        expected = first.sample()
        restored = self.make_replay(seed=999)
        restored.load_state_dict(state)
        actual = restored.sample()
        for field_name in expected.__dataclass_fields__:
            np.testing.assert_array_equal(
                getattr(expected, field_name),
                getattr(actual, field_name),
            )


class RecurrentPricingAgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = load_universal_pricing_protocol(PROTOCOL_PATH)
        cls.bundle = cls.protocol.run_seed_bundle(0)

    def config(self, architecture: AgentArchitecture):
        values = dict(
            architecture=architecture,
            learning_sequence_length=8,
            burn_in_length=8,
            episode_replay_capacity=20,
            batch_size=2,
            actor_hidden_dimension=16,
            critic_hidden_dimension=16,
        )
        if architecture is AgentArchitecture.OE_RSAC:
            values.update(
                opponent_embedding_dimension=4,
                encoder_hidden_dimension=8,
                encoder_learning_rate=3e-4,
                auxiliary_loss_weight=1.0,
            )
        return RecurrentSACPricingAgentConfig(**values)

    def agent(self, architecture: AgentArchitecture):
        cls = (
            RecurrentSACPricingAgent
            if architecture is AgentArchitecture.RSAC
            else OpponentEmbeddingRecurrentSACPricingAgent
        )
        return cls(
            self.config(architecture),
            network_initialization_seed=self.bundle.network_initialization_seed,
            exploration_seed=self.bundle.exploration_seed,
            torch_cpu_seed=self.bundle.torch_cpu_seed,
            torch_cuda_seed=self.bundle.torch_cuda_seed,
        )

    def batch(self):
        replay = UniversalPricingSequenceReplayBuffer(
            capacity_episodes=10,
            learning_sequence_length=8,
            burn_in_length=8,
            batch_size=2,
            replay_sampling_seed=4,
        )
        replay.push_episode(make_episode(20))
        return replay.sample()

    def test_plain_rsac_has_no_encoder_and_oe_rsac_has_target_encoder(self):
        plain = self.agent(AgentArchitecture.RSAC)
        embedding = self.agent(AgentArchitecture.OE_RSAC)
        self.assertIsNone(plain.opponent_encoder)
        self.assertIsNone(plain.encoder_optimizer)
        self.assertIsNotNone(embedding.opponent_encoder)
        self.assertIsNotNone(embedding.target_opponent_encoder)

    def test_both_agents_update_same_sequence_contract(self):
        batch = self.batch()
        for architecture in (
            AgentArchitecture.RSAC,
            AgentArchitecture.OE_RSAC,
        ):
            metrics = self.agent(architecture).update(batch)
            self.assertTrue(
                all(np.isfinite(value) for value in metrics.values())
            )

    def test_online_context_reset_and_checkpoint_continuation(self):
        agent = self.agent(AgentArchitecture.OE_RSAC)
        obs = make_observation(0, decision=True)
        agent.select_action(obs)
        agent.observe_transition(make_episode(1).transitions[0])
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "agent.pt"
            agent.save(checkpoint)
            expected = agent.select_action(make_observation(1))
            restored = self.agent(AgentArchitecture.OE_RSAC)
            restored.load(checkpoint)
            actual = restored.select_action(make_observation(1))
        self.assertEqual(expected, actual)
        restored.reset_recurrent_state()
        self.assertIsNone(restored._actor_hidden)
        np.testing.assert_array_equal(restored._previous_action, np.zeros(5))

    def test_padding_values_do_not_change_update_metrics(self):
        batch = self.batch()
        changed = {
            name: values.copy()
            for name, values in batch.as_mapping().items()
        }
        padding = changed["valid_masks"] == 0
        for name, values in changed.items():
            if name not in {"valid_masks", "loss_masks"}:
                values[np.broadcast_to(padding, values.shape)] = 0.75
        first = self.agent(AgentArchitecture.RSAC).update(batch)
        second = self.agent(AgentArchitecture.RSAC).update(changed)
        for name in first:
            self.assertAlmostEqual(first[name], second[name], places=6)


def small_protocol(artifact_root: Path):
    protocol = load_universal_pricing_protocol(PROTOCOL_PATH)
    profiles = dict(protocol.agent_profiles)
    profiles[AgentArchitecture.SAC] = replace(
        profiles[AgentArchitecture.SAC],
        sac_pricing_config=SACPricingAgentConfig(
            actor_hidden_dimensions=(16, 16),
            critic_hidden_dimensions=(16, 16),
            replay_capacity=200,
            batch_size=8,
        ),
    )
    for architecture in (AgentArchitecture.RSAC, AgentArchitecture.OE_RSAC):
        profile = profiles[architecture]
        recurrent = RecurrentSACPricingAgentConfig(
            architecture=architecture,
            learning_sequence_length=4,
            burn_in_length=4,
            episode_replay_capacity=20,
            batch_size=2,
            actor_hidden_dimension=8,
            critic_hidden_dimension=8,
            **(
                {
                    "opponent_embedding_dimension": 4,
                    "encoder_hidden_dimension": 8,
                    "encoder_learning_rate": 3e-4,
                    "auxiliary_loss_weight": 1.0,
                }
                if architecture is AgentArchitecture.OE_RSAC
                else {}
            ),
        )
        profiles[architecture] = replace(
            profile,
            sequence_length=4,
            episode_replay_capacity=20,
            opponent_embedding_dim=(
                4 if architecture is AgentArchitecture.OE_RSAC else None
            ),
            encoder_hidden_dim=(
                8 if architecture is AgentArchitecture.OE_RSAC else None
            ),
            auxiliary_loss_weight=(
                1.0 if architecture is AgentArchitecture.OE_RSAC else None
            ),
            recurrent_pricing_config=recurrent,
        )
    return replace(
        protocol,
        artifact_root=artifact_root,
        agent_profiles=profiles,
    )


class UniversalRuntimeTests(unittest.TestCase):
    def test_factory_builds_all_three_public_agents(self):
        with tempfile.TemporaryDirectory() as temporary:
            protocol = small_protocol(Path(temporary) / "runs")
            names = {}
            for coordinate in ExperimentMatrix(protocol).coordinates():
                if coordinate.training_seed_index == 0:
                    architecture = coordinate.agent_architecture
                    if architecture not in names:
                        components = UniversalPricingAgentFactory.create(
                            protocol.agent_profiles[architecture],
                            protocol.run_seed_bundle(0),
                        )
                        names[architecture] = type(
                            components.agent
                        ).__name__
            self.assertEqual(
                names,
                {
                    AgentArchitecture.SAC: "SACPricingAgent",
                    AgentArchitecture.RSAC: "RecurrentSACPricingAgent",
                    AgentArchitecture.OE_RSAC: (
                        "OpponentEmbeddingRecurrentSACPricingAgent"
                    ),
                },
            )

    def test_short_shared_training_run_completes_for_every_architecture(self):
        budget = TrainingBudgetConfig(
            environment_steps=100,
            warmup_steps=99,
            updates_per_step=1,
            evaluation_interval_steps=100,
            checkpoint_interval_steps=100,
        )
        with tempfile.TemporaryDirectory() as temporary:
            protocol = small_protocol(Path(temporary) / "runs")
            selected = {}
            for coordinate in ExperimentMatrix(protocol).coordinates():
                if (
                    coordinate.training_seed_index == 0
                    and coordinate.agent_architecture not in selected
                ):
                    selected[coordinate.agent_architecture] = coordinate
            for coordinate in selected.values():
                trainer = UniversalPricingTrainer(
                    protocol,
                    coordinate,
                    budget=budget,
                    run_validation=False,
                    verbose=False,
                )
                manifest = trainer.train()
                self.assertEqual(manifest.status.value, "completed")
                self.assertEqual(trainer.environment_steps, 100)
                self.assertTrue(trainer.final_checkpoint_path.is_file())
                records = [
                    json.loads(line)
                    for line in trainer.metrics_path.read_text(
                        encoding="utf-8"
                    ).splitlines()
                ]
                training_record = next(
                    record
                    for record in records
                    if record["phase"] == "training"
                )
                self.assertTrue(
                    {
                        "agent_bbp_period_fraction",
                        "mean_agent_uniform_price",
                        "mean_agent_bbp_price_spread",
                        "mean_market_share",
                        "mean_retention_rate",
                        "replay_fill_fraction",
                        "replay_bbp_fraction",
                        "replay_decision_fraction",
                        "update_count",
                    }
                    <= set(training_record)
                )
                self.assertTrue(
                    trainer.logger.latest_metrics_path.is_file()
                )

    def test_evaluation_pairs_are_family_balanced(self):
        with tempfile.TemporaryDirectory() as temporary:
            protocol = small_protocol(Path(temporary) / "runs")
            coordinate = next(
                item
                for item in ExperimentMatrix(protocol).coordinates()
                if item.agent_architecture is AgentArchitecture.RSAC
            )
            components = UniversalPricingAgentFactory.create(
                protocol.agent_profiles[AgentArchitecture.RSAC],
                protocol.run_seed_bundle(0),
            )
            checkpoint = Path(temporary) / "final.pt"
            components.agent.save(checkpoint)
            episodes, summary = UniversalPricingEvaluator(
                protocol, coordinate
            ).evaluate_checkpoint(
                checkpoint,
                [1001, 1002],
                suite="validation",
            )
            self.assertEqual(summary["episode_count"], 4)
            self.assertIn("mean_bbp_period_fraction", summary)
            self.assertIn("mean_market_share", summary)
            self.assertEqual(
                set(summary["by_opponent_family"]),
                {"uniform", "bbp"},
            )
            for seed_index in (0, 1):
                families = {
                    item["opponent_family"]
                    for item in episodes
                    if item["evaluation_seed_index"] == seed_index
                }
                self.assertEqual(families, {"uniform", "bbp"})

    def test_interrupted_resume_matches_uninterrupted_parameters(self):
        budget = TrainingBudgetConfig(
            environment_steps=200,
            warmup_steps=99,
            updates_per_step=1,
            evaluation_interval_steps=100,
            checkpoint_interval_steps=100,
        )
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            uninterrupted_protocol = small_protocol(
                base / "uninterrupted"
            )
            resumed_protocol = small_protocol(base / "resumed")
            coordinate = next(
                item
                for item in ExperimentMatrix(uninterrupted_protocol).coordinates()
                if item.agent_architecture is AgentArchitecture.SAC
            )
            uninterrupted = UniversalPricingTrainer(
                uninterrupted_protocol,
                coordinate,
                budget=budget,
                run_validation=False,
                verbose=False,
            )
            uninterrupted.train()

            interrupted = UniversalPricingTrainer(
                resumed_protocol,
                coordinate,
                budget=budget,
                run_validation=False,
                verbose=False,
            )
            interrupted.request_stop()
            self.assertEqual(
                interrupted.train().status.value,
                "interrupted",
            )
            resumed = UniversalPricingTrainer(
                resumed_protocol,
                coordinate,
                budget=budget,
                run_validation=False,
                resume=True,
                verbose=False,
            )
            resumed.train()

            first = UniversalPricingAgentFactory.create(
                uninterrupted_protocol.agent_profiles[
                    AgentArchitecture.SAC
                ],
                uninterrupted_protocol.run_seed_bundle(0),
            ).agent
            second = UniversalPricingAgentFactory.create(
                resumed_protocol.agent_profiles[AgentArchitecture.SAC],
                resumed_protocol.run_seed_bundle(0),
            ).agent
            first.load(uninterrupted.final_checkpoint_path)
            second.load(resumed.final_checkpoint_path)
            for name, parameter in first.actor.state_dict().items():
                torch.testing.assert_close(
                    parameter,
                    second.actor.state_dict()[name],
                    rtol=0,
                    atol=0,
                )


if __name__ == "__main__":
    unittest.main()
