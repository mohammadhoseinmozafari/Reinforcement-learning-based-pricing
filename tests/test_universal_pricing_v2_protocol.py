"""Protocol, seed, pilot, sweep, and research-statistics contracts for v2."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from env.pricing_contracts import AgentArchitecture
from universal_pricing_v2.analysis import (
    V2PairedStatisticalAnalyzer,
    holm_adjust,
    interquartile_mean,
    paired_bootstrap_interval,
)
from universal_pricing_v2.pilot import load_v2_pilot_config
from universal_pricing_v2.protocol import (
    EXPECTED_OPPONENT_ORDER,
    HierarchicalSeedDeriver,
    HierarchicalTrainingPhase,
    V2ArtifactLayout,
    V2ExperimentMatrix,
    V2ExperimentRunId,
    load_universal_pricing_v2_protocol,
)
from universal_pricing_v2.sweep import (
    ProductionSeedWave,
    V2ProductionSweep,
    V2SweepRegistry,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    REPOSITORY_ROOT / "config/protocols/universal_pricing_v2.yaml"
)
PILOT_PATH = (
    REPOSITORY_ROOT / "config/universal_pricing_v2/pilot.yaml"
)


class UniversalPricingV2ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = load_universal_pricing_v2_protocol(PROTOCOL_PATH)

    def test_primary_matrix_has_810_stable_unique_coordinates(self) -> None:
        first = V2ExperimentMatrix(self.protocol).coordinates()
        second = V2ExperimentMatrix(self.protocol).coordinates()
        self.assertEqual(len(first), 810)
        self.assertEqual(first, second)
        run_ids = [
            str(V2ExperimentRunId.from_coordinate(item)) for item in first
        ]
        paths = [
            str(V2ArtifactLayout(self.protocol.artifact_root).run_directory(item))
            for item in first
        ]
        self.assertEqual(len(set(run_ids)), 810)
        self.assertEqual(len(set(paths)), 810)
        self.assertEqual(
            json.dumps([item.to_dict() for item in first], sort_keys=True),
            json.dumps([item.to_dict() for item in second], sort_keys=True),
        )
        for coordinate, run_id in zip(first, run_ids):
            self.assertEqual(V2ExperimentRunId.parse(run_id), coordinate)

    def test_protocol_freezes_budgets_order_and_architectures(self) -> None:
        self.assertEqual(self.protocol.market_timing, "simultaneous")
        self.assertEqual(
            set(self.protocol.agent_profiles), set(AgentArchitecture)
        )
        self.assertEqual(
            self.protocol.training_budget.environment_steps, 600_000
        )
        self.assertEqual(
            tuple(
                stage.opponent_policy_name
                for stage in self.protocol.uniform_curriculum.stages
            ),
            EXPECTED_OPPONENT_ORDER,
        )
        self.assertEqual(
            tuple(
                stage.opponent_policy_name
                for stage in self.protocol.bbp_curriculum.stages
            ),
            EXPECTED_OPPONENT_ORDER,
        )
        self.assertEqual(
            sum(
                stage.maximum_steps
                for stage in self.protocol.uniform_curriculum.stages
            ),
            180_000,
        )
        self.assertEqual(
            sum(
                stage.maximum_steps
                for stage in self.protocol.bbp_curriculum.stages
            ),
            220_000,
        )

    def test_stage_seeds_share_common_prefix_across_architectures(self) -> None:
        coordinates = V2ExperimentMatrix(self.protocol).coordinates()
        selected = [
            item
            for item in coordinates
            if item.training_seed_index == 0
            and item.distribution_combination
            == coordinates[0].distribution_combination
        ]
        bundles = [
            HierarchicalSeedDeriver.episode_bundle(
                self.protocol.run_seed_bundle(
                    item.training_seed_index
                ).run_seed,
                HierarchicalTrainingPhase.BBP_PRICING,
                4,
                17,
            )
            for item in selected
        ]
        self.assertEqual(len(selected), 3)
        self.assertEqual(len(set(bundles)), 1)
        changed = HierarchicalSeedDeriver.episode_bundle(
            self.protocol.run_seed_bundle(0).run_seed,
            HierarchicalTrainingPhase.BBP_PRICING,
            4,
            18,
        )
        self.assertNotEqual(changed, bundles[0])

    def test_short_artifact_layout_is_windows_safe(self) -> None:
        coordinate = V2ExperimentMatrix(self.protocol).coordinates()[-1]
        path = V2ArtifactLayout(Path("experiments/upv2")).run_directory(
            coordinate
        )
        self.assertLess(len(str(path)), 100)
        self.assertRegex(str(path), r"oe_rsac/l-tsn__s-tsn__e-tsn/s09$")

    def test_pilot_declares_nine_300k_runs(self) -> None:
        pilot = load_v2_pilot_config(PILOT_PATH)
        self.assertEqual(len(pilot.coordinates()), 9)
        self.assertEqual(
            pilot.resolved_protocol().training_budget.environment_steps,
            300_000,
        )
        self.assertEqual(
            len(
                {
                    str(V2ExperimentRunId.from_coordinate(item))
                    for item in pilot.coordinates()
                }
            ),
            9,
        )

    def test_sweep_is_split_into_two_complete_five_seed_waves(self) -> None:
        sweep = V2ProductionSweep(
            self.protocol,
            protocol_path=PROTOCOL_PATH,
            python_executable="python",
            device="cuda",
        )
        first = sweep.jobs(ProductionSeedWave.FIRST)
        second = sweep.jobs(ProductionSeedWave.SECOND)
        self.assertEqual(len(first), 405)
        self.assertEqual(len(second), 405)
        self.assertEqual(
            {item.coordinate["training_seed_index"] for item in first},
            set(range(5)),
        )
        self.assertEqual(
            {item.coordinate["training_seed_index"] for item in second},
            set(range(5, 10)),
        )
        self.assertFalse(
            set(item.run_id for item in first)
            & set(item.run_id for item in second)
        )
        self.assertTrue(all(item.launchable for item in first))
        self.assertTrue(
            all("--resume" not in item.command for item in first)
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "wave.json"
            V2SweepRegistry.write(output, first)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(len(payload["jobs"]), 405)

    def test_research_statistics_are_deterministic(self) -> None:
        values = [1.0, 2.0, 3.0, 100.0]
        self.assertEqual(interquartile_mean(values), 2.5)
        first = paired_bootstrap_interval(
            [1.0, 2.0, 3.0], repetitions=500
        )
        second = paired_bootstrap_interval(
            [1.0, 2.0, 3.0], repetitions=500
        )
        self.assertEqual(first, second)
        adjusted = holm_adjust([0.01, 0.03, 0.2])
        self.assertEqual(adjusted, [0.03, 0.06, 0.2])
        economic = V2PairedStatisticalAnalyzer.economic_summaries(
            [
                {
                    "architecture": "sac",
                    "regime_mode": "learned",
                    "gross_agent_profit_total": 10.0,
                    "agent_bbp_operating_cost_total": 1.0,
                    "net_agent_profit_total": 9.0,
                    "net_profit_advantage_total": 2.0,
                    "bbp_period_fraction": 0.5,
                }
            ]
        )
        self.assertEqual(economic[0]["mean_net_agent_profit_total"], 9.0)


if __name__ == "__main__":
    unittest.main()
