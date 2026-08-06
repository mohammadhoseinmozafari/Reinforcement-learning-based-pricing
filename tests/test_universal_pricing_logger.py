"""Observability tests for universal-pricing metrics and logger output."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from train.logger import UniversalPricingTrainingLogger
from train.metrics import UniversalPricingEpisodeMetrics


class UniversalPricingEpisodeMetricsTests(unittest.TestCase):
    def test_summary_contains_economic_regime_and_policy_signals(self) -> None:
        metrics = UniversalPricingEpisodeMetrics()
        metrics.record_step(
            reward=0.4,
            info={
                "raw_agent_profit": 100.0,
                "raw_opponent_profit": 80.0,
                "agent_regime": 1,
                "opponent_regime": 0,
                "regime_changed": True,
                "regime_decision_mask": 1.0,
            },
            agent_firm=SimpleNamespace(
                uniform_price=2.0,
                price_new=3.0,
                price_old=4.0,
                market_share=0.6,
                retention_rate=0.7,
            ),
            opponent_firm=SimpleNamespace(
                uniform_price=4.0,
                price_new=2.0,
                price_old=3.0,
            ),
            policy_diagnostics={
                "uniform_regime_probability": 0.25,
                "bbp_regime_probability": 0.75,
            },
        )
        summary = metrics.summary()
        self.assertEqual(summary["profit_advantage_total"], 20.0)
        self.assertEqual(summary["agent_bbp_period_fraction"], 1.0)
        self.assertEqual(summary["regime_change_count"], 1.0)
        self.assertEqual(summary["mean_agent_bbp_price_spread"], 1.0)
        self.assertEqual(summary["mean_market_share"], 0.6)
        self.assertEqual(summary["mean_bbp_regime_probability"], 0.75)


class UniversalPricingTrainingLoggerTests(unittest.TestCase):
    def test_writes_history_and_latest_record_atomically(self) -> None:
        records = [
            {"phase": "training", "environment_steps": 100, "value": 1.5},
            {"phase": "validation", "environment_steps": 100, "value": 2.5},
        ]
        with tempfile.TemporaryDirectory() as temporary:
            metrics_path = Path(temporary) / "metrics.jsonl"
            logger = UniversalPricingTrainingLogger(
                metrics_path, verbose=False
            )
            logger.write_metric_records(records)
            actual = [
                json.loads(line)
                for line in metrics_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(actual, records)
            self.assertEqual(
                json.loads(
                    logger.latest_metrics_path.read_text(encoding="utf-8")
                ),
                records[-1],
            )

    def test_nonfinite_metrics_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            logger = UniversalPricingTrainingLogger(
                Path(temporary) / "metrics.jsonl",
                verbose=False,
            )
            with self.assertRaises(ValueError):
                logger.write_metric_records(
                    [{"phase": "training", "loss": float("nan")}]
                )

    def test_compact_episode_output_exposes_health_metrics(self) -> None:
        record = {
            "episode_index": 4,
            "environment_steps": 500,
            "opponent_family": "bbp",
            "raw_agent_profit_total": 100.0,
            "profit_advantage_total": 20.0,
            "agent_bbp_period_fraction": 0.6,
            "mean_market_share": 0.55,
            "mean_retention_rate": 0.7,
            "replay_size": 5,
            "replay_unit": "episodes",
            "replay_bbp_fraction": 0.4,
            "episode_wall_seconds": 1.2,
            "mean_critic_loss": 0.4,
            "mean_actor_loss": -0.2,
            "mean_critic_gradient_norm": 0.8,
            "mean_actor_gradient_norm": 0.3,
        }
        output = io.StringIO()
        logger = UniversalPricingTrainingLogger(
            Path("/tmp/unused-universal-metrics.jsonl")
        )
        with redirect_stdout(output):
            logger.log_episode(record, budget_steps=50_000)
        rendered = output.getvalue()
        self.assertIn("profit:", rendered)
        self.assertIn("BBP:", rendered)
        self.assertIn("share:", rendered)
        self.assertIn("replay:", rendered)
        self.assertIn("Lq:", rendered)


if __name__ == "__main__":
    unittest.main()
