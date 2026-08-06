import json
import os
from pathlib import Path
import re
from typing import Any, List, Mapping, Sequence

import numpy as np

from train.curriculum import CurriculumConfig
from train.metrics import TrainingMetrics


# ----------------------------------------------------------------------
# Visual primitives
# ----------------------------------------------------------------------

class Color:
    END = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    WHITE = "\033[37m"


# Single rounded box style used everywhere
BOX = {
    "tl": "╭", "tr": "╮", "bl": "╰", "br": "╯",
    "h": "─", "v": "│", "lt": "├", "rt": "┤",
}

ANSI_RE = re.compile(r"\033\[[0-9;]*m")


def visible_len(text: str) -> int:
    """Length of text excluding ANSI color codes."""
    return len(ANSI_RE.sub("", text))


def fmt_num(value, kind: str = "auto") -> str:
    """Compact human-readable number formatting (1.2K, 3.4M, etc.)."""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        if abs(value) >= 1_000_000:
            return f"{value / 1_000_000:.2f}M"
        if abs(value) >= 1_000:
            return f"{value / 1_000:.1f}K"
        return f"{value:,}"
    if isinstance(value, float):
        if abs(value) < 0.001 or abs(value) > 10000:
            return f"{value:.2e}"
        if abs(value) < 1:
            return f"{value:.6f}".rstrip("0").rstrip(".")
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def _recent_mean(values: List[float], window: int) -> float:
    """Return a stable float mean for the most recent metric window."""
    return float(np.mean(values[-window:])) if values else 0.0


def _regime_label(value: float, compact: bool = False) -> str:
    """Convert an averaged binary regime into a human-readable label."""
    if np.isclose(value, 1.0):
        return "BBP"
    if np.isclose(value, 0.0):
        return "UNI" if compact else "Uniform"
    return f"MIX({value:.1f})" if compact else f"Mixed ({value:.2f})"


def _is_converged(value: Any) -> bool:
    """Extract the stability flag from scheduler ``(stable, change)`` values."""
    return bool(value[0]) if isinstance(value, (tuple, list)) else bool(value)


class Box:
    """
    A simple rounded box renderer that tracks visible width correctly,
    even with embedded ANSI color codes.
    """

    def __init__(self, width: int, color: str = Color.CYAN):
        self.width = width
        self.color = color
        self.inner = width - 2

    def _c(self, text: str, color: str) -> str:
        return f"{color}{text}{Color.END}"

    def top(self, title: str | None = None) -> str:
        if title is None:
            line = BOX["h"] * self.inner
        else:
            title_str = f" {title} "
            pad = self.inner - len(title_str)
            left = pad // 2
            right = pad - left
            line = BOX["h"] * left + title_str + BOX["h"] * right
        return self._c(f"{BOX['tl']}{line}{BOX['tr']}", self.color)

    def bottom(self) -> str:
        return self._c(f"{BOX['bl']}{BOX['h'] * self.inner}{BOX['br']}", self.color)

    def divider(self) -> str:
        return self._c(f"{BOX['lt']}{BOX['h'] * self.inner}{BOX['rt']}", self.color)

    def blank(self) -> str:
        return self._c(BOX["v"], self.color) + " " * self.inner + self._c(BOX["v"], self.color)

    def row(self, content: str = "", align: str = "left") -> str:
        """Render a single row with proper padding accounting for ANSI codes."""
        pad = self.inner - visible_len(content) - 2  # 2 for the leading/trailing space
        pad = max(0, pad)
        if align == "left":
            body = f" {content}{' ' * pad} "
        elif align == "center":
            left = pad // 2
            right = pad - left
            body = f" {' ' * left}{content}{' ' * right} "
        else:  # right
            body = f" {' ' * pad}{content} "
        v = self._c(BOX["v"], self.color)
        return f"{v}{body}{v}"

    def row_cols(self, left: str, right: str, col_width: int) -> str:
        """Render a two-column row, each column padded to col_width (visible chars)."""
        lpad = max(0, col_width - visible_len(left))
        body = f" {left}{' ' * lpad}{right}"
        pad = max(0, self.inner - visible_len(body) - 1)
        v = self._c(BOX["v"], self.color)
        return f"{v}{body}{' ' * pad} {v}"


# ----------------------------------------------------------------------
# Logger
# ----------------------------------------------------------------------

class UniversalPricingTrainingLogger:
    """Console and atomic JSONL observability for universal-pricing runs."""

    def __init__(
        self,
        metrics_path: str | Path,
        *,
        verbose: bool = True,
    ) -> None:
        self.metrics_path = Path(metrics_path)
        self.latest_metrics_path = self.metrics_path.with_name(
            "latest_metrics.json"
        )
        self.verbose = verbose

    @staticmethod
    def _validate_finite(value: Any, location: str = "record") -> None:
        if isinstance(value, Mapping):
            for name, item in value.items():
                UniversalPricingTrainingLogger._validate_finite(
                    item, f"{location}.{name}"
                )
        elif isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            for index, item in enumerate(value):
                UniversalPricingTrainingLogger._validate_finite(
                    item, f"{location}[{index}]"
                )
        elif isinstance(value, (float, np.floating)) and not np.isfinite(value):
            raise ValueError(f"Non-finite metric at {location}: {value}")

    @staticmethod
    def _atomic_json(path: Path, value: Any) -> None:
        temporary = path.with_name(f".{path.name}.temporary")
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)

    def write_metric_records(
        self, records: Sequence[Mapping[str, Any]]
    ) -> None:
        """Atomically persist the metric history and latest status record."""

        for index, record in enumerate(records):
            self._validate_finite(record, f"records[{index}]")
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.metrics_path.with_name(
            f".{self.metrics_path.name}.temporary"
        )
        with temporary.open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(
                    json.dumps(record, sort_keys=True, allow_nan=False) + "\n"
                )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, self.metrics_path)
        if records:
            self._atomic_json(self.latest_metrics_path, records[-1])

    def log_run_start(
        self,
        *,
        run_id: str,
        architecture: str,
        distributions: str,
        environment_steps: int,
        warmup_steps: int,
        device: str,
        parameter_count: int,
        run_directory: str | Path,
        resumed: bool,
    ) -> None:
        if not self.verbose:
            return
        box_width = max(
            88,
            visible_len(run_id) + 18,
            visible_len(str(run_directory)) + 18,
        )
        box = Box(
            box_width,
            color=Color.MAGENTA if resumed else Color.CYAN,
        )
        title = "UNIVERSAL PRICING RESUME" if resumed else "UNIVERSAL PRICING"
        print()
        print(box.top(title))
        print(box.row(f"{Color.BOLD}Run{Color.END}           {run_id}"))
        print(
            box.row_cols(
                f"Agent: {Color.GREEN}{architecture}{Color.END}",
                f"Device: {Color.GREEN}{device}{Color.END}",
                42,
            )
        )
        print(box.row(f"Population     {distributions}"))
        print(
            box.row_cols(
                f"Budget: {Color.GREEN}{fmt_num(environment_steps)} steps"
                f"{Color.END}",
                f"Warmup: {Color.YELLOW}{fmt_num(warmup_steps)}{Color.END}",
                42,
            )
        )
        print(
            box.row(
                f"Parameters     {Color.GREEN}{fmt_num(parameter_count)}"
                f"{Color.END}"
            )
        )
        print(box.row(f"Artifacts      {run_directory}"))
        print(box.bottom())
        print()

    def log_episode(
        self,
        record: Mapping[str, Any],
        *,
        budget_steps: int,
    ) -> None:
        if not self.verbose:
            return
        steps = int(record["environment_steps"])
        progress = 100.0 * steps / max(budget_steps, 1)
        critic = record.get("mean_critic_loss")
        actor = record.get("mean_actor_loss")
        critic_gradient = record.get("mean_critic_gradient_norm")
        actor_gradient = record.get("mean_actor_gradient_norm")
        loss_text = (
            f" Lq:{float(critic):.3g} Lπ:{float(actor):.3g}"
            f" gq:{float(critic_gradient):.2g}"
            f" gπ:{float(actor_gradient):.2g}"
            if critic is not None and actor is not None
            and critic_gradient is not None and actor_gradient is not None
            else " warmup"
        )
        policy_text = (
            f" Pbbp:{100 * float(record['mean_bbp_regime_probability']):4.1f}% "
            if "mean_bbp_regime_probability" in record
            else ""
        )
        print(
            f"{Color.CYAN}[train]{Color.END} "
            f"ep:{int(record['episode_index']):04d} "
            f"step:{steps:>7}/{budget_steps:<7} ({progress:5.1f}%) "
            f"opp:{str(record['opponent_family']):<7} "
            f"profit:{float(record['raw_agent_profit_total']):8.1f} "
            f"Δ:{float(record['profit_advantage_total']):+8.1f} "
            f"BBP:{100 * float(record['agent_bbp_period_fraction']):5.1f}% "
            f"{policy_text}"
            f"share:{float(record['mean_market_share']):.3f} "
            f"retain:{float(record['mean_retention_rate']):.3f} "
            f"replay:{int(record['replay_size']):>5}"
            f" {str(record['replay_unit'])[:3]}"
            f"/{100 * float(record['replay_bbp_fraction']):.0f}%BBP"
            f"{loss_text} "
            f"time:{float(record['episode_wall_seconds']):.2f}s",
            flush=True,
        )

    def log_validation(self, record: Mapping[str, Any]) -> None:
        if not self.verbose:
            return
        box = Box(76, color=Color.GREEN)
        print()
        print(
            box.top(
                f"VALIDATION @ {fmt_num(record['environment_steps'])} STEPS"
            )
        )
        print(
            box.row_cols(
                "Mean episode profit",
                f"{float(record['mean_raw_agent_profit_total']):.2f}",
                42,
            )
        )
        print(
            box.row_cols(
                "Mean profit advantage",
                f"{float(record['mean_profit_advantage_total']):+.2f}",
                42,
            )
        )
        print(
            box.row_cols(
                "Mean normalized reward",
                f"{float(record['mean_normalized_reward_total']):.4f}",
                42,
            )
        )
        if "mean_bbp_period_fraction" in record:
            print(
                box.row_cols(
                    "Mean BBP period fraction",
                    f"{100 * float(record['mean_bbp_period_fraction']):.1f}%",
                    42,
                )
            )
        print(
            box.row_cols(
                "Evaluation episodes",
                str(int(record["episode_count"])),
                42,
            )
        )
        print(box.bottom())
        print()

    def log_checkpoint(self, checkpoint: str | Path, steps: int) -> None:
        if self.verbose:
            print(
                f"{Color.BLUE}[checkpoint]{Color.END} "
                f"step:{steps} path:{checkpoint}",
                flush=True,
            )

    def log_terminal(
        self,
        status: str,
        *,
        environment_steps: int,
        message: str | None = None,
    ) -> None:
        if not self.verbose:
            return
        color = {
            "completed": Color.GREEN,
            "interrupted": Color.YELLOW,
            "failed": Color.RED,
        }.get(status, Color.CYAN)
        suffix = f" — {message}" if message else ""
        print(
            f"{color}[{status}]{Color.END} "
            f"environment_steps:{environment_steps}{suffix}",
            flush=True,
        )


class CurriculumTrainingLogger:

    def __init__(self, curriculum_config: CurriculumConfig, verbose: bool = True) -> None:
        self.curriculum_config = curriculum_config
        self.verbose = verbose

    def c(self, color: str, text: str) -> str:
        return f"{color}{text}{Color.END}"

    # ------------------------------------------------------------
    def print_training_header(self) -> None:
        if not self.verbose:
            return

        cfg = self.curriculum_config
        monitored = []
        if cfg.monitor_critic:
            monitored.append("Critic Loss")
        if cfg.monitor_actor:
            monitored.append("Actor Loss")
        if cfg.monitor_alpha:
            monitored.append("Alpha")

        max_stage_width = max(
            (visible_len(f"Stage {i + 1}: {opp.name}") for i, opp in enumerate(cfg.stages)),
            default=30,
        )
        box_width = max(60, max_stage_width + 16)
        box = Box(box_width)

        print()
        print(box.top("CONVERGENCE-BASED CURRICULUM"))
        print(box.blank())

        monitoring_value = " + ".join(monitored) if monitored else "None"
        print(box.row(f"{self.c(Color.BOLD, 'Monitoring')}  {self.c(Color.GREEN, monitoring_value)}"))
        print(box.row(f"{self.c(Color.BOLD, 'Threshold ')}  {self.c(Color.GREEN, f'{cfg.change_threshold * 100:.1f}% change')}"))
        ep_word = "episode" if cfg.window_size == 1 else "episodes"
        print(box.row(f"{self.c(Color.BOLD, 'Window    ')}  {self.c(Color.GREEN, f'{cfg.window_size} {ep_word}')}"))

        print(box.divider())
        print(box.row(self.c(Color.BOLD + Color.BLUE, "Curriculum Stages")))
        print(box.blank())

        for i, opp in enumerate(cfg.stages):
            stage_num = f"Stage {i + 1}"
            if i == 0:
                marker = self.c(Color.YELLOW, "▶")
                stage_line = f"{marker} {self.c(Color.YELLOW, stage_num)}: {self.c(Color.BOLD, opp.name)}  {self.c(Color.RED, '← CURRENT')}"
            else:
                stage_line = f"   {self.c(Color.CYAN, stage_num)}: {self.c(Color.BOLD, opp.name)}"
            print(box.row(stage_line))

            desc = opp.description
            max_desc_width = box.inner - 6
            if desc and len(desc) > max_desc_width:
                desc = desc[:max_desc_width - 3] + "..."
            if desc:
                print(box.row(f"   {self.c(Color.DIM, desc)}"))

            if i < len(cfg.stages) - 1:
                print(box.row(self.c(Color.DIM, "│")))

        print(box.divider())

        total_stages = len(cfg.stages)
        active_monitors = len(monitored)
        summary = f"{total_stages} stages configured  •  {active_monitors} metrics monitored  •  window size {cfg.window_size}"
        print(box.row(self.c(Color.BOLD, summary), align="center"))
        print(box.bottom())
        print()

    # ------------------------------------------------------------
    def log_replay_buffer(self, replay_buffer: Any) -> None:
        if not self.verbose:
            return

        buffer_info = replay_buffer.get_info()

        max_label_width = max((len(name) for name in buffer_info.keys()), default=20)
        box_width = max(55, max_label_width + 30)
        box = Box(box_width)

        print()
        print(box.top("REPLAY BUFFER INITIALIZATION"))
        print(box.blank())
        print(box.row(self.c(Color.BOLD + Color.BLUE, "Buffer Details")))
        print(box.blank())

        for i, (buffer_name, length) in enumerate(buffer_info.items()):
            length_str = fmt_num(length)
            left = f"  {self.c(Color.CYAN, buffer_name)}"
            right = self.c(Color.GREEN, f"Size: {length_str}")
            print(box.row_cols(left, right, max_label_width + 2))
            if i < len(buffer_info) - 1:
                print(box.row(self.c(Color.DIM, "·" * (box.inner - 4))))

        print(box.divider())

        total_buffers = len(buffer_info)
        total_capacity = sum(buffer_info.values())
        summary = f"Total Buffers: {total_buffers}  |  Combined Size: {fmt_num(total_capacity)}"
        print(box.row(self.c(Color.BOLD, summary), align="center"))
        print(box.bottom())
        print()

    # ------------------------------------------------------------
    def log_environment_config(self, env: Any) -> None:
        if not self.verbose:
            return

        env_info  = {
        "num_consumers" : env.num_consumers,
        "episode_length" : env.episode_length,
        "seed_value" : env.seed_value,
        "environment_type" : env.environment_type.value

        }

        max_param_width = max(  
            (len(param.replace("_", " ").title()) for param in env_info.keys()),
            default=20,
        )
        box_width = max(55, max_param_width + 30)
        box = Box(box_width)

        print()
        print(box.top("ENVIRONMENTS CONFIGURATION"))
        print(box.blank())
        print(box.row(self.c(Color.BOLD + Color.BLUE, "Environment Parameters")))
        print(box.blank())

        for i, (param, value) in enumerate(env_info.items()):
            param_display = param.replace("_", " ").title()
            value_str = fmt_num(value)

            left = f"  {self.c(Color.CYAN, param_display)}:"
            right = self.c(Color.GREEN, value_str)
            print(box.row_cols(left, right, max_param_width + 4))

            if i < len(env_info) - 1 and i % 3 == 2:
                print(box.row(self.c(Color.DIM, "·" * (box.inner - 4))))

        print(box.divider())
        summary = f"Configuration Summary: {len(env_info)} parameters initialized"
        print(box.row(self.c(Color.BOLD, summary), align="center"))
        print(box.bottom())
        print()

    
    def log_agent_config(self, agent: Any) -> None:
        if not self.verbose:
            return

        agent_info = agent.get_info()

        max_param_width = max(
            (len(param.replace("_", " ").title()) for param in agent_info.keys()),
            default=20,
        )
        box_width = max(55, max_param_width + 30)
        box = Box(box_width)

        print()
        print(box.top("AGENT CONFIGURATION"))
        print(box.blank())
        print(box.row(self.c(Color.BOLD + Color.BLUE, "Agent Parameters")))
        print(box.blank())

        for i, (param, value) in enumerate(agent_info.items()):
            param_display = param.replace("_", " ").title()
            value_str = fmt_num(value)

            left = f"  {self.c(Color.CYAN, param_display)}:"
            right = self.c(Color.GREEN, value_str)
            print(box.row_cols(left, right, max_param_width + 4))

            if i < len(agent_info) - 1 and i % 3 == 2:
                print(box.row(self.c(Color.DIM, "·" * (box.inner - 4))))

        print(box.divider())
        summary = f"Configuration Summary: {len(agent_info)} parameters initialized"
        print(box.row(self.c(Color.BOLD, summary), align="center"))
        print(box.bottom())
        print()

    # ------------------------------------------------------------
    def log_warmup_start(self, warmup_steps: int) -> None:
        if not self.verbose:
            return

        steps_str = fmt_num(warmup_steps)
        message = f"Warming up with {steps_str} random steps..."
        box_width = max(55, len(message) + 6)
        box = Box(box_width)

        print()
        print(box.top())
        print(box.row(self.c(Color.BOLD + Color.YELLOW, message), align="center"))
        print(box.bottom())
        print()

    def log_start_training(self) -> None:
        if not self.verbose :
            return
        print("\033[32mStarting training...\033[0m\n")
    # ------------------------------------------------------------
    def log_episode_progress(self, episode: int, metrics: TrainingMetrics, agent: Any,
                              eval_reward: float, curriculum: Any, policy_stats : Any, config: Any) -> None:
        if not self.verbose:
            return

        info = curriculum.get_info()
        conv = info["convergence_status"]

        window = config.eval_freq
        avg_reward = _recent_mean(metrics.episode_rewards, window)
        avg_profit = _recent_mean(metrics.episode_profits, window)
        avg_opp_profit = _recent_mean(metrics.episode_opp_profits, window)
        agent_prices = (
            _recent_mean(metrics.episode_uniform_prices, window),
            _recent_mean(metrics.episode_new_prices, window),
            _recent_mean(metrics.episode_old_prices, window),
        )
        opponent_prices = (
            _recent_mean(metrics.episode_opp_uniform_prices, window),
            _recent_mean(metrics.episode_opp_new_prices, window),
            _recent_mean(metrics.episode_opp_old_prices, window),
        )
        agent_regime = _regime_label(_recent_mean(metrics.episode_regimes, window))
        opp_regime = _regime_label(_recent_mean(metrics.episode_opp_regimes, window))
        avg_share = _recent_mean(metrics.episode_market_shares, window)
        lrs = agent.get_current_lrs()

        box_width = max(82, len(f"Episode {episode + 1}/{config.num_episodes}") + 30)
        box = Box(box_width)
        col_width = box.inner // 2 - 2

        print(box.top(f"EPISODE {episode + 1}/{config.num_episodes}"))

        # Performance metrics
        print(box.row(self.c(Color.BOLD + Color.BLUE, "Performance Metrics")))
        print(box.row_cols(
            f"Avg Reward:  {self.c(Color.GREEN, f'{avg_reward:>8.1f}')}",
            f"Eval Reward: {self.c(Color.GREEN, f'{eval_reward:>8.1f}')}",
            col_width,
        ))
        print(box.row_cols(
            f"Agent Profit: {self.c(Color.GREEN, f'{avg_profit:>8.1f}')}",
            f"Opponent Profit: {self.c(Color.GREEN, f'{avg_opp_profit:>8.1f}')}",
            col_width,
        ))
        print(box.row(f"Market Share: {self.c(Color.GREEN, f'{avg_share:.3f}')}"))

        print(box.divider())

        # Agent pricing
        agent_regime_color = Color.GREEN if agent_regime == "BBP" else Color.CYAN
        print(box.row(self.c(Color.BOLD + Color.BLUE, "Agent Pricing")))
        print(box.row(f"Regime: {self.c(agent_regime_color, agent_regime)}"))
        print(box.row_cols(
            f"Uniform Price: {self.c(Color.GREEN, f'{agent_prices[0]:.2f}')}",
            f"BBP New Price: {self.c(Color.GREEN, f'{agent_prices[1]:.2f}')}",
            col_width,
        ))
        print(box.row(f"BBP Old Price: {self.c(Color.GREEN, f'{agent_prices[2]:.2f}')}"))

        print(box.divider())

        # Opponent information
        regime_color = (
            Color.GREEN if opp_regime == "BBP"
            else Color.YELLOW if "Mixed" in opp_regime
            else Color.CYAN
        )
        print(box.row(self.c(Color.BOLD + Color.BLUE, "Opponent Information")))
        print(box.row(f"Regime: {self.c(regime_color, opp_regime)}"))
        print(box.row_cols(
            f"Uniform Price: {self.c(Color.GREEN, f'{opponent_prices[0]:.2f}')}",
            f"BBP New Price: {self.c(Color.GREEN, f'{opponent_prices[1]:.2f}')}",
            col_width,
        ))
        print(box.row(f"BBP Old Price: {self.c(Color.GREEN, f'{opponent_prices[2]:.2f}')}"))

        print(box.divider())

        # Training status
        stage_color = Color.YELLOW if info["stage_name"] == info.get("initial_stage", "") else Color.GREEN
        print(box.row(self.c(Color.BOLD + Color.BLUE, "Training Status")))
        print(box.row_cols(
            f"Alpha:    {self.c(Color.GREEN, f'{agent.alpha:.4f}')}",
            f"Actor LR: {self.c(Color.GREEN, f'{lrs['actor_lr']:.2e}')}",
            col_width,
        ))
        print(box.row_cols(
            f"Critic LR: {self.c(Color.GREEN, f'{lrs['critic_lr']:.2e}')}",
            f"Current Stage: {self.c(stage_color, info['stage_name'])}",
            col_width,
        ))

        print(box.divider())

        # Policy statistics remain normalized to the SAC action space [-1, 1].
        print(box.row(self.c(Color.BOLD + Color.BLUE, "Policy Heads (Normalized)")))
        for head, label in (("uniform", "Uniform"), ("new", "BBP New"), ("old", "BBP Old")):
            stats = policy_stats[head]
            print(box.row(
                f"{label:<8}  "
                f"Action {self.c(Color.GREEN, f'{stats['action']:+.4f}')}  "
                f"Mean {self.c(Color.GREEN, f'{stats['mean']:+.4f}')}  "
                f"Std {self.c(Color.GREEN, f'{stats['std']:.4f}')}  "
                f"LogStd {self.c(Color.GREEN, f'{stats['log_std']:+.4f}')}"
            ))

        print(box.divider())

        # Convergence status
        critic_ok = _is_converged(conv.get("critic", False))
        actor_ok = _is_converged(conv.get("actor", False))
        alpha_ok = _is_converged(conv.get("alpha", False))

        def status(ok: bool) -> str:
            label = "CONVERGED" if ok else "NOT CONV"
            icon = "✓" if ok else "✗"
            color = Color.GREEN if ok else Color.RED
            return self.c(color, f"{label} {icon}")

        print(box.row(self.c(Color.BOLD + Color.BLUE, "Convergence Status")))
        conv_line = (
            f"Critic: {status(critic_ok)}   "
            f"Actor: {status(actor_ok)}   "
            f"Alpha: {status(alpha_ok)}"
        )
        print(box.row(conv_line, align="center"))      


        print(box.bottom())
        print()

    # ------------------------------------------------------------
    def log_episode_progress_compact(self, episode: int, metrics: Any, agent: Any,
                                      eval_reward: float, curriculum: Any, config: Any) -> None:
        if not self.verbose:
            return

        info = curriculum.get_info()
        conv = info["convergence_status"]

        def icon(ok: bool) -> str:
            return self.c(Color.GREEN, "✓") if ok else self.c(Color.RED, "✗")

        window = config.eval_freq
        avg_reward = _recent_mean(metrics.episode_rewards, window)
        avg_profit = _recent_mean(metrics.episode_profits, window)
        avg_opp_profit = _recent_mean(metrics.episode_opp_profits, window)
        avg_uniform = _recent_mean(metrics.episode_uniform_prices, window)
        avg_new = _recent_mean(metrics.episode_new_prices, window)
        avg_old = _recent_mean(metrics.episode_old_prices, window)
        avg_share = _recent_mean(metrics.episode_market_shares, window)
        opp_regime = _regime_label(
            _recent_mean(metrics.episode_opp_regimes, window), compact=True
        )
        lrs = agent.get_current_lrs()

        critic_ok = _is_converged(conv.get("critic", False))
        actor_ok = _is_converged(conv.get("actor", False))
        alpha_ok = _is_converged(conv.get("alpha", False))

        status_line = (
            f"Ep {episode + 1:>4}/{config.num_episodes:<4} │ "
            f"R:{avg_reward:>7.1f} E:{eval_reward:>7.1f} │ "
            f"Π:{avg_profit:>6.1f}/{avg_opp_profit:>6.1f} S:{avg_share:>5.2f} │ "
            f"U:{avg_uniform:>4.2f} N:{avg_new:>4.2f} O:{avg_old:>4.2f} │ "
            f"Opp:{opp_regime:<6} │ "
            f"α:{agent.alpha:.3f} LR:{lrs['actor_lr']:.1e} │ "
            f"Stg:{info['stage_name']:<10} │ "
            f"C:{icon(critic_ok)} A:{icon(actor_ok)} α:{icon(alpha_ok)}"
        )

        box_width = max(visible_len(status_line) + 4, 80)
        box = Box(box_width)

        print(box.top())
        print(box.row(status_line))
        print(box.bottom())

    def log_stage_transition(self, new_opponent) -> None:
        if not self.verbose:
            return

        message = f"Switching to opponent: {new_opponent.opponent_type}"
        box_width = max(60, visible_len(message) + 6)
        box = Box(box_width, color=Color.YELLOW)

        print()
        print(box.top())
        print(box.row(self.c(Color.BOLD + Color.YELLOW, message), align="center"))
        print(box.bottom())

    def log_mixed_stage_entry(self, opponent_types: List[str]) -> None:
        if not self.verbose:
            return

        message = "Entering Mixed Stage"
        box_width = max(60, max(len(t) for t in opponent_types) + 20)
        box = Box(box_width, color=Color.YELLOW)

        print()
        print(box.top())
        print(box.row(self.c(Color.BOLD + Color.YELLOW, message), align="center"))
        print(box.divider())
        print(box.row(self.c(Color.BOLD + Color.BLUE, "Opponent Pool")))
        print(box.blank())

        for opp_type in opponent_types:
            print(box.row(f"  {self.c(Color.CYAN, '•')} {self.c(Color.GREEN, opp_type)}"))

        print(box.bottom())
        print()
    
    def log_replay_buffer_stage_change(self, current_stage: str) -> None:
        if not self.verbose:
            return

        message = f"Replay buffer stage changed → {current_stage}"
        box_width = max(55, visible_len(message) + 6)
        box = Box(box_width, color=Color.YELLOW)

        print()
        print(box.top())
        print(box.row(self.c(Color.BOLD + Color.YELLOW, message), align="center"))
        print(box.bottom())

    def log_warmup_new_opponent(self, opponent_type: str) -> None:
        if not self.verbose:
            return

        message = f"Warming up agent with new opponent: {opponent_type}"
        box_width = max(55, visible_len(message) + 6)
        box = Box(box_width, color=Color.YELLOW)

        print()
        print(box.top())
        print(box.row(self.c(Color.BOLD + Color.YELLOW, message), align="center"))
        print(box.bottom())
