import re
from typing import Any, List

import numpy as np

from train.curriculum import CurriculumConfig


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
        print(box.top("SAC AGENT CONFIGURATION"))
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
    def log_episode_progress(self, episode: int, metrics: Any, agent: Any,
                              eval_reward: float, curriculum: Any, config: Any) -> None:
        if not self.verbose:
            return

        info = curriculum.get_info()
        conv = info["convergence_status"]

        avg_reward = np.mean(metrics.episode_rewards[-config.eval_freq:])
        avg_price = np.mean(metrics.episode_prices[-config.eval_freq:])
        avg_opp_price_uniform = np.mean(metrics.episode_opp_uniform_prices[-config.eval_freq:])
        avg_opp_price_new = np.mean(metrics.episode_opp_new_prices[-config.eval_freq:])
        avg_opp_price_old = np.mean(metrics.episode_opp_old_prices[-config.eval_freq:])
        avg_opp_regime = np.mean(metrics.episode_opp_regimes[-config.eval_freq:])
        avg_share = np.mean(metrics.episode_market_shares[-config.eval_freq:])
        lrs = agent.get_current_lrs()

        if avg_opp_regime == 1.0:
            opp_regime = "BBP"
        elif avg_opp_regime == 0.0:
            opp_regime = "Uniform"
        else:
            opp_regime = f"Mixed ({avg_opp_regime:.2f})"

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
            f"Avg Price:   {self.c(Color.GREEN, f'{avg_price:>8.2f}')}",
            f"Market Share:{self.c(Color.GREEN, f'{avg_share:>8.2f}')}",
            col_width,
        ))

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
            f"BBP New Price: {self.c(Color.GREEN, f'{avg_opp_price_new:.2f}')}",
            f"BBP Old Price: {self.c(Color.GREEN, f'{avg_opp_price_old:.2f}')}",
            col_width,
        ))
        print(box.row(f"Uniform Price: {self.c(Color.GREEN, f'{avg_opp_price_uniform:.2f}')}"))

        print(box.divider())

        # Training status
        stage_color = Color.YELLOW if info["stage_name"] == info.get("initial_stage", "") else Color.GREEN
        print(box.row(self.c(Color.BOLD + Color.BLUE, "Training Status")))
        print(box.row_cols(
            f"Alpha:    {self.c(Color.GREEN, f'{agent.alpha:.4f}')}",
            f"Actor LR: {self.c(Color.GREEN, f'{lrs['actor_lr']:.2e}')}",
            col_width,
        ))
        print(box.row(f"Current Stage: {self.c(stage_color, info['stage_name'])}"))

        print(box.divider())

        # Convergence status
        critic_ok, actor_ok, alpha_ok = conv.get("critic", False), conv.get("actor", False), conv.get("alpha", False)

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

        avg_reward = np.mean(metrics.episode_rewards[-config.eval_freq:])
        avg_price = np.mean(metrics.episode_prices[-config.eval_freq:])
        avg_share = np.mean(metrics.episode_market_shares[-config.eval_freq:])
        avg_opp_regime = np.mean(metrics.episode_opp_regimes[-config.eval_freq:])
        lrs = agent.get_current_lrs()

        if avg_opp_regime == 1.0:
            opp_regime = "BBP"
        elif avg_opp_regime == 0.0:
            opp_regime = "UNI"
        else:
            opp_regime = f"MIX({avg_opp_regime:.1f})"

        status_line = (
            f"Ep {episode + 1:>4}/{config.num_episodes:<4} │ "
            f"R:{avg_reward:>7.1f} E:{eval_reward:>7.1f} │ "
            f"P:{avg_price:>5.2f} S:{avg_share:>5.2f} │ "
            f"Opp:{opp_regime:<6} │ "
            f"α:{agent.alpha:.3f} LR:{lrs['actor_lr']:.1e} │ "
            f"Stg:{info['stage_name']:<10} │ "
            f"C:{icon(conv.get('critic', False))} "
            f"A:{icon(conv.get('actor', False))} "
            f"α:{icon(conv.get('alpha', False))}"
        )

        box_width = max(visible_len(status_line) + 4, 80)
        box = Box(box_width)

        print(box.top())
        print(box.row(status_line))
        print(box.bottom())

    def log_policy_stats(self, policy_stats) -> None:
        if not self.verbose:
            return
        mean = policy_stats["mean"]
        raw_log_std = policy_stats["raw_log_std"]
        log_std = policy_stats["log_std"]
        std = policy_stats["std"]
        action = policy_stats["action"]
        status_line = (
            f"Policy Stats │ "
            f"mean:{mean:>7.3f} │ "
            f"raw_log_std:{raw_log_std:>7.3f} │ "
            f"log_std:{log_std:>7.3f} │ "
            f"std:{std:>7.3f} │ "
            f"action:{action:>7.3f}"
        )

        box_width = max(visible_len(status_line) + 4, 80)
        box = Box(box_width, color=Color.MAGENTA)

        print(box.top())
        print(box.row(self.c(Color.GREEN, status_line)))
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