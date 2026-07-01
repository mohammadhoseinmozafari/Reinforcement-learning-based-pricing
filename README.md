# Dynamic Pricing BBP

This project explores dynamic pricing in a two-firm Hotelling market with reinforcement learning. It includes a custom market environment, SAC-based training pipelines, evaluation scripts, and a Streamlit dashboard for visualizing training, evaluation, and simulation results.

## What’s Included

- A Hotelling-style duopoly environment with uniform pricing and behavior-based pricing.
- Curriculum-based training pipelines for uniform pricing and BBP opponent setups.
- Evaluation scripts that generate JSON summaries for multiple opponent scenarios.
- A Streamlit dashboard for training, evaluation, and live simulation views.
- Optimization utilities for pricing experiments and grid search.

## Project Structure

- [config/](config/) - global constants and hyperparameters.
- [env/](env/) - market logic, opponent policies, and environment factories.
- [evaluation/](evaluation/) - evaluation config, metrics, and result handling.
- [models/](models/) - SAC agent, replay buffer, and reward normalization.
- [optimization/](optimization/) - pricing optimization and visualization utilities.
- [train/](train/) - curriculum logic, metrics, and training workflows.
- [visualisation/](visualisation/) - Streamlit app and dashboard pages.
- [experiments/](experiments/) - saved runs, checkpoints, metrics, and evaluation outputs.

## Requirements

Use Python 3.10 or newer. The code depends on the core scientific and RL stack used throughout the project, including:

- `numpy`
- `pandas`
- `plotly`
- `streamlit`
- `gymnasium`
- `pettingzoo`
- `stable-baselines3`
Note that StableBaselines is for multi-agent environments, which this project doesn't have a module for training multi agent environments yet, So this project will work without stable baseline for now. If you are running the training or evaluation pipelines, make sure your virtual environment is activated before launching any script.

## Installation

```bash
cd "/home/mamasi/Desktop/article/Dynamic Pricing/bbp"
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt 
```

If you already have a working environment, install only the missing packages.

## Running The Dashboards

Start the Streamlit app from the repository root:

```bash
streamlit run visualisation/app.py
```

The landing page links to:

- Training Dashboard
- Evaluation Dashboard
- Simulation Dashboard

## Running Training

All pricing experiments use the same YAML-driven entrypoint:

```bash
python pricing_train.py --config config/uniform_vs_bbp.yaml
```

Available experiment files cover uniform and BBP agents against uniform, BBP,
or mixed curricula. Useful one-off overrides do not require editing YAML:

```bash
python pricing_train.py --config config/bbp_vs_mixed.yaml --episodes 50 --seed 7 --device cpu
```

Checkpoints and metrics are saved to the experiment's configured `save_dir`.

## Running Evaluation

To regenerate the evaluation results used by the dashboard:

```bash
python uniform_vs_bbp_eval.py
```

This writes JSON summaries into:

```text
experiments/uniform_pricing/bbp_opp/eval/
```

## Pricing Optimization

The optimization entrypoint runs grid searches for multiple opponent scenarios:

```bash
python opt_main.py
```

## Main Outputs

- Training checkpoints are stored under [experiments/](experiments/).
- Final training metrics are written as JSON files next to each run.
- Evaluation results are written as scenario-specific JSON files in the evaluation folder.
- The Streamlit dashboards read these artifacts directly.

## Configuration

Configuration is composed from three YAML layers:

- `config/*_vs_*.yaml` selects the agent strategy, training profile, curriculum, and output path.
- `config/training/default.yaml` contains shared environment, SAC, training, evaluation, and logging settings.
- `config/curricula/*.yaml` contains ordered opponent stages and convergence settings.

Low-level market price bounds and economic constants remain in
`config/constants.py`.

## Notes On The Environment

The `env/` folder is part of the project code, not a virtual environment. It contains the market simulator and environment factory used by the training and evaluation scripts.

## Example Workflows

### Train a policy

1. Activate your environment.
2. Run `python pricing_train.py --config config/uniform_vs_uniform.yaml`.
3. Inspect the saved metrics in [experiments/](experiments/).

### Evaluate a trained policy

1. Make sure the trained model exists in the expected run folder.
2. Run `python uniform_vs_bbp_eval.py`.
3. Open the Streamlit app and view the Evaluation Dashboard.

### Explore the market interactively

1. Start Streamlit with `streamlit run visualisation/app.py`.
2. Open the Simulation Dashboard.
3. Adjust the firm regimes and market parameters from the sidebar.

## Troubleshooting

- If imports such as `env.models` fail in Streamlit, run the app from the repository root.
- If a dashboard looks empty, confirm that the expected JSON or checkpoint files exist under [experiments/](experiments/).
- If you are using a fresh environment, verify that all RL and visualization packages are installed in the active virtual environment.
