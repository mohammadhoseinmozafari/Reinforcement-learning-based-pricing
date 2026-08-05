# One-Week Research Roadmap

## Objective

Build a reproducible research pipeline that compares three universal pricing
agents:

1. `SACPricingAgent`
2. `RecurrentSACPricingAgent` without an opponent encoder
3. `OpponentEmbeddingRecurrentSACPricingAgent`

Each agent must be able to:

- Choose between uniform pricing and behavior-based pricing (BBP).
- Set the prices required by the selected regime.
- Train against a mixed population of uniform and BBP opponents.
- Generalize across consumer populations generated from uniform, truncated
  normal, and truncated skew-normal distributions.
- Be trained and evaluated with preassigned, reproducible random seeds.

The primary design contains all 27 combinations of distributions for consumer
location, strategicness, and exclusivity preference.

## Definition of Done

By the end of the week, the repository should have:

- One unified environment in which the agent selects its pricing regime.
- A tested hybrid discrete-continuous action contract.
- Working SAC, plain RSAC, and opponent-embedding RSAC implementations.
- Reproducible consumer-distribution generation for all 27 combinations.
- A committed seed manifest with separate training, validation, and test banks.
- A shared YAML-driven experiment runner for all three models.
- Fixed-budget training and deterministic evaluation pipelines.
- Passing unit, integration, environment-contract, and reproducibility tests.
- Successful pilot runs demonstrating that every model can learn.
- The production experiment sweep launched, with completed runs collected as
  compute capacity permits.
- Machine-readable run manifests and analysis-ready result tables.

Completing all final training and evaluation runs within one week depends on
parallel compute throughput. Code correctness, pilot validation, experiment
registration, and reproducibility are mandatory even if part of the production
sweep continues after the deadline.

## Decisions to Freeze on Day 1

The following defaults should be treated as the experimental protocol unless a
decision is changed before implementation begins:

- The agent chooses between uniform and BBP regimes.
- A regime is held for **10 market periods** before another regime decision.
- Opponents keep their regime fixed during an episode.
- The primary training curriculum contains both uniform and BBP opponents.
- Consumer attributes remain fixed within an episode and are regenerated
  between episodes.
- The same episode population and opponent seeds are used across models.
- The primary comparison uses an equal maximum environment-step budget.
- Computational expense is measured separately through wall-clock time,
  environment steps, memory, parameter count, and inference latency.
- There are 10 confirmatory training seeds per primary experimental cell.
- There are 25 fixed validation seeds and 100 locked final-evaluation seeds.
- Final evaluation seeds are never used for training, tuning, checkpoint
  selection, or early stopping.
- Raw economic profit is the primary outcome; shaped training reward is not
  reported as profit.

## Target Architecture

### Unified hybrid action

The policy action contains:

```text
regime:        {uniform, BBP}
uniform_price: continuous
new_price:     continuous
old_price:     continuous
```

When the regime is uniform, only `uniform_price` affects the market. When the
regime is BBP, only `new_price` and `old_price` affect the market. Inactive
price heads must not influence market transitions and should be masked where
necessary during optimization.

The regime must be represented by a categorical policy, not by thresholding an
ordinary continuous action. Because there are only two regimes, the SAC target
and policy objectives should evaluate or marginalize the two choices directly.

### Shared observation contract

All three models receive the same observable state:

- Current agent regime
- Remaining regime-commitment time
- Opponent regime, when economically observable
- Agent and opponent prices
- Market shares and demand
- New-versus-established customer composition
- Retention and switching information
- Profit and price history summaries
- Episode progress

SAC uses the current observation. Plain RSAC adds recurrent state without a
dedicated opponent encoder. Embedding RSAC adds its opponent-history encoder
and auxiliary opponent-prediction objective.

## Consumer Distribution Protocol

Each of the three consumer attributes independently selects one family:

- Consumer location
- Strategicness
- Exclusivity preference

The primary families are:

| Family | Initial specification |
|---|---|
| Uniform | `Uniform(0, 1)` |
| Normal | `Normal(0.5, 0.2)`, truncated to `[0, 1]` |
| Skew-normal | Positive shape parameter, truncated to `[0, 1]`, calibrated to match the truncated normal's realized mean and variance |

This produces:

```text
3 location families
× 3 strategicness families
× 3 exclusivity families
= 27 consumer configurations
```

The distribution factory must use an explicit RNG supplied by the experiment
seed manager. It must never read global NumPy randomness implicitly.

For consumer location, a mirrored negative-skew sensitivity experiment should
be registered separately because a single skew direction may favor one side of
the Hotelling market. This sensitivity run is not part of the primary 27-cell
count.

## Reproducibility Protocol

Create and commit a seed manifest before production training. Each training
replicate should have named, isolated streams:

```text
run_seed
network_initialization_seed
consumer_population_seed_root
opponent_seed_root
exploration_seed
replay_sampling_seed
torch_cpu_seed
torch_cuda_seed
```

The environment and opponent seed schedules must be shared across model types.
Model-internal randomness must be isolated so a different number of random
draws in one architecture cannot change later consumer populations.

Every run directory must contain:

- Fully resolved configuration
- Seed manifest
- Model type and parameter count
- Git commit and protocol version
- Hardware and software information
- Checkpoints
- Training curves
- Raw evaluation episodes
- Aggregated metrics
- Completion or failure status

## One-Week Schedule

### Day 1 — Freeze contracts and repair the experiment foundation

Tasks:

- Freeze the action, observation, regime-commitment, seed, and artifact
  contracts.
- Define the exact plain-RSAC architecture: recurrent state with no opponent
  encoder or opponent auxiliary loss.
- Create configuration schemas for model type, distribution triplet, opponent
  curriculum, and seed-bank selection.
- Create the versioned seed manifest.
- Resolve existing configuration blockers, including invalid mixed-curriculum
  opponent names.
- Establish one canonical training entrypoint and one canonical evaluation
  entrypoint.
- Write tests before implementing the new environment behavior.

Exit gate:

- Every planned experimental cell can be enumerated without starting training.
- A run ID maps uniquely to model, distributions, curriculum, and seed.
- Invalid configurations fail before creating an environment or output files.

#### Day 1 classes and responsibilities

The Day 1 implementation uses explicit domain names and reserves the concrete
model names without adding empty model stubs:

| Class | Responsibility |
|---|---|
| `PricingRegime` | Defines the uniform and BBP action regimes. |
| `PricingAction` | Stores one immutable regime decision and three normalized price controls. |
| `PricingActionCodec` | Converts structured actions to Gym dictionaries and fixed five-element replay vectors. |
| `PricingObservationFeature` | Defines the serialized names and stable order of all 18 observation features. |
| `PricingObservationCodec` | Encodes and validates finite, normalized observations. |
| `PricingAgent` | Defines the runtime interface all three future agents implement. |
| `AgentArchitecture` | Defines the stable identifiers `sac`, `rsac`, and `oe_rsac`. |
| `AgentArchitectureSpec` | Reserves implementation class names and declares recurrent, replay, and encoder capabilities. |
| `ConsumerDistributionFamily` | Defines uniform, truncated-normal, and truncated-skew-normal families. |
| `ConsumerDistributionSpec` | Validates one distribution family and its parameters. |
| `ConsumerPopulationSpec` | Groups location, strategicness, and exclusivity specifications. |
| `DistributionCombination` | Identifies one immutable coordinate in the 27-cell distribution grid. |
| `TrainingBudgetConfig` | Defines the fixed step, warmup, update, evaluation, and checkpoint budget. |
| `AgentProfileConfig` | Rejects architecture-incompatible recurrent or encoder settings. |
| `UniversalPricingProtocolConfig` | Holds the fully resolved and validated versioned research protocol. |
| `ExperimentCoordinate` | Identifies one architecture, distribution cell, curriculum, and training seed. |
| `ExperimentMatrix` | Deterministically enumerates the 810 primary coordinates. |
| `OpponentFamily` | Defines uniform and BBP opponent families. |
| `OpponentPoolConfig` | Validates the nine registered policies and equal family weights. |
| `OpponentEpisodeAssignment` | Records the family, policy, and seed assigned to one episode. |
| `BalancedOpponentSchedule` | Produces reproducible two-episode balanced blocks and shuffled round-robin policy coverage. |
| `SeedPurpose` | Defines stable integer identifiers for seven independent random streams. |
| `SeedBankManifest` | Stores the committed 10 training, 25 validation, and 100 final-evaluation seeds. |
| `RunSeedBundle` | Holds all named streams derived for one training replicate. |
| `EpisodeSeedBundle` | Holds consumer and opponent seeds for one episode. |
| `SeedDeriver` | Derives streams without global RNG state or call-order dependence. |
| `RunStatus` | Defines the run lifecycle states. |
| `ExperimentRunId` | Builds and parses deterministic run identifiers. |
| `ArtifactLayout` | Computes output paths without creating them. |
| `ExperimentRunManifest` | Records immutable resolved protocol, seed, source, hardware, status, and artifact facts. |
| `ManifestRepository` | Reads and atomically replaces identity-compatible manifests. |

`SACPricingAgent` is implemented on Day 3.
`RecurrentSACPricingAgent` and
`OpponentEmbeddingRecurrentSACPricingAgent` are implemented on Day 4.

### Day 2 — Unified environment and consumer distributions

Tasks:

- Replace the fixed agent regime with agent-controlled regime selection.
- Implement the 10-period regime commitment.
- Ensure uniform and BBP price heads activate correctly.
- Implement uniform, truncated-normal, and truncated-skew-normal sampling.
- Regenerate consumers between episodes using explicit episode seeds.
- Correct observation bounds and validate initial observations.
- Add reproducibility, truncation, empirical-moment, symmetry, and environment
  contract tests.
- Add an equal-price scenario where uniform and BBP are economically
  equivalent and an established-customer scenario where BBP is preferable.

Exit gate:

- Repeated reset with the same seed produces identical consumers.
- Different episode seeds produce different populations.
- All observations and actions satisfy their declared spaces.
- Uniform and BBP market paths can both run for a complete episode.
- Equal effective prices produce equal no-history outcomes.
- The established-customer scenario produces the expected BBP advantage.

#### Day 2 classes and responsibilities

| Class | Responsibility |
|---|---|
| `ConsumerAttribute` | Defines the three independently seeded consumer attributes. |
| `ConsumerAttributeSampler` | Defines the bounded explicit-generator sampling interface. |
| `UniformConsumerAttributeSampler` | Samples uniform attributes. |
| `TruncatedNormalConsumerAttributeSampler` | Samples true truncated-normal attributes through inverse CDFs. |
| `TruncatedSkewNormalConsumerAttributeSampler` | Samples moment-matched truncated skew-normal attributes. |
| `TruncatedSkewNormalMomentCalibrator` | Reconstructs and validates the frozen skew parameters. |
| `ConsumerAttributeSamplerRegistry` | Validates the family-to-sampler mapping. |
| `ConsumerPopulationSnapshot` | Stores one immutable sampled population. |
| `ConsumerPopulationGenerator` | Derives independent attribute streams and generates a population. |
| `UniversalPricingEpisodeContext` | Records one episode's seeds and opponent assignment. |
| `UniversalPricingEpisodeContextFactory` | Resolves deterministic episode contexts. |
| `PricingPriceTransform` | Converts normalized controls and actual prices. |
| `RegimeDecisionResult` | Records the proposed and effective regime decision. |
| `RegimeCommitmentController` | Enforces immediate selection and ten-period commitments. |
| `UniversalPricingObservationBuilder` | Builds and validates the frozen 18-feature observation. |
| `ProfitRewardNormalizer` | Normalizes raw own profit against the theoretical bound. |
| `UniversalPricingEnv` | Coordinates the population, opponent, market, action, observation, and reward contracts. |
| `UniversalPricingEnvironmentFactory` | Builds an unwrapped universal environment from a protocol coordinate. |

### Day 3 — Hybrid SAC

Status: implemented.

Tasks:

- Implement the categorical regime head and conditional continuous price heads.
- Implement critic inputs and target calculations for hybrid actions.
- Mask inactive price components correctly.
- Use separate entropy treatment for regime and price exploration.
- Update transition replay to store the full hybrid action.
- Add checkpoint save/load support for the new architecture.
- Add numerical tests for action bounds, loss shapes, gradient flow, and
  deterministic inference.

#### Day 3 classes and responsibilities

| Class | Responsibility |
|---|---|
| `SACPricingAgentConfig` | Validates the feed-forward architecture, optimizer, entropy, replay, and numerical hyperparameters. |
| `HybridPricingActionTensorCodec` | Produces canonical critic actions and removes inactive price controls. |
| `HybridPricingPolicyOutput` | Gives stable names to categorical, uniform-price, and BBP-price actor outputs. |
| `SACPricingActor` | Implements the categorical regime head and conditional tanh-Gaussian uniform and BBP price heads. |
| `SACPricingCritic` | Estimates one scalar Q-value from an observation and canonical hybrid action. |
| `UniversalPricingTransition` | Validates and owns one complete universal-environment replay transition. |
| `UniversalPricingReplayBatch` | Gives sampled replay arrays stable names and shapes. |
| `UniversalPricingReplayBuffer` | Stores transitions and samples them through a private committed NumPy RNG. |
| `SACPricingUpdateMetrics` | Records the critic, actor, entropy, Q-value, and decision-mask diagnostics from an update. |
| `SACPricingAgent` | Selects masked hybrid actions, performs exact two-regime SAC updates, manages three entropy temperatures, and saves/restores complete checkpoints. |
| `SACPricingAgentFactory` | Constructs the SAC agent and replay buffer from the validated profile and run seed bundle. |

Exit gate:

- Hybrid SAC overfits a small deterministic environment.
- It learns an economically competitive policy in the equal-price sanity
  scenario; learned regime usage is reported as a diagnostic.
- It learns BBP pricing and a meaningful price spread in the BBP-preferred
  sanity scenario.
- Saving and restoring a checkpoint preserves deterministic actions.

### Day 4 — Plain RSAC and embedding RSAC

Status: implemented.

Tasks:

- Implement plain RSAC by using recurrent market history without an opponent
  encoder.
- Adapt the existing embedding RSAC to the shared hybrid action contract.
- Keep replay sequence construction identical between recurrent models.
- Verify recurrent-state reset at episode and evaluation boundaries.
- Ensure padded sequence steps do not affect any loss.
- Record parameter counts and model-specific compute metrics.

#### Day 4 classes and responsibilities

| Class | Responsibility |
|---|---|
| `RecurrentSACPricingAgentConfig` | Validates shared recurrent and OE-only architecture settings. |
| `RecurrentPricingActor` | Produces hybrid regime and conditional-price policies from recurrent interaction history. |
| `RecurrentPricingCritic` | Encodes action-independent history and evaluates canonical hybrid actions. |
| `OpponentHistoryEncoder` | Learns a pre-decision opponent embedding and predicts opponent controls. |
| `RecurrentSACPricingAgent` | Implements plain recurrent hybrid SAC without an opponent encoder. |
| `OpponentEmbeddingRecurrentSACPricingAgent` | Adds the dedicated opponent encoder, auxiliary loss, and target encoder. |
| `UniversalPricingEpisode` | Validates one complete contiguous universal episode. |
| `UniversalPricingEpisodeBuilder` | Accumulates environment transitions into a complete episode. |
| `UniversalPricingSequenceBatch` | Carries burn-in, learning, padding, and previous-transition context. |
| `UniversalPricingSequenceReplayBuffer` | Samples reproducible 16-step burn-in and learning windows. |
| `UniversalPricingAgentFactory` | Constructs all three agents and their compatible replay buffers. |
| `UniversalPricingTrainer` | Runs fixed-step training, logging, checkpointing, interruption, and exact resume. |
| `UniversalPricingTrainingSnapshot` | Stores the complete episode-boundary continuation state. |
| `UniversalPricingEvaluator` | Runs balanced deterministic validation and final evaluation from the final checkpoint. |

Exit gate:

- Both recurrent models pass sequence, masking, checkpoint, and hidden-state
  isolation tests.
- All three models can train through the same experiment entrypoint.
- No model requires a private environment or incompatible result format.

### Day 5 — End-to-end pilots and learning validation

Run predeclared pilots using three anchor consumer configurations:

1. Uniform / uniform / uniform
2. Normal / normal / normal
3. Skew-normal / skew-normal / skew-normal

Tasks:

- Run every model on every anchor with one pilot seed.
- Evaluate against representative uniform and BBP opponents.
- Inspect regime-use frequency, price heads, raw profit, market share, and
  retention.
- Check for regime collapse, NaNs, invalid actions, replay imbalance, and
  nondeterministic evaluation.
- Tune only using pilot and validation seeds.
- Freeze hyperparameters after the pilot gate.

Exit gate:

- Nine pilot runs complete without numerical or contract failures.
- Every model can execute both regimes; learned mixed-regime usage is reported
  as a diagnostic rather than a pass/fail condition.
- Every model beats random-price and random-regime baselines.
- At least one sanity scenario causes a clear, economically sensible change in
  regime usage.
- Repeating a run from its manifest reproduces its consumer and opponent
  trajectories.

If this gate fails, production training must not start. Day 6 should be used to
repair learning behavior instead.

### Day 6 — Production sweep and evaluation pipeline

Primary training target:

```text
3 models × 27 distribution configurations × 10 seeds = 810 runs
```

Launch in two predeclared waves:

1. First five seeds for all 27 cells and all models: 405 runs.
2. Remaining five seeds for all 27 cells and all models: 405 runs.

Do not expand or drop cells based on early performance. If the deadline prevents
the second wave from finishing, label the five-seed results exploratory rather
than presenting them as the final confirmatory study.

Tasks:

- Launch runs through a resumable job manifest.
- Capture failures without stopping unrelated runs.
- Evaluate completed checkpoints against all uniform and BBP opponent panels.
- Evaluate in-distribution and out-of-distribution consumer configurations.
- Build the 27-by-27 train/evaluation transfer matrix.
- Aggregate distribution shifts by zero, one, two, or three changed consumer
  attributes.

Exit gate:

- All intended jobs are registered exactly once.
- Failed and incomplete jobs are distinguishable from valid low-performing
  runs.
- Evaluation never modifies training replay or model state.
- Raw and aggregated results are traceable to a checkpoint and manifest.

### Day 7 — Statistical analysis and reproducibility freeze

Tasks:

- Produce model-comparison tables and learning curves.
- Report raw profit, profit advantage, generalization gap, and efficiency.
- Generate paired 95% bootstrap confidence intervals.
- Report mean, standard deviation, median, interquartile range, interquartile
  mean, and probability of improvement.
- Apply Holm correction to the three pairwise model comparisons for each
  primary outcome.
- Report economic secondary metrics separately from primary hypotheses.
- Validate a clean reproduction from manifest to result.
- Tag the protocol and archive resolved configurations, seeds, checkpoints,
  logs, and raw evaluation data.
- Document incomplete runs and limitations honestly.

Exit gate:

- Every figure and table can be regenerated from committed analysis code and
  run manifests.
- Training seeds are treated as the units of replication; evaluation episodes
  are not incorrectly counted as independently trained agents.
- No final-test seed influenced model selection or hyperparameter tuning.

## Primary Outcomes

Keep the confirmatory outcome family small:

1. Raw episodic agent profit
2. Agent profit minus opponent profit
3. In-distribution performance
4. Out-of-distribution performance
5. Generalization gap

Secondary economic outcomes:

- Opponent and total industry profit
- Consumer surplus and total welfare
- Average transaction price
- New-customer and established-customer prices
- BBP price spread
- Market share and concentration
- Retention and switching
- Price and profit volatility

Efficiency outcomes:

- Performance versus environment steps
- Area under the learning curve
- Steps and wall-clock time to a preassigned threshold
- Total training time
- Parameter count
- Peak memory
- Inference latency

## Baselines

Every evaluation panel should include:

- Random price and random regime
- Fixed uniform pricing
- Fixed BBP pricing
- Rule-based regime selection
- Best specialized uniform-only policy, when available
- Best specialized BBP-only policy, when available

The specialized policies are evaluation references. They do not need the full
27-cell training sweep during this one-week implementation deadline.

## Scope Control

Must complete this week:

- Universal hybrid environment
- Three functioning model families
- Distribution and seed infrastructure
- Tests and sanity benchmarks
- Pilot runs
- Resumable production sweep
- Reproducible evaluation and result manifests

Defer unless required by the research pipeline:

- Streamlit dashboard redesign
- Legacy optimization scripts
- Cosmetic refactoring
- Additional distribution families
- Correlated consumer attributes
- Dynamic opponent regime switching
- Exhaustive hyperparameter searches
- Full retraining of specialized uniform-only and BBP-only agents

## Main Risks and Responses

| Risk | Response |
|---|---|
| Agent collapses to one regime | Separate regime entropy, balanced warmup, stratified replay diagnostics, and known-regime sanity tasks |
| BBP receives weak delayed credit | Keep sufficiently long episodes, expose retention state, and validate the regime commitment horizon |
| Hybrid SAC becomes numerically unstable | Evaluate the two categorical choices directly and test each loss component independently |
| Architectures receive different market randomness | Use named RNG streams and preassigned episode-level environment seeds |
| Mixed curriculum favors one opponent family | Balance opponent sampling and report per-family results |
| Skewed locations favor one firm | Add the predeclared mirrored-location sensitivity analysis |
| Production sweep exceeds the deadline | Run complete five-seed waves across all cells; never selectively finish favorable cells |
| Evaluation creates pseudoreplication | Aggregate evaluation episodes within each training seed before model-level inference |
| Performance and compute become confounded | Report both equal-step performance and time/steps-to-threshold |

## Recommended Order of Work

Do not start by launching the existing training scripts. The correct sequence is:

```text
Freeze protocol
→ establish seeds and configuration identities
→ implement and test the universal environment
→ implement Hybrid SAC
→ implement both recurrent variants
→ pass known-regime sanity tests
→ run anchor pilots
→ freeze hyperparameters
→ launch complete seed waves
→ run locked evaluation
→ produce paired statistical analysis
```

This order minimizes the risk of spending most of the one-week deadline
producing checkpoints that cannot be compared or reproduced.
