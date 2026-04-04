"""RL Meta-Controller constants and safety bounds."""

# -- Position sizing bounds --
MAX_POSITION_SIZE = 0.10  # Never more than 10% of portfolio per position
MIN_POSITION_SIZE = 0.005  # Below 0.5% is not worth the commission

# -- Kelly divergence cap --
MAX_KELLY_DIVERGENCE = 0.5  # RL size can't differ from Kelly by more than 50%

# -- Feature dimensions --
WEIGHT_CONTEXT_DIM = 11  # Features for weight allocation context
POSITION_OBS_DIM = 25  # Observation space for position sizing
EXIT_OBS_DIM = 20  # Observation space for exit management

# -- Action spaces --
NUM_WEIGHT_CONFIGS = 50  # Number of pre-defined weight configurations
NUM_EXIT_ACTIONS = 5  # HOLD, TIGHTEN_STOP, PARTIAL_25, PARTIAL_50, FULL_EXIT

# -- Promotion criteria --
PROMOTION_MIN_DAYS = 90  # Minimum shadow days before promotion
PROMOTION_MIN_PREDICTIONS = 500  # Minimum predictions before evaluation
PROMOTION_SHARPE_THRESHOLD = 0.1  # RL Sharpe must beat baseline by this margin

# -- Training --
DEFAULT_EXPLORATION_RATE = 0.1  # Epsilon for bandit exploration
DEFAULT_LEARNING_RATE = 0.001

# -- Agent type keys (must match ensemble/weights.py) --
AGENT_TYPES = (
    "technical",
    "statistical",
    "momentum",
    "sentiment",
    "geopolitical",
    "world_leader",
    "crypto",
)
