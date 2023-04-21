# Environment
ENVIRONMENT = 'CartPole-v1' # Name of the environment

# Training parameters
MAX_EPISODES = 3000 # Maximum number of episodes
MAX_TIMESTEPS = 1000 # Maximum number of timesteps per episode
EXPECTED_REWARD = 230 # Expected average reward to solve the environment

# Epsilon in Epsilon-Greedy action selection
EPSILON_START = 1.0 # Starting value of epsilon
EPSILON_END = 0.01 # Minimum value of epsilon
EPSILON_DECAY = 0.995 # Decay rate of epsilon

