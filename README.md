# Rainbow
---
My Deep Q-Learnig Rainbow implementation.
I am implementing it little by little, mainly for learning purposes.

### Credentials
Originally, I tried to implement everything myself, but when I started implementing Prioritized Experience Replay I was too lazy to implement segment trees from scratch :/. Eventually, I took implementations of `segment_tree`, `ReplayBuffer` and `PrioritizedReplayBuffer` from the [Openai baselines GitHub repository](https://github.com/openai/baselines/blob/master/baselines/deepq/).

### Content
Generally speaking every next notebook is built on the top of the previous one. For example notebook `03_DoubleDQL.ipynb` uses both Double DQL and fixed QTargets originally implemented in `02_DQL_with_fixed_QTargets.ipynb`.

- `01_Vanilla_DQL.ipynb` - basic implementation of Deep QLearning. Uses `rainbow/basic_agent.py`
- `02_DQL_with_fixed_QTargets.ipynb` - fixed QTargets added. Uses `rainbow/fixed_qtarget_agent.py`
- `03_DoubleDQL.ipynb` - Double DQL (selects optimal next action using two sets of weights). Uses `rainbow/double_dql_agent.py`
- `04_PER.ipynb` - Prioritized Experience Replay. Uses `PrioritizedReplayBuffer` class and  `rainbow/per_agent.py`.
- `05_Dueling_QLearning.ipynb` - Dueling Q Learning algorithm. Uses `DuelingQNetwork` class and  `rainbow/dueling_agent.py`.
- `06_NStep.ipynb` - N-step bootstrapping. `rainbow/n_step_agent.py`.
- `07_Categorical.ipynb` - C-51 algorithm added (action distributions are estimated by a neural network). Uses `DistributionalQNetwork` class and  `rainbow/categorical_agent.py`.
- `08_Rainbow.ipynb` - Noisy neural networks added. Uses `NoisyQNetwork` class and  `rainbow/rainbow_agent.py`.


### TO DO
Everything works fine with `mse_loss`, but does not want to work with `smooth_l1_loss`. I do not really understand why, I need to dig into it further.