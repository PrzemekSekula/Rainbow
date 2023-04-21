# Rainbow
---
My Deep Q-Learnig Rainbow implementation.
I am implementing it little by little, mainly for learning purposes.

### Credentials
Originally, I tried to implement everything myself, but when I started implementing Prioritized Experience Replay I was too lazy to implement segment trees from scratch :/. Eventually, I took implementations of `segment_tree`, `ReplayBuffer` and `PrioritizedReplayBuffer` from [Openai baselines GitHub repository](https://github.com/openai/baselines/blob/master/baselines/deepq/).

### Content
Generally speaking every next notebook is built on the top of the previous one. For example notebook `03_DoubleDQL.ipynb` uses both Double DQL and fixed QTargets originally implemented in `02_DQL_with_fixed_QTargets.ipynb`
- `01_Vanilla_DQL.ipynb` - basic implementation of Deep QLearning. Uses `lib/basic_agent.py`
- `02_DQL_with_fixed_QTargets.ipynb` - fixed QTargets added. Uses `lib/fixed_qtarget_agent.py`
- `03_DoubleDQL.ipynb` - Double DQL (selects optimal next action using two sets of weights). Uses `lib/double_dql_agent.py`
- `04_PER.ipynb` - Prioritized Experience Replay. `NOT IMPLEMENTED`
- `05_Dueling_QLearning.ipynb` - Dueling Q Learning algorithm. `NOT IMPLEMENTED`
