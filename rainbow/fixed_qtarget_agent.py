import numpy as np
import random
from collections import namedtuple, deque

from rainbow.replay_buffer import ReplayBuffer

from rainbow.dense_network import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size,
                 buffer_size = int(1e5),
                 batch_size = 64,
                 gamma = 0.99,
                 lr = 5e-4,
                 update_every = 4,
                 device = None):
        """Initialize an Agent object.
        Args:
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            buffer_size (int): replay buffer size. Default: int(1e5)
            batch_size (int): minibatch size. Default: 64
            gamma (float): discount factor. Default: 0.99
            lr (float): learning rate. Default: 5e-4
            update_every (int): how often to update the network. Default: 1        
            device (torch.device): device to use. Default: None    
        """
        self.state_size = state_size
        self.action_size = action_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = lr
        self.update_every = update_every
        self.device = device

        # Q-Network
        fc1, fc2 = 64, 16 # Size of the layers
        # TODO: Create a network with 2 hidden layers (fc1 and fc2 nodes)
        self.local_network = QNetwork(state_size, action_size, fc1, fc2).to(device)
        self.target_network = QNetwork(state_size, action_size, fc1, fc2).to(device)
        
        # TODO: Create the optimizer (Adam with learning rate lr)
        self.optimizer = optim.Adam(self.local_network.parameters(), lr=lr)

        # TODO: Create a Replay Buffer
        self.memory = ReplayBuffer(buffer_size)
        
        # Initialize time step (for updating every self.update_every steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        
        # TODO: Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every self.update_every time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample(self.batch_size)
                self.learn(experiences)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.local_network.eval()
        with torch.no_grad():
            action_values = self.local_network(state)
        self.local_network.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences
        
        device = self.device  # for shortening the following lines
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions.reshape(-1, 1)).to(device)
        rewards = torch.FloatTensor(rewards.reshape(-1, 1)).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones.reshape(-1, 1)).to(device)        
        
        Q_targets_next = self.target_network(next_states).detach().max(dim=1, keepdim=True)[0]
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        Q_expected = self.local_network(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def target_hard_update(self):
        """ Assigns Local network weights to target network weights
        """
        target_parameters = list(self.target_network.parameters())
        local_parameters = list(self.local_network.parameters())
        for i in range(len(local_parameters)):
            target_parameters[i].data[:] = local_parameters[i].data[:]

        
    def save(self, path = 'checkpoint.pth'):
        """Saves the network
        Args:
            path (str): path to save the network. Default is 'checkpoint.pth'
        """
        torch.save(self.local_network.state_dict(), path)
 
    def load(self, path = 'checkpoint.pth'):
        """Loads the network
        Args:
            path (str): path with the weights. Default is 'checkpoint.pth'
        """
        self.local_network.load_state_dict(torch.load(path))