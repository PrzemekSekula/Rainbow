import numpy as np
import random
from collections import namedtuple, deque

from rainbow.replay_buffer import PrioritizedReplayBuffer

from rainbow.networks import DuelingQNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

from torch.nn.utils import clip_grad_norm_


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size,
                 buffer_size = int(1e5),
                 batch_size = 64,
                 gamma = 0.99,
                 lr = 5e-4,
                 update_every = 4,
                 device = None,
                 # PER parameters
                 per_alpha = 0.6,
                 per_beta_start = 0.4,
                 per_beta_frames = 1e6,
                 per_prior_eps = 1e-6,
                 #Dueling parameters
                 clip_grad = 10
                 ):
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
            per_alpha (float): PER hyperparameter alpha. Default: 0.6
            per_beta_start (float): PER hyperparameter initial beta. Default: 0.4
            per_beta_frames (float): PER hyperparameter beta frames. Default: 1e6
                beta will be equal to 1.0 after per_beta_frames
            per_prior_eps (float): PER hyperparameter prior_eps. Default: 1e-6  
            clip_grad (int): Gradient clipping. Default: None
        """
        self.state_size = state_size
        self.action_size = action_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = lr
        self.update_every = update_every
        self.device = device
        
        self.per_alpha = per_alpha
        self.per_beta_start = per_beta_start            
        self.per_beta_frames = per_beta_frames
        self.per_prior_eps = per_prior_eps
        
        self.clip_grad = clip_grad

        # Q-Network
        fc1, fc2 = 64, 16 # Size of the layers
        # TODO: Create a network with 2 hidden layers (fc1 and fc2 nodes)
        self.local_network = DuelingQNetwork(state_size, action_size, fc1, fc2).to(device)
        self.target_network = DuelingQNetwork(state_size, action_size, fc1, fc2).to(device)
        self.target_hard_update()       
        
        # TODO: Create the optimizer (Adam with learning rate lr)
        self.optimizer = optim.Adam(self.local_network.parameters(), lr=lr)

        # TODO: Create a Replay Buffer
        self.memory = PrioritizedReplayBuffer(buffer_size, per_alpha)
        
        # Initialize time step (for updating every self.update_every steps
        # and for calculating per_beta 
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        
        # TODO: Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every self.update_every time steps.
        self.t_step += 1

        if (self.t_step % self.update_every) == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                per_beta = min(1.0, self.per_beta_start + self.t_step * (1.0 - self.per_beta_start) / self.per_beta_frames)
                experiences = self.memory.sample(self.batch_size, per_beta)
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
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done, weights, idxes) tuples 
        """
        states, actions, rewards, next_states, dones, weights, idxes = experiences
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions.reshape(-1, 1)).to(self.device)
        rewards = torch.FloatTensor(rewards.reshape(-1, 1)).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones.reshape(-1, 1)).to(self.device)        
        weights = torch.FloatTensor(weights.reshape(-1, 1)).to(self.device)        

        next_actions = self.local_network(next_states).detach().argmax(1, keepdim=True)
        Q_next = self.target_network(next_states).gather(1, next_actions).detach()
                
        Q_targets = rewards + (self.gamma * Q_next * (1 - dones))
        Q_curr = self.local_network(states).gather(1, actions)
                
        elementwise_loss = F.mse_loss(Q_curr, Q_targets, reduction='none')        
        loss = torch.mean(elementwise_loss * weights)
        self.optimizer.zero_grad()
        loss.backward()
        
        if self.clip_grad is not None:
            clip_grad_norm_(self.local_network.parameters(), self.clip_grad)
        
        self.optimizer.step()
        
        # PER: update priorities
        loss_for_prior = elementwise_loss.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.per_prior_eps
        self.memory.update_priorities(idxes, new_priorities)
        
        
    def target_hard_update(self):
        """ Assigns Local network weights to target network weights
        """
        self.target_network.load_state_dict(self.local_network.state_dict())
        self.target_network.eval()        
        
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