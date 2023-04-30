import numpy as np
import random
from collections import deque

from rainbow.replay_buffer import PrioritizedReplayBuffer

from rainbow.networks import NoisyQNetwork

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
                 clip_grad = 10,
                 #N-step parameters
                 n_steps = 3, 
                 #Categorical parameters
                 atom_size = 51,
                 v_min = 0,
                 v_max = 200,
                 # NoisyNet parameters
                 explore_with_noise = True,
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
            n_steps (int): Number of bootstrap steps to consider. Default: 3
            atom_size (int): Number of atoms for categorical DQN. Default: 51
            v_min (int): Minimum value support for categorical DQN. Default: 0
            v_max (int): Maximum value support for categorical DQN. Default: 200
            explore_with_noise: if true, use noisy network for exploration. 
                Otherwise use e-greedy exploration. Default: True
        """
        self.state_size = state_size
        self.action_size = action_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = lr
        self.update_every = update_every
        self.device = device
        
        # PER parameters
        self.per_alpha = per_alpha
        self.per_beta_start = per_beta_start            
        self.per_beta_frames = per_beta_frames
        self.per_prior_eps = per_prior_eps
        
        # Dueling parameters
        self.clip_grad = clip_grad

        #N-step parameters
        self.n_steps = n_steps 
        self.experiences = deque(maxlen=n_steps)   
        
        #Categorical parameters
        self.atom_size = atom_size
        self.v_min = v_min
        self.v_max = v_max
        self.support = torch.linspace(self.v_min, self.v_max, self.atom_size).to(device)
        
        # NoisyNet parameters
        self.explore_with_noise = explore_with_noise
        
             

        # Q-Network
        fc1, fc2 = 64, 16 # Size of the layers
        self.local_network = NoisyQNetwork(
            state_size, action_size, 
            fc1, fc2,
            atom_size=atom_size,
            support=self.support,
            ).to(device)
        self.target_network = NoisyQNetwork(
            state_size, action_size, 
            fc1, fc2,
            atom_size=atom_size,
            support=self.support,
            ).to(device)
        self.target_hard_update()       
        
        self.optimizer = optim.Adam(self.local_network.parameters(), lr=lr)

        # Replay Buffer
        self.memory = PrioritizedReplayBuffer(buffer_size, per_alpha)
        
        # Initialize time step (for updating every self.update_every steps
        # and for calculating per_beta 
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        
        
        # Save experience in replay memory
        if self.n_steps == 1:
            self.memory.add(state, action, reward, next_state, done)
        else:
            self.experiences.append((state, action, reward, next_state, done))
        
            if len(self.experiences) == self.n_steps:
                state = self.experiences[0][0]
                action = self.experiences[0][1]
                reward = self.experiences[0][2]
                for i in range(1, self.n_steps):
                    reward += self.experiences[i][2] * self.gamma ** i
                next_state = self.experiences[-1][3]
                self.memory.add(state, action, reward, next_state, done)            
                    
            if done:
                self.experiences.clear()      
                
            
        
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
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        
        
        if self.explore_with_noise:
            with torch.no_grad():
                action_values = self.local_network(state)
            action = np.argmax(action_values.cpu().data.numpy())
        else:
            self.local_network.eval()
            with torch.no_grad():
                action_values = self.local_network(state)
            self.local_network.train()

            # Epsilon-greedy action selection
            if random.random() > eps:
                action = np.argmax(action_values.cpu().data.numpy())
            else:
                action = random.choice(np.arange(self.action_size))
                
        return action

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

        
        # Distributional part        
        delta_z = float(self.v_max - self.v_min) / (self.atom_size - 1)
        
        with torch.no_grad():
            next_actions = self.local_network(next_states).argmax(1, keepdim=True)
            next_dist = self.target_network.dist(next_states)[range(self.batch_size), next_actions.squeeze(1)]
            target_z = rewards + (1 - dones) * self.gamma * self.support
            target_z = target_z.clamp(min=self.v_min, max=self.v_max)
            
            # projection of the target distribution onto the support set
            # code from https://github.com/Curt-Park/rainbow-is-all-you-need/blob/master/06.categorical_dqn.ipynb
            b = (target_z - self.v_min) / delta_z
            lower = b.floor().long()
            upper = b.ceil().long()
                                    
            offset = torch.linspace(
                0, 
                (self.batch_size - 1) * self.atom_size, 
                self.batch_size
                )\
                .long().unsqueeze(1)\
                .expand(self.batch_size, self.atom_size)\
                .to(self.device)
            
            proj_dist = torch.zeros(next_dist.size(), device=self.device)
                    
            proj_dist.view(-1).index_add_(
                0, 
                (lower + offset).view(-1), 
                (next_dist * (upper.float() - b)).view(-1)
                )
        
        dist = self.local_network.dist(states)
        log_p = torch.log(dist[range(self.batch_size), actions.squeeze(1)])
        
        
        
        elementwise_loss = -(proj_dist * log_p).sum(1)
        
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