import torch 
import torch.nn as nn
import torch.nn.functional as F
import math

class QNetwork(nn.Module):
    """Fully connected (dense) neural network that estimates Q values."""

    def __init__(self, state_size, action_size, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Args:
            state_size (int): Dimension of each state. It equals
                the number of features in the network.
            action_size (int): Dimension of each action. It equals 
                the number of the network outputs
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values.
        Args:
            state (torch.Tensor): The state of the environment
        Returns:
            torch.Tensor: The action values
        """

        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DuelingQNetwork(nn.Module):
    """Dueling neural network that estimates Q values."""

    def __init__(self, state_size, action_size, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Args:
            state_size (int): Dimension of each state. It equals
                the number of features in the network.
            action_size (int): Dimension of each action. It equals 
                the number of the network outputs
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(DuelingQNetwork, self).__init__()
        
       
        self.feature_block = nn.Sequential(
            nn.Linear(state_size, fc1_units), 
            nn.ReLU(),
        )      
        
        self.value_block = nn.Sequential(
            nn.Linear(fc1_units, fc2_units),
            nn.ReLU(),
            nn.Linear(fc2_units, 1),
        )

        self.advantage_block = nn.Sequential(
            nn.Linear(fc1_units, fc2_units),
            nn.ReLU(),
            nn.Linear(fc2_units, action_size),
        )
        
    def forward(self, state):
        """Build a network that maps state -> action values.
        Args:
            state (torch.Tensor): The state of the environment
        Returns:
            torch.Tensor: The action values
        """

        features = self.feature_block(state)
        value = self.value_block(features)
        advantage = self.advantage_block(features)
        
        q = value + advantage - advantage.mean(dim=-1, keepdim=True)

        return q
    
    
class DistributionalQNetwork(nn.Module):
    """Categorical neural network that estimates Q values."""

    def __init__(self, state_size, action_size, 
                 fc1_units=64, fc2_units=64, 
                 atom_size = 51,
                 support = None):
        """Initialize parameters and build model.
        Args:
            state_size (int): Dimension of each state. It equals
                the number of features in the network.
            action_size (int): Dimension of each action. It equals 
                the number of the network outputs
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
            atom_size (int): Number of atoms in the distribution
            support (torch.Tensor): The support of the distribution
        """
        super(DistributionalQNetwork, self).__init__()
                
                
        self.action_size = action_size
        self.atom_size = atom_size
        self.support = support
        
        self.feature_block = nn.Sequential(
            nn.Linear(state_size, fc1_units), 
            nn.ReLU(),
        )      
        
        self.value_block = nn.Sequential(
            nn.Linear(fc1_units, fc2_units),
            nn.ReLU(),
            nn.Linear(fc2_units, atom_size),
        )

        self.advantage_block = nn.Sequential(
            nn.Linear(fc1_units, fc2_units),
            nn.ReLU(),
            nn.Linear(fc2_units, action_size * atom_size),
        )
                

    def forward(self, state):
        dist = self.dist(state)
        q = torch.sum(dist * self.support, dim=2)          
        return q     
                

    def dist(self, state):
        """Build a network that maps state -> action values.
        Args:
            state (torch.Tensor): The state of the environment
        Returns:
            torch.Tensor: The action values
        """

        features = self.feature_block(state)
        value = self.value_block(features).view(-1, 1, self.atom_size)
        advantage = self.advantage_block(features).view(-1, self.action_size, self.atom_size)
        
        q_atoms = value + advantage - advantage.mean(dim=-1, keepdim=True)
        dist = F.softmax(q_atoms, dim=-1)
        dist = dist.clamp(min=1e-3)  # for avoiding nans
        
        return dist
    
    
class NoisyQNetwork(nn.Module):
    """Neural Network with Noise Layers.
    """

    def __init__(self, state_size, action_size, 
                 fc1_units=64, fc2_units=64, 
                 atom_size = 51,
                 support = None):
        """Initialize parameters and build model.
        Args:
            state_size (int): Dimension of each state. It equals
                the number of features in the network.
            action_size (int): Dimension of each action. It equals 
                the number of the network outputs
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
            atom_size (int): Number of atoms in the distribution
            support (torch.Tensor): The support of the distribution
        """
        super(NoisyQNetwork, self).__init__()
                
                
        self.action_size = action_size
        self.atom_size = atom_size
        self.support = support
        
        self.feature_block = nn.Sequential(
            nn.Linear(state_size, fc1_units), 
            nn.ReLU(),
        )      
        
        self.value_block = nn.Sequential(
            NoisyLinear(fc1_units, fc2_units),
            nn.ReLU(),
            NoisyLinear(fc2_units, atom_size),
        )

        self.advantage_block = nn.Sequential(
            NoisyLinear(fc1_units, fc2_units),
            nn.ReLU(),
            NoisyLinear(fc2_units, action_size * atom_size),
        )
        
    def forward(self, state):
        dist = self.dist(state)
        q = torch.sum(dist * self.support, dim=2)          
        return q     
                

    def dist(self, state):
        """Build a network that maps state -> action values.
        Args:
            state (torch.Tensor): The state of the environment
        Returns:
            torch.Tensor: The action values
        """

        features = self.feature_block(state)
        value = self.value_block(features).view(-1, 1, self.atom_size)
        advantage = self.advantage_block(features).view(-1, self.action_size, self.atom_size)
        
        q_atoms = value + advantage - advantage.mean(dim=-1, keepdim=True)
        dist = F.softmax(q_atoms, dim=-1)
        dist = dist.clamp(min=1e-3)  # for avoiding nans
        
        return dist
    
    def reset_noise(self):
        """Reset all noisy layers."""
        self.advantage_block[0].reset_noise()
        self.advantage_block[2].reset_noise()
        self.value_block[0].reset_noise()
        self.value_block[2].reset_noise()
        
    
    
class NoisyLinear(nn.Module):
    """A layer with factorized Gausian noise added to the weights and biases.
    """
    
    def __init__(self, in_features, out_features, std_init=0.4, noise_autoreset = True):
        """Initialize parameters and build a layer.
        Args:
            in_features (int): Number of input features
            out_features (int): Number of output features
            std_init (float): The standard deviation of the initial weights
            noise_autoreset (bool): If True, reset noise after each forward pass.
                Noise is reset in training mode only.
        """
        
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.noise_autoreset = noise_autoreset
        
        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))
        
        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))
        
        self.reset_parameters()
        self.reset_noise()
                
        
    def reset_parameters(self):
        """Initilize the parameters.
        """
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))
        
    def reset_noise(self):
        """Reset the noise.
        """
        epsilon_in = self.scale_noise(self.in_features)
        epsilon_out = self.scale_noise(self.out_features)

        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)     
                
    def forward(self, x):
        if self.training:
            if self.noise_autoreset:
                self.reset_noise()
                            
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
            
        return F.linear(x, weight, bias)        
    
    
    @staticmethod
    def scale_noise(size):
        """Generate a noise tensor.
        Args:
            size (int): The size of the noise tensor

        Returns:
            torch.tensor: The noise tensor
        """
        x = torch.randn(size)
        x = x.sign().mul(x.abs().sqrt())
        return x