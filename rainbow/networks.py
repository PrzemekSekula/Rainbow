import torch 
import torch.nn as nn
import torch.nn.functional as F


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
