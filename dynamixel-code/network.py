import torch
from torch import nn
import numpy as np
from torch.distributions import Categorical
from torch.distributions import normal
import math

def network_factory(in_size, num_actions, env):
    """

    :param in_size:
    :param num_actions:
    :param env: The gym environment. You shouldn't need this, but it's included regardless.
    :return: A network derived from nn.Module
    """
    network = nn.Sequential(nn.Linear(in_size, 64, bias=True), nn.Tanh(), nn.Linear(64, num_actions, bias=True))
    return network

    
class PolicyNetwork():
    def __init__(self, network):
        super(PolicyNetwork, self).__init__()
        self.network = network
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def forward(self, state):
        state = torch.tensor(state, device=self.device, dtype=torch.float32)
        mean, sigma = torch.transpose(self.network(state), 0, -1)
        #LIMITS
        try:
            return torch.clamp(np.pi*torch.tanh(torch.tensor([mean])), -np.pi, np.pi ).requires_grad_(True) , torch.tensor([abs(sigma)]).requires_grad_(True)
        except:
            return torch.clamp(np.pi*torch.tanh(torch.tensor(mean)), -np.pi, np.pi ).requires_grad_(True) , torch.tensor(abs(sigma)).requires_grad_(True)       

    def get_action(self, mean, sigma):
        """
        This function will be used to evaluate your policy.
        :param inputs: environmental inputs. These should be the environment observation wrapped in a tensor:
        torch.tensor(obs, device=device, dtype=torch.float32)
        :return: Should return a single integer specifying the action
        """
        dist = normal.Normal(mean, sigma)
        return dist.sample()


class ValueNetwork():
    def __init__(self, in_size):
        super(ValueNetwork, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network_v = nn.Sequential(nn.Linear(in_size, 64, bias=True), nn.Tanh(), nn.Linear(64, 1, bias=True))
        
    def forward(self, state):
        state = torch.tensor(state, device=self.device, dtype=torch.float32)
        return self.network_v(state)

    def get_value(self, state):
        """
        This function will be used to evaluate your policy.
        :param inputs: environmental inputs. These should be the environment observation wrapped in a tensor:
        torch.tensor(obs, device=device, dtype=torch.float32)
        :return: Should return a single integer specifying the action
        """
        return self.forward(state).item()
