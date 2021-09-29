#!/usr/bin/env python

"""
Place your PPO agent code in here.
"""

import torch
from torch.utils.tensorboard import SummaryWriter
from network import PolicyNetwork
from network import ValueNetwork
import numpy as np
import random
import sys
from torch import nn
from torch import optim
from torch.distributions import normal

class PPO:
    def __init__(self,
                 device,  # cpu or cuda
                 network,  # your network
                 state_size,  # size of your state vector
                 batch_size,  # size of batch
                 mini_batch_div,
                 epoch_count,
                 gamma=0.99,  # discounting
                 l=0.95,  # lambda used in lambda-return
                 eps=0.2,  # epsilon value used in PPO clipping
                 summary_writer: SummaryWriter = None):
        self.device = device

        self.batch_size = batch_size
        self.mini_batch_div = mini_batch_div
        self.epoch_count = epoch_count
        self.gamma = gamma
        self.l = l
        self.eps = eps
        self.summary_writer = summary_writer

        self.state_size = state_size
        self.network = network
        #self.optimizer = torch.optim.Adam(self.network.parameters())
        self.policy_estimator = PolicyNetwork(network)
        self.value_estimator = ValueNetwork(state_size)
        self.optimizer =   optim.Adam(self.policy_estimator.network.parameters())#,  lr=0.00005)
        self.optimizer_v = optim.Adam(self.value_estimator.network_v.parameters())#, lr=0.00005)
        self.sigma_init = torch.tensor(0.5)
        self.sigmaflag = 0                

    def get_estimated_value(self, state):
        return self.value_estimator.forward(state)
    
    def get_estimated_policy(self, state):
        return self.policy_estimator.forward(state)


    def step(self, state):
        """
        You will need some step function which returns the action.
        This is where I saved my transition data in my own code.
        :param state:
        :param r:
        :param terminal:
        :return:
        """
        
        # Takes an action based on policy given the state
             
        mean_start, sigma_start =  self.get_estimated_policy(state)
        mean_start = mean_start.detach()
        
        # Initializes the standard deviation to 0.5 at the start of the first episode
        
        if self.sigmaflag == 0:
            sigma_start = self.sigma_init
            self.sigmaflag = 1
        
        sigma_start = sigma_start.detach()
        action =  self.policy_estimator.get_action(mean_start,sigma_start)
        return action

    @staticmethod
    def compute_return(r_buffer, v_buffer, gamma, l):
        """

        Compute the return. Unit test this function

        :param r_buffer: rewards
        :param v_buffer: values
        :param t_buffer: terminal
        :param l: lambda value
        :param gamma: gamma value
        :return:  the return
        """
        
        # Computes lambda return from rewards and value estimats
        
        r = r_buffer[::-1]
        V = v_buffer.view(1,-1)[0].tolist()[::-1]
        G = [r[0]]
        
        for i in range(1,len(r)):
            G.append(r[i] + gamma* ((1 - l)*V[i-1] + l*G[-1]) )
        G = G[::-1]
        return G


    def compute_advantage(self, g, v):
        """
        Compute the advantage
        :return: the advantage
        """
        advantage = g - v
        return advantage

    def compute_rho(self, old_pi_dist, new_pi_dist, miniindices, action_tensor, miniaction_tensor):
        """
        Compute the ratio between old and new pi
        :param actions:
        :param old_pi:
        :param new_pi:
        :return:
        """
        old_pi = old_pi_dist.log_prob(action_tensor).exp()[miniindices].view(-1,1)
        new_pi = new_pi_dist.log_prob(miniaction_tensor).exp().view(-1,1)
        return new_pi / old_pi

    def learn(self, prob_action_old_dist, normalized_advantage, action_tensor, miniindices, miniaction_tensor, minireturn_tensor, ministate_tensor):
        """
        Here's where you should do your learning and logging.
        :param t: The total number of transitions observed.
        :return:
        """
        
        # Learning on mini batches
        
        mean_current, sigma_current =  self.get_estimated_policy(ministate_tensor)
        prob_action_current_dist = normal.Normal(mean_current,sigma_current)
                        
        r = self.compute_rho(prob_action_old_dist, prob_action_current_dist, miniindices, action_tensor, miniaction_tensor)
        clipped_r = torch.clamp(r, 1 - 0.2, 1 + 0.2)

        final_normalized_clipping = torch.min(r.view(1,-1)[0] * normalized_advantage[miniindices], 
                                              clipped_r.view(1,-1)[0] * normalized_advantage[miniindices])
        
        # Value loss                                      
        loss_v = torch.mean((minireturn_tensor -  self.get_estimated_value(ministate_tensor).view(1,-1)[0])**2)
        
        # Policy loss
        loss = -torch.mean(final_normalized_clipping) + loss_v #+ torch.mean( - 0.01 * normal_dist.entropy())
        
        # Policy update
        self.optimizer.zero_grad()   
        loss.backward(retain_graph=True)                                          
        self.optimizer.step() 
                       
        # Value update                        
        self.optimizer_v.zero_grad()
        loss_v.backward(retain_graph=True)
        self.optimizer_v.step()
