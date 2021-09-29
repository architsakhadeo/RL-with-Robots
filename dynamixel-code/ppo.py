#!/usr/bin/env python

import argparse
import os
import torch
from torch.utils.tensorboard import SummaryWriter
from env import ReacherEnv
from ppo_agent import PPO
from gym import spaces
import time
import senseact.devices.dxl.dxl_utils as dxl
import matplotlib
import matplotlib.pyplot as plt
import gym
from network import network_factory
import numpy as np
import random
import sys
from torch import nn
from torch import optim
import pickle
from torch.distributions import normal



def main(cycle_time, idn, baud, port_str, batch_size, mini_batch_div, epoch_count, gamma, l, max_action, outdir,
         ep_time):
    """
    :param cycle_time: sense-act cycle time
    :param idn: dynamixel motor id
    :param baud: dynamixel baud
    :param batch_size: How many sample to record for each learning update
    :param mini_batch_size: How many samples to sample from each batch
    :param epoch_count: Number of epochs to train each batch on. Is this the number of mini-batches?
    :param gamma: Usual discount value
    :param l: lambda value for lambda returns.


    In the original paper PPO runs N agents each collecting T samples.
    I need to think about how environment resets are going to work. To calculate things correctly we'd technically
    need to run out the episodes to termination. How should we handle termination? We might want to have a max number
    of steps. In our setting we're going to be following a sine wave - I don't see any need to terminate then. So we
    don't need to run this in an episodic fashion, we'll do a continuing task. We'll collect a total of T samples and
    then do an update. I think I will implement the environment as a gym environment just to permit some
    interoperability. If there was an env that had a terminal then we would just track that terminal and reset the env
    and carry on collecting. Hmmm, actually I'm not sure how to think about this as a gym env. So SenseAct uses this
    RTRLBaseEnv, but I'm not sure I want to do that.

    So the changes listed from REINFORCE:
    1. Drop γ^t from the update, but not from G_t
    2. Batch Updates
    3. Multiple Epochs over the same batch
    4. Mini-batch updates
    5. Surrogate objective: - π_θ/π_θ_{old} * G_t
    6. Add Baseline
    7. Use λ-return: can you the real lambda returns or use generalized advantage estimation like they do in the paper.
    8. Normalize the advantage estimates: H = G^λ - v
    9. Proximity constraint:
        ρ = π_θ/π_θ_{old}
        objective:
        -min[ρΗ, clip(ρ, 1-ε, 1+ε)H]

    Also, there is the value function loss and there is an entropy bonus given.

    """

    tag = f"{time.time()}"
    summaries_dir = f"./summaries/{tag}"
    returns_dir = "./returns"
    networks_dir = "./networks"
    if outdir:
        summaries_dir = os.path.join(outdir, f"summaries/{tag}")
        returns_dir = os.path.join(outdir, "returns")
        networks_dir = os.path.join(outdir, "networks")

    os.makedirs(summaries_dir, exist_ok=True)
    os.makedirs(returns_dir, exist_ok=True)
    os.makedirs(networks_dir, exist_ok=True)

    summary_writer = SummaryWriter(log_dir=summaries_dir)

    env = ReacherEnv(cycle_time, ep_time, dxl.get_driver(False), idn, port_str, baud, 100.0)
    in_size = env.observation_space.shape[0]
    num_actions = 2 # mean and sigma
    eps = 0.2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = network_factory(in_size, num_actions, env)  # TODO: create your network
    network.to(device)
        
    agent = PPO(device, network, in_size, batch_size, mini_batch_div, epoch_count, gamma, l, eps, summary_writer)

    num_episodes = 1000
        
    
    # TODO: implement your main loop here. You will want to collect batches of transitions

    total_rewards = []
    finalrewards = []
    finalreturns = []
    batch_returns = []
    batch_actions = []
    batch_states = []
    batch_counter = 0
    
    # Start iterating over episodes

    for ep in range(num_episodes):
        s_0 = env.reset()
        states = []
        rewards = []
        actions = []
        complete = False

        while complete == False:
        
            # Generates trajectory of states, actions, rewards per transition while episode is not over
            
            action = agent.step(s_0)
            s_1, r, complete = env.step(action)

            states.append(s_0)
            rewards.append(r)
            actions.append(action)
            s_0 = s_1        
            
        # When episode is over    

        batch_counter += 1
        
        # Calculates lambda returns

        total_rewards.append(sum(rewards))
        
        value_estimates_step =  agent.get_estimated_value(torch.tensor(states, dtype=torch.float32)).detach()
        batch_returns.extend(agent.compute_return(rewards, value_estimates_step, gamma, l))

        batch_states.extend(states)
        batch_actions.extend(actions)
        
        finalrewards.append(rewards)
        finalreturns.append(agent.compute_return(rewards, value_estimates_step, gamma, l))                
        
        
        # When a whole batch is collected
                
        if batch_counter == batch_size:
            state_tensor = torch.tensor(batch_states, dtype=torch.float32)
            action_tensor = torch.tensor(batch_actions, dtype=torch.float32)
            return_tensor = torch.tensor(batch_returns, dtype=torch.float32)            
                
            value_estimates =  agent.get_estimated_value(state_tensor).detach()
                
            # Computes old policy distribution
                
            mean_old, sigma_old =  agent.get_estimated_policy(state_tensor)
            mean_old = mean_old.detach()
            sigma_old = sigma_old.detach()                        
            prob_action_old_dist = normal.Normal(mean_old,sigma_old)
                
            # Advantage calculation and normalization

            advantage = agent.compute_advantage(return_tensor, value_estimates.view(1,-1)[0])
            normalized_advantage = (advantage - advantage.mean()) / advantage.std()
                
            # Optimization over multiple epochs

            for epoch in range(epoch_count):
                indices = [i for i in range(len(return_tensor))]
                random.shuffle(indices)
                    
                # Updates over mini batches
                    
                for i in range(mini_batch_div):
                    miniindices = indices[int(1.0*i*len(return_tensor)/mini_batch_div):int(1.0*(i+1)*len(return_tensor)/mini_batch_div)]
                    minireturn_tensor = torch.tensor([batch_returns[j] for j in miniindices], dtype=torch.float32)
                    ministate_tensor = torch.tensor([batch_states[j] for j in miniindices], dtype=torch.float32)
                    miniaction_tensor = torch.tensor([batch_actions[j] for j in miniindices], dtype=torch.float32)
                        
                    # Learning
                        
                    agent.learn(prob_action_old_dist, normalized_advantage, action_tensor, miniindices, miniaction_tensor, minireturn_tensor, ministate_tensor)
            
            # Reinitialize batch variables at the start of a new batch
                                                                  
            batch_returns = []
            batch_actions = []
            batch_states = []
            batch_counter = 0
                    
        print("Ep: {} Average of last 1: {:.2f}".format(
                    ep + 1, np.mean(total_rewards[-1:])))


    finalreturns = np.array(finalreturns)
    finalrewards = np.array(finalrewards)
    np.save('./finalreturnsPPO1.npy', finalreturns)
    np.save('./finalrewardsPPO1.npy', finalrewards)    

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycle_time", type=float, default=0.040, help="sense-act cycle time")
    parser.add_argument("--idn", type=int, default=1, help="Dynamixel ID")
    parser.add_argument("--baud", type=int, default=1000000, help="Dynamixel Baud")
    parser.add_argument("--port_str", type=str, default=None,
                        help="Default of None will use the first device it finds. Set this to override.")
    parser.add_argument("--batch_size", type=int, default=40,
                        help="How many samples to record for each learning update")
    parser.add_argument("--mini_batch_div", type=int, default=32, help="Number of division to divide batch into")
    parser.add_argument("--epoch_count", type=int, default=10,
                        help="Number of times to train over the entire batch per update.")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount")
    parser.add_argument("--l", type=float, default=0.95, help="lambda for lambda return")
    parser.add_argument("--max_action", type=float, default=3.14159,
                        help="The maximum value you will output to the motor. "
                             "This should be dependent on the control mode which you select.")
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--ep_time", type=float, default=2.0, help="number of seconds to run for each episode.")

    args = parser.parse_args()
    main(**args.__dict__)
