import argparse
import time
import matplotlib.pyplot as plt
import numpy as np
from gym import spaces

import senseact.devices.dxl.dxl_utils as dxl
from env import ReacherEnv
import math


class PIDReacherEnv(ReacherEnv):

    def __init__(self, cycle_time, ep_time, driver, idn, port_str, baud, max_action):
        super(PIDReacherEnv, self).__init__(cycle_time, ep_time, driver, idn, port_str, baud, max_action)
        self.observation_space = spaces.Box(low=np.inf, high=np.inf, shape=(4,))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--cycle_time", type=float, default=0.040, help="cycle time")
    parser.add_argument("--ep_time", type=float, default=2.0, help="number of seconds to run for each episode.")
    parser.add_argument("--idn", type=int, default=1, help="Dynamixel ID")
    parser.add_argument("--baud", type=int, default=1000000, help="Dynamixel Baud")
    parser.add_argument("--port_str", type=str, default=None,
                        help="Default of None will use the first device it finds. Set this to override.")
    parser.add_argument("--ep_count", type=int, default=100, help="How many episodes to average returns over.")

    args = parser.parse_args()

    max_action = np.pi  # TODO:100
    use_ctypes = True  # TODO: False

    idn = args.idn
    env = PIDReacherEnv(args.cycle_time, args.ep_time, dxl.get_driver(use_ctypes), idn, args.port_str, args.baud, max_action)

    # TODO
    
    # PID constants
    
    kp = 1
    ki = 0.0000001
    kd = 0.0000001

    returns = []
    rewards = []
    
    error_prior = 0
    integral = 0
    derivative = 0
    
    for i in range(args.ep_count):
        print(i)
        start = time.time()
        flag = 0
        observation, reward_at_start = env.reset()
        error = observation[2]
        action_prior = observation[3]

        g = 0.0  # g does not include reward_at_start
        r = [reward_at_start] # only consider this while plotting to show how far the joint was initially from the target
        print('Reward at start ', reward_at_start)

        # TODO: Run one episode of PID control
        while True:
            
            integral = integral + (error*env.cycle_time)
            derivative = derivative +(error)/env.cycle_time
            
            action = kp*error + ki*integral + kd*derivative # + bias
            print('Action ', action, end = '\t')
            
            # action gives change needed in the action
            # action_prior gives previous action
            # position control works with target action and not the change in the action
            # hence, I added them.
            
            action = action + action_prior

            observation, reward, done = env.step(action)
            
            action_prior = observation[3]
            error = observation[2]
            
            print('Observation ', observation, end='\t')
            
            # return
            g += reward
            
            # rewards
            r.append(reward)
            print('Reward ', reward, end='\n')
            
            if flag == 1:
                break

            if done == True:
                flag = 1
                print('Done')
            
            #time.sleep(env.cycle_time)    

        print('Returns ',g)
        returns.append(g)
        rewards.append(r)
        end = time.time()
        print(end-start)

    env.close()

    returns = np.array(returns)
    rewards = np.array(rewards)
    np.save('./returnsoverallepisodes.npy', returns)
    np.save('./rewardsoverallepisodes.npy', rewards)
    avg_return = returns.mean()

    print(f"Avg return: {avg_return}.")
