import gym
import time
import numpy as np
from gym import spaces
import senseact.devices.dxl.dxl_utils as dxl
from senseact.devices.dxl.dxl_mx64 import MX64
import math

class ReacherEnv(gym.core.Env):
    # TODO: you are free to add additional parameters to this function
    def __init__(self, cycle_time, episode_length, driver, idn, port_str, baud, max_action):
        """
        You are responsible for:
        - handling the dynamixel communications
        - implementing the step function
        - implement _reset_motor
        - implement _make_observation

        :param cycle_time: This is the rate of the interaction loop, in seconds.
        :param episode_length: The length of the episode, in seconds.
        :param driver: Dynamixel driver
        :param idn: Dynamixel id
        :param port_str: Dynamixel port
        :param baud: Dynamixel baud rate
        """
        self.cycle_time = cycle_time
        self.episode_length = episode_length
        self.driver = driver
        self.idn = idn
        self.port_str = port_str
        self.baud = baud
        self.port = dxl.make_connection(driver, baudrate=baud, timeout=1, port_str=port_str)

        self.step_count = int(self.episode_length / self.cycle_time)
        self.steps = 0
        self.total_steps = 0
        self.cycle_time_mean = 0.0
        self.cycle_time_var = 0.0
        self.last_step_call_time = None
        self.max_action = np.pi

        self.target = None
        self.range = 0.5 * np.pi * 150 / 180  # duplicate vals from SenseAct.

        self.action_space = spaces.Box(low=-np.pi, high=np.pi, shape=(1,))

        # TODO: you define the observation space - specify dimensions here
        self.observation_space = spaces.Box(low=np.inf, high=np.inf, shape=(4,))




    def reset(self):
        """
        Resets the motor at the start of a new episode
        :return:
        """

        def draw_pos():
            return np.random.uniform(-self.range, self.range)

        self.steps = 0
        self.last_step_call_time = None
        start_pos = 0.0
        self.target = draw_pos()
        print('Target Position ', self.target)
        self._reset_motor(start_pos)

        stddev = np.sqrt(self.cycle_time_var / self.total_steps) if self.total_steps > 0 else 0.0
        print(f"{self.total_steps}: Cycle time (mean, stddev): {self.cycle_time_mean}, {stddev}")
        
        obs = self.read()
        ret_obs = self._make_observation(obs)
        
        # Adds previous action as the 4th observation feature
        ret_obs = np.append(ret_obs, [0]) # first previous action
        
        reward_at_start = self.compute_reward(obs)

        #return ret_obs, reward_at_start   # FOR PID
        return ret_obs                     # FOR PPO




    def close(self):
        """
        DO NOT MODIFY
        :return:
        """

        dxl.close(self.driver, self.port)




    def _reset_motor(self, start_pos):
        # TODO: Reset your motor. Move the motor to the start_pos, then activate your chosen control mode
        dxl.set_joint_mode(self.driver, self.port, 1)
        dxl.write_pos(self.driver, self.port, 1, 0.0)
        time.sleep(0.3)




    def read(self):
        # TODO: read from the motor
        """
        Note that dxl_utils offers different methods for reading. We used read_vals before to read the entire register
        of the motor. However, we can be more efficient by only reading the registers we care about. This
        can be done by creating a read block and calling the 'read' method. Ex.
        read_block = dxl_mx64.MX64.subblock('derivative_gain', 'proportional_gain', ret_dxl_type=driver.is_ctypes_driver)
        obs = dxl.read(driver, port, idn, read_block

        The subblock tells the motor to return all registers starting from 'derivative_gain' up to and including
        'proportional_gain': derivative_gain, integral_gain, proportional_gain. The ordering is specified in dxl_mx64.py

        :return:
        """
        read_block = MX64.subblock('goal_pos', 'present_pos', ret_dxl_type=self.driver.is_ctypes_driver)
        dict_obs = dxl.read(self.driver, self.port, 1, read_block)
        return dict_obs




    def _update_cycle_time_stats(self, delta):
        """
        DO NOT MODIFY
        :param delta:
        :return:
        """
        diff1 = delta - self.cycle_time_mean
        self.cycle_time_mean += diff1 / self.total_steps
        diff2 = delta - self.cycle_time_mean
        self.cycle_time_var += diff1 * diff2




    def compute_reward(self, obs):
        """
        DO NOT MODIFY

        This mimics the reward function used by SenseAct, with the exception that SenseAct further divides this
        value by 0.04.
        :param obs:
        :return:
        """
        cur_pos = obs["present_pos"]
        return -1 * self.cycle_time * abs(self.target - cur_pos)




    def step(self, action):
        """
        DO NOT MODIFY

        :param action:
        :return:
        """
        now = time.time()
        if self.last_step_call_time is not None:
            timed_cycle = now - self.last_step_call_time
            self.total_steps += 1
            self.steps += 1
            self._update_cycle_time_stats(timed_cycle)
        self.last_step_call_time = now
        
        return self._step(action)




    def _step(self, action):
        """
        This function expects you to take an action and then return the resulting observation. So this is where
        your sleep needs to go to maintain your cycle time.
        :return:
        """
        # TODO: fill in this method
        
        # Clips action
        
        #LIMITS 
        if action > self.max_action:
            action = self.max_action
        if action < -self.max_action:
            action = -self.max_action
                   
        dxl.set_joint_mode(self.driver, self.port, 1)
        dxl.write_pos(self.driver, self.port, 1, action)
        time.sleep(0.035)
        obs = self.read()

        observation = self._make_observation(obs)
        
        # Adds previous action as the 4th observation feature
        observation = np.append(observation, [action]) # previous action
                
        reward = self.compute_reward(obs)
        done = self._is_done()
        return observation, reward, done




    def _is_done(self):
        """
        DO NOT MODIFY
        :return:
        """
        return self.steps >= self.step_count




    def _make_observation(self, dxl_observation):
        # Return a numpy array
        # You can get the max values used in each dynamixel field as follows:
        # MX64["present_speed"].unit.y_max
        # See dxl_mx64 for the complete list of registers.
        # TODO: create your observation
        
        # Adding prior action to this observation in reset() and _step()
        obs = np.array([self.target, dxl_observation['present_pos'], self.target - dxl_observation['present_pos']])
        return obs


