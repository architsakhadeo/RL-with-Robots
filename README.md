# Reinforcement Learning with Robots

The code in this repository heavily uses the SenseAct framework https://github.com/kindredresearch/SenseAct.

## Installation instructions:

Please follow the installation instructions for SenseAct as specified in https://github.com/kindredresearch/SenseAct.

## Description

We conduct experiments with two robots:<br>
    1) Create2 Roomba robot,<br>
    2) Dynamixel Servo motor.

SenseAct provides an interface to interact with these robots.


### iRobot Create 2 robots:
- [Create-Mover](https://github.com/kindredresearch/SenseAct/blob/master/senseact/envs/create2/create2_mover_env.py)
- [Create-Docker](https://github.com/kindredresearch/SenseAct/blob/master/senseact/envs/create2/create2_docker_env.py)

| ![Create-Mover](docs/create-mover-ppo.gif) <br />Create-Mover | ![Create-Docker](docs/create-docker-trpo.gif) <br /> Create-Docker |
| --- | --- |

- For the Create2 Roomba robot, we perform the Docker task. In this, the robot learns from scratch to dock to its charging station.
- We use the PPO algorithm (policy-gradient RL algorithm) from OpenAI's repository https://github.com/openai/baselines


### Dynamixel (DXL) actuators:

- [DXL-Reacher](https://github.com/kindredresearch/SenseAct/blob/master/senseact/envs/dxl/dxl_reacher_env.py)
- [DXL-Tracker](https://github.com/kindredresearch/SenseAct/blob/master/senseact/envs/dxl/dxl_tracker_env.py)

| ![DXL-Reacher](docs/dxl-reacher-trpo.gif) <br/>DXL-Reacher | ![DXL-Tracker](docs/dxl-tracker-trpo.gif)<br /> DXL-Tracker |
| --- | --- |

- For the Dynamixel Servo motor, we perform the Reacher task. In this, the motor learns to move the joint to a desired position (angle).
- We implement our own 'minimal' PPO algorithm (policy-gradient RL algorithm) for the Reacher task on the Dynamixel. This minimal PPO algorithm is derived from the base REINFORCE algorithm. The derivation and the pseudocode are given in this PDF file https://github.com/architsakhadeo/RL-with-Robots/blob/main/pdf/pseudocodePPO.pdf
- We also implement a PID controller for the Dynamixel Servo motor.



