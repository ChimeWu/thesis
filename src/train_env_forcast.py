from stable_baselines3 import DDPG, PPO, SAC, TD3, A2C
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np
import pandas as pd
from os.path import abspath, dirname, join
import random
import math
from env_forcast import *

def train_ddpg(timesteps):
    data = Data.load_data('train')
    env = WorkEnv(data)

    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=100 * np.ones(n_actions))

    model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1)
    model.learn(total_timesteps=timesteps)
    model.save("ddpg_my_env")

def test_ddpg():
    data = Data.load_data('test')
    env = WorkEnv(data)
    model = DDPG.load("ddpg_my_env")
    test_model = TestAlgo(model, env)
    test_model.test()
    test_model.print_cost()

def train_a2c(timesteps):
    data = Data.load_data('train')
    env = WorkEnv(data)

    model = A2C("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=timesteps)
    model.save("a2c_my_env")

def test_a2c():
    data = Data.load_data('test')
    env = WorkEnv(data)
    model = A2C.load("a2c_my_env")
    test_model = TestAlgo(model, env)
    test_model.test()
    test_model.print_cost()

def train_ppo(timesteps):
    data = Data.load_data('train')
    env = WorkEnv(data)

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=timesteps)
    model.save("ppo_my_env")

def test_ppo():
    data = Data.load_data('test')
    env = WorkEnv(data)
    model = PPO.load("ppo_my_env")
    test_model = TestAlgo(model, env)
    test_model.test()
    test_model.print_cost()

def train_sac(timesteps):
    data = Data.load_data('train')
    env = WorkEnv(data)

    model = SAC("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=timesteps)
    model.save("sac_my_env")

def test_agent(model, env):
    test_model = TestAlgo(model, env)
    test_model.test()
    test_model.print_cost()

def test_agents():
    data = Data.load_data('test')
    env = WorkEnv(data)
    model = DDPG.load("ddpg_my_env")
    test_agent(model, env)
    model = A2C.load("a2c_my_env")
    test_agent(model, env)
    model = PPO.load("ppo_my_env")
    test_agent(model, env)
    model = SAC.load("sac_my_env")
    test_agent(model, env)
    model = RandomAgent(env.action_space)
    test_agent(model, env)
    model = ConstantAgent(env.action_space)
    test_agent(model, env)
    model = ForcastAgent(env.action_space)
    test_agent(model, env)

def train():
    #train_ddpg(10000)
    #train_a2c(10000)
    #train_ppo(30000)
    train_sac(30000)


def test():
    test_ddpg()
    test_a2c()

if __name__ == "__main__":
    #train()
    #test()
    test_agents()