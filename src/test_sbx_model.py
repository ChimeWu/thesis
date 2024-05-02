import pandas as pd
from os.path import abspath, dirname, join
from datetime import datetime
from sbx import DDPG, PPO, SAC, TD3, CrossQ, TQC, DroQ
import random
from env import *

def test_constant(possible_times,env):
    model = ConstantActionAgent(env.action_space)
    rewards = np.zeros(len(possible_times))
    for start_time in possible_times:
        state = env.reset(start_time=start_time)
        info = {}
        done = False
        while not done:
            action = model.get_action()
            state, reward, done, info = env.step(action)
        rewards[possible_times.index(start_time)] = info['cumulative_reward']

    return rewards

def test_simple(possible_times,env):
    model = SimpleStateBasedAgent(env.action_space)
    rewards = np.zeros(len(possible_times))
    for start_time in possible_times:
        state = env.reset(start_time=start_time)
        info = {}
        done = False
        while not done:
            action = model.get_action(state)
            state, reward, done, info = env.step(action)
        rewards[possible_times.index(start_time)] = info['cumulative_reward']

    return rewards

def test_random(possible_times,env):
    model = RandomActionAgent(env.action_space)
    rewards = np.zeros(len(possible_times))
    for start_time in possible_times:
        state = env.reset(start_time=start_time)
        info = {}
        done = False
        while not done:
            action = model.get_action()
            state, reward, done, info = env.step(action)
        rewards[possible_times.index(start_time)] = info['cumulative_reward']

    return rewards


def test_ddpg(possible_times,env):
    model = DDPG.load("sbx_ddpg_rye_flex_env")
    rewards = np.zeros(len(possible_times))
    for start_time in possible_times:
        state = env.reset(start_time=start_time)
        info = {}
        done = False
        while not done:
            action = model.predict(state)[0]
            state, reward, done, info = env.step(action)
        rewards[possible_times.index(start_time)] = info['cumulative_reward']
    return rewards

def test_ppo(possible_times,env):
    model = PPO.load("sbx_ppo_rye_flex_env")
    rewards = np.zeros(len(possible_times))
    for start_time in possible_times:
        state = env.reset(start_time=start_time)
        info = {}
        done = False
        while not done:
            action = model.predict(state)[0]
            state, reward, done, info = env.step(action)
        rewards[possible_times.index(start_time)] = info['cumulative_reward']
    return rewards

def test_sac(possible_times,env):
    model = SAC.load("sbx_sac_rye_flex_env")
    rewards = np.zeros(len(possible_times))
    for start_time in possible_times:
        state = env.reset(start_time=start_time)
        info = {}
        done = False
        while not done:
            action = model.predict(state)[0]
            state, reward, done, info = env.step(action)
        rewards[possible_times.index(start_time)] = info['cumulative_reward']
    return rewards

def test_td3(possible_times,env):
    model = TD3.load("sbx_td3_rye_flex_env")
    rewards = np.zeros(len(possible_times))
    for start_time in possible_times:
        state = env.reset(start_time=start_time)
        info = {}
        done = False
        while not done:
            action = model.predict(state)[0]
            state, reward, done, info = env.step(action)
        rewards[possible_times.index(start_time)] = info['cumulative_reward']
    return rewards

def test_crossq(possible_times,env):
    model = CrossQ.load("sbx_crossq_rye_flex_env")
    rewards = np.zeros(len(possible_times))
    for start_time in possible_times:
        state = env.reset(start_time=start_time)
        info = {}
        done = False
        while not done:
            action = model.predict(state)[0]
            state, reward, done, info = env.step(action)
        rewards[possible_times.index(start_time)] = info['cumulative_reward']
    return rewards

def test_tqc(possible_times,env):
    model = TQC.load("sbx_tqc_rye_flex_env")
    rewards = np.zeros(len(possible_times))
    for start_time in possible_times:
        state = env.reset(start_time=start_time)
        info = {}
        done = False
        while not done:
            action = model.predict(state)[0]
            state, reward, done, info = env.step(action)
        rewards[possible_times.index(start_time)] = info['cumulative_reward']
    return rewards

def test_droq(possible_times,env):
    model = DroQ.load("sbx_droq_rye_flex_env")
    rewards = np.zeros(len(possible_times))
    for start_time in possible_times:
        state = env.reset(start_time=start_time)
        info = {}
        done = False
        while not done:
            action = model.predict(state)[0]
            state, reward, done, info = env.step(action)
        rewards[possible_times.index(start_time)] = info['cumulative_reward']
    return rewards

def test_all_algo(datapath:str):
    root_dir = dirname(abspath(join(__file__, "../")))
    path = join(root_dir, "data/{}.csv".format(datapath))
    data = pd.read_csv(path, index_col=0, parse_dates=True)
    env = RyeFlexEnv(data)
    possible_times = env.get_possible_start_times()
    #从possible_times中随机采样100个时间点
    possible_times = random.sample(possible_times, 100)

    #constant_rewards = test_constant(possible_times,env)
    simple_rewards = test_simple(possible_times,env)
    random_rewards = test_random(possible_times,env)
    ddpg_rewards = test_ddpg(possible_times,env)
    ppo_rewards = test_ppo(possible_times,env)
    sac_rewards = test_sac(possible_times,env)
    td3_rewards = test_td3(possible_times,env)
    tqc_rewards = test_tqc(possible_times,env)
    #droq_rewards = test_droq(possible_times,env)

    df = pd.DataFrame({
        'simple':simple_rewards,
        'random':random_rewards,
        'ppo':ppo_rewards,
        'ddpg':ddpg_rewards,
        'sac':sac_rewards,
        'td3':td3_rewards,
        'tqc':tqc_rewards,
                       })
    
    df.to_csv('all_algo_rewards_on_{}.csv'.format(datapath))
    df.describe().to_csv('all_algo_rewards_on_{}_describe.csv'.format(datapath))

def test():
    test_all_algo('test')
    test_all_algo('train')

if __name__ == "__main__":
    test()

