from sbx import DDPG, PPO, SAC, TD3, CrossQ, TQC, DroQ
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np
import pandas as pd
from os.path import abspath, dirname, join
from env_allow_forcast import RyeFlexEnv, RyeFlexEnvEpisodePlotter, ForcastAgent
import random
from test_sbx_model import *
import math

def train_sbx_ddpg(timesteps):
    root_dir = dirname(abspath(join(__file__, "../")))
    data = pd.read_csv(join(root_dir, "data/train.csv"), index_col=0, parse_dates=True)
    env = RyeFlexEnv(data)

    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.5 * np.ones(n_actions))

    model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1)
    model.learn(total_timesteps=timesteps)
    model.save("sbx_ddpg_rye_flex_env_allow_forcast")

def train_sbx_ppo(timeseteps):
    root_dir = dirname(abspath(join(__file__, "../")))
    data = pd.read_csv(join(root_dir, "data/train.csv"), index_col=0, parse_dates=True)
    env = RyeFlexEnv(data)

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=timeseteps)
    model.save("sbx_ppo_rye_flex_env")

def train_sbx_sac(timeseteps):
    root_dir = dirname(abspath(join(__file__, "../")))
    data = pd.read_csv(join(root_dir, "data/train.csv"), index_col=0, parse_dates=True)
    env = RyeFlexEnv(data)

    model = SAC("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=timeseteps)
    model.save("sbx_sac_rye_flex_env")

def train_sbx_td3(timeseteps):
    root_dir = dirname(abspath(join(__file__, "../")))
    data = pd.read_csv(join(root_dir, "data/train.csv"), index_col=0, parse_dates=True)
    env = RyeFlexEnv(data)

    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.3 * np.ones(n_actions))

    model = TD3("MlpPolicy", env, action_noise=action_noise, verbose=1)
    model.learn(total_timesteps=timeseteps)
    model.save("sbx_td3_rye_flex_env_allow_forcast")

def train_sbx_crossq(timeseteps):
    root_dir = dirname(abspath(join(__file__, "../")))
    data = pd.read_csv(join(root_dir, "data/train.csv"), index_col=0, parse_dates=True)
    env = RyeFlexEnv(data)

    model = CrossQ("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=timeseteps)
    model.save("sbx_crossq_rye_flex_env")

def train_sbx_tqc(timeseteps):
    root_dir = dirname(abspath(join(__file__, "../")))
    data = pd.read_csv(join(root_dir, "data/train.csv"), index_col=0, parse_dates=True)
    env = RyeFlexEnv(data)

    model = TQC("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=timeseteps)
    model.save("sbx_tqc_rye_flex_env")

def train_sbx_droq(timeseteps):
    root_dir = dirname(abspath(join(__file__, "../")))
    data = pd.read_csv(join(root_dir, "data/train.csv"), index_col=0, parse_dates=True)
    env = RyeFlexEnv(data)

    model = DroQ("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=timeseteps)
    model.save("sbx_droq_rye_flex_env")

def continue_train_ddpg(timeseteps,env):
    model = DDPG.load("sbx_ddpg_rye_flex_env")
    model.set_env(env)
    model.learn(total_timesteps=timeseteps)
    model.save("sbx_ddpg_rye_flex_env")

def continue_train_ppo(timeseteps,env):
    model = PPO.load("sbx_ppo_rye_flex_env")
    model.set_env(env)
    model.learn(total_timesteps=timeseteps)
    model.save("sbx_ppo_rye_flex_env")

def continue_train_sac(timeseteps,env):
    model = SAC.load("sbx_sac_rye_flex_env")
    model.set_env(env)
    model.learn(total_timesteps=timeseteps)
    model.save("sbx_sac_rye_flex_env")

def continue_train_td3(timeseteps,env):
    model = TD3.load("sbx_td3_rye_flex_env")
    model.set_env(env)
    model.learn(total_timesteps=timeseteps)
    model.save("sbx_td3_rye_flex_env")

def continue_train_tqc(timeseteps,env):
    model = TQC.load("sbx_tqc_rye_flex_env")
    model.set_env(env)
    model.learn(total_timesteps=timeseteps)
    model.save("sbx_tqc_rye_flex_env")

def continue_train_crossq(timeseteps,env):
    model = CrossQ.load("sbx_crossq_rye_flex_env")
    model.set_env(env)
    model.learn(total_timesteps=timeseteps)
    model.save("sbx_crossq_rye_flex_env")

def train_sac_by_compare():
    root_dir = dirname(abspath(join(__file__, "../")))
    data = pd.read_csv(join(root_dir, "data/train.csv"), index_col=0, parse_dates=True)
    env = RyeFlexEnv(data)

    possible_times = env.get_possible_start_times()
    #从possible_times中随机采样100个时间点
    possible_times = random.sample(possible_times, 100)
    dppg_rewards = test_ddpg(possible_times,env)
    mean_ddpg_rewards = np.mean(dppg_rewards)
    min_differece = math.inf

    for i in range(40):
        continue_train_sac(2000,env)
        sac_rewards = test_sac(possible_times,env)
        mean_sac_rewards = np.mean(sac_rewards)
        differece = mean_sac_rewards - mean_ddpg_rewards
        if abs(differece) < min_differece:
            min_differece = abs(differece)

        if (min_differece < 2000) and (abs(differece) > 2000):
            print(i)
            break
        
        if differece >= 0:
            model = SAC.load("sbx_sac_rye_flex_env")
            model.save("sbx_sac_rye_flex_env_best")
            print("sac:{} > ddpg:{}".format(mean_sac_rewards,mean_ddpg_rewards))
            print(i)
            break

    print("min_differece:{}".format(min_differece))

def train_ppo_by_compare():
    root_dir = dirname(abspath(join(__file__, "../")))
    data = pd.read_csv(join(root_dir, "data/train.csv"), index_col=0, parse_dates=True)
    env = RyeFlexEnv(data)

    possible_times = env.get_possible_start_times()
    #从possible_times中随机采样100个时间点
    possible_times = random.sample(possible_times, 100)
    dppg_rewards = test_ddpg(possible_times,env)
    mean_ddpg_rewards = np.mean(dppg_rewards)
    min_differece = math.inf

    for i in range(40):
        continue_train_ppo(2000,env)
        ppo_rewards = test_ppo(possible_times,env)
        mean_ppo_rewards = np.mean(ppo_rewards)
        differece = mean_ppo_rewards - mean_ddpg_rewards
        if abs(differece) < min_differece:
            min_differece = abs(differece)

        if (min_differece < 2000) and (abs(differece) > 2000):
            print(i)
            break
        
        if differece >= 0:
            model = PPO.load("sbx_ppo_rye_flex_env")
            model.save("sbx_ppo_rye_flex_env_best")
            print("ppo:{} > ddpg:{}".format(mean_ppo_rewards,mean_ddpg_rewards))
            print(i)
            break

    print("min_differece:{}".format(min_differece))

def train_td3_by_compare():
    root_dir = dirname(abspath(join(__file__, "../")))
    data = pd.read_csv(join(root_dir, "data/train.csv"), index_col=0, parse_dates=True)
    env = RyeFlexEnv(data)

    possible_times = env.get_possible_start_times()
    #从possible_times中随机采样100个时间点
    possible_times = random.sample(possible_times, 100)
    dppg_rewards = test_ddpg(possible_times,env)
    mean_ddpg_rewards = np.mean(dppg_rewards)
    min_differece = math.inf

    for i in range(40):
        continue_train_td3(2000,env)
        td3_rewards = test_td3(possible_times,env)
        mean_td3_rewards = np.mean(td3_rewards)
        differece = mean_td3_rewards - mean_ddpg_rewards
        if abs(differece) < min_differece:
            min_differece = abs(differece)

        if (min_differece < 2000) and (abs(differece) > 2000):
            print(i)
            break
        
        if differece >500:
            model = TD3.load("sbx_td3_rye_flex_env")
            model.save("sbx_td3_rye_flex_env_best")
            print("td3:{} > ddpg:{}".format(mean_td3_rewards,mean_ddpg_rewards))
            print(i)
            break

    print("min_differece:{}".format(min_differece))    

def train_tqc_by_compare():
    root_dir = dirname(abspath(join(__file__, "../")))
    data = pd.read_csv(join(root_dir, "data/train.csv"), index_col=0, parse_dates=True)
    env = RyeFlexEnv(data)

    possible_times = env.get_possible_start_times()
    #从possible_times中随机采样100个时间点
    possible_times = random.sample(possible_times, 100)
    dppg_rewards = test_td3(possible_times,env)
    mean_ddpg_rewards = np.mean(dppg_rewards)
    min_differece = math.inf

    for i in range(6):
        continue_train_tqc(2000,env)
        tqc_rewards = test_tqc(possible_times,env)
        mean_tqc_rewards = np.mean(tqc_rewards)
        differece = mean_tqc_rewards - mean_ddpg_rewards
        if abs(differece) < min_differece:
            min_differece = abs(differece)

        if (min_differece < 2000) and (abs(differece) > 2000):
            print(i)
            break
        
        if differece >= 0:
            model = TQC.load("sbx_tqc_rye_flex_env")
            model.save("sbx_tqc_rye_flex_env_best")
            print("tqc:{} > ddpg:{}".format(mean_tqc_rewards,mean_ddpg_rewards))
            print(i)
            break

    print("min_differece:{}".format(min_differece))

def train():
    #train_sbx_ddpg(32000)
    #train_sbx_ppo(2000) min_differece: 19125.156
    #train_sbx_sac(2000)
    train_sbx_td3(20000)
    #train_sbx_crossq(2000)
    #train_sbx_tqc(8000) #18000次
    #train_sbx_droq(timeseteps)
    #continue_train_ddpg(2000)
    """
    root_dir = dirname(abspath(join(__file__, "../")))
    data = pd.read_csv(join(root_dir, "data/train.csv"), index_col=0, parse_dates=True)
    env = RyeFlexEnv(data)

    continue_train_ppo(1000,env)
    continue_train_sac(1000,env)
    continue_train_td3(1000,env)
    continue_train_crossq(1000,env)
    """
    #train_sac_by_compare()
    #train_ppo_by_compare()
    #train_td3_by_compare()
    #train_tqc_by_compare()

def test_ddpg():
    plotter = RyeFlexEnvEpisodePlotter()

    # 加载并运行模型
    model = DDPG.load("sbx_td3_rye_flex_env_allow_forcast")
    root_dir = dirname(abspath(join(__file__, "../")))
    data = pd.read_csv(join(root_dir, "data/test.csv"), index_col=0, parse_dates=True)
    env = RyeFlexEnv(data)
    state = env.reset(start_time=datetime(2021, 2, 1, 0, 0))
    info = {}
    done = False

    while not done:
        action = model.predict(state)[0]

        state, reward, done, info = env.step(action)

        plotter.update(info)

    print(f"Your score is: {info['cumulative_reward']} NOK")
    plotter.plot_episode()

def test_forcast_agent():
    plotter = RyeFlexEnvEpisodePlotter()

    # 加载并运行模型
    root_dir = dirname(abspath(join(__file__, "../")))
    data = pd.read_csv(join(root_dir, "data/test.csv"), index_col=0, parse_dates=True)
    env = RyeFlexEnv(data)
    state = env.reset(start_time=datetime(2021, 2, 1, 0, 0))
    info = {}
    done = False
    model = ForcastAgent(env.action_space)

    while not done:
        action = model.get_action(state)

        state, reward, done, info = env.step(action)

        plotter.update(info)

    print(f"Your score is: {info['cumulative_reward']} NOK")
    plotter.plot_episode()


if __name__ == "__main__":
    #train()
    #test_ddpg()
    test_forcast_agent()