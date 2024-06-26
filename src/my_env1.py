import logging
from datetime import datetime, timedelta
from random import randrange, seed
from typing import Any, Dict, List, Optional, Tuple, cast, Union
from dataclasses import dataclass
from gym import spaces
from stable_baselines3 import DDPG, PPO, SAC, TD3, A2C
from stable_baselines3.common.noise import NormalActionNoise

import matplotlib.pyplot as plt
import gym
import numpy as np
import pandas as pd
import math
import random


class Action:
    def __init__(self):
        self.battery = 0
        self.hydrogen = 0
        self._battery_charge_max = 400
        self._hydrogen_charge_max = 55
        self._battery_discharge_max = 400
        self._hydrogen_discharge_max = 100

    def update(self, action: np.ndarray):
        self.battery = action[0]
        self.hydrogen = action[1]

    def into_array(self):
        return np.array([self.battery, self.hydrogen], dtype=np.float64)

    def get_max(self):
        return np.array([self._battery_charge_max, self._hydrogen_charge_max])

    def get_min(self):
        return np.array([-self._battery_discharge_max, -self._hydrogen_discharge_max])
    
    def clip(self):
        b = self.battery
        h = self.hydrogen
        b = min(b, self._battery_charge_max)
        b = max(b, -self._battery_discharge_max)
        h = min(h, self._hydrogen_charge_max)
        h = max(h, -self._hydrogen_discharge_max)


class State:
    def __init__(self):
        self.consumption = 0
        self.pv_production = 0
        self.wind_production = 0
        self.spot_market_price = 0
        self.battery_storage = 0
        self.hydrogen_storage = 0
        self.grid_import = 0
        self.energy_shotage = 0

    def into_array(self):
        return np.array(
            [
                self.consumption,
                self.pv_production,
                self.wind_production,
                self.spot_market_price,
                self.battery_storage,
                self.hydrogen_storage,
                self.grid_import,
                self.energy_shotage,
            ],
            dtype=np.float64,
        )
    
    def get_max_state(self):
        return np.array([math.inf, math.inf, math.inf, math.inf, 500,1670, math.inf, math.inf])
    
    def get_min_state(self):
        return np.array([0, 0, 0, 0, 0, 0, 0, 0])
    
    def update(self, state: np.ndarray):
        self.consumption = state[0]
        self.pv_production = state[1]
        self.wind_production = state[2]
        self.spot_market_price = state[3]
        self.battery_storage = state[4]
        self.hydrogen_storage = state[5]
        self.grid_import = state[6]
        self.energy_shotage = state[7]

    def reset(self):
        self.consumption = 0
        self.pv_production = 0
        self.wind_production = 0
        self.spot_market_price = 0
        self.battery_storage = 0
        self.hydrogen_storage = 0
        self.grid_import = 0
        self.energy_shotage = 0
    
class Battery:
    def __init__(self, capacity=500, efficiency=0.85):
        self.capacity = capacity
        self.efficiency = efficiency
        self.the_charge = 0

    def take_action(self, action:np.ndarray):
        action = action[0]
        b = 0.0
        if action >= 0.0:
            b = self.charge(action)
        else:
            b = - self.discharge(-action)

        return b


    def charge(self, charge):
        new_charge = min(self.the_charge + charge*self.efficiency, self.capacity)
        actual_charge = new_charge - self.the_charge
        self.the_charge = new_charge
        return actual_charge

    def discharge(self, discharge):
        new_charge = max(self.the_charge - discharge, 0)
        actual_discharge = self.the_charge - new_charge
        self.the_charge = new_charge
        return actual_discharge

    def reset(self):
        self.the_charge = 0
    
class Hydrogen:
    def __init__(self, capacity=1670, efficiency=0.35):
        self.capacity = capacity
        self.efficiency = efficiency
        self.the_charge = 0

    def take_action(self, action:np.ndarray):
        action = action[1]
        h = 0
        if action > 0:
            h = self.charge(action)
        else:
            h = - self.discharge(-action)
        
        return h

    def charge(self, charge):
        new_charge = min(self.the_charge + charge*self.efficiency, self.capacity)
        actual_charge = new_charge - self.the_charge
        self.the_charge = new_charge
        return actual_charge
    
    def discharge(self, discharge):
        new_charge = max(self.the_charge - discharge, 0)
        actual_discharge = self.the_charge - new_charge
        self.the_charge = new_charge
        return actual_discharge
    
    def reset(self):
        self.the_charge = 0

class SolarPanel:
    def __init__(self, data):
        self.data = data

    def get_production(self, time):
        return self.data[time]

class WindTurbine:
    def __init__(self, data):
        self.data = data

    def get_production(self, time):
        return self.data[time]
    
class SpotMarket:
    def __init__(self, data):
        self.data = data

    def get_price(self, time):
        return self.data[time]
    
class Load:
    def __init__(self, data):
        self.data = data

    def get_consumption(self, time):
        return self.data[time]

class Clock:
    def __init__(self, data, episode=timedelta(days=30),resolution=timedelta(hours=1)):
        self.time_table = data
        self.episode = episode
        self.resolution = resolution
        self.start_time = self.time_table[0]
        self.end_time = self.time_table[data.shape[0]-1]
        self.current_time = self.start_time
        self.episode_end_time = self.current_time + self.episode

    def tick(self):
        self.current_time += self.resolution
        if self.current_time > self.end_time:
            self.reset()
            return True
        return False

    def is_episode_end(self):
        return self.current_time == self.episode_end_time
    
    def is_end(self):
        return self.current_time == self.end_time
    
    def reset(self,time=None):
        if time is None:
            random_index = random.randint(0, len(self.time_table)-721)
            index = random_index//720
            self.current_time = self.time_table[index]
        else:
            self.current_time = time
        
        self.episode_end_time = self.current_time + self.episode

    def get_time(self):
        return self.current_time
    
    def get_time_index(self):
        return self.time_table.tolist().index(self.current_time)
    
class Data:
    def __init__(self, data: pd.DataFrame):
        self.data = data
    
    @staticmethod
    def load_data(path):
        path = "data/{}.csv".format(path)
        data = pd.read_csv(path)
        return Data(data)

    def get_time(self):
        time = self.data.time.values
        for i in range(len(time)):
            time[i] = datetime.strptime(time[i], '%Y-%m-%d %H:%M:%S')
        return time
    
    def get_consumption(self):
        consumption = self.data.consumption.values.astype(np.float64)
        consumption[consumption < 0] *= -1
        return consumption
    
    def get_pv_production(self):
        pv_production = self.data.pv_production.values.astype(np.float64)
        pv_production[pv_production < 0] *= -1
        return pv_production
    
    def get_wind_production(self):
        wind_production = self.data.wind_production.values.astype(np.float64)
        wind_production[wind_production < 0] *= -1
        return wind_production
    
    def get_spot_market_price(self):
        spot_market_price = self.data.spot_market_price.values.astype(np.float64)
        spot_market_price[spot_market_price < 0] *= -1
        return spot_market_price
    

class MicroGrid:
    def __init__(self, data: Data):
        self.state = State()
        self.action = Action()
        self.baterry = Battery()
        self.hydrogen = Hydrogen()
        self.solar_panel = SolarPanel(data.get_pv_production())
        self.wind_turbine = WindTurbine(data.get_wind_production())
        self.spot_market = SpotMarket(data.get_spot_market_price())
        self.load = Load(data.get_consumption())
        self.clock = Clock(data.get_time())
        self.cost = 0
        self.reward = 0
        self.culmulative_cost = 0
        self.culmulative_reward = 0
        self.reset()

    def take_action(self):
        action = self.action.into_array()
        time = self.clock.get_time_index()
        b = self.baterry.take_action(action)
        h = self.hydrogen.take_action(action)
        p = self.solar_panel.get_production(time)
        w = self.wind_turbine.get_production(time)
        c = self.load.get_consumption(time)
        energe_shotage = c - p - w
        prize = 0
        if b + h > 0 and p + w > c:
            prize = b + h

        import_from_grid = c + b + h - p - w
        import_from_grid = max(import_from_grid, 0)
        price = self.spot_market.get_price(time)
        self.cost = import_from_grid * price
        self.reward = price * prize
        self.culmulative_reward += self.reward
        self.culmulative_cost += self.cost
        self.state.update(np.array([c, p, w, price, self.baterry.the_charge, self.hydrogen.the_charge, import_from_grid, energe_shotage]))
        return np.array([b, h], dtype=np.float64)

    def reset(self,time=None):
        self.clock.reset(time)
        self.baterry.reset()
        self.hydrogen.reset()
        self.cost = 0
        self.reward = 0
        self.culmulative_cost = 0
        self.culmulative_reward = 0
        time = self.clock.get_time_index()
        c = self.load.get_consumption(time)
        p = self.solar_panel.get_production(time)
        w = self.wind_turbine.get_production(time)
        price = self.spot_market.get_price(time)
        energe_shotage = max(c - p - w, 0)
        self.state.update(np.array([c, p, w, price, self.baterry.the_charge, self.hydrogen.the_charge, 0, energe_shotage]))


class WorkEnv(gym.Env):
    def __init__(self,data: Data):
        self.microgrid = MicroGrid(data)
        self.action_space = spaces.Box(
            low=self.microgrid.action.get_min(),
            high=self.microgrid.action.get_max(),
            shape=(2,),
            dtype=np.float64,
        )
        self.observation_space = spaces.Box(
            low=self.microgrid.state.get_min_state(),
            high=self.microgrid.state.get_max_state(),
            shape=(8,),
            dtype=np.float64,
        )

    def step(self, action: np.ndarray):
        self.microgrid.clock.tick()
        self.microgrid.action.update(action)
        self.microgrid.action.clip()
        actual_action = self.microgrid.take_action()
        state = self.microgrid.state.into_array()
        reward = - self.microgrid.cost
        done = False
        info = {
            "cost": self.microgrid.cost,
            "culmulative_cost": self.microgrid.culmulative_cost,
            "time": self.microgrid.clock.get_time(),
            "state": state,
            "action": action,
            "actual_action": actual_action,
            "reward": self.microgrid.reward,
            "culmulative_reward": self.microgrid.culmulative_reward,
        }
        if self.microgrid.clock.is_episode_end():
            done = True
            self.reset()

        return state, reward, done, info

    def reset(self, time=None):
        self.microgrid.reset(time)
        return self.microgrid.state.into_array()

class Plotter:
    def __init__(self):
        self.time:List[datetime] = []
        self.cost:List[Dict[str,float]] = []
        self.state:List[Dict[str,float]] = []
        self.action:List[Dict[str,float]] = []

    def update(self, info):
        self.time.append(info["time"])
        self.cost.append({
            "cost": info["cost"],
            "culmulative_cost": info["culmulative_cost"],
            "reward": info["reward"],
            "culmulative_reward": info["culmulative_reward"],
        })
        self.state.append({
            "comsumption": info["state"][0],
            "pv_production": info["state"][1],
            "wind_production": info["state"][2],
            "spot_market_price": info["state"][3],
            "battery_storage": info["state"][4],
            "hydrogen_storage": info["state"][5],
            "energy_storage": info["state"][4] + info["state"][5],
            "grid_import": info["state"][6],
            "energy_shortage": info["state"][7],
        })
        self.action.append({
            "battery": info["action"][0],
            "hydrogen": info["action"][1],
            "actual_battery": info["actual_action"][0],
            "actual_hydrogen": info["actual_action"][1],
        })

    def plot(self,name=""):
        state = pd.DataFrame(self.state, index=self.time)
        action = pd.DataFrame(self.action, index=self.time)
        cost = pd.DataFrame(self.cost, index=self.time)
        state.plot(subplots=True,title="State-{}".format(name))
        plt.savefig("state-{}.png".format(name),dpi=300)
        action.plot(subplots=True,title="Action-{}".format(name))
        plt.savefig("action-{}.png".format(name),dpi=300)
        cost.plot(subplots=True,title="Cost-{}".format(name))
        plt.savefig("cost-{}.png".format(name),dpi=300)
        #plt.show()

    def reset(self):
        self.time = []
        self.cost = []
        self.state = []
        self.action = []

class TestAlgo:
    def __init__(self, model, env,potters=Plotter()):
        self.model = model
        self.env = env
        self.plotter = potters
        self.cost = 0
        self.reward = 0
        self.time_table = []
        self.costs = []
        self.rewards = []

    def test(self,name=""):
        state = self.env.reset()
        done = False
        list1 = []
        while not done:
            action, _states = self.model.predict(state)
            state, reward, done, info = self.env.step(action)
            self.plotter.update(info)
        
        self.plotter.plot(name)
        self.cost = info["culmulative_cost"]
        self.reward = info["culmulative_reward"]

    def test_no_plot(self,time=None):
        state = self.env.reset(time)
        done = False
        while not done:
            action, _states = self.model.predict(state)
            state, reward, done, info = self.env.step(action)
        
        cost = info["culmulative_cost"]
        reward = info["culmulative_reward"]

        self.cost += cost
        self.reward += reward

    def test_multiple(self):
        for time in self.time_table:
            self.test_no_plot(time)
            self.costs.append(self.cost)
            self.rewards.append(self.reward)
            self.cost = 0
            self.reward = 0
    
    def init_time_table(self,k=6):
        time_table = self.env.microgrid.clock.time_table
        t = []
        for i in range(k):
            t.append(time_table[i*24])
        self.time_table = t
    
    def reset(self):
        self.cost = 0
        self.reward = 0
        self.costs = []
        self.rewards = []
        self.plotter.reset()

    def print_cost(self):
        print(self.cost)
        print(self.reward)


class RandomAgent:
    def __init__(self,action_space):
        self.action_space = action_space

    def predict(self, state):
        return self.action_space.sample(), None
    
class SimpleAgent:
    def __init__(self, action_space):
        self.action_space = action_space

    def predict(self, state):
        action = np.array([0,0], dtype=np.float64)
        if state[7] > 20:
            action[0] = -10
            action[1] = -10
        else:
            action[0] = 10
            action[1] = 10
        return action, None
    
class ConstantAgent:
    def __init__(self, action_space):
        self.action_space = action_space

    def predict(self, state):
        action = np.array([0,0], dtype=np.float64)
        return action, None
        
def test_multi_model():
    data = Data.load_data("test")
    env = WorkEnv(data)
    model0 = ConstantAgent(env.action_space)
    model1 = SimpleAgent(env.action_space)
    model2 = RandomAgent(env.action_space)
    model3 = DDPG.load("ddpg")
    model4 = A2C.load("a2c")
    model5 = PPO.load("ppo")
    model6 = SAC.load("sac")

    test = TestAlgo(model0, env)
    test.init_time_table()
    test.test_multiple()
    c0 = test.costs
    r0 = test.rewards
    test.reset()
    
    test.model = model1
    test.test_multiple()
    c1 = test.costs
    r1 = test.rewards
    test.reset()

    test.model = model2
    test.test_multiple()
    c2 = test.costs
    r2 = test.rewards
    test.reset()

    test.model = model3
    test.test_multiple()
    c3 = test.costs
    r3 = test.rewards
    test.reset()

    test.model = model4
    test.test_multiple()
    c4 = test.costs
    r4 = test.rewards
    test.reset()

    test.model = model5
    test.test_multiple()
    c5 = test.costs
    r5 = test.rewards
    test.reset()

    test.model = model6
    test.test_multiple()
    c6 = test.costs
    r6 = test.rewards
    test.reset()

    df = pd.DataFrame({
        "constant": c0,
        "simple": c1,
        "random": c2,
        "ddpg": c3,
        "a2c": c4,
        "ppo": c5,
        "sac": c6,
    })
    df.plot()
    plt.savefig("cost.png",dpi=300)
    df.to_csv("cost.csv")
    df = df.describe()
    df.to_csv("cost_describe.csv")

    df = pd.DataFrame({
        "constant": r0,
        "simple": r1,
        "random": r2,
        "ddpg": r3,
        "a2c": r4,
        "ppo": r5,
        "sac": r6,
    })
    df.plot()
    plt.savefig("reward.png",dpi=300)
    df.to_csv("reward.csv")
    df = df.describe()
    df.to_csv("reward_describe.csv")
        
def plot_cost():
    df = pd.read_csv("cost.csv")
    df.plot()
    plt.show()


def train_model():
    data = Data.load_data("train")
    env = WorkEnv(data)

    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=100 * np.ones(n_actions))

    model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1)
    model.learn(total_timesteps=20000)
    model.save("ddpg")

    model = A2C("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=20000)
    model.save("a2c")

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=20000)
    model.save("ppo")

    model = SAC("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=20000)
    model.save("sac")

def test_model(i):
    data = Data.load_data("test")
    env = WorkEnv(data)
    model = DDPG.load("ddpg")
    test = TestAlgo(model, env)
    test.test("ddpg-{}".format(i))
    test.print_cost()


    model = A2C.load("a2c")
    test.model = model
    test.reset()
    test.test("a2c-{}".format(i))
    test.print_cost()

    model = PPO.load("ppo")
    test.model = model
    test.reset()
    test.test("ppo-{}".format(i))
    test.print_cost()

    model = SAC.load("sac")
    test.model = model
    test.reset()
    test.test("sac-{}".format(i))
    test.print_cost()

    model = RandomAgent(env.action_space)
    test.model = model
    test.reset()
    test.test("random-{}".format(i))
    test.print_cost()

    model = SimpleAgent(env.action_space)
    test.model = model
    test.reset()
    test.test("simple-{}".format(i))
    test.print_cost()

    model = ConstantAgent(env.action_space)
    test.model = model
    test.reset()
    test.test("constant-{}".format(i))
    test.print_cost()

if __name__ == "__main__":
    train_model()
    test_multi_model()
    test_model(1)

