import logging
from datetime import datetime, timedelta
from random import randrange, seed
from typing import Any, Dict, List, Optional, Tuple, cast, Union
from dataclasses import dataclass
from gym import spaces

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
        self.pv_next = 0
        self.wind_next = 0

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
                self.pv_next,
                self.wind_next,
            ],
            dtype=np.float64,
        )

    def get_max_state(self):
        return np.array(
            [
                math.inf,
                math.inf,
                math.inf,
                math.inf,
                500,
                1670,
                math.inf,
                math.inf,
                math.inf,
                math.inf,
            ]
        )

    def get_min_state(self):
        return np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    def update(self, state: np.ndarray):
        self.consumption = state[0]
        self.pv_production = state[1]
        self.wind_production = state[2]
        self.spot_market_price = state[3]
        self.battery_storage = state[4]
        self.hydrogen_storage = state[5]
        self.grid_import = state[6]
        self.energy_shotage = state[7]
        self.pv_next = state[8]
        self.wind_next = state[9]

    def reset(self):
        self.consumption = 0
        self.pv_production = 0
        self.wind_production = 0
        self.spot_market_price = 0
        self.battery_storage = 0
        self.hydrogen_storage = 0
        self.grid_import = 0
        self.energy_shotage = 0
        self.pv_next = 0
        self.wind_next = 0


class Battery:
    def __init__(self, capacity=500, efficiency=0.85):
        self.capacity = capacity
        self.efficiency = efficiency
        self.the_charge = 0

    def take_action(self, action: np.ndarray):
        action = action[0]
        b = 0.0
        if action >= 0.0:
            b = self.charge(action)
        else:
            b = -self.discharge(-action)

        return b

    def charge(self, charge):
        new_charge = min(self.the_charge + charge * self.efficiency, self.capacity)
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

    def take_action(self, action: np.ndarray):
        action = action[1]
        h = 0
        if action > 0:
            h = self.charge(action)
        else:
            h = -self.discharge(-action)

        return h

    def charge(self, charge):
        new_charge = min(self.the_charge + charge * self.efficiency, self.capacity)
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

    def get_next_production(self, time):
        if time == len(self.data) - 1:
            return self.data[time]
        return self.data[time + 1]


class WindTurbine:
    def __init__(self, data):
        self.data = data

    def get_production(self, time):
        return self.data[time]

    def get_next_production(self, time):
        if time == len(self.data) - 1:
            return self.data[time]
        return self.data[time + 1]


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
    def __init__(self, data, episode=timedelta(days=30), resolution=timedelta(hours=1)):
        self.time_table = data
        self.episode = episode
        self.resolution = resolution
        self.start_time = self.time_table[0]
        self.end_time = self.time_table[data.shape[0] - 1]
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

    def reset(self, time=None):
        if time is None:
            random_index = random.randint(0, len(self.time_table) - 721)
            index = random_index // 720
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
            time[i] = datetime.strptime(time[i], "%Y-%m-%d %H:%M:%S")
        return time

    def get_consumption(self):
        consumption = self.data.consumption.values.astype(np.float64)
        consumption[consumption < 0] *= -1
        return consumption

    def get_pv_production(self):
        pv_production = self.data.pv_production.values.astype(np.float64)
        pv_production[pv_production < 0] *= -1
        return 3 * pv_production

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
        punish = (abs(action[0]) - abs(b)) ** 2 + (abs(action[1]) - abs(h)) ** 2
        punish = math.sqrt(punish)
        energe_shotage = c - p - w
        prize = 0
        #if b + h <= 0 and c - p - w > 0 and p + w - c - b - h >= 0:
        #    prize = -b - h
        if b + h > 0 and p + w > c:
            prize = b + h

        import_from_grid = c + b + h - p - w
        import_from_grid = max(import_from_grid, 0)
        price = self.spot_market.get_price(time)
        self.cost = import_from_grid * price
        self.reward = prize*price
        self.culmulative_reward += self.reward
        self.culmulative_cost += self.cost
        pv_next = self.solar_panel.get_next_production(time)
        wind_next = self.wind_turbine.get_next_production(time)
        self.state.update(
            np.array(
                [
                    c,
                    p,
                    w,
                    price,
                    self.baterry.the_charge,
                    self.hydrogen.the_charge,
                    import_from_grid,
                    energe_shotage,
                    pv_next,
                    wind_next,
                ]
            )
        )
        return np.array([b, h], dtype=np.float64)

    def reset(self, time=None):
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
        pv_next = self.solar_panel.get_next_production(time)
        wind_next = self.wind_turbine.get_next_production(time)
        energe_shotage = c - p - w
        grind_import = max(c - p - w, 0)
        self.baterry.take_action(np.array([max(p+w-c, 0), 0]))
        self.state.update(
            np.array(
                [
                    c,
                    p,
                    w,
                    price,
                    self.baterry.the_charge,
                    self.hydrogen.the_charge,
                    grind_import,
                    energe_shotage,
                    pv_next,
                    wind_next,
                ]
            )
        )


class WorkEnv(gym.Env):
    def __init__(self, data: Data):
        self.microgrid = MicroGrid(data)
        self.action_space = spaces.Box(
            low=np.array([0,0], dtype=np.float64),
            high=np.array([1,5], dtype=np.float64),
            shape=(2,),
            dtype=np.float64,
        )
        self.observation_space = spaces.Box(
            low=self.microgrid.state.get_min_state(),
            high=self.microgrid.state.get_max_state(),
            shape=(10,),
            dtype=np.float64,
        )

    def step(self, action: np.ndarray):
        pv = self.microgrid.state.pv_next
        wind = self.microgrid.state.wind_next
        c = self.microgrid.state.consumption
        l = action[0]
        r = 1 - l
        k = action[1]
        c_next = c*k
        ac = np.array([l * (pv + wind - c_next), r * (pv + wind - c_next)], dtype=np.float64)
        self.microgrid.clock.tick()
        self.microgrid.action.update(ac)
        self.microgrid.action.clip()
        actual_action = self.microgrid.take_action()
        state = self.microgrid.state.into_array()
        reward = -self.microgrid.cost
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
        self.time: List[datetime] = []
        self.cost: List[Dict[str, float]] = []
        self.state: List[Dict[str, float]] = []
        self.action: List[Dict[str, float]] = []

    def update(self, info):
        self.time.append(info["time"])
        self.cost.append(
            {
                "cost": info["cost"],
                "culmulative_cost": info["culmulative_cost"],
                "reward": info["reward"],
                "culmulative_reward": info["culmulative_reward"],
            }
        )
        self.state.append(
            {
                "comsumption": info["state"][0],
                "pv_production": info["state"][1],
                "wind_production": info["state"][2],
                "spot_market_price": info["state"][3],
                "battery_storage": info["state"][4],
                "hydrogen_storage": info["state"][5],
                "energy_strorage": info["state"][4] + info["state"][5],
                "grid_import": info["state"][6],
                "energy_shortage": info["state"][7],
            }
        )
        self.action.append(
            {
                "battery/(battery+hydrogen)": info["action"][0],
                "forcast_price_rate": info["action"][1],
                "actual_battery": info["actual_action"][0],
                "actual_hydrogen": info["actual_action"][1],
            }
        )

    def plot(self):
        state = pd.DataFrame(self.state, index=self.time)
        action = pd.DataFrame(self.action, index=self.time)
        cost = pd.DataFrame(self.cost, index=self.time)
        state.plot(subplots=True, title="State")
        action.plot(subplots=True, title="Action")
        cost.plot(subplots=True, title="Cost")
        plt.show()
        self.reset()

    def reset(self):
        self.time = []
        self.cost = []
        self.state = []
        self.action = []


class TestAlgo:
    def __init__(self, model, env, potters=Plotter()):
        self.model = model
        self.env = env
        self.plotter = potters
        self.cost = 0
        self.reward = 0

    def test(self):
        state = self.env.reset()
        done = False
        list1 = []
        while not done:
            action, _states = self.model.predict(state)
            state, reward, done, info = self.env.step(action)
            self.plotter.update(info)

        self.plotter.plot()
        self.cost = info["culmulative_cost"]
        self.reward = info["culmulative_reward"]

    def test_no_plot(self):
        state = self.env.reset()
        done = False
        while not done:
            action, _states = self.model.predict(state)
            state, reward, done, info = self.env.step(action)
        self.cost = info["culmulative_cost"]
        self.reward = info["culmulative_reward"]

    def print_cost(self):
        print(self.cost)
        print(self.reward)


class ForcastAgent:
    def __init__(self, action_space):
        self.action_space = action_space

    def predict(self, state):
        state0 = State()
        state0.update(state)
        c = state0.consumption
        pv = state0.pv_production
        wind = state0.wind_production

        r = random.random()
        l = 1 - r
        k = random.random()*5

        ac = np.array([r,k], dtype=np.float64)
        return ac, state0

class Forcast(gym.Env):
    def _init_(self, env):
        self.env = env
        self.action_space = spaces.Box(
            low=np.array([0,0], dtype=np.float64),
            high=np.array([1,math.inf], dtype=np.float64),
            shape=(2,),
            dtype=np.float64,
        )
        self.observation_space = spaces.Box(
            low=np.array([0,0], dtype=np.float64),
            high=np.array([math.inf,math.inf], dtype=np.float64),
            shape=(2,),
            dtype=np.float64,
        )

    def step(self, action: np.ndarray):
        l = action[0]
        r = 1 - l
        c = self.env.state.consumption
        pv = self.env.state.pv_next
        wind = self.env.state.wind_next
        k = action[1]
        c_next = c*k
        ac = np.array([l * (pv + wind - c_next), r * (pv + wind - c_next)], dtype=np.float64)
        state, reward, done, info = self.env.step(ac)

        mystate = np.array([state[8], state[9]], dtype=np.float64)

        return mystate, reward, done, info
    
    def reset(self,time=None):
        state = self.env.reset(time)
        mystate = np.array([state[8], state[9]], dtype=np.float64)
        return mystate
        






class RandomAgent:
    def __init__(self, action_space):
        self.action_space = action_space

    def predict(self, state):
        return self.action_space.sample(), state


class ConstantAgent:
    def __init__(self, action_space):
        self.action_space = action_space

    def predict(self, state):
        return np.array([0.5, 1], dtype=np.float64), state


class SimpleAgent:
    def __init__(self, action_space):
        self.action_space = action_space

    def predict(self, state):
        state0 = State()
        state0.update(state)
        c = state0.consumption
        pv = state0.pv_production
        wind = state0.wind_production

        if c > pv + wind:
            return np.array([-c + pv + wind, -15], dtype=np.float64), state0
        else:
            return (
                np.array(
                    [(2 * random.random() - 1) * 10, (2 * random.random() - 1) * 15],
                    dtype=np.float64,
                ),
                state0,
            )


def test_clock():
    data = Data.load_data("train")
    clock = Clock(data.get_time())
    print(clock.current_time)


def test_env():
    data = Data.load_data("train")
    env = WorkEnv(data)
    plotter = Plotter()
    state = env.reset()
    done = False
    while not done:
        b = random.randint(-100, 100)
        h = random.randint(-100, 100)
        action = np.array([b, h], dtype=np.float64)
        state, reward, done, info = env.step(action)
        plotter.update(info)
    plotter.plot()


def test_action():
    action = Action()

    c = np.array([-83, 63], dtype=np.float64)
    action.update(c)
    action.clip()
    print(action.into_array())

    battery = Battery()
    b = battery.take_action(c)
    print(b)

    hydrogen = Hydrogen()
    h = hydrogen.take_action(c)
    print(h)

    a = np.array([80, 93], dtype=np.float64)
    action.update(a)
    action.clip()
    print(action.into_array())

    b = battery.take_action(a)
    print(b)

    h = hydrogen.take_action(a)
    print(h)


if __name__ == "__main__":
    test_env()
    # test_action()
