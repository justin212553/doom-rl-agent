import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from environment import VizDoomGym

import os
from vizdoom import *
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env

SCENARIOS_PATH = './'
SCENARIOS = os.listdir(SCENARIOS_PATH)
SCENARIOS_CFG = [SCENARIOS_PATH + file for file in SCENARIOS if file.endswith(".cfg")]

ITER = 100

for file in SCENARIOS_CFG:
    env = make_vec_env(lambda: VizDoomGym(False, file), n_envs=1)
    model = DQN.load('./train/train_basic/best_model.zip', env)
    reward_sum = 0
    
    print(f'\nEvaluation begin: {file}')
    for episode in range(ITER): 
        obs = env.reset()
        done = False
        total_reward = 0
        while not done: 
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
            total_reward += reward
        print(f'Total Reward for episode {total_reward} is {episode}')
        reward_sum += total_reward
        
    print(f'Average reward after {ITER}: {round(reward_sum[0] / ITER, 2)}')
    print(f'Evaluation ended: {file}')
    print('============================')
    env.close()