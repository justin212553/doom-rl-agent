import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from environment import VizDoomGym

import os
from vizdoom import *
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env

CHECKPOINT_DIR = './train/train_basic'
LOG_DIR = './logs/log_basic'

SCENARIOS_PATH = './'
SCENARIOS = os.listdir(SCENARIOS_PATH)
SCENARIOS_CFG = [SCENARIOS_PATH + file for file in SCENARIOS if file.endswith(".cfg")]

env = make_vec_env(lambda: VizDoomGym(False, SCENARIOS_CFG[0]), n_envs=1)
env_eval = make_vec_env(lambda: VizDoomGym(False, SCENARIOS_CFG[0]), n_envs=1)

eval_callback = EvalCallback(
    env_eval, 
    best_model_save_path=CHECKPOINT_DIR,
    log_path=LOG_DIR,
    eval_freq=1000,
    deterministic=True
)

model = DQN(
    'CnnPolicy', 
    env, 
    tensorboard_log=LOG_DIR, 
    verbose=1, 
    learning_rate=0.00005, 
    buffer_size=500000,
    learning_starts=10000,
    target_update_interval=10000,
    exploration_fraction=1,
    exploration_final_eps=0.1
)

model.learn(total_timesteps=300_000, callback=eval_callback)
