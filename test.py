import os
import cv2
import time 
import numpy as np
from vizdoom import *
from vizdoom import gymnasium_wrapper
from gymnasium import Env
from gymnasium.spaces import Discrete, Box
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

import warnings

class VizDoomGym(Env): 
    # Function that is called when we start the env
    def __init__(self, render=False, scenario=str): 
        # Inherit from Env
        super().__init__()
        # Setup the game 
        self.game = DoomGame() # type: ignore
        self.game.load_config(scenario)
        
        # Render frame logic
        if render == False: 
            self.game.set_window_visible(False)
        else:
            self.game.set_window_visible(True)
        
        # Start the game 
        self.game.init()
        
        # Create the action space and observation space
        self.observation_space = Box(low=0, high=255, shape=(100,160,1), dtype=np.uint8) 
        self.action_space = Discrete(3)
        
    # This is how we take a step in the environment
    def step(self, action):
        # Specify action and take step 
        actions = np.identity(3)
        reward = self.game.make_action(actions[action], 4) 
        
        # Get all the other stuff we need to retun 
        if self.game.get_state(): 
            state = self.game.get_state().screen_buffer
            state = self.grayscale(state)
            ammo = self.game.get_state().game_variables[0]
            info = ammo
        else: 
            state = np.zeros(self.observation_space.shape)
            info = 0 
        
        info = {"info":info}
        done = self.game.is_episode_finished()
        
        return state, reward, done, info 
    
    # Define how to render the game or environment 
    def render(): 
        pass
    
    # What happens when we start a new game 
    def reset(self): 
        self.game.new_episode()
        state = self.game.get_state().screen_buffer
        return self.grayscale(state)
    
    # Grayscale the game frame and resize it 
    def grayscale(self, observation):
        # gray = cv2.cvtColor(np.moveaxis(observation, 0, -1), cv2.COLOR_BGR2GRAY)
        processed_image = np.moveaxis(observation, 0, -1)
        ndim = processed_image.ndim
        
        if ndim == 3 and processed_image.shape[-1] == 3:
            gray = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
        elif ndim == 3 and processed_image.shape[-1] == 1:
            gray = processed_image.squeeze()
        else:
            gray = processed_image

        resize = cv2.resize(gray, (160,100), interpolation=cv2.INTER_CUBIC)
        state = np.reshape(resize, (100,160,1))
        return state
    
    # Call to close down the game
    def close(self): 
        self.game.close()

# To run sample tests, create directory shown below and put .cfg files and .wad files
scenarios_path = './VizDoom/scenarios/'
scenarios = os.listdir(scenarios_path)
scenarios_cfg = [scenarios_path + file for file in scenarios if file.endswith(".cfg")]

warnings.filterwarnings(
    "ignore"
)

for file in scenarios_cfg[11:]:
    env = VizDoomGym(True, file)
    model = PPO.load('Basic100k.zip', env)
    reward_sum = 0
    iter = 1
    print(f'\nEvaluation begin: {file}')
    for episode in range(iter): 
        obs = env.reset()
        done = False
        total_reward = 0
        while not done: 
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
            total_reward += reward
        print(f'Total Reward for episode {total_reward} is {episode}')
        reward_sum += total_reward
        time.sleep(2)

    print(f'Average reward after {iter}: {round(reward_sum / iter, 2)}')
    print(f'Evaluation ended: {file}')
    print('============================')
    env.close()