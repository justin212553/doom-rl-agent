import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import cv2
import numpy as np
from vizdoom import *
from gymnasium import Env
from gymnasium.spaces import Discrete, Box

class VizDoomGym(Env): 
    def __init__(self, render=False, scenario=str): 
        # Inherit from Env to interract with SB3
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

    def step(self, action):
        # Specify action and take step
        actions = np.identity(3)
        reward = self.game.make_action(actions[action], 4) 
        
        # Get all game states
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
        truncated = False
        
        return state, reward, done, truncated, info
    
    # Define how to render the game or environment 
    def render(self): 
        pass
    
    # What happens when we start a new game 
    def reset(self, seed=None, options=None): 
        super().reset(seed=seed) 

        self.game.new_episode()
        state = self.game.get_state().screen_buffer
        initial_observation = self.grayscale(state)
        info = {}
        
        return initial_observation, info
    
    # Grayscale the game frame and resize it 
    def grayscale(self, observation):
        gray = cv2.cvtColor(np.moveaxis(observation, 0, -1), cv2.COLOR_BGR2GRAY)
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

