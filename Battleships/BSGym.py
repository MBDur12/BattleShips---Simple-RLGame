import Board as bd
import gym
from gym import Env # 
from gym.spaces import Box, Discrete
import numpy as np
import random
import os
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import DQN

class BattleShipsEnv(Env):
    def __init__(self, board_size=10, fleet={}):
        # Setup game board to play on
        self.board_size = board_size
        self.board = bd.Board(self.board_size)
        self.fleet = fleet
        if not self.fleet:
            self.fleet = {"Patrol Boat": 2, "Submarine": 3, "Destroyer": 3, "Battleship": 4, "Carrier": 5}

        for k,v in self.fleet.items():
            self.board.place_ship(k, v)

        # define action space: an integer value representing a square on the board.
        # Squares are numbered left-to-right, top-to-bottom
        self.action_space = Discrete(self.board_size**2)

        self.observation_space = Box(low=-1, high=1, shape=(self.board_size**2,), dtype=np.int8)

    
    def step(self, action):
        # translate integer action into a coordinate on the board.
        x, y = divmod(action, self.board_size)

        reward = 0

        if self.board.board[x,y] == 0:
            reward += 5
        else:
            reward -= 10
        
        # Take the action to change the board
        self.board.hit((x,y))
        self.render()
        # Update the observation to reflect the change (if any)
        observations = []
        for val in np.nditer(self.board.board):
            observations.append(val)
        observations = np.array(observations, dtype=np.int8)

        

        # Include any further/diagnositc information here.
        info = {}

        # Check if all ships are sunk, in which case flag the game as being done.
        if not self.board._ships:
            done = True
        else:
            done = False

        
        return observations, reward, done, info
        

    def render(self):
        # To Do - maybe in pygame or printing in place.
        print("", end="\r")
        self.board.display()

    def reset(self):
        self.board = bd.Board(self.board_size)
        for k,v in self.fleet.items():
            self.board.place_ship(k, v)

        observations = np.zeros(shape=(self.board_size**2), dtype=np.int8)
        return observations

"""
TEST THE ENVIRONMENT

"""
env = BattleShipsEnv()
"""#check_env(env)
episodes = 5
scores = []
for episode in range(1, episodes+1):
    state = env.reset()
    print(env.board._ships)
    done = False
    score = 0

    while not done:
        env.render()
        action = env.action_space.sample()
        new_state, reward, done, info = env.step(action)
        score += reward
    print(f"Episode: {episode}, Score: {score}")
    scores.append(score)
env.close()
print(f"Scores for each: {scores}")"""


log_path = os.path.join("Trainings", "Logs")
