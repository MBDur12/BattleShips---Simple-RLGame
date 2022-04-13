import torch
import random
import numpy as np
from collections import deque
from AIgame import PaddleGame

MAX_MEMORY = 100_000
LR = 0.001

class Agent:
    def __init__(self):
        self.game_count = 0
        self.exp_rate = 0 # between 0 and 1
        self.dis_rate = 0 # ^
        self.memory = deque(maxlen=MAX_MEMORY)
        # TODO: model, trainer
        pass

    def get_state(self, game):
        # state = [direction of ball L,R,U,D + ball L, R + ball Y-distance]


        pass

    def remember(self, state, action, reward, next_state, done):
        pass

    def train_long_memory(self):
        pass

    def train_short_memory(self, state, action, reward, next_state, done):
        pass

    def get_action(self, state):
        pass

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = PaddleGame()

    while True:
        # get old state
        old_state = agent.get_state(game)

        # get move
        best_action = agent.get_action(old_state)

        # take move and get new state
        reward, done, score = game.take_step(best_action)
        new_state = agent.get_state(game)

        # train short memory
        agent.train_short_memory(old_state, best_action, reward, new_state, done)

        # remember for long memory
        agent.remember(old_state, best_action, reward, new_state, done)

        if done:
            # train long memory, plot result
            game.reset()
            agent.game_count += 1
            agent.train_long_memory()

            if score > reward:
                record = score
                # agent.model.save()

            print('Game:', agent.game_count, 'Score:', score, 'Record:', record)

            # TODO: plot data


    pass

if __name__ == '__main__':
    train()