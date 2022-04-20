import torch
import random
import numpy as np
from collections import deque
from AIgame import PaddleGame
from model import LinQNetwork, Trainer
from helper import plot
# Game Parameters
WIDTH, HEIGHT = 450, 500
# Adjust these parameters to modify exploration vs. exploitation
MAX_EXP_RATE = 1
MIN_EXP_RATE = 0.01
EXP_DECAY_RATE = 0.001

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    def __init__(self):
        self.game_count = 0
        self.exp_rate = MAX_EXP_RATE 
        self.dis_rate = 0.9 # This can be adjusted if necessary.
        self.memory = deque(maxlen=MAX_MEMORY)

        # Model: 8 inputs, 3 outputs. hidden layer size can be adjusted, right now set to 256.
        self.model = LinQNetwork(5, 256, 3)
        self.trainer = Trainer(LR, self.dis_rate, self.model) 
        

    def get_state(self, game):
        paddle_x = game.paddle.x
        ball_x = game.ball.x
        ball_y = game.ball.y
        ball_dx, ball_dy = game.ball_speed

        state = np.interp([paddle_x, ball_x, ball_y, ball_dx, ball_dy], [0, 500], [-1,1])
        return state

    def remember(self, state, action, reward, next_state, done):
        # add parameters into memory. If max capacity is reached, it pops from the left (oldest entry)
        self.memory.append((state, action, reward, next_state, done))


    def train_long_memory(self):
        # Train in larger batches
        if len(self.memory) > BATCH_SIZE:
            sample = random.sample(self.memory, BATCH_SIZE)
        else:
            sample = self.memory

             
        # Aggregate all the states, actions etc. together.
        states, actions, rewards, next_states, statuses = zip(*sample)
        self.trainer.train_step(states,actions,rewards,next_states,statuses)
        

    def train_short_memory(self, state, action, reward, next_state, done):
        # calls the train step function to train model on the single state-action information.
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        """
        Sets the parameters for the exploration/exploitation tradeoff. 
        Setup so that the likelihood of exploiting rather than exploring (random move) becomes greater
        as the model is trained.
        """
        action = [0,0,0]
        exploration_threshold = random.uniform(0,1)
        if exploration_threshold < self.exp_rate:
            # Threshold not reached = explore, so take a random action.
            rand_index = random.randint(0,2)
            action[rand_index] = 1
        else:
            single_state = torch.tensor(state, dtype=torch.float) # setup tensor (MD array)
            # Use the model to make a prediction based on the current state
            # returns a tensor of "weights" for the given actions
            prediction = self.model(single_state) 
            # argmax returns a tensor, so get index with item() method.
            best_action_index = torch.argmax(prediction).item()
            action[best_action_index] = 1
            

        # Decay exploration rate: decreases expontentially as number of games played increases.
        self.exp_rate = MIN_EXP_RATE + (MAX_EXP_RATE - MIN_EXP_RATE) * np.exp(-EXP_DECAY_RATE*self.game_count)


        return action

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = PaddleGame(WIDTH, HEIGHT)

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
                agent.model.save()

            print(f"Game: {agent.game_count}, Score: {score}, Record: {record}, Mean Score: {total_score / agent.game_count}")

            # TODO: plot data
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.game_count
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


    

if __name__ == '__main__':
    train()