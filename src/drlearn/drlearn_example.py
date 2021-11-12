from __future__ import print_function

from typing import Deque
import gym
import numpy as np
import random 

from itertools import count
import matplotlib.pyplot as plt

from dnetworks.layers.activation import Softmax
from dnetworks.model import NNet
from dnetworks.layers import (
    LinearLayer, 
    LeakyReLU, 
    Sigmoid,
    MSELoss
)
from dnetworks.optimizers import Adam


env_name='CartPole-v1'

# Initialize the environment
gym_env = gym.make(env_name)
n_states = gym_env.observation_space.shape[0]
n_actions = gym_env.action_space.n


model = NNet()

# Create the model structure
model.add(LinearLayer(n_states, 64))
model.add(LeakyReLU())

model.add(LinearLayer(64, 32))
model.add(LeakyReLU())

model.add(LinearLayer(32, 8))
model.add(LeakyReLU())

model.add(LinearLayer(8, n_actions))

# set the loss functions and the optimize method
loss = MSELoss()
optim = Adam(lr=0.001)  

class StatesMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, step_env):
        """Save a transition"""
        self.memory.append(step_env)
        drop_condition = len(self.memory) > self.capacity

        if drop_condition:
            self.memory.pop(0)

    def sample(self, batch_size):
        batch_size = min(batch_size, len(self.memory))

        return random.sample(self.memory, batch_size)


class DRLearn:
    def __init__(
        self, 
        model=None,
        env=None,  
        epsilon=1, 
        gamma=0.9, 
        decay_rate=0.005, 
        min_epsilon=0.1,
    ):
        self.epsilon = epsilon
        self.gamma = gamma
        self.decay_rate = decay_rate
        self.min_epsilon = min_epsilon
        self.memory = StatesMemory(capacity=300)

        # Initialize the environment
        self.env = env
        self.n_states = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.n
        self.model = model

    def take_action(self, state):
        if np.random.rand() < self.epsilon:
            # Choose action randomly
            action = np.random.randint(self.n_actions)
        else:
            # Take action with highest predicted utility given state
            action = np.argmax(self.model.forward(state).T, axis=1)[0]

        return action

    def get_xy(self, steps_sample):

        states = np.array([a[0] for a in steps_sample]).T
        new_states = np.array([a[3] for a in steps_sample]).T

        Q = self.model.forward(states)
        Q_new = self.model.forward(new_states)

        sample_len = len(steps_sample)
        X = np.zeros((sample_len, self.n_states))
        y = np.zeros((sample_len, self.n_actions))
        
        for num, params in enumerate(steps_sample):
            state_r, action_r, reward_r, new_state_r, done_r = params

            target = Q.T[num]
            target[action_r] = reward_r
            # If we're done the utility is simply the reward of executing action a in
            # state s, otherwise we add the expected maximum future reward as well
            if not done_r:
                target[action_r] += self.gamma * np.amax(Q_new.T[num])

            X[num] = state_r
            y[num] = target

        return X.T, y.T

    def play(self, batch_size):
        state = self.env.reset()
        total_reward = 0

        epoch_loss = []

        for num in count():
            action = self.take_action(state.reshape(self.n_states, 1))

            new_state, reward, done, _ = self.env.step(action)
            step_list = [state, action, reward, new_state, done]

            self.memory.push(step_list)
            steps_sample = self.memory.sample(batch_size)

            X, Y = self.get_xy(steps_sample)

            self.model.forward(X)
            cost = model.loss(Y, loss)
            model.backward()
            model.optimize(optim)

            epoch_loss.append(cost)

            total_reward += reward
            state = new_state

            if done: break

        epoch_loss = np.mean(epoch_loss)     
        
        return epoch_loss, total_reward

    def train(self, n_epochs=500, batch_size=32):
        self.epoch_losses = []
        max_reward = 0
        for epoch in range(n_epochs):
            epoch_loss, total_reward = self.play(batch_size)

            max_reward = max(max_reward, total_reward)
            self.epsilon = self.min_epsilon + (1.0 - self.min_epsilon) * np.exp(-self.decay_rate * epoch)
            self.epoch_losses.append(epoch_loss)
            if epoch % 20 == 0:
                print('loss: ', epoch_loss, 'max reward: ', max_reward, 'epoch: ', epoch)
            
            if total_reward==500:
                break


    def play_game(self, n_epochs=10):
        # self.env = gym.wrappers.Monitor(self.env, '/tmp/cartpole-experiment-1', force=True)
        for epoch in range(n_epochs):
            state = self.env.reset()
            total_reward = 0
            while True:
                self.env.render()
                state = state.reshape(self.n_states, 1)
                action = np.argmax(self.model.forward(state).T, axis=1)[0]
                state, reward, done, _ = self.env.step(action)
                total_reward += reward
                if done: break
            print ("%d Reward: %s" % (epoch, total_reward))

        self.env.close()


def main():
    dqn = DRLearn(
        model=model,
        env=gym_env,
        epsilon=0.9, 
        gamma=0.8, 
        decay_rate=0.005,
        min_epsilon=0.1
    )

    print()
    dqn.train(n_epochs=500)
    plt.plot(np.convolve(dqn.epoch_losses, np.ones(20), 'valid') / 20)
    plt.show()
    dqn.play_game(n_epochs=10)
    
if __name__ == "__main__":
    main()