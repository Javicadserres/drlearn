import numpy as np

from itertools import count
from utils import Memory


class DRLearn:
    """Deep Reinforcement learning class.

    Parameters
    ----------
    model : dnetworks.model 
        Model class.
    optim : dnetworks.optimizers
        Optimeze class.
    loss : dnetworks.loss
        Loss class .
    env : gym.env
        Gym enviroment
    epsilon : int, default=1
    gamma : int, default=0.9
    decay_rate : int, default=0.005
    min_epsilon : int, default=0.1    
    """
    def __init__(
        self, 
        model=None,
        optim=None,
        loss=None,
        env=None,  
        epsilon=1, 
        gamma=0.9, 
        decay_rate=0.005, 
        min_epsilon=0.1,
    ):
        self.model = model
        self.optim = optim
        self.loss = loss
        self.env = env
        self.epsilon = epsilon
        self.gamma = gamma
        self.decay_rate = decay_rate
        self.min_epsilon = min_epsilon
        self.memory = Memory(capacity=300)

    def take_action(self, state):
        """
        Action taken by the model for a given state. Note that it can 
        be chosen randomly as well for a given epsilon.

        Parameters
        ----------
        state : list
            State to give the action.
        
        Returns
        -------
        action : int
            Action taken by the model.
        """
        # random choice
        if np.random.rand() < self.epsilon:
            action = np.random.randint(self.n_actions)  
        else:
            pred = self.model.forward(state).T
            action = np.argmax(pred, axis=1)[0]

        return action

    def get_xy(self, steps_sample):
        """
        
        """
        
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
            cost = self.model.loss(Y, self.loss)
            self.model.backward()
            self.model.optimize(self.optim)

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