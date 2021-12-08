import numpy as np


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
        memory=None,
        epsilon=1, 
        gamma=0.9, 
        decay_rate=0.005, 
        min_epsilon=0.1,
    ):
        self.model = model
        self.optim = optim
        self.loss = loss
        self.env = env
        self.memory = memory
        self.epsilon = epsilon
        self.gamma = gamma
        self.decay_rate = decay_rate
        self.min_epsilon = min_epsilon
        self.n_states = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.n

    def take_action(self, state):
        """
        Action taken by the model for a given state. Note that it 
        can be chosen randomly as well for a given epsilon.

        Parameters
        ----------
        state : list
            State to give the action.
        
        Returns
        -------
        action : int
            Action taken by the model.
        """
        if np.random.rand() > self.epsilon:
            pred = self.model.forward(state).T
            action = np.argmax(pred, axis=1)[0]
        else:
            action = np.random.randint(self.n_actions)  

        return action

    def get_target(self, steps_sample):
        """
        Gets the target that will be passed to the loss
        function.

        Parameters
        ----------
        steps_sample : list
            List containing the main information of each step
            (actual state, action, reward, new state, is finished).

        Return
        ------
        targets : numpy.array
            Array containing the output to be predicted by the 
            model.
        """
        states, new_states = self._get_states(steps_sample)

        quality_new = self.model.forward(new_states)
        quality = self.model.forward(states)

        targets = np.zeros((len(steps_sample), self.n_actions))
        
        for num, params in enumerate(steps_sample):
            _, action_r, reward_r, _, done_r = params
            target = quality.T[num]
            target[action_r] = reward_r

            if not done_r:
                future_reward = np.amax(quality_new.T[num])
                target[action_r] += self.gamma * future_reward

            targets[num] = target

        return states, targets.T

    def play(self):
        """
        The agent plays takes an action for each state until the
        is finished.

        Returns
        -------
        epoch_loss : int
            Result of the loss function for the given game.
        total_reward : int
            Total reward for the game.
        """
        state = self.env.reset()
        total_reward = 0

        epoch_loss = []
        done = False

        while not done:
            action = self.take_action(state.reshape(self.n_states, 1))

            new_state, reward, done, _ = self.env.step(action)
            step_list = [state, action, reward, new_state, done]

            self.memory.push(step_list)
            steps_sample = self.memory.sample()

            X, Y = self.get_target(steps_sample)
            self.model.forward(X)
            cost = self.model.loss(Y, self.loss)
            self.model.backward()
            self.model.optimize(self.optim)

            epoch_loss.append(cost)

            total_reward += reward
            state = new_state

        epoch_loss = np.mean(epoch_loss)     
        
        return epoch_loss, total_reward

    def update_epsilon(self, epoch):
        """
        Updates the parameter epsilon.

        Parameters
        ----------
        epoch : int
        """
        _exp = np.exp(-self.decay_rate * epoch)
        _mult = (1.0 - self.min_epsilon) * _exp
        self.epsilon = self.min_epsilon + _mult

    def _get_states(self, steps_sample):
        """
        Gets the new and actual states from the steps list.
        """
        states = np.array([a[0] for a in steps_sample]).T
        new_states = np.array([a[3] for a in steps_sample]).T

        return states, new_states