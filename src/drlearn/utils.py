import random 

class Memory:
    """Class to save the enviroment steps.

    Parameters
    ----------
    capacity : int
        Maximum length of the list.

    Atributes
    ---------
    memory : list
        List containing the enviroment steps.
    """
    def __init__(self, capacity, sample_size=32):
        self.capacity = capacity
        self.memory = []
        self.sample_size = sample_size

    def push(self, step_env):
        """Save a new step in the memory.

        Parameters
        ----------
        step_env : list
            List containing the new enviroment step to be saved.
        """
        self.memory.append(step_env)
        drop_condition = len(self.memory) > self.capacity

        if drop_condition:
            self.memory.pop(0)

    def sample(self):
        """Returns a sample of the steps saved.

        Parameters
        ----------
        batch_size : int
            Length of the sample.
        """
        batch_size = min(self.sample_size, len(self.memory))

        return random.sample(self.memory, batch_size)