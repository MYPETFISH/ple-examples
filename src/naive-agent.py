import numpy as np

class NaiveAgent():
    """ This naive agent picks actions at random! """

    def __init__(self, actions):
        self.actions = actions

    def pickAction(self, reward, obs):
        return self.actions[np.random.randint(0, len(self.actions))]

