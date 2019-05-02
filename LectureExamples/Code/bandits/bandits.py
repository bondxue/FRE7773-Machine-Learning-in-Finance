""" Definition of various bandit classes
"""

from abc import ABCMeta, abstractmethod
import numpy as np


class Bandit(metaclass=ABCMeta):
    """ Abstract base for all concrete bandit classes
    """
    def __init__(self, n_actions, best_action = None, best_reward = None):
        self._n = n_actions
        self._ba = best_action
        self._br = best_reward
        
    @abstractmethod
    def get_reward(self, i):
        raise NotImplementedError
        
    @property
    def n_actions(self):
        return self._n

    @property
    def best_action(self):
        return self._ba

    @property
    def best_reward(self):
        return self._br


class FixedBandit(Bandit):
    """ Fixed reward bandit
    """
    def __init__(self, rewards):
        n_actions = len(rewards)
        assert n_actions > 0
        self._rewards = rewards

        super().__init__(
                n_actions, 
                max(range(n_actions), key=rewards.__getitem__),
                max(rewards))

    def get_reward(self, i):
        """ Returns the reward for selecting bandit arm i
        """
        return self.rewards[i]

    @property
    def rewards(self):
        return self._rewards

    
class BernoulliBandit(Bandit):
    """ Bernoulli distribution bandit
    """
    def __init__(self, probs):
        n_actions = len(probs)
        assert n_actions > 0        
        assert all(x >=0 and x <= 1 for x in probs)
        self._probs = probs

        super().__init__(
                n_actions, 
                max(range(n_actions), key=self._probs.__getitem__),
                max(self._probs))

    def get_reward(self, i):
        """ Returns the reward for selecting bandit arm i
        """
        if np.random.random() < self._probs[i]:
            return 1
        else:
            return 0

    @property
    def probs(self):
        return self._probs


class NormalBandit(Bandit):
    """ Normal distribution bandit
    """
    def __init__(self, means, stdevs):
        assert len(stdevs) == len(means)        
        assert all(x >=0 for x in stdevs)
        n_actions = len(means)
        
        self._means = means
        self._stdevs = stdevs
        
        super().__init__(
                n_actions, 
                max(range(n_actions), key=self._means.__getitem__),
                max(self._means))

    def get_reward(self, i):
        """ Returns the reward for selecting bandit arm i
        """
        return np.random.normal(self._means[i], self._stdevs[i])

    @property
    def means(self):
        return self._means

    @property
    def stdevs(self):
        return self._stdevs
