""" Definitions of bandit games that use different action selection policies,
    and different action-value estimation methods
"""

from abc import ABCMeta, abstractmethod
import numpy as np
import time
from scipy.stats import beta

from bandits.bandits import Bandit

class BanditGame(metaclass=ABCMeta):
    """ Abstract base for all bandit games
    """
    def __init__(self, bandit, full_rec):
        """
        bandit (Bandit): the target bandit  
        """
        assert isinstance(bandit, Bandit)
        np.random.seed(int(time.time()))

        self.full_rec = full_rec  #If true, record history of rewards and regrets
        self.bandit = bandit
        self._reset()

    @abstractmethod
    def _reset(self):
        self.actcounts = [0] * self.bandit.n_actions  # Number of visits per action
        self.actvals = [0] * self.bandit.n_actions    # Current action-value estimates
        self.actions = []    # History of actions taken
        self.cumreward = 0.  # Cumulative reward
        self.rewards = []    # History of rewards
        self.cumregret = 0.  # Cumulative regret
        self.regrets = []    # History of regrets

    def _update_history(self, i, reward):
        # i (int): index of the selected action.
        self.actcounts[i] += 1        
        self.actions.append(i)
        self.cumreward += reward
        self.cumregret += self.bandit.best_reward - reward
        if self.full_rec:
            self.rewards.append(reward)
            self.regrets.append(self.bandit.best_reward - reward)

    @abstractmethod
    def _run_one_step(self):
        """Returns the action selected and the reward received """
        raise NotImplementedError

    def run(self, num_steps):
        self._reset()
        assert self.bandit is not None
        for _ in range(num_steps):
            i, reward = self._run_one_step()
            self._update_history(i, reward)


class EpsilonGreedy(BanditGame):
    def __init__(self, bandit, eps, init_actvals=1.0, full_rec=True):
        """
        eps (float): the probability to explore at each time step.
        init_actvals (float): prior action values; default to be 1.0; optimistic initialization
        full_rec (bool): True for collecting full history
        """
        assert 0. <= eps <= 1.0        
        self.eps = eps
        self.init_actvals = init_actvals
        super().__init__(bandit, full_rec)
        
        # these are to be reset after every run
        self.actvals = [init_actvals] * self.bandit.n_actions


    def _run_one_step(self):
        if np.random.random() < self.eps:
            # Do random exploration
            i = np.random.randint(0, self.bandit.n_actions)
        else:
            # Pick the best one according to current action-value estimates
            # Randomly break ties
            n_act = self.bandit.n_actions
            maxactval = max(self.actvals)
            maxidx = [i for i in range(n_act) if self.actvals[i] >= maxactval]
            i = int(np.random.choice(maxidx, 1))

        r = self.bandit.get_reward(i)
        # update the estimated action values
        self.actvals[i] += 1. / (self.actcounts[i] + 1) * (r - self.actvals[i])
        # return action and reward
        return i, r

    def _reset(self):
        self.actvals = [self.init_actvals] * self.bandit.n_actions
        super()._reset()


class UCBBayesian(BanditGame):
    """Assuming Beta prior."""

    def __init__(self, bandit, risk_weight=1.0, init_a=1.0, init_b=1.0, full_rec=True):
        """
        risk_weight (float): how many standard devs to consider as upper confidence bound.
        init_a (int): initial value of a in Beta(a, b)
        init_b (int): initial value of b in Beta(a, b)
        full_rec (bool): True for collecting full history        
        """
        self.c = risk_weight
        self.init_a = init_a
        self.init_b = init_b
        super().__init__(bandit, full_rec)
        
        # these are to be reset after every run
        self._as = [init_a] * self.bandit.n_actions
        self._bs = [init_b] * self.bandit.n_actions

    def _run_one_step(self):
        # Pick the best one with consideration of upper confidence bounds
        # Break ties at random
        n_act = self.bandit.n_actions
        maxi = max(
            range(n_act),
            key=lambda x: 
                self._as[x] / float(self._as[x] + self._bs[x]) 
                + self.c * beta.std(self._as[x], self._bs[x]))
        maxactval = self.actvals[maxi]
        # all max value action indices
        maxidx = [i for i in range(n_act) if self.actvals[i] >= maxactval]
        # choose one at random
        i = int(np.random.choice(maxidx, 1))
        
        r = self.bandit.get_reward(i)

        # Update posterior
        self._as[i] += r
        self._bs[i] += (1 - r)

        # update the estimated action values
        self.actvals = [self._as[i] / float(self._as[i] + self._bs[i]) for i in range(n_act)]
        # return action and reward
        return i, r

    def _reset(self):
        self._as = [self.init_a] * self.bandit.n_actions
        self._bs = [self.init_b] * self.bandit.n_actions
        super()._reset()


class ThompsonBayesian(BanditGame):
    def __init__(self, bandit, init_a=1.0, init_b=1.0, full_rec=True):
        """
        init_a (int): initial value of a in Beta(a, b)
        init_b (int): initial value of b in Beta(a, b)
        full_rec (bool): True for collecting full history        
        """
        
        self.init_a = init_a
        self.init_b = init_b
        super().__init__(bandit, full_rec)
        
        # these are to be reset after every run 
        self._as = [init_a] * self.bandit.n_actions
        self._bs = [init_b] * self.bandit.n_actions

    def _run_one_step(self):
        # Sample from the posterior
        samples = [np.random.beta(self._as[x], self._bs[x]) for x in range(self.bandit.n_actions)]
        # Pick the best action and get the reward
        i = max(range(self.bandit.n_actions), key=lambda x: samples[x])
        r = self.bandit.get_reward(i)

        # Update posterior
        self._as[i] += r
        self._bs[i] += (1 - r)

        # update the estimated action values
        self.actvals = [self._as[i] / float(self._as[i] + self._bs[i]) for i in range(self.bandit.n_actions)]
        # return action and reward
        return i, r

    def _reset(self):
        self._as = [self.init_a] * self.bandit.n_actions
        self._bs = [self.init_b] * self.bandit.n_actions
        super()._reset()
