""" Definitions of the experiment class, used for running bandit games,
    and collecting statistics
"""

import collections
from bandits.games import BanditGame
from math import sqrt
import numpy as np

class Experiment:
    
    def __init__(self, games_dict):
        """
        games_dict (Mapping object): the dictionary of games
        """
        assert isinstance(games_dict, collections.Mapping)
        assert all(map(lambda s: isinstance(s, BanditGame), games_dict.values()))        
        self.gdict = games_dict
        self._reset(0)
        
    def run(self, n_steps, n_runs, verbose=False):
        """
        n_steps (int): number of steps to play each game
        n_runs  (int): number of runs (epochs) to play each game        
        """
        self._reset(n_steps)
        
        for k, v in self.gdict.items():
            cumreward = 0
            cumreward2 = 0
            for n in range(n_runs):
                v.run(n_steps)
                rewards = np.array(v.rewards)
#                self.mean_rewards[k] = [x + y for (x, y) in zip (self.mean_rewards[k], v.rewards)]
                self.mean_rewards[k] += rewards
#                self.stdev_rewards[k] = [x + y**2 for (x, y) in zip (self.stdev_rewards[k], v.rewards)]
                self.stdev_rewards[k] += np.square(rewards)
                cumreward += v.cumreward
                cumreward2 += v.cumreward * v.cumreward
            self.mean_cumreward[k] = cumreward / n_runs
            self.stdev_cumreward[k] = sqrt(cumreward2 / n_runs - np.square(self.mean_cumreward[k]))
#            self.mean_rewards[k] = [x / n_runs for x in self.mean_rewards[k]]    # E[X]
            self.mean_rewards[k] /= n_runs     # E[X]
#            self.stdev_rewards[k] = [x / n_runs for x in self.stdev_rewards[k]]  # E[X^2]
            self.stdev_rewards[k] /= n_runs   # E[X^2]
#            self.stdev_rewards[k] =  [sqrt(x - y**2) for (x, y) in zip (self.stdev_rewards[k], self.mean_rewards[k])]
            self.stdev_rewards[k] -= np.square(self.mean_rewards[k]) # E(X^2) - E(X)^2
            self.stdev_rewards[k] = np.sqrt(self.stdev_rewards[k])
            if verbose:
                print('==> {} for {} steps and {} runs. MeanRewardPerStep={}. StdevRewardPerStep={}'
                      .format(k, n_steps, n_runs, 
                              round(self.mean_cumreward[k]/n_steps, 4), 
                              round(self.stdev_cumreward[k]/n_steps, 4)))

    def _reset(self, nsteps):
        gnames = self.gdict.keys()
        self.mean_cumreward = {key: 0.0 for key in gnames}
        self.stdev_cumreward = {key: 0.0 for key in gnames}                
        self.mean_rewards = {key: np.zeros(nsteps) for key in gnames}
        self.stdev_rewards = {key: np.zeros(nsteps) for key in gnames}
