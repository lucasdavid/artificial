import abc
import random
import time

import numpy as np

from . import base


class Adversarial(base.Base, metaclass=abc.ABCMeta):
    """Adversarial Search.
    
    Parameters
    ----------
    time_limit : float (default=np.inf)
        Time limit (in seconds) for a performance.
        By default, search has infinite time to make a decision.
    
    depth_limit : float (default=np.inf)
        Depth limit (in hops) for a branch search.
        By default, search can keep going until the branch dies.

    dispose : bool (default=False)
        Always dispose memory after a movement.
    
    Attributes
    ----------
    started_at : long
        Time in which performance started.
        `time.time() - started_at` yeilds how much time has
        approximately passed since the `MinMax.perform` was called.
       
    """

    MINIMIZE, MAXIMIZE = (0, 1)

    def __init__(self, agent, root=None,
                 time_limit=np.inf, depth_limit=np.inf,
                 dispose=False):
        super().__init__(agent=agent, root=root)

        self.time_limit = time_limit
        self.depth_limit = depth_limit
        self.dispose = dispose

        self.started_at = None


class Random(Adversarial):
    """Random Adversarial Search.
    
    Actions are taken randomly, achieving a result.
    
    """

    def __init__(self, agent, root=None,
                 time_limit=np.inf, depth_limit=np.inf,
                 dispose=False, random_state=None):
        super().__init__(agent=agent, root=root,
                         time_limit=time_limit, depth_limit=depth_limit,
                         dispose=dispose)
        self.random_state = random_state or random.Random()

    def search(self):
        self.started_at = time.time()

        state = self.root or self.agent.environment.generate_random_state()
        depth = 0

        while (state and depth < self.depth_limit and
               time.time() - self.started_at < self.time_limit):
            children = self.agent.predict(state)
            state = self.random_state.choice(children) if children else None
            depth += 1

        self.solution_candidate = state

        return self


class MinMax(Adversarial):
    """Min Max Adversarial Search.
    
    Notes
    -----
    Not all branches can be completely searched in feasible time.
    `MinMax` assumes that the agent at hand has a "good" utility 
    function to evaluate states, regardless of their position in
    the derivation tree.

    """

    def search(self):
        self.started_at = time.time()
        return self._min_max_policy(self.root, 0)

    def _min_max_policy(self, state, depth):
        if (depth > self.depth_limit or
            time.time() - self.started_at > self.time_limit):
            return self.agent.utility(state)

        children = self.agent.predict(state)

        if not children:
            # Terminal state. Return utility.
            return self.agent.utility(state)

        utilities = [self._min_max_policy(c, depth + 1) for c in children]
        order = max if depth % 2 == self.MAXIMIZE else min

        return order(children, keys=lambda i, e: utilities[i])


class AlphaBetaPruning(MinMax):
    pass
