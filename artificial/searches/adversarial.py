import time
import random
import numpy as np

from . import base


class Adversarial(base.Search, metaclass=abc.ABCMeta):
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

    def perform(self):
        self.started_at = time.time()
        return super().perform()


class Random(Adversarial):
    """Random Adversarial Search.
    
    Actions are taken randomly, achieving a result.
    
    """
    def __init__(self, agent, root=None,
                 time_limit=np.inf, depth_limit=np.inf,
                 dispose=None, random_state=None)
         self.random_state = random_state or random.Random()

    def _perform(self):
        state = self.root
        depth = 0
        
        while state and depth < self.limit_depth and \
              time.time() - self.started_at < self.time_limit:
            children = self.agent.predict(state)
            state = self.random_state.choice(children) if children else None
            depth += 1

        return state


class MinMax(Adversarial):
    """Min Max Adversarial Search.
    
    Notes
    -----
    Not all branches can be completely searched in feasible time.
    `MinMax` assumes that the agent at hand has a "good" utility 
    function to evaluate states, regardless of their position in
    the derivation tree.

    """
    
    def _perform(self):
        return self._min_max_policy(self.root)
    
    def _min_max_policy(self, state, depth=0):
        if self.depth_limit and depth > self.depth_limit or \
           time.time() - self.started_at > self.time_limit:
            return self.agent.utility(self)

        children = self.agent.predict(state)

        if not children:
            # Terminal state. Return utility.
            return self.agent.utility(state)

        utilities = [self._min_max_policy(c, depth + 1) for c in children]
        interest = max if depth % 2 == self.MAXIMIZE else min

        return interest(children, keys=lambda i, e: utilities[i])


class AlphaBetaPrunning(MinMax):
    def _min_max_policy(self, state, depth=0, a=-np.inf, b=np.inf):
        if self.depth_limit and depth > self.depth_limit or \
           time.time() - self.started_at > self.time_limit:
            return self.agent.utility(self)

        children = self.agent.predict(state)
        
        if not children:
            # Terminal state. Return utility.
            return self.agent.utility(state)
        
        v, interest = ((-np.inf, max)
                        if depth % 2 == self.MAXIMIZE 
                        else (np.inf, min))

        for c in children:
            v = interest(v, self._min_max_policy(c, depth + 1, a, b)
                
            if depth % 2 == self.MAXIMIZE:
                a = interest(a, v)
            else:
                b = interest(b, v)

            if b <= a:
                break
       
        return v
