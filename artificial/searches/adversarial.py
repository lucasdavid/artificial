import time
import numpy as np

from . import base


class MinMax(base.Search):
    """Min Max Adversarial Search.
    
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
    
    Notes
    -----
    Not all branches can be completely searched in feasible time.
    `MinMax` assumes that the agent at hand has a "good" utility 
    function to evaluate states, regardless of their position in
    the derivation tree.

    """
    
    MINIMIZE, MAXIMIZE = (0, 1)

    def __init__(self, agent, root=None,
                 time_limit=np.inf, depth_limit=np.inf,
                 dispose=False):
        super().__init__(agent=agent, root=root)
        
        self.time_limit = time_limit
        self.dispose = dispose
        self.started_at = None
       
    def _perform(self):
        self.started_at = time.time()
        return self._min_max(self.root, 0, self.MAXIMIZE)
    
    def _min_max(self, state, depth):
        if self.depth_limit and depth > self.depth_limit or \
           time.time() - self.started_at > self.time_limit:
            return self.agent.utility(self)

        children = self.agent.predict(state)
        
        if not children:
            # Terminal state. Return utility.
            return self.agent.utility(state)

        utilities = [self._min_max(c, depth + 1) for c in children]
        order = max if depth % 2 == self.MAXIMIZE else min

        return order(children, keys=lambda i, e: utilities[i])


class AlphaBetaPrunning(MinMax):
    pass

