"""Artificial Adversarial Searches"""

# Author: Lucas David -- <ld492@drexel.edu>
# License: MIT (c) 2016

import abc
import random
import time

import numpy as np
import six

from . import base


@six.add_metaclass(abc.ABCMeta)
class Adversarial(base.SearchBase):
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

    MAXIMIZE, MINIMIZE = (0, 1)

    def __init__(self, agent, root=None, time_limit=np.inf,
                 depth_limit=np.inf, dispose=False):
        super(Adversarial, self).__init__(agent=agent, root=root)

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
                 dispose=False, random_generator=None):
        super(Random, self).__init__(
            agent=agent, root=root, time_limit=time_limit,
            depth_limit=depth_limit, dispose=dispose)
        self.random_generator = random_generator or random.Random()

    def search(self):
        self.started_at = time.time()
        state = self.root or self.agent.last_state.random()

        children = self.agent.predict(state)
        self.solution_candidate_ = (self.random_generator.choice(children)
                                    if children else None)
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
        self.solution_candidate_ = None
        best_score = -np.inf

        for c in self.agent.predict(self.root):
            c_score = self._selection_policy(c)

            if best_score < c_score:
                self.solution_candidate_ = c
                best_score = c_score

        return self

    def should_maximize_at(self, depth):
        return depth % 2 == self.MAXIMIZE

    def _selection_policy(self, state, depth=1):
        """Min-Max Policy."""
        children = self.agent.predict(state)
        if (depth > self.depth_limit or not children or
                        time.time() - self.started_at > self.time_limit):
            return self.agent.utility(state)

        interest = max if self.should_maximize_at(depth) else min
        return interest(self._selection_policy(c, depth + 1)
                        for c in children)


class AlphaBeta(MinMax):
    """Alpha Beta Pruning Adversarial Search.

    Min-Max search with alpha-beta pruning, a optimization strategy for
    branch cutting.
    """

    def _selection_policy(self, state, depth=1, a=-np.inf, b=np.inf):
        """Alpha Beta Pruning Policy."""
        children = self.agent.predict(state)
        if (depth > self.depth_limit or not children or
                        time.time() - self.started_at > self.time_limit):
            return self.agent.utility(state)

        v, interest = ((-np.inf, max)
                       if self.should_maximize_at(depth)
                       else (np.inf, min))
        for c in children:
            v = interest(v, self._selection_policy(c, depth + 1, a, b))

            if self.should_maximize_at(depth):
                a = interest(a, v)
            else:
                b = interest(b, v)

            if b <= a: break

        return v
