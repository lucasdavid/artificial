"""Artificial Local Searches"""

# Author: Lucas David -- <ld492@drexel.edu>
# License: MIT (c) 2016

import abc
import multiprocessing
import threading

import six

from . import base
from .. import agents


@six.add_metaclass(abc.ABCMeta)
class Local(base.SearchBase):
    """Base Local Search.

    Base class for HillClimbing and LocalBeam searches.

    Parameters
    ----------
    strategy : ('default'|'steepest-ascent')
        Defines the climbing policy.

        Options are:

        --- 'classic' : first child that improves utility is chosen.

        --- 'steepest-ascent' : child that provides greatest
                                utility improvement is chosen.

    restart_limit : int
        Define maximum number of random-restarts.

        If `1`, classic HillClimbing is performed and no restarts occur.
        If limit is passed and it's greater than 1, the agent will restart
        `i` times before returning a solution.

    """

    def __init__(self, agent, root=None,
                 strategy='steepest-ascent', restart_limit=1):
        super(Local, self).__init__(agent=agent, root=root)

        assert isinstance(agent, agents.UtilityBasedAgent), \
            'Local searches require an utility based agent.'

        self.strategy = strategy
        self.restart_limit = restart_limit


class HillClimbing(Local):
    """Hill Climbing Search.

    Perform Hill Climbing search according to a designated strategy.
    """

    def search(self):
        self.solution_candidate_ = self.root
        strategy_is_classic = self.strategy == 'classic'

        current = self.root
        it, limit = 0, self.restart_limit or 1

        while it < limit:
            it += 1
            stalled = False

            if current is None:
                current = self.agent.environment.state_class_.random()

            while not stalled:
                children = self.agent.predict(current)
                stalled = True

                for child in children:
                    if self.agent.utility(child) > self.agent.utility(current):
                        current = child
                        stalled = False

                        # Classic strategy always takes the first
                        # child that improves utility.
                        if strategy_is_classic: break

            if (not self.solution_candidate_ or self.agent.utility(current) >
                self.agent.utility(self.solution_candidate_)):
                # We've just found a better solution!
                self.solution_candidate_ = current

            # Force random restart.
            current = None

        return self


class LocalBeam(Local):
    """Local Beam.

    Parameters
    ----------
    k : ['auto'|int] (default='auto')
        The number of beams to keep track of. If value is `auto`,
        then the number of beams is inferred from the number of processors
        available.

    strategy : ('default'|'steepest-ascent')
        Defines the climbing policy.

        Options are:

        --- 'classic' : first child that improves utility is choosen.

        --- 'steepest-ascent' : child that provides greatest
                                utility improvement is choosen.

    restart_limit : int
        Define maximum number of random-restarts.

        If `1`, classic HillClimbing is performed and no restarts occur.
        If limit is passed and it's greater than 1, the agent will restart
        `i` times before returning a solution.
    """

    class Beam(threading.Thread):
        def __init__(self, manager):
            super(LocalBeam.Beam, self).__init__()

            self.manager = manager
            self.hill_climber = HillClimbing(agent=manager.agent,
                                             strategy=manager.strategy)

        def run(self):
            it, limit = 0, self.manager.restart_limit or 1

            while it < limit:
                it += 1
                state = self.hill_climber.search().solution_candidate_

                with self.manager._solution_update_lock:
                    if (not self.manager.solution_candidate_ or
                                self.manager.agent.utility(state) >
                                self.manager.agent.utility(
                                    self.manager.solution_candidate_)):
                        self.manager.solution_candidate_ = state

    def __init__(self, agent, root=None, k='auto',
                 strategy='steepest-ascent', restart_limit=1):
        super(LocalBeam, self).__init__(agent=agent,
                                        root=root,
                                        strategy=strategy,
                                        restart_limit=restart_limit)
        self.k = k
        self.beams = None
        self._solution_update_lock = threading.Lock()

    def restart(self, root):
        super(LocalBeam, self).restart(root=root)
        self.beams = None
        return self

    def search(self):
        self.solution_candidate_ = self.solution_path_ = None

        if self.k == 'auto':
            k = multiprocessing.cpu_count()
        elif isinstance(self.k, int):
            k = self.k
        else:
            raise ValueError('Unknown value for k (%s)' % str(self.k))

        self.beams = [self.Beam(self) for _ in range(k)]

        for beam in self.beams: beam.start()
        for beam in self.beams: beam.join()

        return self
