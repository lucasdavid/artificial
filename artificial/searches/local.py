import threading
import multiprocessing

from . import base
from .. import agents


class HillClimbing(base.Base):
    """Hill Climbing Search.

    Perform Hill Climbing search according to a designated strategy.

    Parameters
    ----------

    strategy : ('default'|'steepest-ascent')
        Defines the climbing policy.

        Options are:

        --- 'classic' : first child that improves utility is choosen.

        --- 'steepest-ascent' : child that provides greatest
                                utility improvement is choosen.

    restart_limit : (None|int)
        Define maximum number of random-restarts.

        If None, classic HillClimbing is performed and no restarts occurr.
        If limit passed is an integer `i`, the agent will restart `i` times
        before returning a solution.

    """

    backtracks = False

    def __init__(self, agent, root=None,
                 strategy='steepest-ascent', restart_limit=None):
        super().__init__(agent=agent, root=root)

        assert isinstance(agent, agents.UtilityBasedAgent), \
            'Hill Climbing Search requires an utility based agent.'

        self.strategy = strategy
        self.restart_limit = restart_limit

    def _perform(self):
        self.solution_candidate = self.root
        current = self.root
        it, limit = 0, self.restart_limit or 1

        while it < limit:
            stalled = False

            while not stalled:
                children = self.agent.predict(current)
                utility = self.agent.utility(current)
                stalled = True

                for child in children:
                    if self.agent.utility(child) > utility:
                        current = child
                        stalled = False

                        if self.strategy == 'classic':
                            # Classic strategy always takes the first
                            # child that improves utility.
                            # I do NOT appreciate having to check for classic
                            # strategy every child, and I'd very much like
                            # this to change for something more efficient.
                            continue

            if (self.agent.utility(current) >
                self.agent.utility(self.solution_candidate)):
                # Ok, great. Something better was found this iteration.
                self.solution_candidate = current

            if it < limit - 1:
                # There will be a next iteration.
                current = self.solution_candidate.generate_random()

            it += 1

        return self.solution_candidate


class LocalBeam(base.Base):
    """Local Beam.

    Parameters
    ----------
    k : ['auto'|int] (default='auto')
        The number of beams to keep track of. If value is `auto`,
        then the number of beams is infered from the number of processors
        available.

    """
    class Beam(threading.Thread):
        def __init__(self, manager):
            super().__init__()

            self.manager = manager

        def run(self):
            pass

    def __init__(self, agent, k='auto',
                 strategy='steepest-ascent', restart_limit=None):
        super().__init__(agent=agent, root=root)

        assert isinstance(agent, agents.UtilityBasedAgent), \
            'Hill Climbing Search requires an utility based agent.'

        self.k = k
        self.strategy = strategy
        self.restart_limit = restart_limit

        self.beams = None

    def restart(self, root):
        super().restart(root=root)
        self.beams = None

        return self

    def _perform(self):
        if k == 'auto':
            k = multiprocessing.cpu_count()

        self.beams = [self.Beam(self) for _ in range(k)]

        for beam in self.beams:
            beam.start()

        for beam in self.beams:
            beam.join()
