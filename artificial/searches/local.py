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

    def restart(self, root):
        super().restart(root=root)

        return self

    def _perform(self):
        self.solution_candidate = self.root
        current = self.root
        iterations = 0
        restart_limit = self.restart_limit or 1

        while iterations < restart_limit:
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
                            continue

            if (self.agent.utility(current)
                    > self.agent.utility(self.solution_candidate)):
                self.solution_candidate = current

            if iterations < restart_limit - 1:
                # There will be a next iteration.
                current = self.solution_candidate.generate_random()

            iterations += 1

        return self.solution_candidate
