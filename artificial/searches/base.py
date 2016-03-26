import abc
import warnings

from artificial.base.helpers import PriorityQueue

from artificial import agents


class Base(metaclass=abc.ABCMeta):
    """Base Search.

    Common interface for searches, including agent, space and root properties.

    Parameters
    ----------
    agent : Agent-like
            Agent that requested search. This is needed as only an agent
            of the specific domain problem can predict outcomes based on
            their own actions.

    root  : State-like
            The state in which the search should start.

    backtracks : bool (default=True)
                 Once the goal set is reached, backtracking will be performed
                 in order to create a sequence of intermediate states.
                 If this operation is not needed or too much time-consuming,
                 False can be passed and this will be skipped.

    Attributes
    ----------
    space : set
            A set used to contain states and efficient repetition checking.

    solution_candidate : State-like
            A State's subclass found when performing the search.

    solution_path : list
            A list of intermediate states to achieve the goal state.
            This attribute is undefined when `backtracks` parameter
            is set to False.

    """
    def __init__(self, agent, root=None, backtracks=True):
        assert isinstance(agent, agents.GoalBasedAgent), \
            'First Search requires an goal based agent.'

        self.agent = agent
        self.root = self.solution_candidate = self.solution_path = None

        self.space = set()
        self.backtracks = backtracks

        self.restart(root)

    def restart(self, root):
        self.root = root
        self.space = {root} if root else set()
        self.solution_candidate = None
        self.solution_path = None

        return self

    def perform(self):
        self.solution_candidate = self.solution_path = None
        self.solution_candidate = self._perform()

        if not self.solution_candidate and self.agent.verbose:
            warnings.warn('Could not find a path (%s:root)->(:goal)'
                          % self.root)

        if self.backtracks:
            self.backtrack()

        return self

    def _perform(self):
        """Search actual performing.

        Returns
        -------
        State object (or State's subclass' object), A solution candidate
        found by the local search.

        """
        raise NotImplementedError

    def backtrack(self):
        state_sequence = []

        state = self.solution_candidate
        while state:
            state_sequence.insert(0, state)
            state = state.parent

        self.solution_path = state_sequence

        return self

    def solution_path_as_action_list(self):
        """Build a list of actions from the solution candidate path.

        Naturally, must be executed after `perform` call.

        Returns
        -------
        actions : array-like, a list containing the actions performed
                  by the sates in `solution_path`.
        """
        return ([s.action for s in self.solution_path[1:]]
                if self.solution_path
                else None)
