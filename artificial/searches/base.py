"""Artificial Searches Base"""

# Author: Lucas David -- <ld492@drexel.edu>
# License: MIT (c) 2016

import abc

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


    Attributes
    ----------

    space_ : set
            A set used to contain states and efficient repetition checking.

    solution_candidate_ : State-like
            A State's subclass found when performing the search.

    solution_path_ : list
            A list of intermediate states to achieve the goal state.
            This attribute is undefined when `backtracks` parameter
            is set to False.

    """

    def __init__(self, agent, root=None):
        assert isinstance(agent, agents.GoalBasedAgent), \
            'First Search requires an goal based agent.'

        self.a = self.agent = agent
        self.root = self.solution_candidate_ = self.solution_path_ = None

        self.space_ = set()
        self.restart(root)

    def restart(self, root):
        self.root = root
        self.space_ = {root} if root else set()
        self.solution_candidate_ = None
        self.solution_path_ = None

        return self

    def search(self):
        """Search for solution candidate.

        This method should set the `solution_candidate` property
        to the State-like object found by the search at hand and, finally,
        return `self` object.

        """
        raise NotImplementedError

    def backtrack(self):
        """Backtrack answer.

        For problems where the path to the solution matters, users can call
        this method to backtrack the solution candidate found and set the
        `solution_path` property.

        IMPORTANT: this method should always come after searching (the call
        for `search` method), as only there `solution_candidate`
        property is set.

        """
        state_sequence = []

        state = self.solution_candidate_

        if state is None:
            raise RuntimeError('Cannot backtrack a nonexistent state. You are '
                               'most likely backtracking before searching, '
                               'which is illegal.')

        while state:
            state_sequence.insert(0, state)
            state = state.parent

        self.solution_path_ = state_sequence

        return self

    def solution_path_as_action_list(self):
        """Build a list of actions from the solution candidate path.

        IMPORTANT: this method must always be executed after `backtrack` call,
        as only there `solution_path` is set.

        Returns
        -------
        actions : array-like, a list containing the actions performed
                  by the sates in `solution_path`.
        """
        return ([s.action for s in self.solution_path_[1:]]
                if self.solution_path_
                else None)
