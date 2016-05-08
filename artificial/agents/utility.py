"""Utility Based Agent"""

# Author: Lucas David -- <ld492@drexel.edu>
# License: MIT (c) 2016

import abc

import six

from . import goal_based


@six.add_metaclass(abc.ABCMeta)
class UtilityBasedAgent(goal_based.GoalBasedAgent):
    """UtilityBasedAgent Base.

    The difference between agents of this category to goal-based agents
    is the search function, which attempts to find the goal satisfying some
    "utility" (or "happiness") measure.
    """

    def utility(self, state):
        """Utility of a state for the current agent's object.

        By default, agents attempt to minimize the cost function `state.f()`.

        Parameters
        ----------
        state : State-like object
            The state which should have its utility to the agent computed.

        Notes
        -----
        Overriding this method should always be followed by setting
        `state.computed_utility` parameter and re-use it, in order
        to to increase performance.

        """
        state.computed_utility_ = (-state.f()
                                   if state.computed_utility_ is None
                                   else state.computed_utility_)
        return state.computed_utility_
