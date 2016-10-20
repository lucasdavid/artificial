"""Goal Based Agent"""

# Author: Lucas David -- <ld492@drexel.edu>
# License: MIT (c) 2016

import abc
import logging

import six

from . import predicting

logger = logging.getLogger('artificial')


@six.add_metaclass(abc.ABCMeta)
class GoalBasedAgent(predicting.PredictingAgent):
    """GoalBasedAgent Base.

    This class works as a wrapper around goal-based artificial. Obviously,
    `search` must be implemented correctly in order to achieve
    the desired 'goal-seeking' behavior.

    This is also the implementation for Simple-Problem-Solving-Agent,
    as described by S. Russell & P. Norvig in Artificial Intelligence,
    An Modern Approach, 3rd ed. p. 67. I decided to merge both classes
    as they present too many similarities, and separation would be
    a code smell.

    This agent searches for a desired action sequence and executes
    it completely. When it is finally done, a new search is made and a
    new action sequence will be performed.

    Parameters
    ----------
    search : a Search's subclass that will be instantiated
             on the agent's self initialization.

    """

    def __init__(self, search, environment, actions=None,
                 search_params=None):
        super(GoalBasedAgent, self).__init__(
            environment=environment, actions=actions)

        self.search = search(agent=self, **(search_params or {}))
        self.actions_to_perform = []

    def act(self):
        if not self.actions_to_perform:
            self.actions_to_perform = (self.search
                                       .restart(self.last_known_state)
                                       .search()
                                       .backtrack()
                                       .solution_path_as_action_list())
            logger.info('agent has set action path: %s',
                        str(self.actions_to_perform))

        return (self.actions_to_perform.pop(0)
                if self.actions_to_perform
                else None)

    def is_goal(self, state):
        """Checks if state is agent's local goal.

        By default, an agent's local goal it's the environment global goal.
        This should be overridden if required.

        Parameters
        ----------
        state : State-like object
            The state to be checked.

        """
        return state.is_goal
