"""Simple Reflex Agent"""

# Author: Lucas David -- <ld492@drexel.edu>
# License: MIT (c) 2016
import logging

from . import base

logger = logging.getLogger('artificial')


class SimpleReflexAgent(base.AgentBase):
    """Simple Reflex Agent.

    Basic intelligent agent based on decision rules.
    """

    def __init__(self, rules, environment, actions=None,
                 random_state=None):
        super(SimpleReflexAgent, self).__init__(environment=environment,
                                                actions=actions,
                                                random_state=random_state)
        self.rules = rules

    def act(self):
        state = self.last_state

        if state not in self.rules:
            logger.warning('Rule set doesn\'t describe an action for '
                           'state %s', state)
            return None

        return self.rules[state]
