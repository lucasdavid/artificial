"""Simple Reflex Agent"""

# Author: Lucas David -- <ld492@drexel.edu>
# License: MIT (c) 2016

import warnings

from . import base


class SimpleReflexAgent(base.Agent):
    """Simple Reflex Agent.

    Basic intelligent agent based on decision rules.
    """

    def __init__(self, rules, environment, actions=None, verbose=False):
        super(SimpleReflexAgent, self).__init__(
            environment=environment, actions=actions, verbose=verbose)

        self.rules = rules

    def act(self):
        state = self.last_state

        if state not in self.rules:
            warnings.warn('Rule set doesn\'t describe an action for '
                          'state %s' % state)
            return None

        return self.rules[state]
