"""Table Driven Agent"""

# Author: Lucas David -- <ld492@drexel.edu>
# License: MIT (c) 2016

import warnings
from . import base


class TableDrivenAgent(base.AgentBase):
    """Table Driven Agent.

    Basic intelligent agent based table of percepts.
    """

    def __init__(self, action_map, environment, actions=None,
                 random_state=None):
        super(TableDrivenAgent, self).__init__(environment=environment,
                                               actions=actions,
                                               random_state=random_state)
        self.action_map = action_map
        self.percepts = ''

    def perceive(self):
        super(TableDrivenAgent, self).perceive()
        self.percepts += str(hash(self.environment.current_state))

        return self

    def act(self):
        if self.percepts not in self.action_map:
            warnings.warn('Perception sequence {%s} doesn\'t have a '
                          'correspondent on action map.' % self.percepts)
            return None

        return self.action_map[self.percepts]
