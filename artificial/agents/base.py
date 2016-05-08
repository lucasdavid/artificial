"""Artificial Agents Base"""

# Author: Lucas David -- <ld492@drexel.edu>
# License: MIT (c) 2016

import abc

import six


@six.add_metaclass(abc.ABCMeta)
class Agent:
    """Agent Base.

    Arguments
    ---------
    environment : Environment
        The environment upon which the agent will act.

    actions : list-like (optional)
        Which actions an agent has. This is used as a reminder for
        `predict` implementations and it's optional,

    verbose : bool (default=False)
        The mode in which the agent operates.
        If True, errors or warnings are always sent to the output buffer.

    """

    def __init__(self, environment, actions=None, verbose=False):
        self.environment = environment
        self.actions = actions
        self.verbose = verbose
        self.last_state = None
        self.last_known_state = None

    def perceive(self):
        self.last_state = self.environment.current_state

        if self.last_state:
            self.last_known_state = self.last_state

        return self

    @abc.abstractmethod
    def act(self): pass
