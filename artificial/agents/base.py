"""Artificial Agents Base"""

# Author: Lucas David -- <ld492@drexel.edu>
# License: MIT (c) 2016

import abc

import six

from ..base import Environment


@six.add_metaclass(abc.ABCMeta)
class AgentBase:
    """Agent Base Template.

    Defines a basic contract shared between all agents.

    Arguments
    ---------
    environment : Environment
        The environment upon which the agent will act.

    actions : list-like (optional)
        Which actions an agent has. This is used as a reminder for
        `predict` implementations and it's optional,

    """

    def __init__(self, environment, actions=None):
        if environment is not None and not isinstance(environment, Environment):
            raise ValueError('Illegal type (%s) for environment. It should '
                             'be an object of an Environment\'s subclass' %
                             type(environment))
        self.environment = environment
        self.actions = actions
        self.last_state = None
        self.last_known_state = None

    def perceive(self):
        """Perceive the environment and save current state."""
        self.last_state = self.environment.current_state

        if self.last_state:
            self.last_known_state = self.last_state

        return self

    def act(self):
        """Decides which action should be performed over the world,
        and return its code.

        """
        raise NotImplementedError
