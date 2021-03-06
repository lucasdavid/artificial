"""Artificial Agents Base"""

# Author: Lucas David -- <ld492@drexel.edu>
# License: MIT (c) 2016

import abc

import six
from sklearn.utils import check_random_state

from ..base import Environment


@six.add_metaclass(abc.ABCMeta)
class AgentBase(object):
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

    def __init__(self, environment, actions=None, random_state=None):
        self.environment = environment
        self.actions = actions
        self.last_state = None
        self.last_known_state = None
        self.random_state = check_random_state(random_state)

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
