"""Artificial Environments"""

# Author: Lucas David -- <ld492@drexel.edu>
# License: MIT (c) 2016

import abc

import six


@six.add_metaclass(abc.ABCMeta)
class Environment:
    """Environment Base Class.

    Defines how agents and states intertwine, modeling a problem domain
    into the computer.
    Obviously,you must subclass `Environment` for every different problem
    faced.

    Parameters
    ----------
    initial_state: State-like object, default=None
        Initial state of the environment.

    Attributes
    ----------
    current_state : State-like object
        Link to the current domain's State specification, for reference
        purposes.

    agents : list of Agent-like objects
        Contains a list of all agents currently inserted into the domain.

    """

    state_class_ = None

    def __init__(self, initial_state=None):
        self.current_state = self.initial_state = initial_state
        self.agents = []

    def build(self):
        """Build the environment, if necessary"""

    @abc.abstractmethod
    def update(self):
        """Update the environment, should be overridden to reflect the changes
        in the real world.

        """

    def finished(self):
        return self.current_state and self.current_state.is_goal
