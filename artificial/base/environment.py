"""Artificial Environment"""

# Author: Lucas David -- <ld492@drexel.edu>
# License: MIT (c) 2016

import abc
import logging

import six

logger = logging.getLogger('artificial')


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
    state_class_ : State subclass
        A link to the current's domain State specification, for reference
        purposes. Useful for domain-specific methods that create random
        instances of a state with `...state_class_.random()`.

    current_state : State-like object
        The model that represents the current state of the world.

    agents : list of Agent-like objects
        Contains a list of all agents currently inserted into the domain.

    """

    __instance = None
    state_class_ = None

    def __init__(self, initial_state=None):
        self.current_state = self.initial_state = initial_state
        self.agents = []
        Environment.__instance = self

    def build(self):
        """Build the environment, if necessary"""
        return self

    def dispose(self):
        self.current_state = None
        self.agents = []

        Environment.__instance = None

        return self

    def __del__(self):
        self.dispose()

    @classmethod
    def current(cls):
        if cls.__instance is None:
            raise RuntimeError('no %s is currently running', cls.__name__)
        return cls.__instance

    @abc.abstractmethod
    def update(self):
        """Update the environment, should be overridden to reflect the changes
        in the real world.

        """

    def finished(self):
        return self.current_state and self.current_state.is_goal

    def live(self, n_cycles=100):
        """Make the environment alive!

        Bring the Environment to life, and run it through `n_cycles` cycles.

        Parameters
        ----------
        n_cycles: int, default=100
            The number of cycles in which `env.update` will be called.

        """
        self.build()

        logger.info('initial state: %s', str(self.current_state))

        try:
            cycle = 0

            while cycle < n_cycles and not self.finished():
                self.update()
                cycle += 1

                logger.info('#%i: {%s}' % (cycle, str(self.current_state)))

        except KeyboardInterrupt:
            logger.info('canceled by user.')
        finally:
            logger.info('final state: %s', str(self.current_state))
