"""Artificial Environment"""

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
    state_class_ : State subclass
        A link to the current's domain State specification, for reference
        purposes. Useful for domain-specific methods that create random
        instances of a state with `...state_class_.random()`.

    current_state : State-like object
        The model that represents the current state of the world.

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

    def live(self, n_cycles=100, verbose=False):
        """Make the environment alive!

        Bring the Environment to life, and run it through `n_cycles` cycles.

        Parameters
        ----------
        n_cycles: int, default=100
            The number of cycles in which `env.update` will be called.

        verbose: bool, default=True
            Constant info regarding the current state of the environment is
            is displayed to the user, if True.

        """
        self.build()

        if verbose:
            print('Initial state: %s' % str(self.current_state))

        try:
            cycle = 0

            while cycle < n_cycles:
                self.update()
                cycle += 1

        except KeyboardInterrupt:
            if verbose:
                print('canceled by user.')
        finally:
            if verbose:
                print('Final state: %s' % str(self.current_state))
