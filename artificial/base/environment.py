import abc
import random



class Environment(metaclass=abc.ABCMeta):
    """Environment Base Class.

    Defines how agents and states intertwine, modeling a problem domain
    into the computer.
    Obviously,you must subclass `Environment` for every different problem
    faced.


    Parameters
    ----------

    initial_state : State-like object
        Initial state of the environment.

    random_generator : Random object (default=None)
        Random instance used for debugging purposes.

    """

    def __init__(self, initial_state, random_generator=None):
        self.current_state = self.initial_state = initial_state
        self.random_generator = random_generator or random.Random()
        self.agents = []

    def update(self):
        raise NotImplementedError

    def finished(self):
        return self.current_state.is_goal

    def generate_random_state(self):
        """Generate Random State.

        A class method that generates a random state. This is useful for
        optimization problems, where the *solution path* is not important
        (nor the starting point), in opposite to the final state itself.


        Notes
        -----
        Searches that allow random restart, (e.g.: `HillClimbing`) might
        require a valid implementation of this method.

        """
        raise NotImplementedError
