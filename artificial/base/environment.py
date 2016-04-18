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
        Random instance used in debugging, yielding deterministic results.

    Attributes
    ----------
    state_class_ : State subclass
        State sub-class link to the current domain's State specification,
        for reference purposes. Usually required for searches that generate
        random states.

    """

    state_class_ = None

    def __init__(self, initial_state, random_generator=None):
        self.current_state = self.initial_state = initial_state
        self.random_generator = random_generator or random.Random()
        self.agents = []

    def update(self):
        raise NotImplementedError

    def finished(self):
        return self.current_state and self.current_state.is_goal
