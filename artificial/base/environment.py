import abc
import copy


class State(metaclass=abc.ABCMeta):
    """State class.
    
    Keeps track of episodic updates in the environment, maintaining 
    a sequence through a recursive reference to parent states.
    """

    def __init__(self, data, parent=None, action=None, g=0):
        self.data = data
        self.parent = parent
        self.action = action

        self.g = g

    @property
    def is_goal(self):
        raise NotImplementedError

    def h(self):
        return 0

    def f(self):
        return self.g + self.h()

    def mitosis(self, parenting=True, **mutation):
        """Nuclear division of current state into a new one.

        Parameters
        ----------
        parenting : bool (default=True)
            Define clone parent as :self if true. parent will be None,
            otherwise.

        mutation  : dict
            Attributes which should mutate, as well as their mutated values.

        Returns
        -------
        The clone made.
        """

        # self.__class__ is used here instead of `State` directly,
        # as we want to keep the specified class implemented by the user.
        return self.__class__(
            data=copy.deepcopy(self.data),
            parent=self if parenting else None,
            **mutation
        )

    def __eq__(self, other):
        return isinstance(other, State) and self.data == other.data

    def __hash__(self):
        try:
            return hash(self.data)

        except TypeError:
            # Attempts to convert data to str first,
            # as strings are always hashable.
            return hash(str(self.data))

    def __str__(self):
        return ('data: %s, action: %s, g: %d'
                % (str(self.data), self.action, self.g))


class Environment(metaclass=abc.ABCMeta):
    """Environment base class.
    
    Contains artificial and states and defines how these intertwine.
    You must subclass `Environment` for every different problem faced.
    """

    def __init__(self, initial_state):
        self.current_state = self.initial_state = initial_state
        self.agents = []

    def update(self):
        raise NotImplementedError

    def finished(self):
        return self.current_state.is_goal
