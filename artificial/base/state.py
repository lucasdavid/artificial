import abc
import copy as copy_lib


class State:
    """State.

    Keeps track of episodic updates in the environment, maintaining
    a sequence through a recursive reference to parent states.
    
    Attributes
    ----------
    computed_utility_ : float
        The utility of a state which was already computed once.
        The attribute is then set, preventing unnecessary work
        to be done when utility is called multiple times and 
        it's an expensive job (e.g., LocalBeam, GeneticAlgorithm
        searches).
  
    Examples
    --------
    Note: these are merely toy examples on how to represent problems using
    `State`. You will most likely want to extend `State` class and override
    `is_goal` and `h` methods.

    # 1. Romania Routing Problem.
    State(0)

    # 2. Dirt Cleaner Problem.
    # The first four elements in the list represent if the sector
    # is dirt or not. The last one contains the agent's current position.
    State([1, 1, 1, 1, 0])

    """

    def __init__(self, data, parent=None, action=None, g=0):
        self.data = data
        self.parent = parent
        self.action = action

        self.g = g
        self.computed_utility_ = None

    @property
    def is_goal(self):
        """Checks if `State` object is the environment's goal.

        Some problems involve searching if a state is the agent's goal or
        the environment's (i.e. global) goal.
        By default, `GoalBasedAgent.is_goal` property is exactly the state's
        `is_goal` property. Hence this must be overridden according to the
        problem at hand.

        """
        return False

    def h(self):
        """Heuristic Function.

        An heuristic function is used by some searches, such as `GreedyFirst`
        and `AStar`, in an attempt to decrease the process' time and memory
        requirements.

        """
        return 0

    def f(self):
        """F Function.

        A sum of the local cost and the heuristic function.
        """
        return self.g + self.h()

    def mitosis(self, parenting=True, copy=True, **mutation):
        """Nuclear division of current state into a new one.

        Parameters
        ----------
        parenting : bool (default=True)
            Define clone parent as :self if true. parent will be None,
            otherwise.

        copy : bool (default=True)
            Deeply copy data if True. Otherwise, data is simply transfered to
            the newly divided `State`.

        mutation : dict
            Attributes which should mutate, as well as their mutated values.

        Notes
        -----
        By default, divisions create a child s.t. `child.g = parent.g + 1`.
        This can be overridden by simply passing the parameter g in `mutation`
        dict.

        Returns
        -------
        The clone made.

        """

        if 'g' not in mutation:
            mutation['g'] = self.g + 1

        # self.__class__ is used here instead of `State` directly,
        # as we want to keep the specified class implemented by the user.
        return self.__class__(
            data=copy_lib.deepcopy(self.data) if copy else self.data,
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

    @classmethod
    def random(cls):
        """Generate Random State.

        A class method that generates a random state. This is useful for
        optimization problems or genetic algorithm searches, where the
        *solution path* is not important (nor the starting point),
        in opposite to the final state itself.

        Notes
        -----
        Searches that allow random restart, (e.g.: `HillClimbing`) might
        require a valid implementation of this method.

        """
        raise NotImplementedError


class GeneticState(State):
    """Genetic State.

    State abstraction for searches using `GeneticAlgorithm` class.

    """
    def cross(self, other):
        raise NotImplementedError

    def mutate(self, factor, probability):
        raise NotImplementedError
