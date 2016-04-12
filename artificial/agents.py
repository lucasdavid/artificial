import abc
import warnings


class Agent(metaclass=abc.ABCMeta):
    """Agent base class.

    Arguments
    ---------

    environment : (None|Environment)
        The environment upon which the agent will act.

    verbose     : (True|False)
        The mode in which the agent operates.
        If True, errors or warnings are always sent to the output buffer.

    """

    def __init__(self, environment, actions, verbose=False):
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

    def act(self):
        raise NotImplementedError

    def predict(self, state):
        raise NotImplementedError


class TableDrivenAgent(Agent, metaclass=abc.ABCMeta):
    """Table Driven Agent.

    Basic intelligent agent based table of percepts.
    """

    def __init__(self, action_map, environment, actions, verbose=False):
        super().__init__(environment=environment, actions=actions,
                         verbose=verbose)

        self.action_map = action_map
        self.percepts = ''

    def perceive(self):
        super().perceive()
        self.percepts += hash(self.environment.current_state)

        return self

    def act(self):
        if self.percepts not in self.action_map:
            if self.verbose:
                warnings.warn('Perception sequence {%s} doesn\'t have a '
                              'correspondent on action map.' % self.percepts)
            return None

        return self.action_map[self.percepts]


class SimpleReflexAgent(Agent, metaclass=abc.ABCMeta):
    """Simple Reflex Agent.

    Basic intelligent agent based on decision rules.
    """

    def __init__(self, rules, environment, actions, verbose=False):
        super().__init__(environment=environment, actions=actions,
                         verbose=verbose)

        self.rules = rules

    def act(self):
        state = self.last_state

        if state not in self.rules:
            if self.verbose:
                warnings.warn('Rule set doesn\'t describe an action for '
                              'state %s' % state)
            return None

        return self.rules[state]


class ModelBasedAgent(Agent, metaclass=abc.ABCMeta):
    """ModelBasedAgent base class.

    Basic model-based agent for partially observable environments.
    This agent will perceive the environment when possible and
    infer it when otherwise.

    Notes:
        The method `act` overrides the superclass implementation in order
        to retrieve the last action being performed. This value must be
        stored as it might be used when inferring the environment state.

        Naturally, ModelBasedAgent must come before any classes that implement
        `act` and `perceive` methods in the inheritance list. E.g.:

        ```
        class ModelBasedReflexAgent(ModelBasedAgent, SimpleReflexAgent):
            ...
        ```
    """

    def __init__(self, model, environment, actions, verbose=False):
        super().__init__(environment=environment, actions=actions,
                         verbose=verbose)
        self.model = model
        self.last_action = None

    def perceive(self):
        # Method's simple sketch.
        #
        # super().perceive()
        #
        # if not self.last_state:
        #     # Current state is unknown. We must infer it.
        #     for state in self.predict(self.last_known_state):
        #         if state.action == self.last_action:
        #             self.last_known_state = state
        #             break
        #
        # return self
        raise NotImplementedError

    def act(self):
        action = super().act()
        self.last_action = action

        return action


class GoalBasedAgent(Agent, metaclass=abc.ABCMeta):
    """GoalBasedAgent base class.

    This class works as a wrapper around goal-based artificial. Obviously,
    `search` must be implemented correctly in order to achieve
    the desired 'goal-seeking' behavior.

    This is also the implementation for Simple-Problem-Solving-Agent,
    as described by S. Russell & P. Norvig in Artificial Intelligence,
    An Modern Approach, 3rd ed. p. 67. I decided to merge both classes
    as they present too many similarities, and separation would be
    a code smell.

    This agent searches for a desired action sequence and executes
    it completely. When it is finally done, a new search is made and a
    new action sequence will be performed.

    Parameters
    ----------
        goal   : the state which the agent seeks to achieve.

        search : a Search's subclass that will be instantiated
                 on the agent's self initialization.
    """

    def __init__(self, search, environment, actions,
                 search_params=None, verbose=False):
        super().__init__(environment=environment, actions=actions,
                         verbose=verbose)

        self.search = search(agent=self, **(search_params or {}))
        self.actions_to_perform = []

    def act(self):
        if not self.actions_to_perform:
            self.actions_to_perform = (self.search
                                       .restart(self.last_known_state)
                                       .perform()
                                       .solution_path_as_action_list())

            if self.verbose:
                print('Agent has set action path: %s'
                      % str(self.actions_to_perform))

        return (self.actions_to_perform.pop(0)
                if self.actions_to_perform
                else None)

    def is_goal(self, state):
        """Checks if state is agent's local goal.

        By default, an agent's local goal it's the environment global goal.
        This should be overridden if required.

        Parameters
        ----------
        state : State-like object
            The state to be checked.

        """
        return state.is_goal


class UtilityBasedAgent(GoalBasedAgent, metaclass=abc.ABCMeta):
    """UtilityBasedAgent base class.

    The difference between agents of this category to goal-based agents
    is the search function, which attempts to find the goal satisfying some
    "utility" (or "happiness") measure.
    """

    def utility(self, state):
        """Utility of a state for the current agent's object.

        By default, agents attempt to minimize the cost function `state.f()`.

        Parameters
        ----------
        state : State-like object
            The state which should have its utility to the agent computed.

        """
        return -state.f()


class LearningAgent(Agent, metaclass=abc.ABCMeta):
    pass
