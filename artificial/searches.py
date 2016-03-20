import abc
import warnings

from artificial.base.helpers import PriorityQueue

from artificial import agents


class Base(metaclass=abc.ABCMeta):
    """Base Search.

    Common interface for searches, including agent, space and root properties.

    Parameters
    ----------
    agent : Agent-like
            Agent that requested search. This is needed as only an agent
            of the specific domain problem can predict outcomes based on
            their own actions.

    root  : State-like
            The state in which the search should start.

    backtracks : bool (default=True)
                 Once the goal set is reached, backtracking will be performed
                 in order to create a sequence of intermediate states.
                 If this operation is not needed or too much time-consuming,
                 False can be passed and this will be skipped.

    Attributes
    ----------
    space : set
            A set used to contain states and efficient repetition checking.

    solution_candidate : State-like
            A State's subclass found when performing the search.

    solution_path : list
            A list of intermediate states to achieve the goal state.
            This attribute is undefined when `backtracks` parameter
            is set to False.

    """
    def __init__(self, agent, root=None, backtracks=True):
        assert isinstance(agent, agents.GoalBasedAgent), \
            'First Search requires an goal based agent.'

        self.agent = agent
        self.root = self.solution_candidate = self.solution_path = None

        self.space = set()
        self.backtracks = backtracks

        self.restart(root)

    def restart(self, root):
        self.root = root
        self.space = {root} if root else set()
        self.solution_candidate = None
        self.solution_path = None

        return self

    def perform(self):
        self.solution_candidate = self.solution_path = None
        self.solution_candidate = self._perform()

        if not self.solution_candidate and self.agent.verbose:
            warnings.warn('Could not find a path (%s:root)->(:goal)'
                          % self.root)

        if self.backtracks:
            self.backtrack()

        return self

    def _perform(self):
        """Actual search performing.

        Returns
        -------
        State-like, A solution candidate for the search.
        Usually, the state's `is_goal` property is True, but that's not really
        required. Base's `perform` method will then attempt to backtrack the
        state, recording a sequence of intermediate states.

        """
        raise NotImplementedError

    def backtrack(self):
        state_sequence = []

        state = self.solution_candidate
        while state:
            state_sequence.insert(0, state)
            state = state.parent

        self.solution_path = state_sequence

        return self

    def solution_path_as_action_list(self):
        """Build a list of actions from the solution candidate path.

        Naturally, must be executed after `perform` call.

        Returns
        -------
        actions : array-like, a list containing the actions performed
                  by the sates in `solution_path`.
        """
        return ([s.action for s in self.solution_path[1:]]
                if self.solution_path
                else None)


class FringeBase(Base, metaclass=abc.ABCMeta):
    """Fringe Base Search.

    Base class for searchers that rely on the concept of fringe.
    Fringes are, by default, `lists`, but can be freely overridden
    for other structures, such as `sets` or `PriorityQueues`.

    Attributes
    ----------
    fringe : list
             A collection of states in the search fringe.

    """
    def __init__(self, agent, root=None):
        super().__init__(agent=agent, root=root)

        self.fringe = list(self.space) if self.space else []

    def restart(self, root):
        super().restart(root=root)
        self.fringe = list(self.space)

        return self

    def _perform(self):
        while self.fringe:
            state = self.extract()

            if state is None:
                continue

            if state.is_goal:
                return state

            self.expand(state)

    def extract(self):
        """Fringe extraction policy.
        """
        raise NotImplementedError

    def expand(self, state):
        """Fringe expansion policy.

        Parameters
        ----------
            state : (State)
                State that should be expanded.
        """
        raise NotImplementedError


class BreadthFirst(FringeBase):
    """Breadth First Search.

    Extract elements from the beginning of the fringe and add expanded states'
    children to the end of it. It's complete and minimal.

    """

    def extract(self):
        return self.fringe.pop(0)

    def expand(self, state):
        unseen_children = [s for s in self.agent.predict(state)
                           if s not in self.space]
        self.space = self.space.union(unseen_children)
        self.fringe += unseen_children


class UniformCost(FringeBase):
    """Uniform Cost Search.

    Uses a PriorityQueue as fringe, adding and removing states
    according to the path cost required to reach them. This search is
    complete, minimal and optimal.

    """

    def __init__(self, agent, root=None):
        super().__init__(agent=agent, root=root)

        assert isinstance(agent, agents.UtilityBasedAgent), \
            'Uniform Cost Search requires an utility based agent.'

        self.fringe = PriorityQueue()

        if self.root:
            self.fringe.add(self.root)

    def restart(self, root):
        super().restart(root=root)

        self.fringe = PriorityQueue()
        self.fringe.add(self.root)

        return self

    def extract(self):
        return self.fringe.pop()

    def expand(self, state):
        self.space.add(state)

        for child in self.agent.predict(state):
            if child not in self.space and (child not in self.fringe or
                                            child.g < self.fringe[child][0]):
                # Expanded nodes were already optimally reached.
                # Just ignore these instances instance.
                # This is either a new state or its costs is smaller than
                # the instance found in the fringe, being a shorter path.
                # Relax edge (thank you for this, Dijkstra).
                self.fringe.add(child, priority=child.g)


class GreedyBestFirst(UniformCost):
    """Greedy Best First Search.

    Like Uniform Cost, uses a PriorityQueue as fringe, but adds and extracts
    states based on their evaluation by a predefined heuristic function.
    This is NOT complete, optimal or minimal; but will likely achieve a
    solution quickly and without the need to expand too many states.

    """
    def expand(self, state):
        self.space.add(state)

        for child in self.agent.predict(state):
            if child not in self.space and child not in self.fringe:
                # Only add states that aren't in the fringe yet.
                # Recurrent states are likely to have the same heuristic value,
                # but we chose to keep the one that was added first
                # (less hops <=> closer to the root).
                self.fringe.add(child, priority=child.h())


class AStar(GreedyBestFirst):
    """A Star (A*) Search.

    Combines Uniform cost and Greedy best first to add/remove states
    from the priority queue based on their distance from the current node
    and their evaluation of the heuristic function.

    This search is complete, minimal and optimal given that the heuristic
    is admissible and consistent.

    """
    def expand(self, state):
        self.space.add(state)

        for child in self.agent.predict(state):
            if child not in self.space and (child not in self.fringe or
                                            child.f() < self.fringe[child][0]):
                self.fringe.add(child, priority=child.f())


class DepthFirst(FringeBase):
    """Depth First Search.

    Parameters
    ----------
    prevent_cycles : (False|'branch'|'tree')
        Prevent cyclical searches.

        Options are:
            False : classic Depth First Search. Algorithm will NOT keep
            tab on repetitions and cycles may occur.

            'branch' : repetitions in current branch will not be allowed.
            Requires `O(2d)` memory, as references to predecessors and
            a set of states in current path are kept.

            'tree' : no repetitions are allowed. This option requires
            `O(b^d + d)`, being no better than Breadth-First search
            in memory requirements.
            It can still perform better, however, given a problem domain
            where the solutions is "far" from the root and an optimal goal
            in number of hops is not necessary.
    """

    def __init__(self, agent, root=None, prevent_cycles=False):
        super().__init__(agent=agent, root=root)

        self.prevent_cycles = prevent_cycles
        self.last_expanded = None

    def extract(self):
        previous = self.last_expanded
        current = self.fringe.pop(0)
        common = current.parent
        self.last_expanded = current

        if self.prevent_cycles == 'branch':
            # Remove other branches from the search space.
            if previous and common and previous != common:
                # We switched branches, perform removal.
                while previous and previous != common:
                    self.space.remove(previous)
                    previous = previous.parent

        return current

    def expand(self, state):
        children = self.agent.predict(state)

        if self.prevent_cycles:
            children = [s for s in children if s not in self.space]
            self.space = self.space.union(children)

        self.fringe = children + self.fringe


class DepthLimited(DepthFirst):
    """Depth Limited Search.

    Executes Depth-first search up to a limit and shortcuts branches
    that violate this limit. If no solution candidate is found before
    the limit, `DepthLimited` won't be able to properly answer
    the environment with a action list.

    Obviously, this search is not complete, minimal, or optimal.

    Default cycles prevention policy is `False`, as depth limitation
    is a cycle prevention in itself. Still, 'branch' and 'tree' options
    are still available when further refining is desired.

    Parameters
    ----------
    limit : int (default=10)
            Maximum depth allowed.

    """
    def __init__(self, agent, root=None, prevent_cycles=False, limit=10):
        super().__init__(agent=agent, root=root, prevent_cycles=prevent_cycles)

        self.limit = limit

    def extract(self):
        state = super().extract()

        n_level = 0
        parent = state.parent
        while parent:
            n_level += 1
            parent = parent.parent

        return state if n_level <= self.limit else None


class IterativeDeepening(Base):
    """Iterative Deepening Search.

    Taking an iterative, executes `DepthLimited` passing the iteration's
    value as the `limit` parameter. This search is minimal, given the
    iterative includes the left-side of the Natural set
    (i.e., 1, 2, 3, 4, ...), but not complete nor necessarily optimal.

    Parameters
    ----------
    iterative : (array-like|range)
                list of limits passed to `DepthLimited`.

    """

    def __init__(self, agent, root=None, prevent_cycles=False,
                 iterative=range(10)):
        super().__init__(agent=agent, root=root)

        self.iterative = iterative
        self.depth_limited = DepthLimited(agent=agent, root=root,
                                          prevent_cycles=prevent_cycles)

    def restart(self, root):
        super().restart(root)
        self.depth_limited.restart(root)

        return self

    def _perform(self):
        for limit in self.iterative:
            self.depth_limited.restart(self.root).limit = limit

            state = self.depth_limited.perform().solution_candidate
            if state:
                return state


class HillClimbing(Base):
    """Hill Climbing Search.

    Perform Hill Climbing search according to a designated strategy.

    Attributes
    ----------

    strategy : ('default'|'random-restart')
        Defines which strategy HillClimbing will follow.

        Options are:

        --- 'default' : agent will navigate until find a local optimal point.

        --- 'random-start' : agent will restart randomly after finding a
                             local optimal point.

    """
    def __init__(self, agent, root=None, strategy='default'):
        super().__init__(agent=agent, root=root)

        assert isinstance(agent, agents.UtilityBasedAgent), \
            'Hill Climbing Search requires an utility based agent.'

        self.strategy = strategy
        self.current = root

    def restart(self, root):
        super().restart(root=root)
        self.current = root

        return self

    def _perform(self):
        stable = False

        while not stable:
            children = self.agent.predict(self.current)
            utility = self.agent.utility(self.current)
            stable = True

            for c in children:
                c_utility = self.agent.utility(c)
                if c_utility > utility:
                    self.current = c
                    stable = False

        return self.current
