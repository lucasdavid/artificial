import abc
from . import base
from .. import agents
from ..base import helpers

class FringeBase(base.Base, metaclass=abc.ABCMeta):
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

        self.fringe = helpers.PriorityQueue()

        if self.root:
            self.fringe.add(self.root)

    def restart(self, root):
        super().restart(root=root)

        self.fringe = helpers.PriorityQueue()
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

    limit : (None|int)
            If a positive integer, executes Limited Depth First Search
            up to a limit and shortcuts branches that violate this limit.
            If no solution candidate is found before the limit,
            `DepthLimited` won't be able to properly answer the environment
            with a action list.

            Obviously, this search is not complete, minimal, or optimal.

            If None, no limit is imposed and original Depth First algorithm
            is executed.

    Notes
    -----

        If limit parameter is not None, the state's `g` property is used
        to assert its depth in the search tree. Users are then oblidge to
        correctly

    """

    def __init__(self, agent, root=None, prevent_cycles=False, limit=None):
        super().__init__(agent=agent, root=root)

        self.prevent_cycles = prevent_cycles
        self.limit = limit

        self.last_expanded = None

    def restart(self, root, limit=None):
        super().restart(root=root)
        self.limit = limit
        self.last_expanded = None

        return self

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

        # Checks if current depth violates limit constraint.
        return (current
                if self.limit is None or current.g <= self.limit
                else None)

    def expand(self, state):
        children = self.agent.predict(state)

        if self.prevent_cycles:
            children = [s for s in children if s not in self.space]
            self.space = self.space.union(children)

        self.fringe = children + self.fringe


class IterativeDeepening(base.Base):
    """Iterative Deepening Search.

    Taking an iterative, executes `DepthLimited` passing the iteration's
    value as the `limit` parameter. This search is minimal, given the
    iterative includes the left-side of the Natural set
    (i.e., 1, 2, 3, 4, ...), but not complete nor necessarily optimal.

    Parameters
    ----------
    iterations : [array-like|range] (default=range(10))
                list of limits passed to `DepthFirst`.

    """

    def __init__(self, agent, root=None, prevent_cycles=False,
                 iterations=range(10)):
        super().__init__(agent=agent, root=root)

        self.iterations = iterations
        self.depth_limited = DepthFirst(agent=agent, root=root,
                                        prevent_cycles=prevent_cycles)

    def restart(self, root):
        super().restart(root)
        self.depth_limited.restart(root)

        return self

    def _perform(self):
        for limit in self.iterations:
            self.depth_limited.restart(root=self.root, limit=limit)

            state = self.depth_limited.perform().solution_candidate
            if state:
                return state
