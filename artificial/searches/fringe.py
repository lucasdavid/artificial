"""Artificial Fringe Searches"""

# Author: Lucas David -- <ld492@drexel.edu>
# License: MIT (c) 2016

import abc

import six

from . import base
from .. import agents
from ..utils import PriorityQueue


@six.add_metaclass(abc.ABCMeta)
class FringeBase(base.SearchBase):
    """Fringe Base Search.

    Base class for searchers that rely on the concept of fringe.
    Fringes are, by default, `lists`, but can be freely overridden
    for other structures, such as `sets` or `PriorityQueues`.

    Attributes
    ----------
    fringe_ : list
        A collection of states in the search fringe.

    """

    def __init__(self, agent, root=None):
        super(FringeBase, self).__init__(agent=agent, root=root)

        self.fringe_ = list(self.space_) if self.space_ else []

    def restart(self, root):
        super(FringeBase, self).restart(root=root)
        self.fringe_ = list(self.space_)

        return self

    def search(self):
        while self.fringe_:
            state = self.extract()

            if state is None:
                continue

            if state.is_goal:
                self.solution_candidate_ = state
                break

            self.expand(state)

        return self

    @abc.abstractmethod
    def extract(self):
        """Fringe extraction policy"""

    @abc.abstractmethod
    def expand(self, state):
        """Fringe expansion policy.

        Parameters
        ----------
            state : (State)
                State that should be expanded.

        """


class BreadthFirst(FringeBase):
    """Breadth First Search.

    Extract elements from the beginning of the fringe and add expanded states'
    children to the end of it. It's complete and minimal.

    """

    def extract(self):
        return self.fringe_.pop(0)

    def expand(self, state):
        unseen_children = [s for s in self.agent.predict(state)
                           if s not in self.space_]
        self.space_.update(unseen_children)
        self.fringe_ += unseen_children


class UniformCost(FringeBase):
    """Uniform Cost Search.

    Uses a `PriorityQueue` as fringe, adding and removing states
    according to the path cost required to reach them. This search is
    complete, minimal and optimal.

    """

    def __init__(self, agent, root=None):
        super(UniformCost, self).__init__(agent=agent, root=root)

        assert isinstance(agent, agents.UtilityBasedAgent), \
            'Uniform Cost Search requires an utility based agent.'

        self.fringe_ = PriorityQueue()

        if self.root:
            self.fringe_.add(self.root)

    def restart(self, root):
        super(UniformCost, self).restart(root=root)

        self.fringe_ = PriorityQueue()
        self.fringe_.add(self.root)

        return self

    def extract(self):
        return self.fringe_.pop()

    def expand(self, state):
        self.space_.add(state)

        for child in self.agent.predict(state):
            if (child not in self.space_ and
                (child not in self.fringe_ or
                 child.g < self.fringe_[child][0])):
                # Expanded nodes were already optimally reached.
                # Just ignore these instances instance.
                # This is either a new state or its costs is smaller than
                # the instance found in the fringe, being a shorter path.
                # Relax edge (thank you for this, Dijkstra).
                self.fringe_.add(child, priority=child.g)


class GreedyBestFirst(UniformCost):
    """Greedy Best First Search.

    Like Uniform Cost, uses a PriorityQueue as fringe, but adds and extracts
    states based on their evaluation by a predefined heuristic function.
    This is NOT complete, optimal or minimal; but will likely achieve a
    solution quickly and without the need to expand too many states.

    """

    def expand(self, state):
        self.space_.add(state)

        for child in self.agent.predict(state):
            if child not in self.space_ and child not in self.fringe_:
                # Only add states that aren't in the fringe yet.
                # Recurrent states are likely to have the same heuristic value,
                # but we chose to keep the one that was added first
                # (less hops <=> closer to the root).
                self.fringe_.add(child, priority=child.h())


class AStar(UniformCost):
    """A Star (A*) Search.

    Combines Uniform cost and Greedy best first to add/remove states
    from the priority queue based on their distance from the current node
    and their evaluation of the heuristic function.

    This search is complete, minimal and optimal given that the heuristic
    is admissible and consistent.

    """

    def expand(self, state):
        self.space_.add(state)

        for child in self.agent.predict(state):
            if (child not in self.space_ and
                (child not in self.fringe_ or
                 child.f() < self.fringe_[child][0])):
                self.fringe_.add(child, priority=child.f())


class DepthFirst(FringeBase):
    """Depth First Search.

    Parameters
    ----------
    prevent_cycles : [False|'branch'|'tree'] (default=False)
        Prevent cyclical searches.

        Options are:
            False : classic Depth First Search. Algorithm will NOT keep
            tab on repetitions and cycles may occur.

            'branch' : repetitions in current branch will not be allowed.
            Requires :math:`O(2d)` memory, as references to predecessors and
            a set of states in the current path are kept.

            'tree' : no repetitions are allowed. This option requires
            :math:`O(b^d + d)`, being no better than Breadth-First search
            in terms of memory requirement.
            It can still perform better, though, given a problem domain
            where solutions are "far" from the root and minimizing the
            number of hops to the solution is not necessary (something
            which is guaranteed by `BreadthFirst`).

    limit : [None|int] (default=None)
        If a positive integer, executes Limited Depth First Search
        up to a limit and shortcuts branches that violate this limit.
        If no solution candidate is found before the limit,
        `DepthLimited` won't be able to properly answer the environment
        with a action list.

        Obviously, this search is not complete, minimal, or optimal.

        If None, no limit is imposed and original Depth First algorithm
        is executed.

    """

    def __init__(self, agent, root=None, prevent_cycles=False, limit=None):
        super(DepthFirst, self).__init__(agent=agent, root=root)

        self.prevent_cycles = prevent_cycles
        self.limit = limit

        self.last_expanded = None

    def extract(self):
        previous = self.last_expanded
        current = self.fringe_.pop(0)
        common = current.parent
        self.last_expanded = current

        if self.prevent_cycles == 'branch':
            # Remove other branches from the search space.
            if previous and common and previous != common:
                # We switched branches, perform removal.
                while previous and previous != common:
                    self.space_.remove(previous)
                    previous = previous.parent

        if self.limit is None:
            return current

        # Checks if current depth violates limit constraint.
        d = 0
        p = current.parent
        while p:
            p, d = p.parent, d + 1
        if d <= self.limit:
            return current

    def expand(self, state):
        children = self.agent.predict(state)

        if self.prevent_cycles:
            children = [s for s in children if s not in self.space_]
            self.space_.update(children)

        self.fringe_ = children + self.fringe_


class IterativeDeepening(base.SearchBase):
    """Iterative Deepening Search.

    Taking an iterative, executes `DepthLimited` passing the iteration's
    value as the `limit` parameter. This search is minimal, given the
    iterative includes the left-side of the Natural set
    (i.e., 1, 2, 3, 4, ...), but not complete nor necessarily optimal.

    Parameters
    ----------
    prevent_cycles : [False|'branch'|'tree'] (default=False)
        Prevent cyclical searches.

        Options are:
            --- False : classic Depth First Search. Algorithm will NOT keep
            tab on repetitions and cycles may occur.

            --- 'branch' : repetitions in current branch will not be allowed.
            Requires :math:`O(2d)` memory, as references to predecessors and
            a set of states in the current path are kept.

            --- 'tree' : no repetitions are allowed. This option requires
            :math:`O(b^d + d)`, being no better than Breadth-First search in
            terms of memory requirement. It can still perform better, though,
            given a problem domain where solutions are "far" from the root and
            minimizing the number of hops to the solution is not necessary
            (something which is guaranteed by `BreadthFirst`).

    iterations : [array-like|range] (default=range(10))
        List of limits passed to `DepthFirst`.

    """

    def __init__(self, agent, root=None, prevent_cycles=False,
                 iterations=range(10)):
        super(IterativeDeepening, self).__init__(agent=agent, root=root)

        self.iterations = iterations
        self.depth_limited = DepthFirst(agent=agent, root=root,
                                        prevent_cycles=prevent_cycles)

    def search(self):
        for limit in self.iterations:
            self.depth_limited.limit = limit
            self.depth_limited.restart(root=self.root)
            self.solution_candidate_ = (self.depth_limited.search()
                                        .solution_candidate_)
            if self.solution_candidate_:
                break

        return self
