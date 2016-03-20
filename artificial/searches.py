import abc
import warnings

from artificial.base.helpers import PriorityQueue

from artificial import agents


class Search(metaclass=abc.ABCMeta):
    requires_backtracking = True

    def __init__(self, agent, root=None):
        assert isinstance(agent, agents.GoalBasedAgent), \
            'First Search requires an goal based agent.'

        self.agent = agent
        self.space = set()
        self.root = self.solution_candidate = self.solution_path = None

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

        if self.requires_backtracking:
            self.backtrack()

        return self

    def _perform(self):
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
        return ([s.action for s in self.solution_path[1:]]
                if self.solution_path
                else None)


class FirstSearch(Search, metaclass=abc.ABCMeta):
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

        if self.agent.verbose:
            warnings.warn('Could not find a path (%s:root)->(:goal)'
                          % self.root)

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


class BreadthFirstSearch(FirstSearch):
    """Breadth-First Search.

    """

    def extract(self):
        return self.fringe.pop(0)

    def expand(self, state):
        unseen_children = [s for s in self.agent.predict(state)
                           if s not in self.space]
        self.space = self.space.union(unseen_children)
        self.fringe += unseen_children


class UniformCostSearch(FirstSearch):
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
            if child in self.space:
                # Expanded nodes were already optimally reached.
                # Just ignore this new instance.
                continue

            if child not in self.fringe or child.g < self.fringe[child][0]:
                # This is either a new state or its costs is smaller than
                # the instance found in the fringe, being a shorter path.
                # Relax edge (thank you for this, Dijkstra).
                self.fringe.add(child, priority=child.g)


class GreedyBestFirstSearch(UniformCostSearch):
    def expand(self, state):
        self.space.add(state)

        for child in self.agent.predict(state):
            if child not in self.space and child not in self.fringe:
                self.fringe.add(child, priority=child.h)


class AStar(UniformCostSearch):
    def heuristic(self, state):
        raise NotImplementedError


class DepthFirstSearch(FirstSearch):
    """Depth-First Search.

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


class DepthLimitedSearch(DepthFirstSearch):
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


class IterativeDeepeningSearch(Search):
    def __init__(self, agent, root=None, limit=10):
        super().__init__(agent=agent, root=root)

        self.limit = limit

    def _perform(self):
        for limit in range(1, self.limit + 1):
            state = (DepthLimitedSearch(self.agent,
                                        root=self.root, limit=limit)
                     .perform().solution_candidate)

            if state:
                return state
