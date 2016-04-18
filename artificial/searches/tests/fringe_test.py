from unittest import TestCase

from artificial import agents, base
from artificial.searches import fringe as fringe_searches


class _TState(base.State):
    @property
    def is_goal(self):
        return self.g == 2


class _UtilityTestAgent(agents.UtilityBasedAgent):
    def predict(self, state):
        if state.g >= 2:
            return []

        children = [state.mitosis(action='m') for _ in range(2 - state.g)]
        for i, c in enumerate(children):
            c.data += state.data * (i + 1)

        return children


class UniformCostSearchTest(TestCase):
    def test_sanity(self):
        with self.assertRaises(AssertionError):
            fringe_searches.UniformCost(agent=None, root=_TState(10))

        a = _UtilityTestAgent(fringe_searches.UniformCost, None, None)
        s = fringe_searches.UniformCost(agent=a, root=_TState(10))
        self.assertIsNotNone(s)

    def test_search(self):
        a = _UtilityTestAgent(fringe_searches.UniformCost, None, None)
        s = fringe_searches.UniformCost(agent=a).restart(_TState(10))
        s.search().backtrack()

        self.assertTrue(s.solution_candidate_.is_goal)
        self.assertTrue(s.solution_path_)

    def test_solution_path_as_action_list(self):
        a = _UtilityTestAgent(fringe_searches.UniformCost, None, None)
        s = fringe_searches.UniformCost(agent=a, root=_TState(10))
        actions = s.search().backtrack().solution_path_as_action_list()

        self.assertListEqual(actions, ['m', 'm'])


class GreedyBestSearchTest(TestCase):
    def test_sanity(self):
        with self.assertRaises(AssertionError):
            fringe_searches.GreedyBestFirst(agent=None, root=_TState(10))

        a = _UtilityTestAgent(fringe_searches.UniformCost, None, None)
        s = fringe_searches.GreedyBestFirst(agent=a, root=_TState(10))
        self.assertIsNotNone(s)

    def test_search(self):
        a = _UtilityTestAgent(fringe_searches.GreedyBestFirst, None, None)
        s = fringe_searches.GreedyBestFirst(agent=a).restart(_TState(10))
        s.search().backtrack()

        self.assertTrue(s.solution_candidate_.is_goal)
        self.assertTrue(s.solution_path_)


class AStarTest(TestCase):
    def test_sanity(self):
        with self.assertRaises(AssertionError):
            fringe_searches.AStar(agent=None, root=_TState(10))

        a = _UtilityTestAgent(fringe_searches.AStar, None, None)
        s = fringe_searches.AStar(agent=a, root=_TState(10))
        self.assertIsNotNone(s)

    def test_search(self):
        a = _UtilityTestAgent(fringe_searches.AStar, None, None)
        s = fringe_searches.AStar(agent=a).restart(_TState(10))
        s.search().backtrack()

        self.assertTrue(s.solution_candidate_.is_goal)
        self.assertTrue(s.solution_path_)


class BreadthFirstSearchTest(TestCase):
    def test_sanity(self):
        with self.assertRaises(AssertionError):
            fringe_searches.BreadthFirst(agent=None, root=_TState(10))

        a = _UtilityTestAgent(fringe_searches.BreadthFirst, None, None)
        s = fringe_searches.BreadthFirst(agent=a, root=_TState(10))
        self.assertIsNotNone(s)

    def test_search(self):
        a = _UtilityTestAgent(fringe_searches.BreadthFirst, None, None)
        s = fringe_searches.BreadthFirst(agent=a).restart(_TState(10))
        s.search()

        self.assertTrue(s.solution_candidate_.is_goal)

        # Search space size should be:
        # |(root) + 2 (root children) + 2 (solution candidates)| = 5.
        self.assertEqual(len(s.space_), 5)

        s.backtrack()
        self.assertTrue(s.solution_path_)

    def test_solution_path_as_action_list(self):
        a = _UtilityTestAgent(fringe_searches.BreadthFirst, None, None)
        s = fringe_searches.BreadthFirst(agent=a, root=_TState(10))
        actions = s.search().backtrack().solution_path_as_action_list()

        self.assertListEqual(actions, ['m', 'm'])


class DepthFirstSearchTest(TestCase):
    def test_sanity(self):
        with self.assertRaises(AssertionError):
            fringe_searches.DepthFirst(agent=None, root=_TState(10))

        a = _UtilityTestAgent(fringe_searches.DepthFirst, None, None)
        s = fringe_searches.DepthFirst(agent=a, root=_TState(10))
        self.assertIsNotNone(s)

    def test_search(self):
        a = _UtilityTestAgent(fringe_searches.DepthFirst, None, None)
        s = (fringe_searches
             .DepthFirst(agent=a)
             .restart(_TState(10))
             .search()
             .backtrack())

        self.assertTrue(s.solution_candidate_.is_goal)
        self.assertTrue(s.solution_path_)

        # Search space_ stores only root, as we didn't initialized
        # Depth-first search with cycle prevention.
        self.assertEqual(len(s.space_), 1)

        # All cycle prevention policies should
        # find a `solution_candidate_`.
        s.prevent_cycles = 'branch'
        s.restart(_TState(10)).search().backtrack()
        self.assertTrue(s.solution_candidate_.is_goal)
        self.assertTrue(s.solution_path_)

        # Search space:
        # ((root:g=0)->(g=1)->(g=2:goal)
        #            ->(g=1)
        self.assertEqual(len(s.space_), 4)

        s.prevent_cycles = 'tree'
        s.restart(_TState(10)).search().backtrack()
        self.assertTrue(s.solution_candidate_.is_goal)
        self.assertTrue(s.solution_path_)

        # Search space:
        # ((root:g=0)->(g=1)->(g=2:goal)
        #            ->(g=1)
        # Unfortunately, this is not such a great pedagogical example, as it
        # matched the 'branch' option because a solution could be found at the
        # first branch. This might not be always the case, and the 'tree' cycle
        # prevention policy might require higher memory amounts.
        self.assertEqual(len(s.space_), 4)

    def test_solution_path_as_action_list(self):
        a = _UtilityTestAgent(fringe_searches.DepthFirst, None, None)
        s = fringe_searches.DepthFirst(agent=a, root=_TState(10))
        actions = s.search().backtrack().solution_path_as_action_list()

        self.assertListEqual(actions, ['m', 'm'])


class IterativeDeepeningTest(TestCase):
    def test_sanity(self):
        with self.assertRaises(AssertionError):
            fringe_searches.IterativeDeepening(agent=None, root=_TState(10))

        a = _UtilityTestAgent(fringe_searches.IterativeDeepening, None, None)
        s = fringe_searches.IterativeDeepening(agent=a, root=_TState(10))
        self.assertIsNotNone(s)

    def test_search(self):
        a = _UtilityTestAgent(fringe_searches.IterativeDeepening, None, None)
        s = (fringe_searches
             .IterativeDeepening(agent=a, iterations=range(2))
             .restart(_TState(0))
             .search())

        # Cannot find a solution in depths :math:`[0, 2) \in \mathbb{N}`.
        self.assertIsNone(s.solution_candidate_)

        s.iterations = range(3)
        s.restart(_TState(0)).search().backtrack()

        self.assertTrue(s.solution_candidate_.is_goal)
        self.assertTrue(s.solution_path_)

    def test_solution_path_as_action_list(self):
        a = _UtilityTestAgent(fringe_searches.IterativeDeepening, None, None)
        s = fringe_searches.IterativeDeepening(agent=a, root=_TState(0))
        actions = s.search().backtrack().solution_path_as_action_list()

        self.assertListEqual(actions, ['m', 'm'])
