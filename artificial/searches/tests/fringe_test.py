import random
from unittest import TestCase

from artificial import agents, base
from artificial.searches import fringe as fringe_searches

random_generator = random.Random(0)


class _S(base.State):
    def h(self):
        return abs(self.data - 2)

    @property
    def is_goal(self):
        return self.data == 2

    @classmethod
    def random(cls):
        return _S(data=random_generator(0, 4))


class _UtilityTestAgent(agents.UtilityBasedAgent):
    def predict(self, state):
        return [
            _S(state.data - 1, parent=state, action='m', g=state.g + 1),
            _S(state.data + 1, parent=state, action='m', g=state.g + 1)
        ]


class UniformCostSearchTest(TestCase):
    def test_sanity(self):
        with self.assertRaises(AssertionError):
            fringe_searches.UniformCost(agent=None, root=_S(10))

        a = _UtilityTestAgent(fringe_searches.UniformCost, None, None)
        s = fringe_searches.UniformCost(agent=a, root=_S(10))
        self.assertIsNotNone(s)

    def test_search(self):
        a = _UtilityTestAgent(fringe_searches.UniformCost, None, None)
        s = fringe_searches.UniformCost(agent=a).restart(_S(10))
        s.search().backtrack()

        self.assertTrue(s.solution_candidate_.is_goal)
        self.assertTrue(s.solution_path_)

    def test_solution_path_as_action_list(self):
        a = _UtilityTestAgent(fringe_searches.UniformCost, None, None)
        s = fringe_searches.UniformCost(agent=a, root=_S(10))
        actions = s.search().backtrack().solution_path_as_action_list()

        # Eight consecutive m's, because 10 is at a distance of 8 from 2.
        self.assertListEqual(actions, 8 * ['m'])


class GreedyBestSearchTest(TestCase):
    def test_sanity(self):
        with self.assertRaises(AssertionError):
            fringe_searches.GreedyBestFirst(agent=None, root=_S(10))

        a = _UtilityTestAgent(fringe_searches.UniformCost, None, None)
        s = fringe_searches.GreedyBestFirst(agent=a, root=_S(10))
        self.assertIsNotNone(s)

    def test_search(self):
        a = _UtilityTestAgent(fringe_searches.GreedyBestFirst, None, None)
        s = fringe_searches.GreedyBestFirst(agent=a).restart(_S(10))
        s.search().backtrack()

        self.assertTrue(s.solution_candidate_.is_goal)
        self.assertTrue(s.solution_path_)


class AStarTest(TestCase):
    def test_sanity(self):
        with self.assertRaises(AssertionError):
            fringe_searches.AStar(agent=None, root=_S(10))

        a = _UtilityTestAgent(fringe_searches.AStar, None, None)
        s = fringe_searches.AStar(agent=a, root=_S(10))
        self.assertIsNotNone(s)

    def test_search(self):
        a = _UtilityTestAgent(fringe_searches.AStar, None, None)
        s = fringe_searches.AStar(agent=a).restart(_S(10))
        s.search().backtrack()

        self.assertTrue(s.solution_candidate_.is_goal)
        self.assertTrue(s.solution_path_)


class BreadthFirstSearchTest(TestCase):
    def test_sanity(self):
        with self.assertRaises(AssertionError):
            fringe_searches.BreadthFirst(agent=None, root=_S(10))

        a = _UtilityTestAgent(fringe_searches.BreadthFirst, None, None)
        s = fringe_searches.BreadthFirst(agent=a, root=_S(10))
        self.assertIsNotNone(s)

    def test_search(self):
        a = _UtilityTestAgent(fringe_searches.BreadthFirst, None, None)
        s = fringe_searches.BreadthFirst(agent=a).restart(_S(4))
        s.search()

        self.assertTrue(s.solution_candidate_.is_goal)

        # Search space size should be:
        # |(root) + 2 (root children) + 2 (solution candidates)| = 5.
        self.assertEqual(len(s.space_), 5)

        s.backtrack()
        self.assertTrue(s.solution_path_)

    def test_solution_path_as_action_list(self):
        a = _UtilityTestAgent(fringe_searches.BreadthFirst, None, None)
        s = fringe_searches.BreadthFirst(agent=a, root=_S(10))
        actions = s.search().backtrack().solution_path_as_action_list()

        # Eight consecutive m's, because 10 is at a distance 8 from 2.
        self.assertListEqual(actions, 8 * ['m'])

    def test_solution_path_before_search_raises_error(self):
        a = _UtilityTestAgent(fringe_searches.BreadthFirst, None, None)
        s = fringe_searches.BreadthFirst(agent=a, root=_S(10))

        with self.assertRaises(RuntimeError):
            s.backtrack()


class DepthFirstSearchTest(TestCase):
    def test_sanity(self):
        with self.assertRaises(AssertionError):
            fringe_searches.DepthFirst(agent=None, root=_S(10))

        a = _UtilityTestAgent(fringe_searches.DepthFirst, None, None)
        s = fringe_searches.DepthFirst(agent=a, root=_S(10))
        self.assertIsNotNone(s)

    def test_search(self):
        a = _UtilityTestAgent(fringe_searches.DepthFirst, None, None)
        s = (fringe_searches
             .DepthFirst(agent=a).restart(_S(10)).search().backtrack())

        self.assertTrue(s.solution_candidate_.is_goal)
        self.assertTrue(s.solution_path_)

        # Search space_ stores only root, as we didn't initialized `DepthFirst`
        # with any cycle prevention policy.
        self.assertEqual(len(s.space_), 1)

        # All cycle prevention policies should find a `solution_candidate_`!
        s.prevent_cycles = 'branch'
        # We don't need to limit the search depth because goal is at left-most
        # from root.
        s.restart(_S(4)).search().backtrack()
        self.assertTrue(s.solution_candidate_.is_goal)
        self.assertTrue(s.solution_path_)

        # Search space:
        # (root:4)->(3)->(2:goal)
        #         ->(5)
        self.assertEqual(len(s.space_), 4)
        # Notice that, as 5 is never explored (solution is at most-left of 4),
        # it's never removed from space. We still have a single branch
        # (the most-left) stored in space, though.

        # Same policy. Only this time it will backtrack because we are starting
        # from position 0 (it will first go to -1, -2 to only then go 1 and 2.
        s.prevent_cycles = 'branch'
        s.limit = 4
        s.restart(_S(0)).search().backtrack()
        self.assertTrue(s.solution_candidate_.is_goal)
        self.assertTrue(s.solution_path_)

        # Search order:
        # (root:0)->(-1)->(-2)->(-3)->(-4)
        #         ->( 1)->(0:repeated-in-branch, ignored)
        #               ->(2:goal)
        #
        # Final search space:
        # (root:0)->(1)->(2:goal)
        self.assertEqual(len(s.space_), 3)

        s.prevent_cycles = 'tree'
        s.limit = 4
        s.restart(_S(0)).search().backtrack()
        self.assertTrue(s.solution_candidate_.is_goal)
        self.assertTrue(s.solution_path_)

        # Search space:
        # (root:0)->(-1)->(-2)->(-3)->(-4)->(-5)
        #         ->( 1)->(2)
        # `DepthFirst` travels to most-left branches first, adding values
        # smaller than 0. Only then, it starts exploring right branches.
        self.assertEqual(len(s.space_), 8)

    def test_solution_path_as_action_list(self):
        a = _UtilityTestAgent(fringe_searches.DepthFirst, None, None)
        s = fringe_searches.DepthFirst(agent=a, root=_S(4))
        actions = s.search().backtrack().solution_path_as_action_list()

        self.assertListEqual(actions, ['m', 'm'])


class IterativeDeepeningTest(TestCase):
    def test_sanity(self):
        with self.assertRaises(AssertionError):
            fringe_searches.IterativeDeepening(agent=None, root=_S(10))

        a = _UtilityTestAgent(fringe_searches.IterativeDeepening, None, None)
        s = fringe_searches.IterativeDeepening(agent=a, root=_S(10))
        self.assertIsNotNone(s)

    def test_search(self):
        a = _UtilityTestAgent(fringe_searches.IterativeDeepening, None, None)
        s = (fringe_searches
             .IterativeDeepening(agent=a, iterations=range(2))
             .restart(_S(0))
             .search())

        # Cannot find a solution in depths :math:`[0, 2) \in \mathbb{N}`.
        self.assertIsNone(s.solution_candidate_)

        s.iterations = range(1, 3)
        s.restart(_S(0)).search().backtrack()

        self.assertTrue(s.solution_candidate_.is_goal)
        self.assertTrue(s.solution_path_)

    def test_solution_path_as_action_list(self):
        a = _UtilityTestAgent(fringe_searches.IterativeDeepening, None, None)
        s = fringe_searches.IterativeDeepening(agent=a, root=_S(0))
        actions = s.search().backtrack().solution_path_as_action_list()

        self.assertListEqual(actions, ['m', 'm'])
