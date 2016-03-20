from unittest import TestCase

from artificial import agents, searches, base


class _TestState(base.State):
    @property
    def is_goal(self):
        return self.g == 3


class _UtilityTestAgent(agents.UtilityBasedAgent):
    def predict(self, state):
        children = [state.mitosis(g=state.g + 1, action='m') for _ in range(3)]
        for c in children:
            c.data += 1

        return children


class UniformCostSearchTest(TestCase):
    def test_sanity(self):
        with self.assertRaises(AssertionError):
            searches.UniformCost(agent=None, root=_TestState(10))

        a = _UtilityTestAgent(searches.UniformCost, None, None)
        s = searches.UniformCost(agent=a, root=_TestState(10))
        self.assertIsNotNone(s)

    def test_perform(self):
        a = _UtilityTestAgent(searches.UniformCost, None, None)
        s = searches.UniformCost(agent=a).restart(_TestState(10))
        s.perform()

        self.assertTrue(s.solution_candidate.is_goal)
        self.assertTrue(s.solution_path)

    def test_solution_path_as_action_list(self):
        a = _UtilityTestAgent(searches.UniformCost, None, None)
        s = searches.UniformCost(agent=a, root=_TestState(10))
        actions = s.perform().solution_path_as_action_list()

        self.assertListEqual(actions, ['m', 'm', 'm'])


class BreadthFirstSearchTest(TestCase):
    def test_sanity(self):
        with self.assertRaises(AssertionError):
            searches.BreadthFirst(agent=None, root=_TestState(10))

        a = _UtilityTestAgent(searches.BreadthFirst, None, None)
        s = searches.BreadthFirst(agent=a, root=_TestState(10))
        self.assertIsNotNone(s)

    def test_perform(self):
        a = _UtilityTestAgent(searches.BreadthFirst, None, None)
        s = searches.BreadthFirst(agent=a).restart(_TestState(10))
        s.perform()

        self.assertTrue(s.solution_candidate.is_goal)
        self.assertTrue(s.solution_path)

        # Search space is 2^3 - # extracted nodes.
        self.assertEqual(len(s.space) + len(s.fringe), 9 - 3)

    def test_solution_path_as_action_list(self):
        a = _UtilityTestAgent(searches.BreadthFirst, None, None)
        s = searches.BreadthFirst(agent=a, root=_TestState(10))
        actions = s.perform().solution_path_as_action_list()

        self.assertListEqual(actions, ['m', 'm', 'm'])

        # Search space is 2^3 - # extracted nodes.
        self.assertEqual(len(s.space) + len(s.fringe), 9 - 3)


class DepthFirstSearchTest(TestCase):
    def test_sanity(self):
        with self.assertRaises(AssertionError):
            searches.DepthFirst(agent=None, root=_TestState(10))

        a = _UtilityTestAgent(searches.DepthFirst, None, None)
        s = searches.DepthFirst(agent=a, root=_TestState(10))
        self.assertIsNotNone(s)

    def test_perform(self):
        a = _UtilityTestAgent(searches.DepthFirst, None, None)
        s = (searches
             .DepthFirst(agent=a)
             .restart(_TestState(10))
             .perform())

        self.assertTrue(s.solution_candidate.is_goal)
        self.assertTrue(s.solution_path)

        # Search space stores only root, as we didn't initialized
        # Depth-first search with cycle prevention.
        self.assertEqual(len(s.space), 1)

        s = (searches
             .DepthFirst(agent=a, prevent_cycles='tree')
             .restart(_TestState(10))
             .perform())

        self.assertTrue(s.solution_candidate.is_goal)
        self.assertTrue(s.solution_path)

        # Search space has O(d) = 4 nodes
        # ((root:g=0)->(g=1)->(g=2)->(goal:g=3)).
        self.assertEqual(len(s.space), 4)

    def test_solution_path_as_action_list(self):
        a = _UtilityTestAgent(searches.DepthFirst, None, None)
        s = searches.DepthFirst(agent=a, root=_TestState(10))
        actions = s.perform().solution_path_as_action_list()

        self.assertListEqual(actions, ['m', 'm', 'm'])
