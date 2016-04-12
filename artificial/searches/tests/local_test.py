from unittest import TestCase

from artificial import agents, base
from artificial.searches.local import HillClimbing, LocalBeam


class _TestState(base.State):
    @property
    def is_goal(self):
        return self.h() == 0

    def h(self):
        return abs(self.data - 2)


class _TestEnvironment(base.Environment):
    def generate_random_state(self):
        return _TestState(self.random_state.randint(-10, 10))

    def update(self):
        pass


class _UtilityTestAgent(agents.UtilityBasedAgent):
    def predict(self, state):
        children = [state.mitosis(action=1, parenting=False, g=0)
                    for _ in range(3)]

        for i, c in enumerate(children):
            c.data += i

        return children


class HillClimbingTest(TestCase):
    def setUp(self):
        self.env = _TestEnvironment(_TestState(0))
        self.agent = _UtilityTestAgent(HillClimbing, self.env, actions=None)

    def test_sanity(self):
        with self.assertRaises(AssertionError):
            HillClimbing(agent=None, root=_TestState(10))

        s = HillClimbing(agent=self.agent, root=_TestState(10))
        self.assertIsNotNone(s)

    def test_search(self):
        s = (HillClimbing(agent=self.agent)
             .restart(_TestState(0))
             .search())

        self.assertTrue(s.solution_candidate.is_goal,
                        str(s.solution_candidate))
        self.assertEqual(s.solution_candidate.data, 2)

    def test_classic_strategy(self):
        s = (HillClimbing(agent=self.agent, strategy='classic')
             .restart(_TestState(0))
             .search())

        self.assertTrue(s.solution_candidate.is_goal,
                        str(s.solution_candidate))
        self.assertEqual(s.solution_candidate.data, 2)

    def test_random_restart(self):
        s = (HillClimbing(agent=self.agent, restart_limit=2)
             .restart(_TestState(0))
             .search())

        self.assertTrue(s.solution_candidate.is_goal,
                        str(s.solution_candidate))
        self.assertEqual(s.solution_candidate.data, 2)


class LocalBeamTest(TestCase):
    def setUp(self):
        self.env = _TestEnvironment(_TestState(0))
        self.agent = _UtilityTestAgent(LocalBeam, self.env, actions=None)

    def test_sanity(self):
        with self.assertRaises(AssertionError):
            LocalBeam(agent=None, root=_TestState(10), k=2)

        with self.assertRaises(ValueError):
            (LocalBeam(agent=self.agent, root=_TestState(10), k='invalid')
             .search())

        s = LocalBeam(agent=self.agent, root=_TestState(10), k=2)
        self.assertIsNotNone(s)

    def test_search(self):
        s = (LocalBeam(agent=self.agent, k=2, root=_TestState(0))
             .search())

        self.assertTrue(s.solution_candidate.is_goal,
                        str(s.solution_candidate))
        self.assertEqual(s.solution_candidate.data, 2)

    def test_classic_strategy(self):
        s = (LocalBeam(agent=self.agent, root=self.env.current_state,
                       strategy='classic', k=2)
             .search())

        self.assertTrue(s.solution_candidate.is_goal,
                        str(s.solution_candidate))
        self.assertEqual(s.solution_candidate.data, 2)

    def test_random_restart(self):
        s = (LocalBeam(agent=self.agent, root=self.env.current_state,
                       k=2, restart_limit=2)
             .search())

        self.assertTrue(s.solution_candidate.is_goal,
                        str(s.solution_candidate))
        self.assertEqual(s.solution_candidate.data, 2)
