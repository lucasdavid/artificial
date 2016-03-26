from unittest import TestCase

from artificial import agents, base
from artificial.searches.local import HillClimbing


class _TestState(base.State):
    @property
    def is_goal(self):
        return self.h() == 0

    def h(self):
        return abs(self.data - 2)


class _UtilityTestAgent(agents.UtilityBasedAgent):
    def predict(self, state):
        children = [state.mitosis(action=1, parenting=False, g=0)
                    for i in range(3)]

        for i, c in enumerate(children):
            c.data += i

        return children


class HillClimbingTest(TestCase):
    def test_sanity(self):
        with self.assertRaises(AssertionError):
            HillClimbing(agent=None, root=_TestState(10))

        a = _UtilityTestAgent(HillClimbing, None, None)
        s = HillClimbing(agent=a, root=_TestState(10))
        self.assertIsNotNone(s)

    def test_perform(self):
        a = _UtilityTestAgent(HillClimbing, None, None)
        s = (HillClimbing(agent=a)
             .restart(_TestState(0))
             .perform())

        self.assertTrue(s.solution_candidate.is_goal, str(s.solution_candidate))
        self.assertEqual(s.solution_candidate.data, 2)
