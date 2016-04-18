from unittest import TestCase

from artificial import base, agents
from artificial.searches.adversarial import Random, MinMax, AlphaBeta


class _TestState(base.State):
    def h(self):
        return self.data


class _UtilityTestAgent(agents.UtilityBasedAgent):
    maximizing_player = True

    def predict(self, state):
        return [
            _TestState(state.data - 1),
            _TestState(state.data + 1),
        ]

    def utility(self, state):
        return (1 if self.maximizing_player else -1) * state.f()


class RandomTest(TestCase):
    def test_sanitize(self):
        a = _UtilityTestAgent(Random, None, None)
        s = Random(agent=a)

        self.assertIsNotNone(s)

    def test_search(self):
        a = _UtilityTestAgent(Random, None, None)
        s = (Random(agent=a, depth_limit=10)
             .restart(_TestState(50))
             .search())

        self.assertIsNotNone(s.solution_candidate_)


class MinMaxTest(TestCase):
    def test_sanitize(self):
        a = _UtilityTestAgent(MinMax, None, None)
        s = MinMax(agent=a, depth_limit=10)

        self.assertIsNotNone(s)

    def test_search(self):
        a = _UtilityTestAgent(MinMax, None, None)
        s = (MinMax(agent=a, depth_limit=10)
             .restart(_TestState(50))
             .search())

        self.assertIsNotNone(s.solution_candidate_)


class AlphaBetaTest(TestCase):
    def test_sanitize(self):
        a = _UtilityTestAgent(AlphaBeta, None, None)
        s = AlphaBeta(agent=a, depth_limit=10)

        self.assertIsNotNone(s)

    def test_search(self):
        a = _UtilityTestAgent(AlphaBeta, None, None)
        s = (AlphaBeta(agent=a, depth_limit=10)
             .restart(_TestState(50))
             .search())

        self.assertIsNotNone(s.solution_candidate_)
