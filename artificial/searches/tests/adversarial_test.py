from unittest import TestCase

from artificial import base
from artificial.searches.adversarial import Random, MinMax, AlphaBeta


class _TestState(base.State):
    def h(self):
        return self.data


class _UtilityTestAgent(agents.UtilityAgent):
    maximizing_player = True

    def predict(self, state):
        return [
            _TestState(self.data - 1),
            _TestState(self.data + 1),
        ]

    def utility(self, state):
        return (1 if maximizing_player else -1) * state.f()


class RandomTest(TestCase):
    def test_sanitize(self):
        a = _UtilityTestAgent(Random, None, None)
        s = Random(agent=a)

        self.assertIsNotNone(s)

    def test_perform(self):
        a = _UtilityTestAgent(Random, None, None)
        s = (Random(agent=a)
             .restart(_TestState(50))
             .perform())

        self.assertIsNotNone(s.solution_candidate)


class MinMaxTest(TestCase):
    def test_sanitize(self):
        a = _UtilityTestAgent(MinMax, None, None)
        s = MinMax(agent=a)

        self.assertIsNotNone(s)

    def test_perform(self):
        a = _UtilityTestAgent(MinMax, None, None)
        s = (MinMax(agent=a)
             .restart(_TestState(50))
             .perform())

        self.assertIsNotNone(s.solution_candidate)


class AlphaBetaTest(TestCase):
    def test_sanitize(self):
        a = _UtilityTestAgent(AlphaBeta, None, None)
        s = AlphaBeta(agent=a)

        self.assertIsNotNone(s)

    def test_perform(self):
        a = _UtilityTestAgent(AlphaBeta, None, None)
        s = (AlphaBeta(agent=a)
             .restart(_TestState(50))
             .perform())

        self.assertIsNotNone(s.solution_candidate)
