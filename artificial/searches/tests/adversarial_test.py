"""Artificial Adversarial Searches Test"""

# Author: Lucas David -- <ld492@drexel.edu>
# License: MIT (c) 2016

import random
from unittest import TestCase

from artificial import base, agents
from artificial.searches.adversarial import Random, MinMax, AlphaBeta

random_generator = random.Random(0)


class _S(base.State):
    """Generic Game State"""


class _Player(agents.UtilityBasedAgent):
    maximizing_player = True

    def predict(self, state):
        if state.g == 12:
            return []

        return [
            _S(state.data - 1, g=state.g + 1),
            _S(state.data + 1, g=state.g + 1),
        ]

    def utility(self, state):
        state.computed_utility_ = (state.computed_utility_ or
                                   (self.maximizing_player and 1 or -1) *
                                   state.data)
        return state.computed_utility_


class RandomTest(TestCase):
    def test_sanitize(self):
        a = _Player(Random, None, None)
        s = Random(agent=a)

        self.assertIsNotNone(s)

    def test_search(self):
        a = _Player(Random, None, None)

        s = (Random(agent=a, random_generator=random_generator)
             .restart(_S(50))
             .search())

        self.assertIsNotNone(s.solution_candidate_)


class MinMaxTest(TestCase):
    def test_sanitize(self):
        a = _Player(MinMax, None, None)
        s = MinMax(agent=a, depth_limit=10)

        self.assertIsNotNone(s)

    def test_search(self):
        a = _Player(MinMax, None, None)

        for depth_limit in (10, 20):
            s = (MinMax(agent=a, depth_limit=depth_limit)
                 .restart(_S(50))
                 .search())

            self.assertIsNotNone(s.solution_candidate_)


class AlphaBetaTest(TestCase):
    def test_sanitize(self):
        a = _Player(AlphaBeta, None, None)
        s = AlphaBeta(agent=a, depth_limit=10)

        self.assertIsNotNone(s)

    def test_search(self):
        a = _Player(AlphaBeta, None, None)

        for depth_limit in (10, 20):
            s = (AlphaBeta(agent=a, depth_limit=depth_limit)
                 .restart(_S(50))
                 .search())

            self.assertIsNotNone(s.solution_candidate_)
