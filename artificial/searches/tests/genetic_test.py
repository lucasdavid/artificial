from unittest import TestCase

from artificial import base, agents
from artificial.searches import genetic


class _TState(base.State):
    word = 'Hello World'

    def h(self):
        return sum((self.data[i] == self.word[i] and 1 or 0
                    for i in range(min(len(self.data), len(self.word)))))


class _TAgent(agents.UtilityBasedAgent):
    pass


class _TEnv(base.Environment):
    def update(self):
        pass


class GeneticAlgorithmTest(TestCase):
    def setUp(self):
        self.env = _TEnv(_TState('UkDmEmaPCvK'))
        self.agent = _TAgent(search=genetic.GeneticAlgorithm,
                             environment=self.env,
                             actions=None)

    def test_sanity(self):
        ga = genetic.GeneticAlgorithm(self.agent)
        self.assertIsNotNone(ga)

    def test_search(self):
        ga = genetic.GeneticAlgorithm(self.agent)
        solution = ga.search().solution_candidate_

        self.assertIsNotNone(solution)
        self.assertGreaterEqual(solution.h() / len(_TState.word), .9)
