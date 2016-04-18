import random
import string
from unittest import TestCase

import numpy as np

from artificial import base, agents
from artificial.searches import genetic


class _TState(base.GeneticState):
    expected = 'hello world'

    def h(self):
        return sum((1 if self.data[i] != self.expected[i] else 0
                    for i in range(min(len(self.data), len(self.expected)))))

    def cross(self, other):
        cross_point = random.randint(0, len(_TState.expected))
        return _TState(self.data[:cross_point] + other.data[cross_point:])

    def mutate(self, factor, probability):
        m = np.random.rand(len(self.data)) < factor * probability

        if np.any(m):
            d = np.array(list(self.data))
            d[m] = [random.choice(string.ascii_lowercase + ' ')
                    for mutated in m if mutated]

            self.data = ''.join(d)

        return self

    @property
    def is_goal(self):
        return self.data == _TState.expected


class _TAgent(agents.UtilityBasedAgent):
    def predict(self, state):
        pass


class _TEnv(base.Environment):
    def update(self):
        pass

    def generate_random_state(self):
        return _TState(''.join(random.choice(string.ascii_lowercase + ' ')
                               for _ in _TState.expected))


class GeneticAlgorithmTest(TestCase):
    def setUp(self):
        self.env = _TEnv(_TState('UkDmEmaPCvK'),
                         random_generator=random.Random(0))
        self.agent = _TAgent(search=genetic.GeneticAlgorithm,
                             environment=self.env,
                             actions=None)

    def test_sanity(self):
        ga = genetic.GeneticAlgorithm(self.agent)
        self.assertIsNotNone(ga)

    def test_generate_population(self):
        ga = genetic.GeneticAlgorithm(self.agent)
        ga.generate_population()
        self.assertEqual(len(ga.population_), 1000)
        self.assertEqual(len(ga.population_), ga.population_size_)

        ga = genetic.GeneticAlgorithm(self.agent, population_size=20)
        ga.generate_population()
        self.assertEqual(len(ga.population_), 20)
        self.assertEqual(len(ga.population_), ga.population_size_)

        ga = genetic.GeneticAlgorithm(self.agent,
                                      max_evolution_cycles=10,
                                      max_evolution_duration=1,
                                      n_jobs=1)
        ga.generate_population()
        self.assertGreater(len(ga.population_), 100)
        self.assertEqual(len(ga.population_), ga.population_size_)

        with self.assertRaises(ValueError):
            (genetic
             .GeneticAlgorithm(self.agent, population_size=.5)
             .generate_population())

    def test_breeding_selection(self):
        for method in ('random', 'tournament', 'roulette', 'gattaca'):
            ga = genetic.GeneticAlgorithm(self.agent,
                                          n_selected=20,
                                          breeding_selection=method)
            ga.generate_population().breeding_selection()
            self.assertEqual(len(ga.selected_), 20)

    def test_breed(self):
        ga = genetic.GeneticAlgorithm(self.agent,
                                      population_size=100, n_selected=100)

        ga.generate_population().breeding_selection().breed()
        self.assertEqual(len(ga.population_), 100)
        self.assertEqual(len(ga.offspring_), 50)

    def test_search(self):
        np.random.seed(0)

        ga = genetic.GeneticAlgorithm(
            self.agent, max_evolution_duration=60,
            mutation_factor=.5, mutation_probability=1)
        solution = ga.search().solution_candidate_

        # There is a solution.
        self.assertIsNotNone(solution)

        # Got only three or less letters wrong.
        self.assertEqual(solution.data, 'hello world')
 