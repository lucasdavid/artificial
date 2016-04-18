import random
import string
from unittest import TestCase

import numpy as np

from artificial import base, agents
from artificial.searches import genetic

random_generator = random.Random(0)


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

    @classmethod
    def random(cls):
        return cls(''.join(random.choice(string.ascii_lowercase + ' ')
                           for _ in cls.expected))

    @property
    def is_goal(self):
        return self.data == _TState.expected


class _TAgent(agents.UtilityBasedAgent):
    def predict(self, state):
        """Predicts nothing"""


class _TEnv(base.Environment):
    state_class_ = _TState
    
    def update(self):
        """Updates nothing"""


class GeneticAlgorithmTest(TestCase):
    def setUp(self):
        self.env = _TEnv(_TState('UkDmEmaPCvK'),
                         random_generator=random_generator)
        self.agent = _TAgent(search=genetic.GeneticAlgorithm,
                             environment=self.env,
                             actions=None)

    def test_sanity(self):
        ga = genetic.GeneticAlgorithm(self.agent)
        self.assertIsNotNone(ga)

    def test_generate_population(self):
        ga = genetic.GeneticAlgorithm(self.agent, max_evolution_cycles=1)
        ga.search()
        self.assertEqual(len(ga.population_), 1000)
        self.assertEqual(len(ga.population_), ga.population_size_)

        ga = genetic.GeneticAlgorithm(self.agent, population_size=20,
                                      max_evolution_cycles=1)
        ga.search()
        self.assertEqual(len(ga.population_), 20)
        self.assertEqual(len(ga.population_), ga.population_size_)

        ga = genetic.GeneticAlgorithm(self.agent,
                                      max_evolution_cycles=10,
                                      max_evolution_duration=1,
                                      n_jobs=1)
        ga.search()
        self.assertGreater(len(ga.population_), 100)
        self.assertEqual(len(ga.population_), ga.population_size_)

        with self.assertRaises(ValueError):
            (genetic
             .GeneticAlgorithm(self.agent, population_size=.5,
                               max_evolution_cycles=1)
             .search())

    def test_select_for_breeding(self):
        for method in ('random', 'tournament', 'roulette', 'gattaca'):
            ga = genetic.GeneticAlgorithm(self.agent,
                                          n_selected=20,
                                          breeding_selection=method,
                                          max_evolution_cycles=1)
            
            (ga.search_start().generate_population().cycle_start()
               .select_for_breeding())
            self.assertEqual(len(ga.selected_), 20)

        with self.assertRaises(ValueError):
            ga = genetic.GeneticAlgorithm(self.agent,
                                          population_size=100,
                                          breeding_selection='tournament',
                                          tournament_size=200)
            ga.search()

    def test_breed(self):
        ga = genetic.GeneticAlgorithm(self.agent,
                                      population_size=100, n_selected=100)

        (ga.search_start().generate_population().cycle_start()
           .select_for_breeding().breed())
        self.assertEqual(len(ga.population_), 100)
        self.assertEqual(len(ga.offspring_), 50)

    def test_search(self):
        np.random.seed(0)

        ga = genetic.GeneticAlgorithm(
            self.agent, mutation_factor=.5, mutation_probability=1)
        solution = ga.search().solution_candidate_

        # There is a solution.
        self.assertIsNotNone(solution)

        self.assertEqual(ga.population_size_, 1000)
        self.assertEqual(ga.n_selected_, 500)

        # Assert clean-up was made.
        self.assertIsNone(ga.offspring_)
        self.assertIsNone(ga.selected_)

        # Got only three or less letters wrong.
        self.assertEqual(solution.data, 'hello world')
