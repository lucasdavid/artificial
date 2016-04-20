import random
import string
from unittest import TestCase

import numpy as np

from artificial import base, agents
from artificial.searches import genetic

random_generator = random.Random(0)


# Classes for Hello World spelling problem.
class _S(base.GeneticState):
    expected = 'hello world'

    def h(self):
        return sum((1 if self.data[i] != self.expected[i] else 0
                    for i in range(min(len(self.data), len(self.expected)))))

    def cross(self, other):
        cross_point = random.randint(0, len(_S.expected))
        return _S(self.data[:cross_point] + other.data[cross_point:])

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
        return self.data == _S.expected


class _A(agents.UtilityBasedAgent):
    pass


class _E(base.Environment):
    state_class_ = _S


# Classes for a simple numerical optimization.
class _S2(base.GeneticState):
    @classmethod
    def random(cls):
        return _S2([random_generator.randint(0, 1) for _ in range(100)])

    def mutate(self, factor, probability):
        n_mutations = round(factor * len(self.data))

        for _ in range(n_mutations):
            if random_generator.random() > probability:
                continue

            index = random_generator.randint(0, len(self.data) - 1)
            self.data[index] = 1 - self.data[index]

        return self

    def cross(self, other):
        cross_point = random_generator.randint(0, len(self.data))
        return _S2(self.data[:cross_point] + self.data[cross_point:])

    def h(self):
        return -sum(1 if i == 1 else 0 for i in self.data)

    @property
    def is_goal(self):
        # Every element is 1.
        return sum(self.data) == len(self.data)


class _E2(base.Environment):
    state_class_ = _S2


class GeneticAlgorithmTest(TestCase):
    def setUp(self):
        random_generator = random.Random(0)

        self.env = _E(_S('UkDmEmaPCvK'),
                      random_generator=random_generator)
        self.agent = _A(search=genetic.GeneticAlgorithm,
                        environment=self.env,
                        actions=None)

    def test_sanity(self):
        ga = genetic.GeneticAlgorithm(self.agent)
        self.assertIsNotNone(ga)

    def test_generate_population(self):
        ga = genetic.GeneticAlgorithm(self.agent, max_evolution_cycles=1)
        ga.search()
        self.assertEqual(ga.population_size_, 1000)

        # Assert that the arrays necessary for the search were disposed.
        self.assertIsNone(ga.population_)
        self.assertIsNone(ga.selected_)
        self.assertIsNone(ga.offspring_)

        ga = genetic.GeneticAlgorithm(self.agent, population_size=20,
                                      max_evolution_cycles=1)
        ga.search()
        self.assertEqual(ga.population_size_, 20)

        ga = genetic.GeneticAlgorithm(self.agent,
                                      max_evolution_cycles=10,
                                      max_evolution_duration=1,
                                      n_jobs=1)
        ga.search()
        self.assertGreater(ga.population_size_, 100)

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

        # Attributes were set as expected.
        self.assertEqual(ga.population_size_, 1000)
        self.assertEqual(ga.n_selected_, 1000)

        # Assert clean-up was made.
        self.assertIsNone(ga.offspring_)
        self.assertIsNone(ga.selected_)

        # Found a solution.
        self.assertIsNotNone(solution)
        self.assertEqual(solution.data, 'hello world')

        ga = genetic.GeneticAlgorithm(
            self.agent, natural_selection='elitism', max_evolution_duration=5,
            mutation_factor=.5, mutation_probability=1)
        self.assertIsNotNone(ga.search().solution_candidate_)

        ga = genetic.GeneticAlgorithm(
            self.agent, natural_selection='random', max_evolution_duration=5,
            mutation_factor=.5, mutation_probability=1)
        self.assertIsNotNone(ga.search().solution_candidate_)

    def test_preemption_by_genetic_similarity(self):
        a = _A(search=genetic.GeneticAlgorithm,
               environment=_E2(_S2(100 * [0])),
               actions=None)

        ga = (genetic.GeneticAlgorithm(a, min_genetic_similarity=.01,
                                       max_evolution_duration=5)
              .search())

        self.assertIsNotNone(ga.solution_candidate_)
        self.assertEqual(a.utility(ga.solution_candidate_), 100)

    def test_raises_errors(self):
        np.random.seed(0)

        # Assert raises ValueError when parameters are incorrect.
        with self.assertRaises(ValueError):
            genetic.GeneticAlgorithm(self.agent, n_selected='all').search()
        with self.assertRaises(ValueError):
            (genetic.GeneticAlgorithm(self.agent, breeding_selection='rand0m')
             .search())
        with self.assertRaises(ValueError):
            (genetic
             .GeneticAlgorithm(self.agent, natural_selection='steady_state')
             .search())
