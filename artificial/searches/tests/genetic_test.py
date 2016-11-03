"""Artificial Genetic Algorithm Tests"""

# Author: Lucas David -- <ld492@drexel.edu>
# License: MIT (c) 2016

import string
import time
from unittest import TestCase

import numpy as np
from artificial import base, agents
from artificial.searches import genetic
from nose_parameterized import parameterized

random_state = np.random.RandomState(0)


# Classes for Hello World spelling problem.
class _S(base.GeneticState):
    expected = 'hello world'
    alphabet = list(string.ascii_lowercase + ' ')

    def h(self):
        return sum((1 if self.data[i] != self.expected[i] else 0
                    for i in range(min(len(self.data), len(self.expected)))))

    def cross(self, other):
        cross_point = random_state.randint(0, len(_S.expected))
        return _S(self.data[:cross_point] + other.data[cross_point:])

    def mutate(self, factor, probability):
        # We ignore factor, as we are dealing with binary genes.
        m = random_state.rand(len(self.data)) < probability

        if np.any(m):
            d = np.array(list(self.data))
            d[m] = random_state.choice(self.alphabet, size=m.sum())
            self.data = ''.join(d)

        return self

    @classmethod
    def random(cls):
        return cls(''.join(random_state.choice(cls.alphabet,
                                               size=len(cls.expected))))

    @property
    def is_goal(self):
        return self.data == _S.expected


class _A(agents.UtilityBasedAgent):
    def predict(self, state):
        """Predicts nothing."""


class _E(base.Environment):
    state_class_ = _S

    def update(self):
        """Updates nothing."""


# Classes for a simple numerical optimization.
class _S2(base.GeneticState):
    @classmethod
    def random(cls):
        return _S2([random_state.randint(0, 2) for _ in range(10)])

    def mutate(self, factor, probability):
        m = random_state.rand(len(self.data)) < probability

        if m.any():
            data = np.array(self.data)
            data[m] = 1 - data[m]

            self.data = data.tolist()

        return self

    def cross(self, other):
        cross_point = random_state.randint(0, len(self.data))
        return _S2(self.data[:cross_point] + self.data[cross_point:])

    def h(self):
        return -sum(1 if i == 1 else 0 for i in self.data)

    @property
    def is_goal(self):
        # Every element is 1.
        return sum(self.data) == len(self.data)


class _E2(base.Environment):
    state_class_ = _S2

    def update(self):
        """Updates nothing."""


class GeneticAlgorithmTest(TestCase):
    def setUp(self):
        self.env = _E(_S('UkDmEmaPCvK'))
        self.agent = _A(search=genetic.GeneticAlgorithm,
                        environment=self.env,
                        actions=None)

        self.random_state = np.random.RandomState(0)

    def test_sanity(self):
        ga = genetic.GeneticAlgorithm(self.agent,
                                      random_state=self.random_state)
        self.assertIsNotNone(ga)

    def test_generate_population(self):
        ga = genetic.GeneticAlgorithm(self.agent, max_evolution_cycles=1,
                                      random_state=self.random_state)
        ga.search()
        self.assertEqual(ga.population_size_, 1000)

        # Assert that the arrays necessary for the search were disposed.
        self.assertIsNone(ga.population_)
        self.assertIsNone(ga.selected_)
        self.assertIsNone(ga.offspring_)

        ga = genetic.GeneticAlgorithm(self.agent, population_size=20,
                                      max_evolution_cycles=1,
                                      random_state=self.random_state)
        ga.search()
        self.assertEqual(ga.population_size_, 20)

        ga = genetic.GeneticAlgorithm(self.agent,
                                      max_evolution_cycles=10,
                                      max_evolution_duration=1,
                                      n_jobs=1,
                                      random_state=self.random_state)
        ga.search()
        self.assertGreater(ga.population_size_, 100)

    @parameterized.expand([
        'random', 'tournament', 'roulette', 'gattaca'
    ])
    def test_select_for_breeding(self, method):
        ga = genetic.GeneticAlgorithm(self.agent,
                                      n_selected=20,
                                      breeding_selection=method,
                                      max_evolution_cycles=1)

        (ga.search_start().generate_population().cycle_start()
         .select_for_breeding())
        self.assertEqual(len(ga.selected_), 20)

    def test_breed(self):
        ga = genetic.GeneticAlgorithm(self.agent,
                                      population_size=100, n_selected=100)

        (ga.search_start().generate_population().cycle_start()
         .select_for_breeding().breed())
        self.assertEqual(len(ga.population_), 100)
        self.assertEqual(len(ga.offspring_), 50)

    @parameterized.expand([
        (dict(mutation_probability=.2), dict(population_size_=1000,
                                             n_selected_=1000)),
    ])
    def test_search(self, params, expected):
        ga = genetic.GeneticAlgorithm(self.agent,
                                      random_state=self.random_state, **params)
        solution = ga.search().solution_candidate_

        # Attributes were set as expected.
        for key, value in expected.items():
            self.assertEqual(getattr(ga, key), value)

        # Assert clean-up was made.
        self.assertIsNone(ga.offspring_)
        self.assertIsNone(ga.selected_)

        # Assert it eventually finds a solution.
        self.assertIsNotNone(solution)
        self.assertEqual(solution.data, 'hello world')

    @parameterized.expand([
        ({
             'natural_selection': 'elitism',
             'max_evolution_duration': 5,
             'mutation_probability': .2
         }, 5.5),
        ({
             'natural_selection': 'random',
             'max_evolution_duration': 5,
             'mutation_probability': .2
         }, 5.5),
        ({
             'max_evolution_duration': 5,
             'mutation_probability': .2,
             'n_jobs': 4,
             'debug': True
         }, 5.5)
    ])
    def test_search_duration_constraint(self, params, acceptable_elapsed):
        ga = genetic.GeneticAlgorithm(self.agent,
                                      random_state=self.random_state,
                                      **params)

        elapsed = time.time()
        ga.search()
        elapsed = time.time() - elapsed

        # Assert that the duration constraint was respected.
        self.assertLess(elapsed, acceptable_elapsed)
        self.assertIsNotNone(ga.solution_candidate_)

    def test_preemption_by_genetic_similarity(self):
        expected_variability = .4

        a = _A(search=genetic.GeneticAlgorithm, environment=_E2())

        ga = genetic.GeneticAlgorithm(
            a, max_evolution_duration=60,
            min_genetic_similarity=expected_variability,
            population_size=50,
            mutation_probability=0,
            random_state=self.random_state,
            debug=True).search()

        # Assert that the last population's variability is smaller
        # than the `min_genetic_similarity` parameter passed.
        self.assertLessEqual(ga.variability_, expected_variability)

        self.assertIsNotNone(ga.solution_candidate_)
        self.assertGreaterEqual(a.utility(ga.solution_candidate_), 7)

    def test_genetic_similarity_raises_error(self):
        ga = genetic.GeneticAlgorithm(
            self.agent, mutation_factor=.5, mutation_probability=1,
            max_evolution_duration=4, min_genetic_similarity=.5,
            random_state=self.random_state)

        with self.assertRaises(RuntimeError):
            ga.genetic_similarity()

    @parameterized.expand([
        (dict(n_selected='all'),),
        (dict(breeding_selection='rand0m'),),
        (dict(natural_selection='steady_state'),),
        (dict(population_size=100, breeding_selection='tournament',
              tournament_size=200),),
        (dict(population_size=.5, max_evolution_cycles=1),),
    ])
    def test_raises_value_errors(self, params):
        print(params)
        # Assert raises ValueError when parameters are incorrect.
        with self.assertRaises(ValueError):
            genetic.GeneticAlgorithm(self.agent,
                                     random_state=self.random_state,
                                     **params).search()
