"""Genetic algorithm for problem solving"""

# Author: Lucas David -- <ld492@drexel.edu>
# License: MIT (c) 2016

import multiprocessing
import time

import numpy as np
from scipy.spatial import distance

from . import base
from .. import agents


class GeneticAlgorithm(base.Base):
    """Genetic Algorithm Search.

    Parameters
    ----------
    agent : inherited

    population_size : [int|'auto'] (default='auto')
        The size of the population to be kept in memory.

    max_evolution_cycles : [int, np.inf] (default=np.inf]
        The upper bound for evolution cycles, in cycle count.
        If infinity, no bound is set and evolution cycles won't
        be directly a factor for the evolution process stop.

    max_evolution_duration : float (default=np.inf)
        The upper bound for evolution duration, in seconds.
        If infinity, no bound is set and evolution will continue
        regardless the amount of time spent on the process.

    min_genetic_similarity : float (default=0)
        The lower bound for individuals' genetic similarity, in percentage.
        If 0, no bound is set and evolution continues regardless the
        individuals genetic similarity.

    breeding_selection : ['random'|'tournament'|'roulette'|'gattaca']
        (default='roulette')
        The method employed when selecting individuals for breeding.

        Options are:
            --- 'random'
                Individuals are selected at random.

            --- 'tournament'
                `k` tournaments are made to decide which individuals should be
                chosen for breeding. In every one of them, `n_selected`
                individuals are chosen from the population and the best of them
                is selected for breeding.

            --- 'roulette'
                Each individual `i` has an associated rate
                :math:`\frac{U_i}{\sum_{k=0}{|population|} U_k}`

                It's worth mentioning that "windowing" is performed if any
                negative utilities happen to appear: every individual has added
                to their utility the value :math:`\min_k U_k` and utilities
                that assume negative values are then fixed, entailing a correct
                probability distribution.

            --- 'gattaca'
                Individuals are paired up accordingly to their "rank" (fitness)
                in the population.

                Note: this option is just a joke. Don't expect great results
                from this method.

    n_selected : [int|float] (default=1.0)
        Number of individuals selected for breeding. If `int`, exactly
        `n_selected` individuals are selected. If `float`,
        `int(n_selected * self.population_size_)` individuals are
        considered instead.

        This parameter is ignored and automatically assumes the value
        `population_size_ -1` when `natural_selection` chosen is 'elitism',
        as this procedure must always keep only the fittest of every iteration.

    tournament_size : [int|float] (default=.5)
        Number of individuals fighting in a single tournament. If `int`,
        exactly `tournament_size` individuals are selected. If `float`,
        `int(n_selected * self.population_size_)` individuals are
        considered instead.

        This parameter is ignored when `breeding_selection != 'tournament'`.

    mutation_factor : float (default=.05)
        How many genes of a given individual are susceptible to mutation.

     mutation_probability : float (default=.05)
        The probability of a given gene of a individual to mutate.

    natural_selection : ['steady-state', 'elitism'] (default='steady-state')
        The method employed when selecting which individuals will survive the
        evolution process.

        Options are:
            --- 'random'
                `population_size_` individuals are randomly selected from all
                individuals in population and the offspring.

            --- 'steady-state'
                `M` individuals in the current generation are selected to
                replace the `M` worse individuals from the previous generation.

            --- 'elitism'
                All individuals from the previous generation are killed, except
                for the fittest, which replaces the worse in the current one.

    n_jobs : int (default=1)
        Number of jobs to be executed in parallel. If `-1`, execute exactly
        `cpu_count` jobs.

    Attributes
    ----------
    cycle_: int
        The number of evolution cycles that the search went through.

    generations_average_utility_: float
        The average utilities in every single generation.

    generations_highest_utility_: float
        The highest utilities in every single generation.

    generations_lowest_utility_: float
        The lowest utilities in every single generation.

    n_selected_ : int
        Real number of individuals selected for breeding.

    offspring_ : GeneticState's list
        Individuals created on this generation.
        This option is cleared after every generation's natural selection
        process.

    population_ : GeneticState's list
        Current set of individuals.

    population_size_ : int
        Real population size.

    solution_candidate_ : State-like object
        Fittest individual from all iterations. Additionally, we ought to
        mention that `fittest_` isn't necessarily contained in the last
        `population_` batch, although this is true for
        `breeding_selection == 'elitism'` option.

    selected_ : GeneticState's list
        Individuals selected for breeding in the current generation.
        This option is cleared after every generation's natural selection
        process.

    tournament_size_ : int
        Real tournament size. Assumes value `None` when
        `breeding_selection != 'tournament'`,

    variability_: float
        The variability of the last evolved population.

    """

    def __init__(self, agent,
                 population_size='auto',
                 max_evolution_cycles=np.inf,
                 max_evolution_duration=np.inf,
                 min_genetic_similarity=0,
                 breeding_selection='roulette',
                 n_selected=1.0,
                 tournament_size=.5,
                 mutation_factor=.05,
                 mutation_probability=.05,
                 natural_selection='steady-state',
                 n_jobs=1):
        super().__init__(agent=agent)

        assert isinstance(agent, agents.UtilityBasedAgent), \
            'Local searches require an utility based agent.'

        self.population_size = population_size
        self.max_evolution_cycles = max_evolution_cycles
        self.max_evolution_duration = max_evolution_duration
        self.min_genetic_similarity = min_genetic_similarity
        self.breeding_selection = breeding_selection
        self.n_selected = n_selected
        self.tournament_size = tournament_size

        self.mutation_factor = mutation_factor
        self.mutation_probability = mutation_probability

        self.natural_selection = natural_selection

        self.n_jobs = n_jobs

        self.population_size_ = self.population_ = self.tournament_size_ = None
        self.selected_ = self.n_selected_ = self.offspring_ = None
        self.started_at_ = None
        self.variability_ = 0
        self.generations_variability_ = []
        self.generations_average_utility_ = []
        self.generations_highest_utility_ = []
        self.generations_lowest_utility_ = []
        self.cycle_ = 0

    def search(self):
        """Evolve generations while the `continue_evolving` condition is
        satisfied.

        In its default form, `continue_evolving` involves satisfying three
        conditions: (1) the time spent on evolution isn't greater than the
        limit imposed by the user; (2) the number of evolution cycles doesn't
        overflow the limit imposed by the user; (3) the goal hasn't been
        reached (i.e., a `GeneticState` s.t. its `is_goal` property is `True`).

        The fittest individual from all generations is stored in
        `solution_candidate_` attribute.

        Returns
        -------
        self

        """
        self.cycle_ = 0
        self.started_at_ = time.time()
        self.solution_candidate_ = None

        self.search_start().generate_population()

        while self.continue_evolving():
            self.cycle_ += 1
            self.evolve()

        self.search_dispose()

        return self

    def search_start(self):
        if isinstance(self.population_size, int):
            self.population_size_ = self.population_size

        elif self.population_size == 'auto':
            # default population size.
            self.population_size_ = 1000

            cycles = self.max_evolution_cycles
            duration = self.max_evolution_duration
            n_jobs = (self.n_jobs
                      if self.n_jobs > 0
                      else multiprocessing.cpu_count())

            if cycles != np.inf and duration != np.inf:
                # Both cycles count and evolution duration were defined, let's
                # choose the population size that optimizes these constraints.
                # Fist, estimates the time for evaluating a individual.
                utility_elapsed = time.time()
                self.agent.utility(
                    self.agent.environment.state_class_.random())
                utility_elapsed = time.time() - utility_elapsed

                # Only 90% is used, as there are other time consuming jobs
                # during a cycle, such as breeding, mutation and selection.
                # Finally, lower bound value by 100.
                self.population_size_ = max(
                    100,
                    int(.9 * n_jobs * duration / utility_elapsed / cycles))
        else:
            raise ValueError('Illegal value for population size {%i}.'
                             % self.population_size)

        if self.natural_selection == 'elitism':
            # On elitist natural selection, every individual is replaced by the
            # offspring, saved by the fittest. We therefore must select
            # `2 * (population_size_ - 1)` individuals, which will generate an
            # offspring of exactly `population_size_ - 1` new samples.
            self.n_selected_ = 2 * (self.population_size_ - 1)
        elif isinstance(self.n_selected, int):
            self.n_selected_ = self.n_selected
        elif isinstance(self.n_selected, float):
            self.n_selected_ = int(self.n_selected * self.population_size_)
        else:
            raise ValueError('Illegal value for n_selected {%s}'
                             % str(self.n_selected))

        if self.breeding_selection not in ('random', 'tournament', 'roulette',
                                           'gattaca'):
            raise ValueError('Illegal value for breeding selection method '
                             '{%s}.' % self.breeding_selection)

        if self.natural_selection not in ('random', 'steady-state', 'elitism'):
            raise ValueError('Illegal value for natural selection method {%s}'
                             % self.natural_selection)

        if self.breeding_selection == 'tournament':
            self.tournament_size_ = (self.tournament_size
                                     if isinstance(self.tournament_size, int)
                                     else int(self.tournament_size *
                                              self.population_size_))

            if self.tournament_size_ > self.population_size_:
                raise ValueError('Illegal value for tournament size {%i}.'
                                 % self.tournament_size_)

        self.generations_average_utility_ = []
        self.generations_highest_utility_ = []
        self.generations_lowest_utility_ = []
        self.generations_variability_ = []
        self.variability_ = None

        return self

    def search_dispose(self):
        self.population_ = None
        self.selected_ = self.offspring_ = None
        return self

    def cycle_start(self):
        utilities = [self.agent.utility(i) for i in self.population_]
        self.generations_average_utility_.append(sum(utilities) /
                                                 self.population_size_)
        self.generations_highest_utility_.append(max(utilities))
        self.generations_lowest_utility_.append(min(utilities))

        return self

    def cycle_dispose(self):
        self.selected_ = self.offspring_ = None
        return self

    def generate_population(self):
        """Generate initial random population.

        The generations' size are defined by the `self.population_size`
        parameter. If its value is 'auto', a value that best fits the
        constrains is chosen.

        Notes
        -----
        This method assumes the agent is in an environment capable of
        generating random states.

        Returns
        -------
        self

        """
        self.population_ = [self.agent.environment.state_class_.random()
                            for _ in range(self.population_size_)]

        self.solution_candidate_ = max(self.population_,
                                       key=lambda i: self.agent.utility(i))

        return self

    def evolve(self):
        """Evolve a generation of individuals.

        Returns
        -------
        self

        """
        return (self
                .cycle_start()
                .select_for_breeding()
                .breed()
                .naturally_select()
                .cycle_dispose())

    def continue_evolving(self):
        """Checks if evolution process should continue.

        Returns
        -------
        True, if evolution should continue, False otherwise.

        """
        return (time.time() - self.started_at_ < self.max_evolution_duration and
                self.cycle_ < self.max_evolution_cycles and
                not self.solution_candidate_.is_goal and
                (self.min_genetic_similarity == 0 or
                 self.min_genetic_similarity <= self.genetic_similarity()))

    def select_for_breeding(self):
        """Breeding Selection.

        Select individuals for breeding by adding them to the `selected_`
        array. The order in which the individuals are in the array determines
        the breeding pairs (`2i-th` and `(2i+1)-th` are breeding partners, for
        every :math:`i \in [0, n_selected_ / 2)`).

        Returns
        -------
        self
        
        """
        if self.breeding_selection == 'random':
            self.selected_ = np.random.choice(self.population_,
                                              size=self.n_selected_)

        elif self.breeding_selection == 'tournament':
            self.selected_ = [max(np.random.choice(self.population_,
                                                   size=self.tournament_size_),
                                  key=lambda i: -self.agent.utility(i))
                              for _ in range(self.n_selected_)]

        elif self.breeding_selection == 'roulette':
            # If there are negative values, perform windowing by adding the
            # minimum value to all individuals' fitness. Any negative utilities
            # will vanish and the probabilities will sum to 1.
            p = np.array([self.agent.utility(i)
                          for i in self.population_]).astype(float)
            minimum = np.min(p)
            if minimum < 0:
                p -= minimum
            p /= np.sum(p)

            self.selected_ = np.random.choice(self.population_,
                                              size=self.n_selected_, p=p)

        elif self.breeding_selection == 'gattaca':
            self.population_.sort(key=lambda i: -self.agent.utility(i))
            self.selected_ = self.population_[:self.n_selected_]

        return self

    def breed(self):
        self.offspring_ = []
        selected = iter(self.selected_)

        try:
            while True:
                a, b = next(selected), next(selected)
                self.offspring_.append(
                    a.cross(b).mutate(self.mutation_factor,
                                      self.mutation_probability))

        except StopIteration:
            pass

        return self

    def naturally_select(self):
        self.offspring_.sort(key=lambda i: -self.agent.utility(i))
        self.population_.sort(key=lambda i: -self.agent.utility(i))

        self.solution_candidate_ = max(self.offspring_[0],
                                       self.solution_candidate_,
                                       key=lambda i: self.agent.utility(i))

        if self.natural_selection == 'random':
            self.population_ += self.offspring_
            np.random.shuffle(self.population_)
            self.population_ = self.population_[:self.population_size_]

        elif self.natural_selection == 'steady-state':
            self.population_ = (self.population_[:-len(self.offspring_)] +
                                self.offspring_)

        elif self.natural_selection == 'elitism':
            self.population_ = self.population_[:1] + self.offspring_

        return self

    def genetic_similarity(self):
        if not self.population_:
            raise ValueError('Cannot compute genetic similarity of empty '
                             'population. Are you calling genetic_similarity'
                             'method before searching?')

        individuals = np.array([i.data for i in self.population_])

        centroid = (np.sum(individuals, axis=0).astype(float) /
                    individuals.shape[0])

        # Average distance from the centroid, relative to the maximum variance
        # (all genes being different).
        # This assumes that genes assume integer values in {0, 1} set,
        # and it should be improved. :-(
        self.variability_ = np.mean(
            distance.cdist(individuals, np.array([centroid])) /
            np.sqrt(individuals.shape[1]))

        self.generations_variability_.append(self.variability_)

        return self.variability_
