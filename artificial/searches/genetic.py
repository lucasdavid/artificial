import multiprocessing
import time

import numpy as np

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

                It's worth mentioning that "windowing" is performed: every
                individual's added to its utility the value :math:`\min_k U_k`
                and utilities that assume negative values are then fixed,
                entailing a correct probability distribution.

            --- 'gattaca'
                Individuals are paired up accordingly to their "rank" (fitness)
                in the population.

                Note: this option is just a joke. Don't expect great results
                from this method.

    n_selected : [int|float] (default=.5)
        Number of individuals selected for breeding. If `int`, exactly
        `n_selected` individuals are selected. If `float`,
        `int(n_selected * self.population_size_)` individuals are
        considered instead.

        This parameter is ignored and automatically assumes the value
        `population_size_ -1` when `natural_selection` chosen is 'elitism',
        as this procedure must always keep only the fittest of every iteration.

    natural_selection : ['steady-state', 'elitism'] (default='steady-state')
        The method employed when selecting which individuals will survive the
        evolution process.

        Options are:
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

    population_size_ : int
        Real population size.

    population_ : GeneticState's list
        Current set of individuals.

    selected_ : GeneticState's list
        Individuals selected for breeding in the current generation.

    offspring_ : GeneticState's list
        Individuals created on this generation.

    """

    def __init__(self, agent,
                 population_size='auto',
                 max_evolution_cycles=np.inf,
                 max_evolution_duration=np.inf,
                 breeding_selection='roulette',
                 n_selected=.5,
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
        self.breeding_selection_method = breeding_selection
        self.n_selected = n_selected

        self.mutation_factor = mutation_factor
        self.mutation_probability = mutation_probability

        self.natural_selection = natural_selection

        self.n_jobs = n_jobs

        self.population_size_ = self.population_ = None
        self.selected_ = self.offspring_ = None
        self.started_at_ = None
        self.cycle_ = 0

    def search(self):
        self.started_at_ = time.time()
        self.solution_candidate_ = None

        return self.generate_population().evolve()

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
        if isinstance(self.population_size, int):
            self.population_size_ = self.population_size

        elif self.population_size == 'auto':
            # default population size.
            self.population_size_ = 1000

            cycles = self.max_evolution_cycles
            duration = self.max_evolution_duration
            n_jobs = (self.n_jobs
                      if self.n_jobs > -1
                      else multiprocessing.cpu_count())

            if cycles != np.inf and duration != np.inf:
                # Both cycles count and evolution duration were defined, let's
                # choose the population size that optimizes these constraints.
                # Fist, estimates the time for evaluating a individual.
                utility_elapsed = time.time()
                self.agent.utility(
                    self.agent.environment.generate_random_state())
                utility_elapsed = time.time() - utility_elapsed

                # Only 90% is used, as there are other time consuming jobs
                # during a cycle, such as breeding, mutation and selection.
                self.population_size_ = round(.9 * n_jobs * duration /
                                              utility_elapsed / cycles)
        else:
            raise ValueError('Illegal value for population size {%s}.'
                             % self.population_size)

        self.population_ = [self.agent.environment.generate_random_state()
                            for _ in range(self.population_size_)]

        self.solution_candidate_ = max(self.population_,
                                       key=lambda i: self.agent.utility(i))

        return self

    def evolve(self):
        """Evolve while a `continue_evolving` condition is satisfied.

        Returns
        -------
        self

        """
        s = self
        self.cycle_ = 0

        while self.continue_evolving():
            self.cycle_ += 1
            s.breeding_selection().breed().select()

            del self.selected_, self.offspring_

        return self

    def continue_evolving(self):
        """Checks if evolution process should continue.

        Returns
        -------
        True, if evolution should continue, False otherwise.

        """
        return (time.time() - self.started_at_ < self.max_evolution_duration and
                self.cycle_ < self.max_evolution_cycles and
                not self.solution_candidate_.is_goal)

    def breeding_selection(self):
        """Breeding Selection.

        Select individuals for breeding by adding them to the
        `selected_` array.

        Returns
        -------
        self
        """

        n_selected = (self.population_size_ - 1 if self.natural_selection == 'elitism'
                      else self.n_selected if isinstance(self.n_selected, int)
                      else int(self.n_selected * self.population_size_))

        if n_selected > self.population_size_:
            raise ValueError('Illegal value for n_selected: {%i}' % n_selected)

        if self.breeding_selection_method == 'random':
            np.random.shuffle(self.population_)
            self.selected_ = self.population_[:n_selected]

        elif self.breeding_selection_method == 'tournament':
            self.selected_ = []

            for _ in range(n_selected):
                np.random.shuffle(self.population_)
                winner = max(self.population_[:n_selected],
                             key=lambda i: -self.agent.utility(i))

                self.selected_.append(winner)

        elif self.breeding_selection_method == 'roulette':
            # Add minimum value. If there are any negative utilities,
            # they will vanish and the probabilities will sum to 1.
            p = np.array([self.agent.utility(i)
                          for i in self.population_]).astype(float)
            p -= np.min(p)
            p /= np.sum(p)

            self.selected_ = np.random.choice(self.population_,
                                              size=n_selected, p=p)

        elif self.breeding_selection_method == 'gattaca':
            self.population_.sort(key=lambda i: -self.agent.utility(i))
            self.selected_ = self.population_[:n_selected]

        else:
            raise ValueError('Illegal value for breeding selection method '
                             '{%s}.' % self.breeding_selection_method)

        return self

    def breed(self):
        self.offspring_ = []
        selected = iter(self.selected_)

        try:
            while True:
                a, b = next(selected), next(selected)
                self.offspring_.append(
                    a
                    .cross(b)
                    .mutate(self.mutation_factor,
                            self.mutation_probability))

        except StopIteration:
            pass

        return self

    def select(self):
        self.offspring_.sort(key=lambda i: -self.agent.utility(i))

        self.solution_candidate_ = max(self.offspring_[0],
                                       self.solution_candidate_,
                                       key=lambda i: self.agent.utility(i))

        if self.natural_selection == 'steady-state':
            self.population_.sort(key=lambda i: -self.agent.utility(i))
            self.population_ = (self.population_[:len(self.offspring_)] +
                                self.offspring_)

        elif self.natural_selection == 'elitism':
            fittest = self.population_[0]

            self.population_ = self.offspring_
            self.population_.append(fittest)

        else:
            raise ValueError('Illegal option for natural selection {%s}'
                             % self.natural_selection)
