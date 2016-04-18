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

    n_selected_ : int
        Real number of individuals selected for breeding.

    tournament_size_ : int
        Real tournament size. Assumes value `None` when
        `breeding_selection != 'tournament'`,

    population_ : GeneticState's list
        Current set of individuals.

    selected_ : GeneticState's list
        Individuals selected for breeding in the current generation.
        This option is cleared after every generation's natural selection
        process.

    offspring_ : GeneticState's list
        Individuals created on this generation.
        This option is cleared after every generation's natural selection
        process.

    solution_candidate_ : State-like object
        Fittest individual from all iterations. Additionally, we ought to
        mention that `fittest_` isn't necessarily contained in the last
        `population_` batch, although this is true for
        `breeding_selection == 'elitism'` option.

    """

    def __init__(self, agent,
                 population_size='auto',
                 max_evolution_cycles=np.inf,
                 max_evolution_duration=np.inf,
                 breeding_selection='roulette',
                 n_selected=.5,
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
                    int(.9 * n_jobs * duration / utility_elapsed / cycles)
                )
        else:
            raise ValueError('Illegal value for population size {%i}.'
                             % self.population_size)

        if self.natural_selection == 'elitism':
            self.n_selected_ = self.population_size_ - 1
        elif isinstance(self.n_selected, int):
            self.n_selected_ = self.n_selected
        elif isinstance(self.n_selected, float):
            self.n_selected_ = int(self.n_selected * self.population_size_)
        else:
            raise ValueError('Illegal value for n_selected {%s}'
                             % str(self.n_selected))

        if self.breeding_selection == 'tournament':
            self.tournament_size_ = (self.tournament_size
                                     if isinstance(self.tournament_size, int)
                                     else int(self.tournament_size *
                                              self.population_size_))

            if self.tournament_size_ > self.population_size_:
                raise ValueError('Illegal value for tournament size {%i}.'
                                 % self.tournament_size_)

        return self

    def search_dispose(self):
        return self

    def cycle_start(self):
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
                not self.solution_candidate_.is_goal)

    def select_for_breeding(self):
        """Breeding Selection.

        Select individuals for breeding by adding them to the
        `selected_` array.

        Returns
        -------
        self
        
        """
        if self.breeding_selection == 'random':
            self.selected_ = np.random.choice(self.population_,
                                              size=self.n_selected_)

        elif self.breeding_selection == 'tournament':
            self.selected_ = [
                max(np.random.choice(self.population_,
                                     size=self.tournament_size_),
                    key=lambda i: -self.agent.utility(i))
                for _ in range(self.n_selected_)
            ]

        elif self.breeding_selection == 'roulette':
            # Perform windowing by adding the minimum value to all individuals'
            # fitness. If there are any negative utilities, they will vanish
            # and the probabilities will sum to 1.
            p = np.array([self.agent.utility(i)
                          for i in self.population_]).astype(float)
            p -= np.min(p)
            p /= np.sum(p)

            self.selected_ = np.random.choice(self.population_,
                                              size=self.n_selected_, p=p)

        elif self.breeding_selection == 'gattaca':
            self.population_.sort(key=lambda i: -self.agent.utility(i))
            self.selected_ = self.population_[:self.n_selected_]

        else:
            raise ValueError('Illegal value for breeding selection method '
                             '{%s}.' % self.breeding_selection)

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

    def naturally_select(self):
        self.offspring_.sort(key=lambda i: -self.agent.utility(i))
        self.population_.sort(key=lambda i: -self.agent.utility(i))

        self.solution_candidate_ = max(self.offspring_[0],
                                       self.solution_candidate_,
                                       key=lambda i: self.agent.utility(i))

        if self.natural_selection == 'steady-state':
            self.population_ = (self.population_[:-len(self.offspring_)] +
                                self.offspring_)

        elif self.natural_selection == 'elitism':
            self.population_ = self.population_[:1] + self.offspring_

        else:
            raise ValueError('Illegal option for natural selection {%s}'
                             % self.natural_selection)

        return self
