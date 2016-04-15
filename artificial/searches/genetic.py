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

    root  : inherited
    
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
    
    breeding_selection_method : ['random'|'roulette'] (default='roulette')
        The method employed on the selection of individuals for
        breeding.
        
        Options are:
            --- 'random'
                Individuals are selected at random.

            --- 'roulette'
                Each individual `i` has an associated rate
                :math:`\frac{U_i}{\sum_{k=0}{|population|} U_k}`
                
                If negative values are allowed for the utilities,
                every individual has added to their utility the
                value :math:`\min_k U_k`.

    n_jobs : int (default=1)
        Number of jobs to be executed in parallel. If `-1`, execute exactly
        `cpu_count` jobs.

    """

    def __init__(self, agent, root=None,
                 population_size='auto',
                 max_evolution_cycles=np.inf,
                 max_evolution_duration=np.inf,
                 breeding_selection_method='roulette',
                 n_jobs=1):
        super().__init__(agent=agent, root=root)

        assert isinstance(agent, agents.UtilityBasedAgent), \
            'Local searches require an utility based agent.'

        self.population_size = population_size
        self.max_evolution_cycles = max_evolution_cycles
        self.max_evolution_duration = max_evolution_duration
        self.breeding_selection_method = breeding_selection_method
        self.n_jobs = n_jobs

        self.real_population_size_ = self.population_ = None
        self.started_at_ = None

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
        if self.population_size == 'auto':
            # default population size.
            population_size = 10000

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
                population_size = round(.9 * n_jobs * duration /
                                        utility_elapsed / cycles)

        elif isinstance(self.population_size, int):
            population_size = self.population_size
        else:
            raise ValueError('Illegal value for population size {%s}.'
                             % self.population_size)

        self.population_ = [self.agent.environment.generate_random_state()
                            for _ in range(population_size)]

        return self

    def evolve(self):
        s = self
        cycle = 0

        while self.continue_evolving(cycle):
            cycle += 1

            s.breeding_selection().breed().natural_selection()

        return self

    def continue_evolving(self, evolution_cycle):
        """Checks if evolution process should continue.

        Parameters
        ----------
        evolution_cycle : int
            Current cycle of evolution.

        Returns
        -------
        True, if evolution should continue, False otherwise.

        """
        return (time.time() - self.started_at_ < self.max_evolution_cycles and
                evolution_cycle < self.max_evolution_cycles)

    def breeding_selection(self):
        raise NotImplementedError

    def natural_selection(self, children):
        raise NotImplementedError

    def breed(self, individuals):
        raise NotImplementedError
