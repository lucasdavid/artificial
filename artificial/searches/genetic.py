import time

from . import base


class GeneticAlgorithm(base.Base):
    """Genetic Algorithm Search.
    
    Parameters
    ----------
    
    agent : inherited

    root  : inherited
    
    population_size : [int|'auto'] (default='auto')
        The size of the population to be kept in memory.
    
    max_evolutive_cycles : [int, np.inf] (default=np.inf]
        The upper bound for evolution cycles, in cycle count. 
        If infinity, no bound is set and evolution cycles won't
        be directly a factor for the evolutive process stop.
    
    max_evolutive_duration : [float, np.inf] (default=np.inf)
        The upper bound for evolution duration, in seconds.
        If infinity, no bound is set and evolution will continue 
        regardless the amount of time spent on the process.
    
    breeding_selection_method : string ['random'|'roulette'] 
                                       (default='roulette')
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
    
    """
    
    def __init__(self, agent, root=None,
                 population_size='auto',
                 max_evolutive_cycles=np.inf, 
                 max_evolutive_duration=np.inf,
                 breeding_selection_method='roulette',
                 n_jobs=1):
        super().__init__(agent=agent, root=root)
        
        assert isinstance(agent, agents.UtilityBasedAgent), \
            'Local searches require an utility based agent.'
        
        self.population_size = population_size
        self.max_evolutive_cycles = max_evolutive_cycles
        self.max_evolutive_duration = max_evolutive_duration
        self.breeding_selection_method = breeding_selection_method
        self.n_jobs = n_jobs
        
        self.real_population_size_ = None
        
    def search(self):
        self.started_at = time.time()
        self.solution_candidate_ = None
        
        return (self
                    .generate_population()
                    .evolve())
    
    def generate_population(self):
        self.population = [self.agent.env.generate_random_state()
                           for _ in range(population_size)]
                           
        return self
      
    def evolve(self):
        s = self
        cycle = 0
        
        while self.continue_evolving(cycle):
            cycle += 1
            
            s.population = s.natural_selection(
                s.breed(
                    s.breeding_selection()))

        return self
    
    def continue_evolving(self, evolution_cycle):
        return (time.time() - self.started_at < self.max_evolution_time and
                evolution_cycle < self.max_evolution_cycles)
    
    def breeding_selection(self):
        raise NotImplementedError
    
    def natural_selection(self, children):
        raise NotImplementedError
   
    def breed(self, individuals):
        raise NotImplementedError
