"""Knapsack Example.

This example demonstrates how to approximate the Knapsack problem
using genetic algorithms.

Author: Lucas David -- <ld492@drexel.edu>
License: MIT (c) 2016

"""

import time

import artificial as at
import matplotlib.pyplot as plt
import numpy as np
from artificial import searches

random_state = np.random.RandomState(0)

search_params = dict(
    breeding_selection='random',
    # natural_selection='steady-state',
    mutation_probability=.1,
    max_evolution_duration=10,
    min_genetic_similarity=.001,
    debug=True)


class Bag(at.base.GeneticState):
    def h(self):
        bag_weight = sum(World.items_info_expanded[i][1]
                         for i, is_in_the_bag in enumerate(self.data)
                         if is_in_the_bag)

        # overweight bags have their heuristic increased by 200%.
        bag_weight = 1 if bag_weight <= World.bag_max_weight else 3

        return -sum(World.items_info_expanded[i][2]
                    for i, is_in_the_bag in enumerate(self.data)
                    if is_in_the_bag) / bag_weight

    def cross(self, other):
        cross_point = random_state.randint(0, len(World.items_info_expanded))
        return Bag(self.data[:cross_point] + other.data[cross_point:])

    def mutate(self, factor, probability):
        m = random_state.rand(len(self.data)) <= probability

        data = np.array(list(self.data))
        data[m] = 1 - data[m]

        self.data = data.tolist()

        return self

    @classmethod
    def random(cls):
        return Bag(np
                   .round(random_state.rand(len(World.items_info_expanded)))
                   .astype(int).tolist())


class World(at.base.Environment):
    state_class_ = Bag

    bag_max_weight = 400
    items_info = (
        ("map", 9, 150, 1),
        ("compass", 13, 35, 1),
        ("water", 153, 200, 3),
        ("sandwich", 50, 60, 2),
        ("glucose", 15, 60, 2),
        ("tin", 68, 45, 3),
        ("banana", 27, 60, 3),
        ("apple", 39, 40, 3),
        ("cheese", 23, 30, 1),
        ("beer", 52, 10, 3),
        ("suntan cream", 11, 70, 1),
        ("camera", 32, 30, 1),
        ("t-shirt", 24, 15, 2),
        ("trousers", 48, 10, 2),
        ("umbrella", 73, 40, 1),
        ("waterproof trousers", 42, 70, 1),
        ("waterproof overclothes", 43, 75, 1),
        ("note-case", 22, 80, 1),
        ("sunglasses", 7, 20, 1),
        ("towel", 18, 12, 2),
        ("socks", 4, 50, 1),
        ("book", 30, 10, 2),
    )
    items_info_expanded = sum(([(i, w, c)] * q for i, w, c, q in items_info),
                              [])

    def update(self):
        for agent in self.agents:
            solution = agent.act()
            print('Bag: {%s}' % ', '.join(World.items_info_expanded[item][0]
                                          for item, in_the_bag
                                          in enumerate(solution.data)
                                          if item))
            print('Utility: %i' % agent.utility(solution))
            print('Weight: %i' % sum(World.items_info_expanded[i][1]
                                     for i, is_in_the_bag
                                     in enumerate(solution.data)
                                     if is_in_the_bag))

            # print('\n'.join(str(i.data)
            #                 for i in env.agents[0].search.population_))

            print('Cycles: %i' % agent.search.cycle_)
            print('Variability: %f' % agent.search.variability_)


class Hiker(at.agents.UtilityBasedAgent):
    def act(self):
        return (self.search
                .search()
                .solution_candidate_)

    def predict(self, state):
        raise RuntimeError('Sorry! I don\'t know how to predict states!')


def main():
    print(__doc__)

    env = World(initial_state=Bag.random())
    agent = Hiker(environment=env, search=searches.genetic.GeneticAlgorithm,
                  search_params=search_params)
    env.agents = [agent]

    start = time.time()

    try:
        env.update()

        print('\nTime elapsed: %.2f s' % (time.time() - start))

        search = env.agents[0].search

        plt.subplot(1, 2, 1)
        plt.plot(search.highest_utility_, linewidth=2, color='r')
        plt.plot(search.average_utility_, linewidth=4,
                 color='orange')
        plt.plot(search.lowest_utility_, linewidth=2, color='y')
        plt.ylabel('Generations utility')
        plt.title('Utility')

        plt.subplot(1, 2, 2)
        plt.plot(search.variability_, linewidth=4, color='orange')
        plt.ylabel('Generations\' variability')
        plt.title('Variability')

        plt.show()

    except KeyboardInterrupt: print('canceled by user')


if __name__ == '__main__':
    main()
