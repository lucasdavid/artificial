import random
import string
import time

import numpy as np

from artificial import base, agents
from artificial.searches.genetic import GeneticAlgorithm


class WordIndividual(base.GeneticState):
    expected = 'i like cookies'

    def h(self):
        return sum((1 if self.data[i] != self.expected[i] else 0
                    for i in range(min(len(self.data), len(self.expected)))))

    def cross(self, other):
        cross_point = random.randint(0, len(WordIndividual.expected))
        return WordIndividual(
            self.data[:cross_point] + other.data[cross_point:])

    def mutate(self, factor, probability):
        m = np.random.rand(len(self.data)) < factor * probability

        if np.any(m):
            data = np.array(list(self.data))
            data[m] = [random.choice(string.ascii_lowercase + ' ')
                       for mutated in m if mutated]

            self.data = ''.join(data)

        return self

    @property
    def is_goal(self):
        return self.data == self.expected

    def __str__(self):
        return '%s' % str(self.data)

    @classmethod
    def random(cls):
        return cls(''.join(random.choice(string.ascii_lowercase + ' ')
                           for _ in WordIndividual.expected))


class Speller(agents.UtilityBasedAgent):
    def act(self):
        return (self.search
                .search()
                .solution_candidate_)

    def predict(self, state):
        pass


class World(base.Environment):
    state_class_ = WordIndividual

    def update(self):
        for a in self.agents:
            self.current_state = a.act()


def main():
    print('====================')
    print('Word Speller Example')
    print('====================\n')

    env = World(initial_state=None)
    env.agents += [
        Speller(environment=env,
                search=GeneticAlgorithm,
                search_params=dict(mutation_factor=.25, mutation_probability=1),
                actions=(0, 1))]

    print('Initial: {%s}' % str(env.current_state))

    start = time.time()

    try:
        env.update()
        print('Solution: {%s}' % str(env.current_state))

    except KeyboardInterrupt:
        pass

    finally:
        search = env.agents[0].search

        print('\nPopulation:')
        print('\n'.join(str(p) for p in search.population_[:30]))

        print('\nTime elapsed: %.2f s' % (time.time() - start))
        print('Cycles: %i' % search.cycle_)


if __name__ == '__main__':
    main()
