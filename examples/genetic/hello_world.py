import string
import time

import numpy as np
import random
from artificial import base, agents
from artificial.searches.genetic import GeneticAlgorithm

random_state = np.random.RandomState(0)

search_params = dict(mutation_probability=.25,
                     max_evolution_duration=10)


class WordIndividual(base.GeneticState):
    expected = 'hello world'
    alphabet = list(string.ascii_lowercase + ' ')

    def h(self):
        return sum(1 if self.data[i] != self.expected[i] else 0
                   for i in range(min(len(self.data), len(self.expected))))

    def cross(self, other):
        cross_point = random.randint(0, len(WordIndividual.expected))
        return WordIndividual(
            self.data[:cross_point] + other.data[cross_point:])

    def mutate(self, factor, probability):
        m = random_state.rand(len(self.data)) < probability

        if np.any(m):
            data = np.array(list(self.data))
            data[m] = random_state.choice(self.alphabet, size=m.sum())
            self.data = ''.join(data)

        return self

    @property
    def is_goal(self):
        return self.data == self.expected

    def __str__(self):
        return '%s' % str(self.data)

    @classmethod
    def random(cls):
        return cls(''.join(random_state.choice(cls.alphabet,
                                               size=len(cls.expected))))


class Speller(agents.UtilityBasedAgent):
    def act(self):
        return (self.search
                .search()
                .solution_candidate_)

    def predict(self, state):
        raise RuntimeError('Sorry! I don\'t know how to predict states!')


class World(base.Environment):
    state_class_ = WordIndividual

    def update(self):
        print('Initial: {%s}' % str(self.current_state))

        for a in self.agents:
            self.current_state = a.act()
            print('Agent found the solution: {%s}' % self.current_state)


def main():
    print('====================')
    print('Word Speller Example')
    print('====================\n')

    env = World(initial_state=WordIndividual.random())
    agent = Speller(environment=env, search=GeneticAlgorithm,
                    search_params=search_params)
    env.agents = [agent]
    start = time.time()

    try:
        env.update()
    except KeyboardInterrupt:
        pass
    finally:
        print('\nTime elapsed: %.2f s' % (time.time() - start))


if __name__ == '__main__':
    main()
