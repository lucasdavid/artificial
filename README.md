# Artificial

[![Build Status](https://github.com/lucasdavid/artificial/actions/workflows/python-app.yml/badge.svg)

A basic API for intelligent agents.

## Introduction

We want to solve problems, but to do so, we must first represent them in a
reasonable fashion. In `artificial`, there are three basic classes used to
describe a problem (and its solution):

* **World** Encapsulates out-of-computer problem domains and translate them to
            a digital representation.
* **State** A model that holds data for both world and agents. It can be used
            to represent a state of the World or to predict states by agents.
* **Agent** Abstraction that presents an intelligent behavior.

## Installing

`artificial` is written on Python, and requires `numpy` and `scipy`.
You can install everything with:

```shell
git clone https://github.com/lucasdavid/artificial
cd artificial

python setup.py install --user
```

To run the examples, matplotlib and scikit-learn are required:
```
pip install -r docs/requirements-examples.txt --upgrade --user

cd examples
python dirt_cleaner.py
```

To run tests, additional packages are required:
```shell
pip install -r docs/requirements-dev.txt --upgrade --user

# Run tests!
nosetests
```

The coverage info will be placed inside `coverage` folder.

## Examples
### Genetic Algorithm: Speller Agent

Let's say we want to create an agent that generate strings randomly until it
finds the word "hello world". We start by writing a `GeneticState` and
`Environment` that define our problem.

```py
class WordIndividual(base.GeneticState):
    def cross(self, other):
        """Simple crossover between two individuals"""
        cross_point = random.randint(0, 11)
        return WordIndividual(self.data[:cross_point] + other.data[cross_point:])

    def mutate(self, factor, probability):
        """Mutation of an individual's genes"""
        m = np.random.rand(len(self.data)) < factor * probability  # Defines which genes will mutate.

        if np.any(m):
            data = np.array(list(self.data))
            data[m] = [random.choice(string.ascii_lowercase + ' ') for mutated in m if mutated]  # Mutate!
            self.data = ''.join(data)
        return self

    @property
    def is_goal(self):
        return self.data == 'hello world'

    @classmethod
    def random(cls):
        return cls(''.join(random.choice(string.ascii_lowercase + ' ') for _ in range(11))


class World(base.Environment):
    state_class_ = WordIndividual

    def update(self):
        # My world consists on printing agents' solutions. Exciting.
        for agent in self.agents:
            print('Solution found: %s' % agent.act())

```

Now we define an `UtilityBasedAgent` that considers samples similar to the
sentence `"hello world"` as "good":

```py
class Speller(agents.UtilityBasedAgent):
    def utility(self, state):
        """Measures how "good" is a word (sum of the matching letters)"""
        expected = 'hello world'
        return sum(self.data[i] == expected[i] and 1 or 0 for i in range(11)))

    def act(self):
        # We aren't really interest in acting over the world, but more in
        # finding solutions. Hence simply returns the solution candidate,
        # which is the fittest individual for the `GeneticAlgorithm` case.
        return self.search().solution_candidate_

```

The only thing left is to connect the dots! :-)

```py
world = World(initial_state=WordCandidate.random())
env.agents += [
    Speller(environment=world, search=GeneticAlgorithm,
            search_params=dict(mutation_factor=.25, mutation_probability=1))]

print('Initial: {%s}' % str(env.current_state))
world.update()

```

**Note:** take a look at the
[examples](https://github.com/lucasdavid/artificial/tree/master/examples)
folder to see more concrete usages of searches, optimizations, evolutions and
other intelligent stuff.
