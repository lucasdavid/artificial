"""Neural Networks initialization and training with genetic algorithms"""

# Author: Lucas David -- <ld492@drexel.edu>
# License: MIT (c) 2016

from time import time

import artificial as at
import matplotlib.pyplot as plt
import numpy as np
from artificial import searches
from artificial.utils import (preprocessing, helpers, model_selection,
                              datasets)

random_state = np.random.RandomState(0)


def plot_decision_boundary(X, y, l):
    """Plot the decision boundaries of a classifier."""
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = l(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)


# Parameters which define the problem.
data_params = dict(
    n_samples=300,
    n_components=2,
    n_classes=3,
    random_state=random_state
)

# Parameters for `GeneticAlgorithm` search.
search_params = dict(
    max_evolution_duration=10,
    # max_evolution_cycles=np.inf,
    # min_genetic_similarity=0,
    population_size=100,
    mutation_factor=.2,
    mutation_probability=.1,
    # n_selected=1.0,
    # breeding_selection='roulette',
    # tournament_size=.5,
    # natural_selection='steady-state',
    # n_jobs=1,
    debug=True,
    random_state=random_state,
)

# Parameters used to instantiate NNs.
nn_params = dict(
    architecture=(2, 40, 3),
    n_epochs=50,
    learning_rate=.1,
    regularization=.001,
    random_state=random_state,
)

# Plotting decision boundaries.
exec_params = dict(
    plotting=True
)


class LayeredNN:
    """Fully Connected layered Neural Network."""

    def __init__(self, architecture, learning_rate=1e0,
                 regularization=1e-4, n_epochs=1000, copy=True, verbose=False,
                 random_state=None):
        self.architecture = architecture
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.n_epochs = n_epochs
        self.copy = copy
        self.verbose = verbose
        self.random_state = random_state or np.random.RandomState()

        self.W_ = self.b_ = None

    def random_build(self):
        """Build NN architecture randomly"""
        arc = self.architecture

        self.W_, self.b_ = [], []

        for i in range(1, len(arc)):
            # Build random matrices based on the number of units in the
            # previous and current layers.
            self.W_.append(self.random_state.randn(arc[i - 1], arc[i]) /
                           np.sqrt(arc[i - 1]))
            self.b_.append(np.zeros((1, arc[i])))

        return self

    def custom_build(self, W, b):
        """Set a custom architecture for the NN"""
        self.W_ = W
        self.b_ = b

        return self

    def fit(self, X, y):
        if self.W_ is None:
            # If architecture graph's weights aren't defined, set it randomly.
            self.random_build()

        X = X.copy() if self.copy else X

        for i in range(0, self.n_epochs):
            W1, W2 = self.W_
            b1, b2 = self.b_

            # Forward propagation.
            z1 = X.dot(W1) + b1
            a1 = np.tanh(z1)
            z2 = a1.dot(W2) + b2
            exp_scores = np.exp(z2)
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

            # Backpropagation.
            delta3 = probs
            delta3[range(X.shape[0]), y] -= 1
            dW2 = a1.T.dot(delta3)
            db2 = np.sum(delta3, axis=0, keepdims=True)
            delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
            dW1 = np.dot(X.T, delta2)
            db1 = np.sum(delta2, axis=0)

            # Add regularization terms.
            dW2 += self.regularization * W2
            dW1 += self.regularization * W1

            # Gradient descent parameter update.
            W1 += -self.learning_rate * dW1
            b1 += -self.learning_rate * db1
            W2 += -self.learning_rate * dW2
            b2 += -self.learning_rate * db2

            if self.verbose and i % 1000 == 0:
                print("Loss after iteration %i: %f" % (i, self.loss(X)))

        return self

    def loss(self, X):
        W1, W2 = self.W_
        b1, b2 = self.b_

        z1 = X.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        # Calculating the loss
        corect_logprobs = -np.log(probs[range(X.shape[0]), y])
        data_loss = np.sum(corect_logprobs)
        # Add regulatization term to loss (optional)
        data_loss += (self.regularization / 2 *
                      (np.sum(np.square(W1)) + np.sum(np.square(W2))))

        return 1. / X.shape[0] * data_loss

    def predict(self, X):
        W1, W2 = self.W_
        b1, b2 = self.b_

        # Forward propagation
        z1 = X.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        return np.argmax(probs, axis=1)

    def score(self, X, y):
        return (self.predict(X).flatten() == y).sum() / X.shape[0]


class TrainingCandidate(at.base.GeneticState):
    @property
    def W_(self):
        return self.data.W_

    @property
    def b_(self):
        return self.data.b_

    def h(self):
        """Heuristic associated to Training Candidate.

        `GeneticAlgorithm` naturally selects strong candidates in population
        :math:`P`. That is, candidates with high associated utility
        :math:`u: P \to \mathbb{R}`:

        :math:`u(c) := -cost(c) = -(g(c) + h(c)) = -h(c)`

        We therefore minimize the heuristic to maximize the utility.

        """
        gnn = self.data
        y = gnn.predict(World.X)

        return np.sum((y - World.y) ** 2)

    def cross(self, other):
        """Crossover operator.

        Each `n_layers` biparted section of the network is crossovered between
        two neuralnets of same architecture given `n_layers` cut points.

        Returns
        -------
        A `TrainingCandidate` containig a new neural network with the same
        architecture of its parents.

        """
        W_c_ = []
        b_c_ = []

        for layer, (W_a, b_a, W_b, b_b) in enumerate(zip(self.W_, self.b_,
                                                         other.W_, other.b_)):
            shape = W_a.shape
            n_elements = shape[0] * shape[1]

            cut = int(random_state.rand() * n_elements)

            W_c = np.hstack((W_a.flatten()[:cut], W_b.flatten()[cut:]))
            W_c.resize(shape)
            W_c_.append(W_c)

            cut = int(random_state.rand() * b_a.shape[1])

            b_c = np.hstack((b_a[:, :cut], b_b[:, cut:]))
            b_c_.append(b_c)

        return TrainingCandidate(LayeredNN(**nn_params)
                                 .custom_build(W_c_, b_c_))

    def mutate(self, factor, probability):
        for layer, (W, b) in enumerate(zip(self.W_, self.b_)):
            mutates = random_state.rand(*W.shape) < probability
            W[mutates] += (2 * random_state.rand() - 1) * factor

            mutates = random_state.rand(*b.shape) < probability
            b[mutates] += (2 * random_state.rand() - 1) * factor

        return self

    @classmethod
    def random(cls):
        return TrainingCandidate(LayeredNN(**nn_params).random_build())


class World(at.base.Environment):
    state_class_ = TrainingCandidate

    def build(self):
        X, y = datasets.make_vortex(**data_params)
        X = preprocessing.scale(X)

        plt.scatter(X[:, 0], X[:, 1], c=y)
        plt.show()

        # Separate train and test data.
        X, X_test, y, y_test = model_selection.train_test_split(
            X, y, train_size=.8, random_state=random_state)

        # Set class attributes.
        World.X, World.X_test, World.y, World.y_test = X, X_test, y, y_test

        # Build a regularly trained Neural Network once.
        # We'll use it as base for our benchmarks.
        World.cnn = LayeredNN(**nn_params)
        self.train(World.cnn)

        # Build a genetically initialized trained Neural Network.
        World.gnn = LayeredNN(**nn_params)

    def update(self):
        for agent in self.agents:
            # Ask agent to find a trained net for us.
            print('Genetic initialization has started...')

            t = time()
            training = agent.act()
            print('Genetic initiation complete (%i cycles, %f s)'
                  % (agent.search.cycle_, time() - t))

            if exec_params['plotting']:
                # Plotting generations' utilities.
                plt.plot(agent.search.lowest_utility_,
                         color='blue', linewidth=4, label='Lowest')
                plt.plot(agent.search.average_utility_,
                         color='orange', linewidth=4, label='Average')
                plt.plot(agent.search.highest_utility_,
                         color='red', linewidth=4, label='Highest')
                plt.legend()

                plt.xlabel('generation')
                plt.ylabel('utility')

                plt.tight_layout()
                plt.show()

            if not training:
                continue

            # Let's cross our fingers! (yn)
            World.gnn.custom_build(training.W_, training.b_)

            # Measures how training goes.
            self.train(World.gnn)

    def train(self, nn):
        print('Regular training ongoing...')

        t = time()
        nn.fit(World.X, World.y)
        print('Training complete (%f s)' % (time() - t))

        if exec_params['plotting']:
            plot_decision_boundary(World.X, World.y, lambda x: nn.predict(x))
            plt.tight_layout()
            plt.show()

        print('Accuracy score: %f'
              % nn.score(World.X_test, World.y_test))
        print('-----------------------\n')


class NNTrainer(at.agents.UtilityBasedAgent):
    def predict(self, state):
        """Predicts nothing"""

    def act(self):
        """Find a training candidate for a ANN"""
        return (self.search
                .restart(root=self.last_state)
                .search()
                .solution_candidate_)


def main():
    print('=========================================')
    print('Neural Networks TrainingCandidate Example')
    print('=========================================\n')

    world = World()
    world.agents = [NNTrainer(searches.genetic.GeneticAlgorithm, world,
                              search_params=search_params)]

    # Run world for 1 life-cycle.
    helpers.live(world, n_cycles=1)


if __name__ == '__main__':
    main()
