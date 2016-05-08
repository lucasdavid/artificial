"""Neural Networks Training with Genetic Algorithms."""

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


execute_following_training = [
    # Training Iris data set.
    dict(
        dataset=lambda: datasets.iris(n_features=2),
        nn_params=dict(
            architecture=(2, 40, 3),
            n_epochs=200,
            learning_rate=.1,
            regularization=.001,
            random_state=random_state,
        ),
        search_params=dict(
            max_evolution_duration=10 * 60,
            # max_evolution_cycles=np.inf,
            # min_genetic_similarity=0,
            population_size=1000,
            mutation_factor=.2,
            mutation_probability=.1,
            # n_selected=1.0,
            # breeding_selection='roulette',
            # tournament_size=.5,
            # natural_selection='steady-state',
            # n_jobs=1,
            # debug=True,
            random_state=random_state,
        ),
        settings=dict(
            plotting=True
        ),
    ),

    # Training Spiral data set.
    dict(
        dataset=lambda: datasets.make_spiral(
            n_samples=300, n_components=2, n_classes=3,
            random_state=random_state),
        nn_params=dict(
            architecture=(2, 40, 3),
            n_epochs=200,
            learning_rate=.1,
            regularization=.001,
            random_state=random_state,
        ),
        search_params=dict(
            max_evolution_duration=10 * 60,
            mutation_factor=1,
            mutation_probability=.1,
            debug=True,
            random_state=random_state,
        ),
        settings=dict(
            plotting=True
        ),
    )
]


class NN:
    """Fully Connected, 3-Layered Artificial Neural Network.

    This class is an adaption from Britz original implementation in [1].

    References
    ----------
    [1] D. Britz. (2015). IMPLEMENTING A NEURAL NETWORK FROM SCRATCH IN
    PYTHON – AN INTRODUCTION [Online]. Available:
    http://www.wildml.com/2015/09/implementing-a-neural-network-from-scratch/

    """

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
        self._X = self._y = None

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
        X = X.copy() if self.copy else X

        self._X, self._y = X, y

        if self.W_ is None:
            # If architecture graph's weights aren't defined, set it randomly.
            self.random_build()

        for i in range(0, self.n_epochs):
            W1, W2 = self.W_
            b1, b2 = self.b_

            # Forward propagation.
            z1 = X.dot(W1) + b1
            a1 = np.tanh(z1)
            z2 = a1.dot(W2) + b2
            exp_scores = np.exp(z2)
            probabilities = (exp_scores /
                             np.sum(exp_scores, axis=1, keepdims=True))

            # Back-propagation.
            delta3 = probabilities
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
                print("Loss after iteration %i: %f" % (i, self.loss(X, y)))

        return self

    def loss(self, X, y):
        W1, W2 = self.W_
        b1, b2 = self.b_

        z1 = X.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)
        probabilities = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        # Calculating the loss
        correct_log_probs = -np.log(probabilities[range(X.shape[0]), y])
        data_loss = np.sum(correct_log_probs)

        # Add regularization term to loss (optional)
        data_loss += (self.regularization / 2 *
                      (np.sum(np.square(W1)) + np.sum(np.square(W2))))

        return 1. / X.shape[0] * data_loss

    def feedforward(self, X):
        W1, W2 = self.W_
        b1, b2 = self.b_

        z1 = X.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def predict(self, X):
        return np.argmax(self.feedforward(X), axis=1)

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
        """The Heuristic Associated to the Training Candidate.

        `GeneticAlgorithm` naturally selects strong candidates in the
        population. That is, candidates with high associated utility
        :math:`u: P \to \mathbb{R}`:

        :math:`u(c) := -cost(c) = -(g(c) + h(c)) = -h(c)`

        Hence, candidates with low associated heuristic. That means that
        candidates with low heuristic cost are fitter than candidates with
        higher heuristic cost.

        """
        gnn = self.data
        return np.sum((gnn.predict(World.X) - World.y) ** 2)

    def cross(self, other):
        """Crossover Operator.

        This operator is applied to each `n_layers` bipartite section of this
        network and other - of same architecture - given `n_layers` cut points.

        Returns
        -------
        A `TrainingCandidate` containing a new neural network with the same
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

        nn_params = World.current_execution_params['nn_params']
        return TrainingCandidate(NN(**nn_params).custom_build(W_c_, b_c_))

    def mutate(self, factor, probability):
        """Mutation Operator.

        Each layer has its connection matrix and bias affected by mutation.
        Mutation occurs by adding to the current value a random `n \in [-1, 1]`
        scaled by the mutation `factor`.

        The motivation for this operator is that very good values would
        only be slightly mutated, instead of corrupted altogether. In the other
        hand, weak values might be slightly increased, increasing the overall
        score and maybe securing the individual's survival to the following
        generation.

        Notice that the expectancy of mutations in `W` and `b` are NOT 0,
        because each evolution cycle reaps the most weak individuals.

        """
        for layer, (W, b) in enumerate(zip(self.W_, self.b_)):
            mutates = random_state.rand(*W.shape) < probability
            W[mutates] += (2 * random_state.rand() - 1) * factor

            mutates = random_state.rand(*b.shape) < probability
            b[mutates] += (2 * random_state.rand() - 1) * factor

        return self

    @classmethod
    def random(cls):
        """Randomly Build Individual."""
        nn_params = World.current_execution_params['nn_params']
        return TrainingCandidate(NN(**nn_params).random_build())


class World(at.base.Environment):
    state_class_ = TrainingCandidate
    executions = execute_following_training.copy()
    current_execution_params = None

    def update(self):
        if not World.executions:
            raise RuntimeError('Cannot update. No executions left to work on.')

        # Define workspace (data set of interest).
        params = World.executions.pop(0)
        World.current_execution_params = params

        X, y = params['dataset']()
        X = preprocessing.scale(X)

        if params['settings']['plotting']:
            plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
            plt.show()

        # Separate train (80%) and test data (20%).
        X, X_test, y, y_test = model_selection.train_test_split(
            X, y, train_size=.8, random_state=random_state)

        # Set class attributes. We need this for the genetic algorithm.
        World.X, World.X_test, World.y, World.y_test = X, X_test, y, y_test

        trainer = NNTrainer(searches.genetic.GeneticAlgorithm, self,
                            search_params=params['search_params'])

        # Build a regularly trained Neural Network once.
        # We'll use it as base for our benchmarks.
        cnn = NN(**params['nn_params'])
        self.backpropagation_training(cnn)

        # Ask agent to find a trained net for us.
        print('Genetic training has started...')

        t = time()
        training = trainer.act()
        print('Genetic initiation complete (%i cycles, %f s)'
              % (trainer.search.cycle_, time() - t))

        if params['settings']['plotting']:
            # Plotting generations' utilities.
            plt.plot(trainer.search.lowest_utility_,
                     color='blue', linewidth=4, label='Lowest')
            plt.plot(trainer.search.average_utility_,
                     color='orange', linewidth=4, label='Average')
            plt.plot(trainer.search.highest_utility_,
                     color='red', linewidth=4, label='Highest')
            plt.legend()

            plt.xlabel('generation')
            plt.ylabel('utility')

            plt.tight_layout()
            plt.show()

        # Let's cross our fingers!
        # Build a genetically initialized trained Neural Network.
        gnn = (NN(**params['nn_params'])
               .custom_build(training.W_, training.b_))
        self.evaluate(gnn)

    @staticmethod
    def backpropagation_training(nn):
        print('Regular training ongoing...')

        t = time()
        nn.fit(World.X, World.y)
        print('Training complete (%f s)' % (time() - t))

        World.evaluate(nn)

    @staticmethod
    def evaluate(nn):
        if World.current_execution_params['settings']['plotting']:
            plot_decision_boundary(World.X, World.y, lambda x: nn.predict(x))
            plt.tight_layout()
            plt.show()

        print('Accuracy score: %.2f' % nn.score(World.X_test, World.y_test))


class NNTrainer(at.agents.UtilityBasedAgent):
    def predict(self, state):
        """Predicts nothing."""

    def act(self):
        """Find a training candidate for a Neural Network.

        We have only one action, which is to answer a question:
        What is a neural network that best predicts my data set classes?

        """
        return (self.search
                .restart(root=self.last_state)
                .search()
                .solution_candidate_)


def main():
    print('=========================================')
    print('Neural Networks TrainingCandidate Example')
    print('=========================================\n')

    # Build and update world relative to the data defined by the user. At each
    # iteration, the world will train and compare neural nets to a specific
    # data set.
    helpers.live(World(), n_cycles=len(execute_following_training))


if __name__ == '__main__':
    main()
