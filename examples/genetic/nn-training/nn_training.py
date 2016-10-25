"""Neural Networks Training with Genetic Algorithms Example.

This example consists on:

* Loading every data set in `benchmarks` and plot its first dimensions.
* Train a NN with gradient descent and compute it's score.
* Color the decision regions and boundaries of the NN.
* Train a NN with every search parameters in the current execution and store
  info about the best of them.
* Plot fitness over time graphs.
* Color the decision regions and boundaries of the best estimator.

As you can see, there's a lot to do. Hold tight: this is going to take a while.

Author: Lucas David -- <ld492@drexel.edu>
License: MIT (c) 2016

"""
import logging
from time import time

import artificial as art
import matplotlib.pyplot as plt
import numpy as np
from sklearn import decomposition, datasets, model_selection
from sklearn.neural_network import MLPClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('artificial')

random_state = np.random.RandomState(0)

plotting_count = 1


def plot_decision_boundary(X, y, model, file_name=''):
    """Plot the decision boundaries of a classifier.

    This function was originally implemented by Britz [1].

    References
    ----------
    [1] D. Britz. (2015). IMPLEMENTING A NEURAL NETWORK FROM SCRATCH IN
    PYTHON – AN INTRODUCTION [Online]. Available:
    http://www.wildml.com/2015/09/implementing-a-neural-network-from-scratch/

    """
    global plotting_count

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    data = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(data)
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)

    plt.savefig('report/%s%i.png' % (file_name, plotting_count))
    plt.close()

    plotting_count += 1


benchmarks = [
    # Training on Digits.
    {
        'name': 'digits',
        'dataset': lambda: datasets.load_digits(),
        'nn_params': {
            'hidden_layer_sizes': (100,),
            'random_state': np.random.RandomState(76), 'max_iter': 1},
        'searches': [
            {'population_size': 1000, 'debug': True,
             'max_evolution_cycles': 10000, 'max_evolution_duration': 10 * 60,
             'mutation_factor': .1, 'mutation_probability': .1,
             'random_state': np.random.RandomState(82)}
        ],
        'settings': {'pc_decomposing': True, 'whiten': False,
                     'plotting': False}},
]


class TrainingCandidate(art.base.GeneticState):
    @property
    def coefs_(self):
        return self.data.coefs_

    @property
    def intercepts_(self):
        return self.data.intercepts_

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
        return self.data.loss_

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

        for layer, (W_a, b_a, W_b, b_b) in enumerate(
                zip(self.coefs_, self.intercepts_,
                    other.coefs_,
                    other.intercepts_)):
            shape = W_a.shape
            n_elements = shape[0] * shape[1]

            cut = int(self.data.random_state.rand() * n_elements)

            W_c = np.hstack((W_a.ravel()[:cut], W_b.ravel()[cut:]))
            W_c.resize(shape)
            W_c_.append(W_c)

            cut = int(self.data.random_state.rand() * b_a.shape[1])

            b_c = np.hstack((b_a[:, :cut], b_b[:, cut:]))
            b_c_.append(b_c)

        return TrainingCandidate(NN(**World.params['nn_params'])
                                 .custom_build(W_c_, b_c_))

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
        for layer, (W, b) in enumerate(zip(self.coefs_, self.intercepts_)):
            mutates = self.data.random_state.rand(*W.shape) < probability
            W[mutates] += (2 * self.data.random_state.rand() - 1) * factor

            mutates = self.data.random_state.rand(*b.shape) < probability
            b[mutates] += (2 * self.data.random_state.rand() - 1) * factor

        return self

    @classmethod
    def random(cls):
        """Randomly Build Individual."""
        return TrainingCandidate(MLPClassifier(**World.params['nn_params'])
                                 .fit(World.X, World.y))


class World(art.base.Environment):
    state_class_ = TrainingCandidate
    executions = benchmarks.copy()
    params = None

    def update(self):
        if not World.executions:
            raise RuntimeError('Cannot update. No executions left to work on.')

        # Define workspace (data set of interest).
        params = World.executions.pop(0)
        World.params = params

        t_time = time()

        print('Data set %s %f' % (params['name'], t_time))

        ds = params['dataset']()

        X, y = ds.data, ds.target

        if params['settings'].get('pc_decomposing', False):
            X = decomposition.PCA(
                whiten=params['settings'].get('whiten', False),
                random_state=0).fit_transform(X)

        if params['settings']['plotting']:
            plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
            plt.tight_layout()
            plt.savefig('report/ds-%s.png' % params['name'])
            plt.close()

        # Separate train (80%) and test data (20%).
        X, X_test, y, y_test = model_selection.train_test_split(
            X, y, train_size=.8, random_state=random_state)

        # Set class attributes. We need this for the genetic algorithm.
        World.X, World.X_test, World.y, World.y_test = X, X_test, y, y_test

        # Build a regularly trained Neural Network once.
        # We'll use it as base for our benchmarks.
        mpl = MLPClassifier(**params['nn_params'])

        print('Regular training ongoing...')
        t = time()
        mpl.fit(X, y)
        print('Training complete (elapsed: %f s)' % (time() - t))
        self.evaluate(mpl, label='NN')

        best_i, best_model, best_score = -1, None, -np.inf

        for i, search_params in enumerate(params['searches']):
            trainer = NNTrainer(art.searches.genetic.GeneticAlgorithm,
                                self,
                                search_params=search_params)

            # Ask agent to find a trained net for us.
            print('Genetic training has started. Parameters: \n%s'
                  % search_params)
            t = time()
            training = trainer.act()
            print('Evolution complete (%i cycles, %f s elapsed, '
                  'candidate utility: %f)'
                  % (trainer.search.cycle_, time() - t,
                     trainer.utility(training)))

            if (params['settings']['plotting'] and
                    search_params.get('debug', False)):
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
                plt.savefig('report/ut-%s-%i.png' % (params['name'], i))
                plt.close()

            # Let's cross our fingers! Build a Neural Network with the
            # parameters selected by the evolutionary process.
            gmpl = MLPClassifier(**params['nn_params'])
            gmpl.coefs_ = training.coefs_
            gmpl.intercepts_ = training.intercepts_

            score = self.evaluate(gmpl, label='GA')
            if score > best_score:
                best_i, best_model, best_score = i, gmpl, score

            gmpl.fit(X, y)
            self.evaluate(gmpl, label='trained-gnn')

        print('%s\'s report:\n'
              '\tBest estimator id: %i\n'
              'Score: %.2f\n'
              'Total time elapsed: %f s\n'
              '---\n'
              % (params['name'], best_i, best_score, (time() - t_time)))

    @staticmethod
    def evaluate(nn, label='estimator'):
        if World.params['settings']['plotting'] and World.X.shape[1] == 2:
            file_name = '%s-%s-' % (World.params['name'], label)

            plot_decision_boundary(World.X, World.y, nn, file_name=file_name)

        score = nn.score(World.X_test, World.y_test)
        print('%s accuracy score: %.2f\n' % (label, score))

        return score


class NNTrainer(art.agents.ResponderAgent, art.agents.UtilityBasedAgent):
    """Trainer Agent"""


def main():
    print(__doc__)

    # Build and update world relative to the data defined by the user. At each
    # iteration, the world will train and compare neural nets to a specific
    # data set.
    World().live(n_cycles=len(benchmarks))


if __name__ == '__main__':
    main()
