"""Connoisseur Utils.


Author: Lucas David -- <ld492@drexel.edu>
Licence: MIT License 2016 (c)

"""
import argparse
import gc
from jsmin import jsmin
import logging
from datetime import datetime

from .timer import Timer

logger = logging.getLogger('artificial')


class Constants:
    """Container for constants of an experiment, as well as an automatic
    loader for these.

    Parameters:
        data   -- the dict containing info regarding an execution.

    """

    def __init__(self, data):
        self._data = data

    def __getattr__(self, item):
        return self[item]

    def __getitem__(self, item):
        if item not in self._data:
            raise AttributeError('%s not an attribute of constants: %s'
                                 % (item, list(self._data.keys())))
        return self._data[item]

    def __str__(self):
        return str(self._data)

    def __contains__(self, item):
        return item in self._data


class Experiment(object):
    """Base Class for Experiments.

    Notes:
        the `run` method should be overridden for the experiment to be
        actually performed.

    Usage:

    >>> class ToyExperiment(Experiment):
    ...     def run(self):
    ...         print('Hello World!')
    ...
    >>> with ToyExperiment() as t:
    ...     t.run()
    Hello World
    >>> print(t.started_at)
    2016-10-11 14:40:22.454985
    >>> print(t.ended_at)
    2016-10-11 14:40:22.455061

    """

    def __init__(self, consts=None):
        self.consts = consts
        self.started_at = self.ended_at = None

    def setup(self):
        """Setup experiment to run.

        This event is always called right before the execution of the
        `Experiment.run` method.

        """

    def teardown(self):
        """Teardown experiment.

        This event is called after `Experiment.run` method has executed.

        """

    def run(self):
        raise NotImplementedError

    def __enter__(self):
        self.started_at = datetime.now()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.ended_at = datetime.now()


class ExperimentSet(object):
    """Container for experiments.

    Useful when executing multiple experiments or executing from multiple
    different environments.

    Usage:

    >>> class ToyExperiment(Experiment):
    ...     def run(self):
    ...         print('Hello World!')
    ...
    >>> experiments = ExperimentSet(ToyExperiment)
    >>> experiments.load_from_json('toy-experiments.json')
    >>> # Run all experiments!
    >>> experiments.run()
    >>> # Find out more about when these experiments started and ended.
    >>> for e in experiments:
    ...     print('experiment %s started at %s and ended at %s'
    ...           % (e.consts.code_name, e.started_at, e.ended_at))
    ...
    experiment julia started at 2016-10-11 14:52:12.216573
    and ended at 2016-10-11 14:52:14.688444

    """

    def __init__(self, experiment_cls, data=None):
        self.experiment_cls = experiment_cls
        self._data = None
        self._experiment_constants = None
        self.current_experiment_ = -1

        if data: self.load_from_object(data)

    def load_from_json(self, filename='./constants.json',
                       raise_on_not_found=False):
        data = {}

        try:
            with open(filename) as f:
                data = jsmin(f)
        except IOError:
            if raise_on_not_found:
                raise

        return self.load_from_object(data)

    def load_from_object(self, data):
        self._data = data = data.copy()

        if isinstance(data, dict):
            base_params = (data['base_parameters']
                           if 'base_parameters' in data
                           else data)

            experiments_params_lst = (data['executions']
                                      if 'executions' in data
                                      else [{'code_name': 'julia'}])
        else:
            base_params = []
            experiments_params_lst = data

        self._experiment_constants = []

        for experiment_params in experiments_params_lst:
            params = base_params.copy()
            params.update(experiment_params)

            self._experiment_constants.append(Constants(params))

        return self

    def __iter__(self):
        self.current_experiment_ = -1
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        return len(self._experiment_constants)

    def next(self):
        self.current_experiment_ += 1

        if self.current_experiment_ >= len(self._experiment_constants):
            self.current_experiment_ = -1
            raise StopIteration

        consts = self._experiment_constants[self.current_experiment_]
        return self.experiment_cls(consts)

    def run(self):
        t = Timer()

        try:
            logger.info('%i experiments in set', len(self))

            for i, e in enumerate(self):
                try:
                    with e:
                        e.setup()

                        logger.info('Experiment #%i: %s (%s)',
                                    i, e.consts.code_name, e.started_at)
                        logger.info('constants: %s', e.consts)

                        e.run()
                        e.teardown()
                    logger.info('experiment completed (%s)\n'
                                '-------------------------' % e.ended_at)

                    del e
                    gc.collect()
                except KeyboardInterrupt:
                    logger.warning('experiment interrupted by user')
                except Exception as e:
                    logger.error('an error was raised (%s): %s',
                                 e.__class__.__name__, e)
        except KeyboardInterrupt:
            logger.warning('experiment set interrupted by user')
        finally:
            logger.info('time elapsed for all experiments: %s\n', t)


arg_parser = argparse.ArgumentParser(
    description='Experiments on Art Connoisseurship')

arg_parser.add_argument('--constants', type=str, default='./constants.json',
                        help='JSON file containing definitions for the '
                             'experiment.')
