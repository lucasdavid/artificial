"""Learning Agent"""

# Author: Lucas David -- <ld492@drexel.edu>
# License: MIT (c) 2016

import abc
from . import base


class LearningAgent(base.Agent, metaclass=abc.ABCMeta):
    """Agent Base.

    Arguments
    ---------
    environment : Environment
        The environment upon which the agent will act.

    actions : list-like (optional)
        Which actions an agent has. This is used as a reminder for
        `predict` implementations and it's optional,

    model : machine-learning-estimator-like (optional)
        A model used by the agent to take actions.

    verbose : bool (default=False)
        The mode in which the agent operates.
        If True, errors or warnings are always sent to the output buffer.

    """

    def __init__(self, environment, actions=None, model=None, verbose=False):
        super(LearningAgent, self).__init__(
            environment=environment, actions=actions, verbose=verbose)

        self.model = model
