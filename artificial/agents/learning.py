"""Learning Agent"""

# Author: Lucas David -- <ld492@drexel.edu>
# License: MIT (c) 2016

import abc

import six

from . import base


@six.add_metaclass(abc.ABCMeta)
class LearningAgent(base.AgentBase):
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

    """

    def __init__(self, environment, actions=None, model=None,
                 random_state=None):
        super(LearningAgent, self).__init__(environment=environment,
                                            actions=actions,
                                            random_state=random_state)
        self.model = model

    def act(self):
        y = self.model.predict(self.last_state.data)
        return self.actions[y]
