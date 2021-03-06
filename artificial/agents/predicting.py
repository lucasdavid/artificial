"""Predictable Agent"""

# Author: Lucas David -- <ld492@drexel.edu>
# License: MIT (c) 2016


import abc

import six

from . import base


@six.add_metaclass(abc.ABCMeta)
class PredictingAgent(base.AgentBase):
    """Predictable Agent Base.

    Base for agents that can predict states based on their current perception
    of the environment.
    """

    def predict(self, state):
        """Predicts states based on the current perceived one and the
        agent's possible actions.

        Returns
        -------
        State-like list.

        """
        raise NotImplementedError
