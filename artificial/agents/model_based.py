"""Predictable Agent"""

# Author: Lucas David -- <ld492@drexel.edu>
# License: MIT (c) 2016

import abc
import warnings

import six

from . import predicting


@six.add_metaclass(abc.ABCMeta)
class ModelBasedAgent(predicting.PredictingAgent):
    """ModelBasedAgent Base.

    Basic model-based agent for partially observable environments.
    This agent will perceive the environment when possible and
    infer it when it is not.

    This class must be abstract, as its `perceive` and `act` don't
    really understand or affect the environment in any way (such behavior
    must be implemented by the class with which `ModelBasedAgent` is being
    mixed).


    Notes
    -----

    The method `act` overrides the superclass implementation in order
    to retrieve the last action being performed. This value must be
    stored as it might be used when inferring the environment state.

    Naturally, `ModelBasedAgent` must come before any classes that implement
    `act` and `perceive` methods in the inheritance list.


    Examples
    --------

    ```
    class ModelBasedReflexAgent(ModelBasedAgent, SimpleReflexAgent):
        ...
    ```

    """

    last_action = None

    def perceive(self):
        super(ModelBasedAgent, self).perceive()

        if not self.last_state:
            # Current state is unknown. We must infer it.
            for state in self.predict(self.last_known_state):
                if state.action is None:
                    warnings.warn('Unable to infer which action results in '
                                  '{%s}. A ModelBasedAgent needs states to '
                                  'keep track of their entailing actions and '
                                  'you most likely forgot to generate states '
                                  'setting their action property in your '
                                  'agent.predict() implementation.'
                                  % (str(state)))

                if state.action == self.last_action:
                    self.last_state = state
                    break

        return self

    def act(self):
        action = super(ModelBasedAgent, self).act()
        self.last_action = action

        return action
