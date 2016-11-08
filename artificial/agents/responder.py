from .utility import UtilityBasedAgent


class ResponderAgent(UtilityBasedAgent):
    """Responder Agent.

    The Responder is a goal-based agent concerned with one (and only) thing:
    responding a question. It achieves that using its search mechanism and
    returning the `solution_candidate_` found.

    """

    def __init__(self, search, environment, search_params=None,
                 random_state=None):
        super(ResponderAgent, self).__init__(
            search=search, environment=environment,
            search_params=search_params, actions=None,
            random_state=random_state)

    def act(self):
        """Answers the question."""
        return (self.search
                .restart(root=self.last_state)
                .search()
                .solution_candidate_)
