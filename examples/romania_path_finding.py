"""Romania Path Finding Example.

This example shows how A* can be used by an agent to find the best path
between two cities (defined by `source_city_id` and `target_city_id`
class attributes in `Romania` class).

Author: Lucas David -- <ld492@drexel.edu>
License: MIT (c) 2016

"""

import logging

import artificial as art

logger = logging.getLogger('artificial')
logger.setLevel(logging.DEBUG)


class Graph:
    def __init__(self, nodes, edges=None, directed=False):
        self.nodes = nodes
        self.edges = edges or {}
        self.n_nodes = len(nodes)
        self.directed = directed


class CityState(art.base.State):
    @property
    def is_goal(self):
        return self.data == Romania.target_city_id

    def h(self):
        return 0

    def __str__(self):
        return '%s, g: %d, h: %d' % (
            Romania.g.nodes[self.data], self.g, self.h())


class Romania(art.base.Environment):
    g = Graph(
        nodes=['Arad', 'Oradea', 'Zerind', 'Timisoara', 'Lugoj', 'Mehadia',
               'Dorbeta', 'Sibiu', 'Fagaras', 'Rimnicu Vilcea', 'Craiova',
               'Pitesti', 'Bucharest', 'Giurgiu', 'Urziceni', 'Hirsova',
               'Eforie', 'Vaslui', 'Iasi', 'Neamt'],
        edges={
            0: {2: 75, 3: 118, 7: 140},
            1: {2: 71, 7: 151},
            2: {0: 75, 1: 71},
            3: {0: 118, 4: 111},
            4: {3: 111, 5: 70},
            5: {4: 70, 6: 75},
            6: {5: 75, 10: 120},
            7: {0: 140, 1: 151, 8: 99, 9: 80},
            8: {7: 99, 12: 211},
            9: {7: 80, 10: 146, 11: 97},
            10: {6: 120, 9: 146, 11: 138},
            11: {9: 97, 10: 138, 12: 101},
            12: {8: 211, 11: 101, 13: 90, 14: 85},
            13: {12: 90},
            14: {12: 85, 15: 98, 17: 142},
            15: {14: 98, 16: 86},
            16: {15: 86},
            17: {14: 142, 18: 92},
            18: {17: 92, 19: 87},
            19: {18: 87}
        })

    source_city_id = 0
    target_city_id = 19
    real_cost = 0

    def build(self):
        self.real_cost = 0
        return self

    def update(self):
        for agent in self.agents:
            agent.perceive()
            next_city = agent.act()

            current = self.current_state.data

            if (next_city is None or current == next_city or
                        next_city not in Romania.g.edges[current]):
                # Invalid transition or non-existent road.
                continue

            self.real_cost += Romania.g.edges[current][next_city]
            self.current_state = CityState(next_city, g=self.real_cost)


class RoutePlanner(art.agents.UtilityBasedAgent):
    def predict(self, state):
        current = state.data
        neighbors = [city for city in self.actions
                     if city in Romania.g.edges[current]]

        return [CityState(neighbor, parent=state, action=neighbor,
                          g=state.g + Romania.g.edges[current][neighbor])
                for neighbor in neighbors]


def main():
    print(__doc__)

    env = Romania(initial_state=CityState(0))

    env.agents += [
        RoutePlanner(
            environment=env,
            # A* Search.
            search=art.searches.fringe.AStar,
            # Signal that the agent can travel to ANY city
            # (see `RoutePlanner.predict` method)
            actions=list(range(20)))
    ]

    i, max_iterations = 0, 14

    print('Initial state: {%s}\n' % str(env.current_state))

    env.build()

    while i < max_iterations and not env.finished():
        env.update()

        print('#%i: {%s}' % (i, str(env.current_state)))
        i += 1

    print('\nSolution found! :-)'
          if env.current_state.is_goal
          else 'Solution not found. :-(')


if __name__ == '__main__':
    main()
