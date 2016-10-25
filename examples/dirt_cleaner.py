"""Dirt Cleaner Example.

This example shows how A* can be used by an agent to find the best
course of action to clean a room with multiple sectors.

Author: Lucas David -- <ld492@drexel.edu>
License: MIT (c) 2016

"""

import random
import time

import artificial as art


class DirtyState(art.base.State):
    """Dirty State.

    Describes a state of the DirtyEnvironment. Data has always this format:
        `(is_dirty, is_dirty, ..., agent_position_in_grid)`, where the i-th
            `is_dirty` is a flag (0 or 1) indicating whether or not sector `i`
            `is dirty.

    """

    @property
    def is_goal(self):
        """All sectors are 0 (cleaned)"""
        return sum(self.data[:-1]) == 0

    def h(self):
        """Heuristic function indicating the cost of cleaning a sector.

        For each dirty sector, the agent will need to move to it and clean it.
        This requires one operation for the sector in which the agent is
        currently in (to-clean), and at least two operations for any other
        sector (move, move, ..., clean).

        Dirty sectors have the last position in their data set to 1.

        """
        return 2 * sum(self.data[:-1]) - 1


class DirtyEnvironment(art.base.Environment):
    shape = (4, 4)

    def __init__(self, initial_state='random'):
        if initial_state == 'random':
            initial_state = DirtyState([round(min(random.random() + .3, 1))
                                        for _ in range(self.shape[0] *
                                                       self.shape[1])] + [0])

        super().__init__(initial_state=initial_state)

        self.real_cost = 0
        self.actions = (
            {
                'label': 'move-left',
                'cost': 1,
            },
            {
                'label': 'move-right',
                'cost': 1,
            },
            {
                'label': 'move-up',
                'cost': 1,
            },
            {
                'label': 'move-down',
                'cost': 1,
            },
            {
                'label': 'sweep',
                'cost': 1,
            },
        )

    def update(self):
        rows, columns = self.shape

        for agent in self.agents:
            agent.perceive()
            action_id = agent.act()

            if action_id is None:
                continue

            action = self.actions[action_id]

            pos = self.current_state.data[-1]
            grid_pos = (pos // columns, pos % columns)
            s = self.current_state.mitosis(parenting=False, action=action_id)

            if grid_pos[1] > 0 and action_id == 0:
                s.data[-1] -= 1

            elif grid_pos[1] < columns - 1 and action_id == 1:
                s.data[-1] += 1

            elif grid_pos[0] > 0 and action_id == 2:
                s.data[-1] -= columns

            elif grid_pos[0] < rows - 1 and action_id == 3:
                s.data[-1] += columns

            elif s.data[pos] != 0 and action_id == 4:
                s.data[pos] = 0

            if s != self.current_state:
                self.current_state = s
                self.real_cost += action['cost']


class DirtCleanerUtilityAgent(art.agents.UtilityBasedAgent):
    def predict(self, state):
        """Predict how actions affect a given `state`. That is, which new
        states could be reached from `state`.

        :param state: the root state from which new
                      states should be predicted.
        :return: a list of reachable states from the current one.

        Notes
        -----

        Now, you might read this method's code and think: "that looks" awfully
        like the code in `DirtyEnvironment.update`. So, what's the difference?
        `DirtyEnvironment` is a wrapper for the real world. The code in its
        `update` method is a simulation on what would really happen to the
        world given those actions were taken.

        The code in this method ESTIMATES what would succeed from the taken
        of a specific action.

        """
        rows, columns = self.environment.shape
        children = []

        pos = state.data[-1]
        grid_pos = (pos // columns, pos % columns)

        for action_id in self.actions:
            action = self.environment.actions[action_id]

            s = state.mitosis(action=action_id,
                              g=state.g + action['cost'])

            if action_id == 0:
                if grid_pos[1] == 0:
                    # Left corner. cannot move left.
                    continue

                s.data[-1] -= 1

            elif action_id == 1:
                if grid_pos[1] == columns - 1:
                    # Right corner. cannot move right.
                    continue

                s.data[-1] += 1

            elif action_id == 2:
                if grid_pos[0] == 0:
                    # First row. Cannot move up.
                    continue

                s.data[-1] -= columns

            elif action_id == 3:
                if grid_pos[0] == rows - 1:
                    # Last row. Cannot move down.
                    continue

                s.data[-1] += columns

            elif action_id == 4:
                if s.data[pos] == 0:
                    # Does not sweep clean sectors.
                    continue

                s.data[pos] = 0

            children.append(s)

        return children


def main():
    print(__doc__)

    i, max_iterations = 0, 100

    env = DirtyEnvironment(initial_state='random')

    env.agents += [
        DirtCleanerUtilityAgent(environment=env,
                                search=art.searches.fringe.AStar,
                                actions=(0, 1, 2, 3, 4))]

    print('Initial state: {%s}\n' % str(env.current_state))

    start = time.time()

    try:
        while i < max_iterations and not env.finished():
            env.update()

            print('#%i: {%s}' % (i, str(env.current_state)))
            i += 1

        print('\nSolution found! :-)'
              if env.current_state.is_goal
              else 'Solution not found. :-(')
    except KeyboardInterrupt:
        pass
    finally:
        print('Time elapsed: %.2f s' % (time.time() - start))


if __name__ == '__main__':
    main()
