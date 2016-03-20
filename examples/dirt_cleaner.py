import time
from artificial import base, searches, agents


class DirtyState(base.State):
    @property
    def is_goal(self):
        return sum(self.data[:-1]) == 0

    def h(self):
        # We need, at least, 2 operations to clean each dirty sector.
        return 2 * sum(self.data[:-1]) - 1


class DirtyEnvironment(base.Environment):
    shape = (4, 4)

    def __init__(self, initial_state):
        if initial_state == 'random':
            initial_state = DirtyState([1 for _ in range(
                self.shape[0] * self.shape[1])] + [0])

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
            s = self.current_state.mitosis(parenting=False)

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

    def finished(self):
        return self.current_state.is_goal


class DirtCleanerUtilityAgent(agents.UtilityBasedAgent):
    def predict(self, state):
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
    iteration = 0
    max_iterations = 100

    print('==========================')
    print('Dirt cleaner agent example')
    print('==========================\n')

    env = DirtyEnvironment(initial_state='random')

    env.agents += [
        DirtCleanerUtilityAgent(environment=env,
                                search=searches.AStar,
                                actions=(0, 1, 2, 3, 4),
                                verbose=True)]

    print('Initial state: {%s}\n' % str(env.current_state))

    start = time.time()

    try:
        while iteration < max_iterations and not env.finished():
            iteration += 1
            print('Iteration %i' % iteration)
            env.update()

            print('Current state: {%s}\n' % str(env.current_state))

        print('Solution found! (cost: %.1f) :-)' % env.real_cost
              if env.current_state.is_goal
              else 'Solution not found. :-(')

    except KeyboardInterrupt:
        pass

    finally:
        print('Time elapsed: %.2f s' % (time.time() - start))


if __name__ == '__main__':
    main()
