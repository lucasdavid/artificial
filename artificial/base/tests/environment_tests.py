from unittest import TestCase

from artificial.base.environment import State, Environment


class SpaceTest(TestCase):
    def test_sanity(self):
        s = State(data=None)
        self.assertIsNotNone(s)

    def test_equals(self):
        s1, s2, s3 = (State(data=[1, 2, 3, 4]), State(data=[1, 2, 3, 4]),
                      State(data=[1, 3, 2, 1]))

        self.assertEqual(s1, s1)
        self.assertEqual(s1, s2)
        self.assertNotEqual(s1, s3)

        s1, s2 = State(data={1, 2, 3}), State(data={1, 2, 3})
        self.assertEqual(s1, s2)

        s1, s2 = State(data={1, 2, 3}), State(data=[1, 2, 3])
        self.assertNotEqual(s1, s2)

    def test_clone(self):
        s1 = State(data=[1, 2, 3, 4])
        s2 = s1.clone()

        s2.data[0] = -1
        self.assertNotEqual(s1.data[0], s2.data[0])

        s2.data[0] = 1
        self.assertListEqual(s1.data, s2.data)


class _LocalTestEnvironment(Environment):
    def update(self):
        pass


class EnvironmentTest(TestCase):
    def test_sanity(self):
        env = _LocalTestEnvironment(None)
        self.assertIsNotNone(env)

    def test_add_agents(self):
        env = _LocalTestEnvironment(None)
        expected = ['A', 'B', 'C']
        env.agents += expected

        self.assertListEqual(env.agents, expected)
