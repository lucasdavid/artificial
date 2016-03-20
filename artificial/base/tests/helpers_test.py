from unittest import TestCase

from artificial.base.helpers import Graph, PriorityQueue


class GraphTest(TestCase):
    def test_sanity(self):
        g = Graph(
            nodes=['Bagda', 'Sabaa Al Bour', 'Sabiat'],
            edges={
                0: {1: 10, 2: 50},
                1: {0: 10, 2: 40},
                2: {0: 50, 1: 40}
            }
        )

        self.assertIsNotNone(g)
        self.assertFalse(g.directed)
        self.assertEqual(g.n_nodes, 3)

        self.assertEqual(g.edges[0][1], 10)
        self.assertEqual(g.edges[1][2], 40)
        self.assertEqual(g.edges[2][0], 50)

        self.assertListEqual(g.nodes, ['Bagda', 'Sabaa Al Bour', 'Sabiat'])


class PriorityQueueTest(TestCase):
    def setUp(self):
        self.sample_sequence = (('first', 1), ('second', 10),
                                ('third', 5), ('forth', 1))
        self.sample_expected_pop_sequence = ('first', 'forth',
                                             'third', 'second')

    def test_sanity(self):
        p = PriorityQueue()
        self.assertIsNotNone(p)

    def test_add(self):
        queue = PriorityQueue()
        [queue.add(entry, p) for entry, p in self.sample_sequence]

        self.assertEqual(len(queue), len(self.sample_sequence))

<<<<<<< HEAD
=======
        # Test inserting repetitions.
        [queue.add(entry, p) for entry, p in self.sample_sequence]
        self.assertEqual(len(queue), len(self.sample_sequence))

>>>>>>> tests
    def test_pop(self):
        queue = PriorityQueue()
        [queue.add(e, p) for e, p in self.sample_sequence]
        self.assertEqual(len(queue), len(self.sample_expected_pop_sequence))

        for expected in self.sample_expected_pop_sequence:
            actual = queue.pop()
            self.assertEqual(actual, expected)

        self.assertEqual(len(queue), 0)

<<<<<<< HEAD
=======
        with self.assertRaises(KeyError):
            queue.pop()

>>>>>>> tests
    def test_remove(self):
        queue = PriorityQueue()
        entries = ('first', 'forth', 'third', 'second')

        [queue.add(entry) for entry in entries]
        self.assertEqual(len(queue), 4)

        queue.remove('first')
        self.assertEqual(len(queue), 3)
        self.assertNotIn('first', queue)
        [self.assertIn(e, queue) for e in ('second', 'third', 'forth')]

        queue.remove('third')
        self.assertEqual(len(queue), 2)
        self.assertNotIn('third', queue)
        [self.assertIn(e, queue) for e in ('second', 'forth')]

    def test___contains__(self):
        queue = PriorityQueue()
        expected = (('first', 1), ('second', 10), ('third', 5))
        [queue.add(entry, priority) for entry, priority in expected]
        self.assertTrue('first' in queue)

    def test___getitem__(self):
        queue = PriorityQueue()
        [queue.add(e, p) for e, p in self.sample_sequence]

        node = queue['third']
        self.assertListEqual(node, [5, 2, 'third'])

    def test___len__(self):
        q = PriorityQueue()
        [q.add(e, p) for e, p in self.sample_sequence]
        self.assertEqual(len(q), len(self.sample_sequence))

    def test__bool__(self):
        q = PriorityQueue()
        self.assertFalse(q)

        q.add('one')
        self.assertTrue(q)
