from unittest import TestCase

try:
    from unittest.mock import MagicMock
except ImportError:
    from mock import MagicMock

from artificial.utils import PriorityQueue


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

        # Test inserting repetitions.
        [queue.add(entry, p) for entry, p in self.sample_sequence]
        self.assertEqual(len(queue), len(self.sample_sequence))

    def test_pop(self):
        queue = PriorityQueue()
        [queue.add(e, p) for e, p in self.sample_sequence]
        self.assertEqual(len(queue), len(self.sample_expected_pop_sequence))

        for expected in self.sample_expected_pop_sequence:
            actual = queue.pop()
            self.assertEqual(actual, expected)

        self.assertEqual(len(queue), 0)

        with self.assertRaises(KeyError):
            queue.pop()

        [queue.add(e, p) for e, p in self.sample_sequence]
        self.assertEqual(len(queue), len(self.sample_expected_pop_sequence))

        queue.remove('first')
        actual = queue.pop()
        self.assertEqual(actual, 'forth')

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
