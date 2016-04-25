"""Basic Priority Queue"""

# Author: Lucas David -- <ld492@drexel.edu>
# License: MIT (c) 2016

import heapq

import itertools


class PriorityQueue:
    REMOVED = '<removed-e>'

    def __init__(self):
        self.priority_queue = []
        self.map = {}
        self.counter = itertools.count()

    def add(self, entry, priority=0):
        if entry in self.map:
            self.remove(entry)

        count = next(self.counter)
        node = [priority, count, entry]
        self.map[entry] = node
        heapq.heappush(self.priority_queue, node)

    def get(self, entry):
        return self.map[entry]

    def remove(self, entry):
        node = self.map.pop(entry)
        node[-1] = self.REMOVED

    def pop(self):
        while self.priority_queue:
            priority, count, entry = heapq.heappop(self.priority_queue)
            if entry is not self.REMOVED:
                del self.map[entry]
                return entry

        raise KeyError('pop from an empty priority queue')

    def __contains__(self, item):
        return item in self.map and self.map[item] != self.REMOVED

    def __getitem__(self, item):
        return self.get(item)

    def __len__(self):
        return len(self.map)

    def __bool__(self):
        return len(self) != 0
