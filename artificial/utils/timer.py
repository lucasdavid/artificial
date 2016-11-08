"""Artificial Utils Base Module"""

# Author: Lucas David -- <ld492@drexel.edu>
# License: MIT (c) 2016

import time


class Timer:
    """Pretty Time Counter.

    Usage:

    >>> dt = Timer()
    >>> # Do some processing...
    >>> print('time elapsed: ', dt)
    time elapsed: 00:00:05
    """

    def __init__(self):
        self.start = time.time()

    def restart(self):
        self.start = time.time()

    def elapsed(self):
        return time.time() - self.start

    def pretty_elapsed(self):
        m, s = divmod(self.elapsed(), 60)
        h, m = divmod(m, 60)
        time_str = "%02d:%02d:%02d" % (h, m, s)
        return time_str

    def __str__(self):
        return self.pretty_elapsed()
