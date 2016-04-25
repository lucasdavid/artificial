"""Artificial Helper Classes"""

# Author: Lucas David -- <ld492@drexel.edu>
# License: MIT (c) 2016


def live(env, n_cycles=100, verbose=False):
    """Live Environment.

    Bring an Environment to life, and run it through `n_cycles` cycles.

    Parameters
    ----------
    env: Environment-like object
        The environment that will live.

    n_cycles: int, default=100
        The number of cycles in which `env.update` will be called.

    verbose: bool, default=True
        Constant info regarding the current state of the environment is
        is displayed to the user, if True.

    """
    env.build()

    if verbose:
        print('Initial state: %s' % str(env.current_state))

    try:
        cycle = 0

        while cycle < n_cycles:
            env.update()
            cycle += 1

    except KeyboardInterrupt:
        if verbose:
            print('canceled by user.')
    finally:
        if verbose:
            print('Final state: %s' % str(env.current_state))
