import numpy as np
import functools
from time import perf_counter
from collections import defaultdict
from contextlib import contextmanager

_timings = defaultdict(lambda: [])

ENABLE_TIMING: bool = False


@contextmanager
def timed_ctx(name: str = 'Timer', always_print: bool = False):
    if not ENABLE_TIMING:
        yield
        return

    start_time = perf_counter()
    yield
    duration = (perf_counter() - start_time) * 1000
    _timings[name].append(duration)
    if always_print:
        print(f"{name} took {duration:.2f}ms.")


def timed_decorator(f):
    if not ENABLE_TIMING:
        return f

    @functools.wraps(f)
    def timed_function(*args, **kwargs):
        with timed_ctx(f.__name__):
            return f(*args, **kwargs)

    return timed_function


def timed(name_or_fn, *args, **kwargs):
    if callable(name_or_fn):
        return timed_decorator(name_or_fn, *args, **kwargs)
    else:
        return timed_ctx(name_or_fn, *args, **kwargs)


def print_timings(clear: bool = True):
    if not ENABLE_TIMING:
        return

    if len(_timings) == 0:
        print("No timings yet.")

    name_len = max(map(len, _timings.keys()))
    data = [['Name'.center(name_len, ' ')] + [x.rjust(10)
                                              for x in ['Total', 'Mean', 'Stdev', 'Calls']]]

    for name, calls in sorted(_timings.items(), key=lambda data: sum(data[1]), reverse=True):
        data.append([name.ljust(name_len, ' '), np.sum(calls),
                    np.mean(calls), np.std(calls), len(calls)])

    if clear:
        clear_timings()

    print('Timings:')
    print('\n'.join('\t'.join([value if isinstance(
        value, str) else f"{value:10.2f}" for value in row]) for row in data))


def clear_timings():
    _timings.clear()


def normalize(v, *args, **kwargs):
    return v / np.linalg.norm(v, *args, **kwargs)


def projections_to_poses(Ps):
    cur_pose = np.array([[1, 0, 0, 0], [0, 1, 0, 0],
                        [0, 0, 1, 0], [0, 0, 0, 1]])
    poses = []

    for P in Ps:
        # Normal projection matrices are 3x4 to project to 2D homogenous space, but we don't want that
        if P.shape[0] == 3:
            P = np.vstack((P, [0, 0, 0, 1]))
        cur_pose = P @ cur_pose
        poses.append(cur_pose)

    return np.array(poses)


def outliers_to_nan(data, percentile: float = 90):
    data = data.copy()
    data_abs = np.abs(data)
    cutoff = np.nanpercentile(data_abs, percentile)
    mask = data_abs > cutoff
    data[mask] = np.nan
    return data


def invert_P(P):
    P = P.copy()
    P[:, :3, :3] = P[:, :3, :3].transpose((0, 2, 1))
    P[:, 0, 3] *= -1
    P[:, 1, 3] *= -1
    P[:, 2, 3] *= -1
    return P
