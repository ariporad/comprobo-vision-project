import numpy as np
import functools
from time import perf_counter
from collections import defaultdict
from contextlib import contextmanager
from typing import Optional

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


def plot_trajectory(ax, poses, label: Optional[str] = None, scale_factor=1.0, line_color='r', arrow_color='b', arrow_size=5, arrow_prop=5, show_arrows=True, autoscale=True):
	XYZ = np.array([P @ np.array([[0, 0, 0, 1]]).transpose()
				   for P in poses]).squeeze(axis=2)
	UVW = np.array([normalize(P @ np.array([[0, 0, 1, 1]]).transpose())
				   * arrow_size for P in poses]).squeeze(axis=2)

	XYZ *= float(scale_factor)
	UVW *= float(scale_factor)

	if autoscale:
		MIN = np.min([0, np.min(XYZ), *ax.get_xlim(),
					 *ax.get_ylim(), *ax.get_zlim()])
		MAX = np.max([np.max(XYZ), *ax.get_xlim(), *
					 ax.get_ylim(), *ax.get_zlim()]) * 1.10

	if show_arrows:
		ax.quiver(
		    XYZ[::arrow_prop, 0], XYZ[::arrow_prop, 1], XYZ[::arrow_prop, 2],
		    UVW[::arrow_prop, 0], UVW[::arrow_prop, 1], UVW[::arrow_prop, 2], color=arrow_color)
	ax.plot(XYZ[:, 0], XYZ[:, 1], XYZ[:, 2], line_color, label=label)

	if autoscale:
		ax.set_xlim(MIN, MAX)
		ax.set_ylim(MIN, MAX)
		ax.set_zlim(MIN, MAX)