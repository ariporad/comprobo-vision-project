import numpy as np
from typing import Tuple, Dict, Iterable
from dataclasses import dataclass, field

from helpers import timed

@dataclass
class PointCloud:
	# Storing all the data twice is inefficient memory-wise, but allows for fast lookups in either direction
	_points3d: Dict[Tuple[float, float, float], Dict[int,
													 Tuple[float, float]]] = field(default_factory=lambda: {})
	_points2d: Dict[int, Dict[Tuple[float, float], Tuple[float,
														 float, float]]] = field(default_factory=lambda: {})
	_num_observations: int = 0

	@timed
	def _canonicalize2d(self, point2d):
		point2d = np.array(point2d).squeeze()

		if point2d.shape == (3,):
			assert point2d[2] == 1.0, "Invalid homogeneous 2d point! 3rd value must be 1.0!"
			point2d = point2d[:2]

		assert point2d.shape == (2,), "Invalid 2d point!"

		return point2d

	@timed
	def _canonicalize3d(self, point3d):
		point3d = np.array(point3d).squeeze()

		if point3d.shape == (4,):
			assert point3d[3] == 1.0, "Invalid homogeneous 3d point! 4th value must be 1.0!"
			point3d = point3d[:3]

		assert point3d.shape == (3,), "Invalid 3d point!"

		return point3d

	@timed
	def set2d(self, frame_id, point3d, point2d) -> bool:
		""" Record an observation. Returns True if this observation is new, False otherwise."""
		key3d = tuple(self._canonicalize3d(point3d))
		value2d = tuple(self._canonicalize2d(point2d))

		if key3d not in self._points3d:
			self._points3d[key3d] = {}

		if frame_id not in self._points2d:
			self._points2d[frame_id] = {}

		if self._points3d[key3d].get(frame_id, None) == value2d:
			return False

		self._points3d[key3d][frame_id] = value2d
		self._points2d[frame_id][value2d] = key3d
		self._num_observations += 1

		return True

	@timed
	def lookup2d(self, frame_id, point2d):
		value2d = tuple(self._canonicalize2d(point2d))

		return self._points2d.get(frame_id, {}).get(value2d, None)

	@timed
	def lookup3d(self, point3d):
		key3d = tuple(self._canonicalize3d(point3d))

		for frame_id, value2d in self._points3d[key3d].items():
			yield frame_id, np.array(value2d)

	@property
	def num_observations(self) -> int:
		return self._num_observations

	@property
	def num_points3d(self) -> int:
		return len(self._points3d)

	@property
	def points3d(self) -> Iterable[np.array]:
		return (np.array(key3d) for key3d in self._points3d.keys())

	@property
	def observations(self) -> Iterable[Tuple[int, np.array, np.array]]:
		""" Return (frame_id, point2d, point3d) tuples of each observation. """
		for key3d, frame_dict in self._points3d.items():
			point3d = np.array(key3d)
			for frame_id, tuple2d in frame_dict:
				yield frame_id, point3d, np.array(tuple2d)