# %%
## Setup ##
# %matplotlib notebook
# %load_ext autoreload
%autoreload 2

from functools import cached_property
import cv2
import pykitti
import numpy as np
from matplotlib import pyplot as plt
from typing import Iterable, List
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from helpers import timed, plot_trajectory
from point_cloud import PointCloud

KITTI_DIR = 'kitti'

### Feature Detection ###

class VOdom:
	sift = cv2.SIFT_create()
	bf_matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

	K: np.array
	imgs: Iterable[np.array]
	known_poses: List[np.array]

	point_cloud = PointCloud()

	def __init__(self, imgs: Iterable[np.array], known_poses: Iterable[np.array], K: np.array):
		self.imgs = imgs
		self.known_poses = list(known_poses)
		self.K = K

	@timed
	def detect_matches_and_E(self, img1, img2, draw=True):
		# Find Matches
		# From: https://stackoverflow.com/a/33670318
		with timed('SIFT'):
			keypoints1, descriptors1 = self.sift.detectAndCompute(img1, None)
			keypoints2, descriptors2 = self.sift.detectAndCompute(img2, None)
		with timed('Matching'):
			matches = self.bf_matcher.match(descriptors1, descriptors2)

		if draw:
			match_img = cv2.drawMatches(
				img1, keypoints1, img2, keypoints2, matches, None, flags=2)
			plt.figure(figsize=(9, 3))
			plt.imshow(match_img)
			plt.show()

		# Find Points
		# From book
		pts1 = []
		pts2 = []
		for match in matches:
			pts1.append(keypoints1[match.queryIdx].pt)
			pts2.append(keypoints2[match.trainIdx].pt)

		points1 = np.array(pts1)
		points2 = np.array(pts2)

		with timed('findFundamentalMat'):
			F, status_mask = cv2.findFundamentalMat(
				points1, points2, cv2.FM_RANSAC, 1, 0.99, 100000)

		E = self.K.T @ F @ self.K

		if draw:
			print(
				f"Keeping {np.sum(status_mask)}/{status_mask.size} points that match the fundamental matrix")

		status_mask = status_mask[:, 0] == 1
		points1 = points1[status_mask]
		points2 = points2[status_mask]

		return points1, points2, matches, E

	# From book
	@timed
	def P_from_E(self, E):
		w, u, vt = cv2.SVDecomp(E)

		W = np.array([
			[0, -1, 0],
			[1, 0, 0],
			[0, 0, 1]
		])

		R = u @ W @ vt
		t = u[:, 2]

		assert np.abs(np.linalg.det(R)) - \
			1.0 <= 1e-07, "det(R) != ±1.0, this isn't a rotation matrix!"

		P = np.hstack((R, t[:, np.newaxis]))
		return P

	# From book
	def triangulate(
		self,
			# NOTE: u and u1 need to be normalized (multiplied by K, or P0 and P1 do)
			u0: np.array,  # point in image 1: (x, y, 1)
			P0: np.array,  # camera 1 matrix
			u1: np.array,  # point in image 2: (x, y, 1)
			P1: np.array,  # camera 2 matrix
	):
		# XXX: I don't quite understand this
		A = np.array([
			[u0[0]*P0[2, 0]-P0[0, 0], u0[0]*P0[2, 1]-P0[0, 1], u0[0]*P0[2, 2]-P0[0, 2]],
			[u0[1]*P0[2, 0]-P0[1, 0], u0[1]*P0[2, 1] -
			 P0[1, 1], u0[1]*P0[2, 2]-P0[1, 2]],
			[u1[0]*P1[2, 0]-P1[0, 0], u1[0]*P1[2, 1] -
			 P1[0, 1], u1[0]*P1[2, 2]-P1[0, 2]],
			[u1[1]*P1[2, 0]-P1[1, 0], u1[1]*P1[2, 1] -
			 P1[1, 1], u1[1]*P1[2, 2]-P1[1, 2]]
		])

		B = np.array([
			-(u0[0]*P0[2, 3]-P0[0, 3]),
			-(u0[1]*P0[2, 3]-P0[1, 3]),
			-(u1[0]*P1[2, 3]-P1[0, 3]),
			-(u1[1]*P1[2, 3]-P1[1, 3])
		])

		_, X = cv2.solve(A, B, flags=cv2.DECOMP_SVD)
		return np.array([*X[:, 0], 1.0])

	@cached_property
	def K_inv(self):
		return np.linalg.inv(self.K)

	# From book
	@timed
	def triangulate_points(
		self,
			frame_id: int,
			points0: np.array,
			points1: np.array,
			P0: np.array,
			P1: np.array,
	):
		for point0, point1 in zip(points0, points1):
			# Convert to normalized, homogeneous coordinates
			u0 = self.K_inv @ np.array([*point0, 1.0])
			u1 = self.K_inv @ np.array([*point1, 1.0])

			# Triangulate
			X = self.triangulate(u0, P0, u1, P1)

			self.point_cloud.set2d(frame_id, X, point1)

	@timed
	def P_from_PnP(self, points3d, points2d):
		# Not really from book, because the book's implementation of this is incomprehensible
		success, rvec, t, inliers = cv2.solvePnPRansac(
			np.array(points3d), np.array(points2d), self.K, None)
		assert success, "PnP failed!"

		R, _ = cv2.Rodrigues(rvec)

		P = np.hstack((R, t))
		return P, R, t


	# I made this part up, although it's an amalgamation of code from above which came from other places
	@timed
	def determine_projections(self):
		""" Returns Ps """
		### Setup ###
		img0 = next(self.imgs)
		img1 = next(self.imgs)

		points0, points1, matches, E = self.detect_matches_and_E(img0, img1)
		# P0 is assumed to be fixed to start
		P0 = np.array([
			[1, 0, 0, 0],
			[0, 1, 0, 0],
			[0, 0, 1, 0]
		])
		P1 = self.P_from_E(E)
		Ps = [P0, P1]
		self.err = self.triangulate_points(1, points0, points1, P0, P1)

		for frame_id, img in enumerate(self.imgs, start=2):
			with timed('Frame'):
				img0, img1 = img1, img
				points0, points1, matches, _ = self.detect_matches_and_E(
					img0, img1, draw=False)

				points3d_valid = []
				points2d_valid = [] # All in frame1

				for point0, point1 in zip(points0, points1):
					point3d = self.point_cloud.lookup2d(frame_id - 1, point0)
					if point3d is not None:
						points3d_valid.append(point3d)
						points2d_valid.append(point1)

				P0 = P1
				P1, R, t = self.P_from_PnP(points3d_valid, points2d_valid)
				Ps.append(P1)

				self.triangulate_points(frame_id, points0, points1, P0, P1) # updates point cloud

		return Ps

	def projections_to_poses(self, Ps):
		cur_pose = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
		poses = []

		for P in Ps:
			# Normal projection matrices are 3x4 to project to 2D homogenous space, but we don't want that
			P_noproj = np.vstack((P, [0, 0, 0, 1]))
			cur_pose = P_noproj @ cur_pose
			poses.append(cur_pose)

		return poses


	## Bundle Adjustment
	# Based on this tutorial: https://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html, but mostly re-written (except as otherwise marked).

	
	# Relevant data, in their implementation:
	# - camera_params with shape (n_cameras, 9) contains initial estimates of parameters for all cameras. First 3 components in each row form a rotation vector (https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula), next 3 components form a translation vector, then a focal distance and two distortion parameters.
	# - points_3d with shape (n_points, 3) contains initial estimates of point coordinates in the world frame.
	# - camera_ind with shape (n_observations,) contains indices of cameras (from 0 to n_cameras - 1) involved in each observation.
	# - point_ind with shape (n_observations,) contatins indices of points (from 0 to n_points - 1) involved in each observation.
	# - points_2d with shape (n_observations, 2) contains measured 2-D coordinates of points projected on images in each observations.

	@timed
	def bundle_adjustment(self, Ps, draw=False):
		frame_Ps = np.array(list(Ps))
		points3d = np.array(list(self.point_cloud.points3d))
		frame_idx = np.zeros((self.point_cloud.num_observations,), dtype=int)
		point3d_idx = np.zeros((self.point_cloud.num_observations,), dtype=int)
		points2d = np.zeros((self.point_cloud.num_observations, 2))

		n_frames = frame_Ps.shape[0]
		n_observations = points3d.shape[0]

		observation_id = 0
		for point3d_id, point3d in enumerate(points3d):
			for frame_id, point2d in self.point_cloud.lookup3d(point3d):
				frame_idx[observation_id] = frame_id
				point3d_idx[observation_id] = point3d_id
				points2d[observation_id, :] = point2d
				observation_id += 1

		state0 = np.hstack((frame_Ps.ravel(), points3d.ravel()))

		FRAME_P_SIZE = 12
		POINT3D_SIZE = 3

		def _residual(state, K, num_Ps, num_points3d, frame_idx, point3d_idx, points2d):
			frame_state_len = (num_Ps * FRAME_P_SIZE)
			frame_Ps = state[:frame_state_len].reshape((num_Ps, 3, 4))
			points3d = state[frame_state_len:].reshape((num_points3d, POINT3D_SIZE))

			P_by_frame = frame_Ps[frame_idx, :, :]
			point3d_by_frame = points3d[point3d_idx]
			point2d_by_frame = points2d
			# Add a 4th column to make homogenous multiplication work
			point3d_by_frame = np.hstack(
				(point3d_by_frame, [[1]] * point3d_by_frame.shape[0]))

			# Credit to this SO answer for making the math work so nicely: https://stackoverflow.com/a/66971088
			reprojected = np.einsum('kij,kj->ki', (K @ P_by_frame), point3d_by_frame)
			point2d_reproj = reprojected[:, :2] / reprojected[:, 2, None]

			return (point2d_reproj - point2d_by_frame).ravel()

		if draw:
			residual0 = _residual(
				state0, self.K, frame_Ps.shape[0], points3d.shape[0], frame_idx, point3d_idx, points2d)
			plt.figure()
			plt.plot(residual0)
			plt.title('Residuals before bundle adjustment')
			plt.show()

		### Generate Sparsity Matrix ###
		# This logic is completely taken from the SciPy tutorial
		resid_len = frame_idx.size * 2
		state_len = (n_frames * FRAME_P_SIZE) + (n_observations * POINT3D_SIZE)
		A = lil_matrix((resid_len, state_len), dtype=int)
		i = np.arange(frame_idx.size)
		for offset in range(FRAME_P_SIZE):
			A[2 * i, (frame_idx * FRAME_P_SIZE) + offset] = 1
			A[2 * i + 1, (frame_idx * FRAME_P_SIZE) + offset] = 1
		for offset in range(POINT3D_SIZE):
			A[2 * i, (n_frames * FRAME_P_SIZE) +
			  (point3d_idx * POINT3D_SIZE) + offset] = 1
			A[2 * i + 1, (n_frames * FRAME_P_SIZE) +
			  (point3d_idx * POINT3D_SIZE) + offset] = 1

		res = least_squares(_residual, state0, verbose=2 if draw else 0, x_scale='jac',
							jac_sparsity=A, method='trf', args=(self.K, n_frames, n_observations, frame_idx, point3d_idx, points2d))

		if draw:
			plt.figure()
			plt.plot(res.fun)
			plt.title('Residuals after bundle adjustment')
			plt.show()

		Ps_flat = res.x[:(FRAME_P_SIZE * n_frames)]
		points3d_flat = res.x[(FRAME_P_SIZE * n_frames):]

		Ps = Ps_flat.reshape((n_frames, 3, 4))
		points3d = points3d_flat.reshape((self.point_cloud.num_points3d, 3))

		return Ps


	# poses = bundle_adjustment(kitti.calib.K_cam0, Ps, point_cloud, True)


	# ## Scale Factor (Attempt)
	#
	# The idea: for some scale factor $C$, the following must be true:
	#
	# $$PC = P_{truth}$$
	#
	# We know $P$ and $P_{truth}$, so solve for $C$ (an approximate solution, not an exact one).
	#
	# But: what shape should $C$ have? Not sure, trying 4x4, we'll see if that works.

	@timed
	def scale_factor(self, poses):
		P_calculated = np.array(poses).reshape((len(poses), -1))
		P_truth = np.array(self.known_poses).reshape((len(poses), -1))

		assert P_calculated.shape == P_truth.shape

		C, *_ = np.linalg.lstsq(P_calculated, P_truth, rcond=None)

		P_corrected = (P_calculated @ C).reshape((-1, 4, 4))
		corrected_poses = list(P_corrected)
		return corrected_poses


	# corrected_poses = scale_factor(poses, kitti)

	# ## Error Calculation
	#
	# Attempts to quantify the accuracy of the algorithm. Currently, it does so per-axis using the following algorithm:
	#
	# $$\text{Error} = \text{mean}\left(\frac{P_{\text{ground truth, t}} - P_{\text{calculated, t}}}{P_{\text{ground truth, t}}}\right)$$
	#
	# Where $P$ is the robot's position (note that, currently, this doesn't consider orientation), and $t$ is the current timestep.

	def calculate_error(self, actual_poses, calculated_poses):
		loc_odom = np.array(calculated_poses) @ np.transpose([0, 0, 0, 1])
		loc_true = np.array(actual_poses) @ np.transpose([0, 0, 0, 1])
		diff = ((loc_true - loc_odom) / loc_true)[:, :3]  # drop heterogenous 1
		diff = diff[5:, :]  # drop first point (which is nonsense)
		x_err, y_err, z_err = np.mean(np.abs(diff), axis=0) * 100

		return x_err, y_err, z_err


	# x_err, y_err, z_err = calculate_error(
	# 	corrected_poses, kitti.poses[:len(corrected_poses)])

	# %% [markdown]
	# **For Drive 00:**
	# 1.1% on the Z-axis is incredible! I'm really happy with that. Ditto for the Y-axis!
	#
	# I'm skeptical of their data for the X-axis... looking at the video, there's some real shake. Very confused about this.
	#
	# **For Drive 01 (every other frame):**
	# 8% all around is still great! (And seems much more plausible). Really happy with this
	#
	# **For Drive 01 (every frame):**
	# 7-9% on Y and Z is still great! Wtf is going on with X.

	def run(self, draw: bool = False):
		P_original = self.determine_projections()
		print("Projections done!")
		P_adjusted = self.bundle_adjustment(P_original, draw=draw)
		print("Adjustment done")
		raw_poses = self.projections_to_poses(P_adjusted)
		scaled_poses = self.scale_factor(raw_poses)
		x_err, y_err, z_err = self.calculate_error(self.known_poses, scaled_poses)
		print(f"Mean error: X: {x_err:2.1f}%, Y: {y_err:2.1f}%, Z: {z_err:2.1f}%")

		if draw:
			fig = plt.figure()
			ax = fig.add_subplot(projection='3d')
			ax.set_xlim(0, 1)
			ax.set_ylim(0, 1)
			ax.set_zlim(0, 1)

			# plot_trajectory(ax, raw_poses, label='Unscaled', line_color='b', show_arrows=False)
			plot_trajectory(ax, scaled_poses, label='Scaled', line_color='r', show_arrows=False)
			plot_trajectory(ax, self.known_poses, label='Truth', line_color='g--', show_arrows=False)

			ax.set_xlabel('X')
			ax.set_ylabel('Y')
			ax.set_zlabel('Z')

			ax.view_init(0, 90)
			ax.set_title("Trajectory (Bird's-Eye View, Z is forward)")
			plt.show()
			
	
	@classmethod
	def kitti(cls, sequence, start: int, stop: int, step: int=1):
		if not isinstance(sequence, str):
			sequence = f"{sequence:02d}"
		kitti = pykitti.odometry(KITTI_DIR, sequence, frames=range(start, stop, step))
		return VOdom((np.array(img) for img in kitti.cam0), kitti.poses, kitti.calib.K_cam0)

if __name__ == '__main__':
	vodom = VOdom.kitti(1, 0, 50, 1)
	vodom.run(draw=True)
	


# %%
