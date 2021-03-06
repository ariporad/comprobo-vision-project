# %%
## Setup Jupyter ##
# %matplotlib notebook
from graphics import GraphicsContext, NullGraphicsContext
from point_cloud import PointCloud
from helpers import invert_P, timed
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares
from typing import Iterable, List
import numpy as np
import pykitti
import cv2
from functools import cached_property
# %%
## Setup ##


KITTI_DIR = 'kitti'

### Feature Detection ###


class VOdom:
    sift = cv2.SIFT_create()
    bf_matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    K: np.array
    imgs: Iterable[np.array]
    known_poses: List[np.array]

    point_cloud = PointCloud()

    gctx: GraphicsContext

    def __init__(self, imgs: Iterable[np.array], known_poses: Iterable[np.array], K: np.array, gctx: GraphicsContext = NullGraphicsContext()):
        self.imgs = imgs
        self.known_poses = list(known_poses)
        self.K = K
        self.gctx = gctx

    @timed
    def detect_matches_and_E(self, img0, img2):
        # Find Matches
        # From: https://stackoverflow.com/a/33670318
        with timed('SIFT'):
            keypoints0, descriptors0 = self.sift.detectAndCompute(img0, None)
            keypoints1, descriptors1 = self.sift.detectAndCompute(img2, None)
        with timed('Matching'):
            matches = self.bf_matcher.match(descriptors0, descriptors1)

        self.gctx.set_img_feature_matches(
            img0, keypoints0, img2, keypoints1, matches)

        # Find Points
        # From book
        pts1 = []
        pts2 = []
        for match in matches:
            pts1.append(keypoints0[match.queryIdx].pt)
            pts2.append(keypoints1[match.trainIdx].pt)

        points1 = np.array(pts1)
        points2 = np.array(pts2)

        with timed('findEssentialMat'):
            E, status_mask = cv2.findEssentialMat(
                points1, points2, self.K, cv2.RANSAC, .99, 1)

        status_mask = status_mask[:, 0] == 1
        points1 = points1[status_mask]
        points2 = points2[status_mask]

        return points1, points2, matches, E

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
            [u0[0]*P0[2, 0]-P0[0, 0], u0[0]*P0[2, 1] -
                P0[0, 1], u0[0]*P0[2, 2]-P0[0, 2]],
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

    @property
    # FIXME: projections is a misnomer
    # FIXME: This produces correct results for translation, but total nonsense for rotation (which we don't care about right now)
    def known_projections(self) -> Iterable[np.array]:
        # First frame has no projection
        prev_pose = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])

        for pose in self.known_poses:
            yield pose - prev_pose
            prev_pose = pose

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
        success, R, t, _ = cv2.recoverPose(E, points0, points1, self.K)
        assert success, "recoverPose failed!"
        P1 = np.hstack((R, t))
        Ps = [P0, P1]
        self.err = self.triangulate_points(1, points0, points1, P0, P1)

        for frame_id, img in enumerate(self.imgs, start=2):
            with timed('Frame'):
                img0, img1 = img1, img
                points0, points1, matches, E = self.detect_matches_and_E(
                    img0, img1)

                points3d_valid = []
                points2d_valid = []  # All in frame1

                for point0, point1 in zip(points0, points1):
                    point3d = self.point_cloud.lookup2d(frame_id - 1, point0)
                    if point3d is not None:
                        points3d_valid.append(point3d)
                        points2d_valid.append(point1)

                P0 = P1
                success, R, t, _ = cv2.recoverPose(E, points0, points1, self.K)
                assert success, "failed to recover pose"
                P1 = np.hstack((R, t))
                Ps.append(P1)

                # updates point cloud
                self.triangulate_points(frame_id, points0, points1, P0, P1)

        return Ps

    # Bundle Adjustment
    # Based on this tutorial: https://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html, but mostly re-written (except as otherwise marked).

    FRAME_P_SIZE = 12
    POINT3D_SIZE = 3

    def _residual(self, state, num_frames, num_points3d, frame_idxs, point3d_idxs, points2d):
        frame_state_len = (num_frames * self.FRAME_P_SIZE)
        frame_Ps = state[:frame_state_len].reshape((num_frames, 3, 4))
        points3d = state[frame_state_len:].reshape(
            (num_points3d, self.POINT3D_SIZE))

        P_by_frame = frame_Ps[frame_idxs, :, :]
        point3d_by_frame = points3d[point3d_idxs]
        point2d_by_frame = points2d
        # Add a 4th column to make homogenous multiplication work
        point3d_by_frame = np.hstack(
            (point3d_by_frame, [[1]] * point3d_by_frame.shape[0]))

        # Credit to this SO answer for making the math work so nicely: https://stackoverflow.com/a/66971088
        reprojected = np.einsum(
            'kij,kj->ki', (self.K @ P_by_frame), point3d_by_frame)
        point2d_reproj = reprojected[:, :2] / reprojected[:, 2, None]

        return (point2d_reproj - point2d_by_frame).ravel()

    @timed
    def bundle_adjustment(self, Ps, quiet=True):
        num_frames = len(Ps)
        num_observations = self.point_cloud.num_observations
        num_points3d = self.point_cloud.num_points3d

        frame_Ps = np.array(list(Ps))
        points3d = np.array(list(self.point_cloud.points3d))
        frame_idxs = np.zeros((num_observations,), dtype=int)
        point3d_idxs = np.zeros((num_observations,), dtype=int)
        points2d = np.zeros((num_observations, 2))

        observation_id = 0
        for point3d_id, point3d in enumerate(self.point_cloud.points3d):
            for frame_id, point2d in self.point_cloud.lookup3d(point3d):
                frame_idxs[observation_id] = frame_id
                point3d_idxs[observation_id] = point3d_id
                points2d[observation_id, :] = point2d
                observation_id += 1

        state0 = np.hstack((frame_Ps.ravel(), points3d.ravel()))
        residual0 = self._residual(
            state0, num_frames, num_points3d, frame_idxs, point3d_idxs, points2d)

        ### Generate Sparsity Matrix ###
        # This logic is completely taken from the SciPy tutorial
        resid_len = num_observations * 2
        state_len = (num_frames * self.FRAME_P_SIZE) + \
            (num_observations * self.POINT3D_SIZE)
        sparsity = lil_matrix((resid_len, state_len), dtype=int)
        # I can't figure out why we need the 2*i nonsense, but we do
        i = np.arange(frame_idxs.size)
        for offset in range(self.FRAME_P_SIZE):
            sparsity[2 * i, (frame_idxs * self.FRAME_P_SIZE) + offset] = 1
            sparsity[2 * i + 1, (frame_idxs * self.FRAME_P_SIZE) + offset] = 1
        for offset in range(self.POINT3D_SIZE):
            sparsity[2 * i, (num_frames * self.FRAME_P_SIZE) +
                     (point3d_idxs * self.POINT3D_SIZE) + offset] = 1
            sparsity[2 * i + 1, (num_frames * self.FRAME_P_SIZE) +
                     (point3d_idxs * self.POINT3D_SIZE) + offset] = 1

        res = least_squares(self._residual, state0, verbose=2 if not quiet else 0, x_scale='jac',
                            jac_sparsity=sparsity, method='trf', args=(num_frames, num_points3d, frame_idxs, point3d_idxs, points2d))

        self.gctx.set_bundle_adjustment_residuals(residual0, res.fun)

        Ps_flat = res.x[:(self.FRAME_P_SIZE * num_frames)]
        points3d_flat = res.x[(self.FRAME_P_SIZE * num_frames):]

        Ps = Ps_flat.reshape((num_frames, 3, 4))
        points3d = points3d_flat.reshape((self.point_cloud.num_points3d, 3))

        return Ps

    # Scale Factor

    @timed
    def calculate_scale_factor(self, known_projs, projs):
        ts_calc = np.array(projs)[:, 0:3, 3]
        ts_true = np.array(known_projs)[:, 0:3, 3]

        assert ts_calc.shape == ts_true.shape

        with np.errstate(divide='ignore'):
            factors = np.linalg.norm(ts_true, axis=1) / \
                np.linalg.norm(ts_calc, axis=1)
        factors[~np.isfinite(factors)] = 1

        factors2 = np.zeros_like(projs)
        factors2.fill(1)
        factors2[:, 0, 3] = factors
        factors2[:, 1, 3] = factors
        factors2[:, 2, 3] = factors

        corrected_poses = list(np.array(projs) * factors2)

        return corrected_poses

    # Error Calculation
    #
    # Attempts to quantify the accuracy of the algorithm. Currently, it does so per-axis using the following algorithm:
    #
    # Error = mean((P_truth - P_calc) / P_truth)$$
    #
    # Where P is the robot's position (note that, currently, this doesn't consider orientation)

    def calculate_error(self, actual_poses, calculated_poses):
        loc_odom = np.abs(np.array(calculated_poses)[:, :3, 3])
        loc_true = np.abs(np.array(actual_poses)[:, :3, 3])
        return ((loc_true - loc_odom) / loc_true)

    def run(self):
        self.gctx.set_ground_truth(self.known_poses, self.known_projections)
        P_original = np.array(self.determine_projections())

        # For some reason, OpenCV has an inverted Z-axis from the ground truth. We need to invert it
        # to make them align, but we can't do that till after bundle adjustment.
        P_original_uninv = P_original
        P_original = invert_P(P_original)
        self.gctx.set_projections_original(P_original)

        # Do bundle adjustment for the sake of graphing, but don't actually use it
        P_adjusted = self.bundle_adjustment(P_original_uninv)
        P_adjusted = np.array([np.vstack((x, [0, 0, 0, 1]))
                              for x in P_adjusted])
        P_adjusted = invert_P(P_adjusted)
        self.gctx.set_projections_adjusted(P_adjusted)

        P_scaled = np.array(self.calculate_scale_factor(
            list(self.known_projections), P_original))
        self.gctx.set_projections_scaled(P_scaled)
        error = self.calculate_error(list(self.known_projections), P_scaled)
        self.gctx.set_error(error)
        error = error[5:, :]  # drop few points, which are usually garbage
        x_err, y_err, z_err = np.median(np.abs(error), axis=0) * 100

        print(
            f"Median error: X: {x_err:2.1f}%, Y: {y_err:2.1f}%, Z: {z_err:2.1f}%")

    @classmethod
    def kitti(cls, sequence, start: int, stop: int, step: int = 1, gctx: GraphicsContext = NullGraphicsContext()):
        if not isinstance(sequence, str):
            sequence = f"{sequence:02d}"
        kitti = pykitti.odometry(
            KITTI_DIR, sequence, frames=range(start, stop, step))
        return VOdom((np.array(img) for img in kitti.cam0), kitti.poses, kitti.calib.K_cam0, gctx)


if __name__ == '__main__':
    vodom = VOdom.kitti(1, 0, 50, 1)
    vodom.run()


# %%
