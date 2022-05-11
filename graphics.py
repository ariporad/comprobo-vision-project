from this import d
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Iterable
from abc import abstractmethod, ABC
import cv2

from helpers import normalize, projections_to_poses, outliers_to_nan


class GraphicsContext(ABC):
    @abstractmethod
    def set_img_feature_matches(self, img0, keypoints0, img1, keypoints1, matches):
        pass

    @abstractmethod
    def set_bundle_adjustment_residuals(self, resid_initial, resid_final):
        pass

    @abstractmethod
    def set_projections_original(self, P_original):
        pass

    @abstractmethod
    def set_projections_adjusted(self, P_adjusted):
        pass

    @abstractmethod
    def set_projections_scaled(self, P_scaled):
        pass

    @abstractmethod
    def set_ground_truth(self, known_poses, P_known):
        pass

    @abstractmethod
    def set_error(self, error):
        pass


class RealGraphicsContext(GraphicsContext):
    feature_matches_img = None

    # (initial, final)
    bundle_adjustment_residuals: Optional[Tuple[np.array, np.array]] = None

    P_original: Optional[np.array] = None
    P_adjusted: Optional[np.array] = None
    P_known: Optional[np.array] = None

    known_poses: Optional[np.array] = None
    P_known: Optional[np.array] = None

    error: Optional[np.array] = None

    def set_img_feature_matches(self, img0, keypoints0, img1, keypoints1, matches):
        # Only store the first one
        if self.feature_matches_img is not None:
            return

        # Only show the first 50 matches
        matches = list(sorted(matches, key=lambda match: match.distance))[:200]

        # We render the image here to avoid storing 4 different values (for simplicity)
        self.feature_matches_img = cv2.drawMatches(
            img0, keypoints0, img1, keypoints1, matches, None, flags=2)

    def set_bundle_adjustment_residuals(self, resid_initial, resid_final):
        self.bundle_adjustment_residuals = resid_initial, resid_final

    def set_projections_original(self, P_original):
        self.P_original = np.array(P_original)

    def set_projections_adjusted(self, P_adjusted):
        self.P_adjusted = np.array(P_adjusted)

    def set_projections_scaled(self, P_scaled):
        self.P_scaled = np.array(P_scaled)

    def set_ground_truth(self, known_poses, P_known):
        self.known_poses = np.array(list(known_poses))
        self.P_known = np.array(list(P_known))

    def set_error(self, error):
        self.error = error

    def plot_img_feature_matches(self):
        fig = plt.figure()
        ax = fig.subplots()
        ax.imshow(self.feature_matches_img)
        ax.set_title('Identified Feature Matches')

    def plot_bundle_adjustment_residuals(self):
        raise NotImplemented

    def plot_error(self, skip=2, percentile=90):
        fig = plt.figure()
        ax = fig.add_subplot()
        err_percent = np.abs(self.error[skip:]) * 100
        ax.plot(outliers_to_nan(
            err_percent[:, 0], percentile), label='X (Right)')
        ax.plot(outliers_to_nan(
            err_percent[:, 1], percentile), label='Y (Down)')
        ax.plot(outliers_to_nan(
            err_percent[:, 2], percentile), label='Z (Forward)')
        ax.set_title(f"Accuracy (≤{percentile} Percentile)")
        ax.set_xlabel('Frame')
        ax.set_ylabel('Accuracy (Calculated Value as % of True Value)')
        ax.legend()

    def plot_trajectories(self, trajectories: Iterable[str], title: str = "Trajectory (Bird's-Eye View)", autoscale: bool = False, arrows: bool = True):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        if autoscale:
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_zlim(0, 1)

        original_poses = projections_to_poses(self.P_original)
        adjusted_poses = projections_to_poses(self.P_adjusted)
        scaled_poses = projections_to_poses(self.P_scaled)

        original_ts = original_poses[:, :3, 3] * [1, 1, -1]
        adjusted_ts = adjusted_poses[:, :3, 3] * [1, 1, -1]
        scaled_ts = scaled_poses[:, :3, 3] * [1, 1, -1]

        # KLUDGE: this really ought to be an enum, but this is good enough
        for name in set(trajectories):
            if name == 'original':
                self._plot_trajectory(ax, original_poses, label='Calculated (Original)',
                                      line_color='b--', autoscale=autoscale, show_arrows=arrows)
            elif name == 'adjusted':
                self._plot_trajectory(ax, adjusted_poses, label='Calculated (Adjusted)',
                                      line_color='k--', autoscale=autoscale, show_arrows=arrows)
            elif name == 'scaled':
                self._plot_trajectory(ax, scaled_poses, label='Calculated (Scaled)',
                                      line_color='m', autoscale=autoscale, show_arrows=arrows)
            elif name == 'known':
                self._plot_trajectory(ax, self.known_poses, label='Known',
                                      line_color='g--', autoscale=autoscale, show_arrows=arrows)

        ax.set_xlabel('X (Right)')
        ax.set_ylabel('Y (Down)')
        ax.set_zlabel('Z (Forward)')

        ax.view_init(180, -90)
        ax.set_title(title)
        ax.legend(loc='lower center')

        # scaled_ts = scaled_projs[:, :3, 3]
        # known_ts = np.array(list(self.known_projections))[:, :3, 3]

        # plt.figure()
        # plt.plot(scaled_ts[:, 0], label='Computed')
        # plt.plot(known_ts[:, 0], label='Known')
        # plt.title('X')
        # plt.legend()
        # plt.show()

        # plt.figure()
        # plt.plot(scaled_ts[:, 1], label='Computed')
        # plt.plot(known_ts[:, 1], label='Known')
        # plt.title('Y')
        # plt.legend()
        # plt.show()

        # plt.figure()
        # plt.plot(scaled_ts[:, 2], label='Computed')
        # plt.plot(known_ts[:, 2], label='Known')
        # plt.title('Z')
        # plt.legend()
        # plt.show()

    def _plot_trajectory(
            self,
            ax,
            poses,
            label: Optional[str] = None,
            scale_factor=1.0,
            line_color='r',
            arrow_color=None,
            arrow_size=3,
            arrow_prop=3,
            show_arrows=True,
            autoscale=True
    ):
        if arrow_color is None:
            arrow_color = line_color
        XYZ = poses[:, :3, 3]
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
                UVW[::arrow_prop, 0], UVW[::arrow_prop, 1], UVW[::arrow_prop, 2], color=arrow_color[0])

        ax.plot(XYZ[:, 0], XYZ[:, 1], XYZ[:, 2], line_color, label=label)

        if autoscale:
            ax.set_xlim(MIN, MAX)
            ax.set_ylim(MIN, MAX)
            ax.set_zlim(MIN, MAX)


class NullGraphicsContext(GraphicsContext):
    """ GraphicsContext implementation that does nothing. """

    def set_img_feature_matches(self, img0, keypoints0, img1, keypoints1, matches):
        pass

    def set_bundle_adjustment_residuals(self, resid_initial, resid_final):
        pass

    def set_projections_original(self, P_original):
        pass

    def set_projections_adjusted(self, P_adjusted):
        pass

    def set_projections_scaled(self, P_scaled):
        pass

    def set_ground_truth(self, known_poses, P_known):
        pass

    def set_error(self, error):
        pass
