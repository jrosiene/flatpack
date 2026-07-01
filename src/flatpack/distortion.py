"""Per-triangle distortion of a flattening (3D patch -> 2D uv).

For each triangle we compute the 2x2 Jacobian J of the linear map from
the triangle's local 3D frame to uv, and its singular values
sigma1 >= sigma2:

- sigma1 * sigma2 is the area ratio (1 = area preserved),
- sigma1 / sigma2 measures anisotropy (1 = angles preserved; LSCM keeps
  this near 1 and pushes all distortion into area),
- sigma - 1 is the strain the fabric would need along each principal
  direction: positive means the flat panel is bigger than the surface
  (fabric must stretch or you ease it in), negative means smaller
  (excess material on the 3D side -> dart or gather territory).

The principal stretch direction in uv is also reported so the fabric
model can compare it against the stretch axis of the material.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from flatpack.meshutil import triangle_frames


@dataclass
class DistortionReport:
    """Per-triangle distortion arrays plus convenience summaries."""

    sigma1: np.ndarray  # larger singular value per triangle
    sigma2: np.ndarray  # smaller singular value per triangle
    stretch_dir_uv: np.ndarray  # (t, 2) unit vector of max stretch, in uv
    centers_uv: np.ndarray  # (t, 2) triangle centroids in uv
    area_3d: np.ndarray  # per-triangle surface area
    flipped: np.ndarray = field(default=None)  # bool per triangle (det J < 0)

    @property
    def area_ratio(self) -> np.ndarray:
        return self.sigma1 * self.sigma2

    @property
    def anisotropy(self) -> np.ndarray:
        return self.sigma1 / self.sigma2

    @property
    def max_angle_error_deg(self) -> np.ndarray:
        """Worst angle distortion per triangle, in degrees.

        A linear map with singular values s1, s2 changes angles by at most
        2 * atan2(s1 - s2, 2 * sqrt(s1 * s2)) (from the theory of
        quasiconformal maps); exact enough for deciding where a panel
        needs relief.
        """
        return np.degrees(
            2.0 * np.arctan2(self.sigma1 - self.sigma2, 2.0 * np.sqrt(self.sigma1 * self.sigma2))
        )

    def summary(self) -> dict:
        weights = self.area_3d / self.area_3d.sum()
        return {
            "triangles": int(len(self.sigma1)),
            "flipped_triangles": int(self.flipped.sum()),
            "area_ratio_mean": float(weights @ self.area_ratio),
            "area_ratio_worst_high": float(self.area_ratio.max()),
            "area_ratio_worst_low": float(self.area_ratio.min()),
            "max_stretch_strain": float((self.sigma1 - 1.0).max()),
            "max_compress_strain": float((1.0 - self.sigma2).max()),
            "angle_error_deg_mean": float(weights @ self.max_angle_error_deg),
            "angle_error_deg_max": float(self.max_angle_error_deg.max()),
        }

    def worst_triangle_uv(self) -> np.ndarray:
        """uv location of the worst-distorted triangle (for 'add a dart here')."""
        badness = np.maximum(self.sigma1 - 1.0, 1.0 - self.sigma2)
        return self.centers_uv[int(np.argmax(badness))]


def distortion_report(
    vertices: np.ndarray, faces: np.ndarray, uv: np.ndarray
) -> DistortionReport:
    """Compare each triangle's uv image against its true 3D shape."""
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=np.int64)
    uv = np.asarray(uv, dtype=float)

    x1, x2, y2, area = triangle_frames(vertices, faces)

    # Local-frame edge matrix S = [[x1, x2], [0, y2]] and uv edge matrix U;
    # the Jacobian is J = U @ S^-1, built here explicitly per triangle.
    du1 = uv[faces[:, 1]] - uv[faces[:, 0]]  # image of (x1, 0)
    du2 = uv[faces[:, 2]] - uv[faces[:, 0]]  # image of (x2, y2)
    inv_s = np.zeros((len(faces), 2, 2))
    inv_s[:, 0, 0] = 1.0 / x1
    inv_s[:, 0, 1] = -x2 / (x1 * y2)
    inv_s[:, 1, 1] = 1.0 / y2
    u_mat = np.stack([du1, du2], axis=2)  # (t, 2, 2), columns are edge images
    jac = u_mat @ inv_s

    u_svd, s_svd, _ = np.linalg.svd(jac)
    sigma1 = s_svd[:, 0]
    sigma2 = s_svd[:, 1]
    stretch_dir = u_svd[:, :, 0]  # left singular vector: max-stretch dir in uv

    det = np.linalg.det(jac)
    centers = uv[faces].mean(axis=1)

    return DistortionReport(
        sigma1=sigma1,
        sigma2=sigma2,
        stretch_dir_uv=stretch_dir,
        centers_uv=centers,
        area_3d=area,
        flipped=det < 0,
    )
