"""Original analytic skeleton; no third-party body assets."""

import numpy as np
import torch
from scipy.spatial.transform import Rotation

NAMES = [
    "pelvis",
    "spine",
    "chest",
    "neck",
    "head",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "left_hand",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
    "right_hand",
    "left_hip",
    "left_knee",
    "left_ankle",
    "left_toe",
    "right_hip",
    "right_knee",
    "right_ankle",
    "right_toe",
]
PARENTS = [-1, 0, 1, 2, 3, 2, 5, 6, 7, 2, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19]
OFFSETS = np.array(
    [
        [0, 0, 0],
        [0, 0, 0.18],
        [0, 0, 0.2],
        [0, 0, 0.15],
        [0, 0, 0.15],
        [0, 0.18, 0],
        [0, 0.28, 0],
        [0, 0.26, 0],
        [0, 0.09, 0],
        [0, -0.18, 0],
        [0, -0.28, 0],
        [0, -0.26, 0],
        [0, -0.09, 0],
        [0, 0.1, -0.05],
        [0, 0, -0.42],
        [0, 0, -0.42],
        [0.16, 0, -0.05],
        [0, -0.1, -0.05],
        [0, 0, -0.42],
        [0, 0, -0.42],
        [0.16, 0, -0.05],
    ]
)
SKELETON_ID = "analytic_skeleton_v1"
EVALUATED = list(range(1, 21))
UNESTIMATED = ["left_hand", "right_hand", "left_toe", "right_toe", "fingers"]


def fk(local, offsets=OFFSETS):
    """NumPy or differentiable torch FK; root position is always zero."""
    is_torch = isinstance(local, torch.Tensor)
    offsets = torch.as_tensor(offsets, dtype=local.dtype, device=local.device) if is_torch else offsets
    rs, ps = [], []
    for j, p in enumerate(PARENTS):
        if p < 0:
            r = local[..., j, :, :]
            pos = r[..., 0] * 0
        else:
            r = rs[p] @ local[..., j, :, :]
            pos = ps[p] + (rs[p] @ offsets[j][..., None])[..., 0]
        rs.append(r)
        ps.append(pos)
    stack = torch.stack if is_torch else np.stack
    return stack(rs, -3), stack(ps, -2)


def to6(r):
    return r[..., :, :2].swapaxes(-1, -2).reshape(*r.shape[:-2], 6)


def from6(x):
    """Stable Gram-Schmidt, including zero/collinear inputs."""
    a, b = x[..., :3], x[..., 3:]
    fallback = torch.zeros_like(a)
    fallback[..., 0] = 1
    a = torch.where(torch.linalg.vector_norm(a, dim=-1, keepdim=True) > 1e-6, a, fallback)
    u = torch.nn.functional.normalize(a, dim=-1)
    b = b - (u * b).sum(-1, keepdim=True) * u
    basis = torch.nn.functional.one_hot(u.abs().argmin(-1), 3).to(u.dtype)
    alt = basis - (basis * u).sum(-1, keepdim=True) * u
    b = torch.where(torch.linalg.vector_norm(b, dim=-1, keepdim=True) > 1e-6, b, alt)
    v = torch.nn.functional.normalize(b, dim=-1)
    return torch.stack([u, v, torch.linalg.cross(u, v)], -1)


def geodesic(a, b):
    d = a.swapaxes(-1, -2) @ b
    return np.arccos(np.clip((np.trace(d, axis1=-2, axis2=-1) - 1) / 2, -1, 1))


def rotation_mean(rotations):
    return Rotation.from_matrix(rotations).mean().as_matrix()


def skeleton_metadata():
    return {
        "id": SKELETON_ID,
        "joints": [
            {
                "name": n,
                "parent": p,
                "rest_offset_m": o.tolist(),
                "local_basis": np.eye(3).tolist(),
                "dof": 3 if i else 0,
                "evaluation": i in EVALUATED,
            }
            for i, (n, p, o) in enumerate(zip(NAMES, PARENTS, OFFSETS))
        ],
    }
