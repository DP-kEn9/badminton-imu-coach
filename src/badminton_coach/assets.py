"""Explicit licensed analytic_npz_v1 + analytic_skeleton_v1 adapter.

No SMPL/AMASS support is claimed. One-to-one mapping must cover all 21 joints.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .body import NAMES, PARENTS, SKELETON_ID
from .schemas import EvaluationTruth


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_licensed(manifest_path, asset_id, model_id):
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    selected = []
    for ident, category in ((asset_id, "motion"), (model_id, "body_model")):
        asset = next((a for a in manifest["assets"] if a["asset_id"] == ident), None)
        if asset is None or asset["category"] != category:
            raise ValueError("missing asset/category")
        if asset["authorization_status"] != "approved" or "local_research" not in asset["allowed_processing"]:
            raise PermissionError("authorization gate rejected asset")
        path = (manifest_path.parent / asset["path"]).resolve()
        if file_hash(path) != asset["sha256"]:
            raise ValueError("asset hash mismatch")
        selected.append((asset, path))
    motion, model = selected
    if motion[0]["format"] != "analytic_npz_v1" or model[0]["format"] != SKELETON_ID:
        raise ValueError("unsupported data/model combination")
    sk = json.loads(model[1].read_text())
    if sk["id"] != SKELETON_ID or sk["parents"] != PARENTS or sk["shape_dimension"] != 0:
        raise ValueError("unsupported topology or shape dimension")
    with np.load(motion[1], allow_pickle=False) as data:
        names = data["joint_names"].tolist()
        if set(names) != set(NAMES) or len(names) != 21:
            raise ValueError("joint mapping must be bijective")
        mapping = [names.index(n) for n in NAMES]
        q = data["local_rotations_wxyz"][:, mapping]
        t = data["timestamp_s"].copy()
        if q.shape != (len(t), 21, 4) or np.any(np.diff(t) <= 0) or not np.isfinite(q).all():
            raise ValueError("motion dimensions/timestamps")
        if not np.allclose(np.linalg.norm(q, axis=-1), 1, atol=1e-5):
            raise ValueError("non-unit quaternion")
        r = Rotation.from_quat(q.reshape(-1, 4)[:, [1, 2, 3, 0]]).as_matrix().reshape(-1, 21, 3, 3)
    offsets = np.asarray(sk["offsets_m"])
    if offsets.shape != (21, 3) or not np.isfinite(offsets).all():
        raise ValueError("invalid model offsets")
    return {
        "timestamp_s": t,
        "skeleton_id": SKELETON_ID,
        "hand_policy": "no articulated fingers",
        "truth": EvaluationTruth(r, offsets, {}, {"integration": "fixture_only_until_user_assets"}),
        "manifest_hash": file_hash(manifest_path),
    }
