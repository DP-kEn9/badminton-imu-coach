"""Six-axis fusion sees only inference inputs. Oracle is a separate benchmark operation."""

import numpy as np
from scipy.spatial.transform import Rotation

from .body import to6
from .schemas import NODES
from .simulation import G


def synchronize(stream, max_gap_s=0.05):
    anchors = np.asarray(stream["sync_anchors"])
    if anchors.shape[0] < 2 or np.ptp(anchors[:, 0]) <= 0:
        raise ValueError("insufficient sync anchors")
    a, b = np.polyfit(anchors[:, 0], anchors[:, 1], 1)
    timestamps = a * stream["timestamp_device_s"] + b
    target = stream["timestamp_session_s"]
    valid = stream["valid_mask"].copy()
    good = np.flatnonzero(valid)
    if len(good) < 3 or np.max(np.diff(timestamps[good])) > max_gap_s:
        raise ValueError("long gap: split or reject")
    out = {}
    for field in ("accel_specific_force_sensor_mps2", "gyro_sensor_radps"):
        out[field] = np.stack(
            [np.interp(target, timestamps[good], stream[field][good, j]) for j in range(3)], 1
        )
    out["sync_metadata"] = {
        "method": "declared_anchor_affine",
        "slope": a,
        "offset": b,
        "interpolated_frames": int((~valid).sum()),
    }
    return out


def fuse(session):
    session.validate()
    from vqf import offlineVQF

    out = {}
    for node, s in session.streams.items():
        clean = synchronize(s)
        a = np.ascontiguousarray(clean["accel_specific_force_sensor_mps2"], dtype=np.float64)
        g = np.ascontiguousarray(clean["gyro_sensor_radps"], dtype=np.float64)
        t = s["timestamp_session_s"]
        cal = (t >= 0.05) & (t < session.metadata["calibration_end_s"] - 0.05) & s["valid_mask"]
        if cal.sum() < 20 or np.linalg.norm(g[cal], axis=1).max() > 0.1:
            raise ValueError("insufficient/nonstatic calibration")
        bias = g[cal].mean(0)
        result = offlineVQF(g - bias, a, None, 1 / s["sample_rate_hz"])
        q = result["quat6D"]  # official VQF wxyz; active sensor->reference rotation
        r = Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()
        # Known neutral pose and nominal mounting fix the arbitrary initial common heading.
        initial = Rotation.from_matrix(r[cal]).mean().as_matrix()
        r = np.asarray(s["nominal_R_BS"]) @ initial.T @ r
        q = Rotation.from_matrix(r).as_quat()[:, [3, 0, 1, 2]]
        for k in range(1, len(q)):
            if q[k - 1] @ q[k] < 0:
                q[k] *= -1
        clean.update(
            {
                "orientation_sensor_to_reference_wxyz": q,
                "rotation": r,
                "linear_accel_reference_mps2": np.einsum("tij,tj->ti", r, a) + G,
                "fusion_backend": "vqf_offline_six_axis",
                "calibration_metadata": {
                    "source": "independent_static_signals_and_nominal_neutral_pose",
                    "gyro_bias_radps": bias.tolist(),
                    "heading": "assumed_common_initial_zero; absolute heading unavailable",
                    "yaw_mount_error": "unobservable_from_static_gravity",
                },
            }
        )
        out[node] = clean
    return out


def features(session, processed):
    if set(session.streams) != set(NODES):
        raise ValueError("all four nodes required; direct analysis remains available")
    cols = []
    for i, node in enumerate(NODES):
        s, p = session.streams[node], processed[node]
        n = len(s["valid_mask"])
        cols.append(
            np.column_stack(
                [
                    p["accel_specific_force_sensor_mps2"],
                    p["gyro_sensor_radps"],
                    to6(p["rotation"]),
                    np.tile(np.eye(4)[i], (n, 1)),
                    s["valid_mask"],
                    s["saturation_mask"],
                ]
            )
        )
    return np.concatenate(cols, 1).astype("float32")


def oracle_features(session, truth):
    # Benchmark-only entry, never invoked by deployment predict().
    processed = {}
    for node, s in session.streams.items():
        processed[node] = synchronize(s)
        processed[node]["rotation"] = truth.ideal_sensor_orientations[node]
    return features(session, processed)
