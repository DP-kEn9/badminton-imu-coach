import numpy as np
from scipy.spatial.transform import Rotation

from .body import OFFSETS, fk
from .schemas import EvaluationTruth, InferenceSession

G = np.array([0, 0, -9.80665])


def rigid_imu(r, p, dt):
    acc = np.gradient(np.gradient(p, dt, axis=0, edge_order=2), dt, axis=0, edge_order=2)
    if len(p) >= 5:
        acc[2:-2] = (-p[4:] + 16 * p[3:-1] - 30 * p[2:-2] + 16 * p[1:-3] - p[:-4]) / (12 * dt * dt)
    force = np.einsum("tji,tj->ti", r, acc - G)
    # Log increments centered onto sample timestamps; two edge frames are invalid.
    intervals = Rotation.from_matrix(r[:-1].swapaxes(-1, -2) @ r[1:]).as_rotvec() / dt
    omega = np.vstack([intervals[0], (intervals[:-1] + intervals[1:]) / 2, intervals[-1]])
    return force, omega


def binding(hand="right", layout="distal"):
    return {
        "pelvis": (0, np.array([0.04, 0, 0.02])),
        "left_shank": (14, np.array([0, 0, -(0.36 if layout == "distal" else 0.06)])),
        "right_shank": (18, np.array([0, 0, -(0.36 if layout == "distal" else 0.06)])),
        "racket_forearm": (
            10 if hand == "right" else 6,
            np.array([0, -0.22 if hand == "right" else 0.22, 0]),
        ),
    }


def generate(group=0, rate=100, duration=4, layout="distal", perturb=False, kind=None):
    rng = np.random.default_rng(1000 + group)
    t = np.arange(int(rate * duration)) / rate
    u = np.maximum(t - 1, 0)
    # Smooth onset preserves an independent, exactly static calibration interval.
    env = np.where(u < 0.5, np.sin(np.pi * np.minimum(u, 0.5)) ** 2, 1)
    speed = 0.6 + 0.12 * (group % 5)
    amplitude = 0.22 + 0.035 * (group % 4)
    wave = env * np.sin(2 * np.pi * speed * u)
    kinds = ["static", "translation", "rotation", "single", "multi", "multi"]
    kind = kind or kinds[group % 6]
    hand = "left" if group % 2 else "right"
    rv = np.zeros((len(t), 21, 3))
    if kind in ("single", "multi"):
        rv[:, 10 if hand == "right" else 6, 0] = amplitude * wave
    if kind == "multi":
        for j in (1, 2, 5, 9, 13, 14, 17, 18):
            rv[:, j, (j + group) % 3] = amplitude * wave * np.cos(j)
    if kind == "rotation":
        rv[:, 0, 2] = 0.5 * env * u
    local = Rotation.from_rotvec(rv.reshape(-1, 3)).as_matrix().reshape(len(t), 21, 3, 3)
    scale = 0.92 + 0.04 * (group % 5)
    offsets = OFFSETS * scale
    world, positions = fk(local, offsets)
    root = np.zeros((len(t), 3))
    if kind == "translation":
        root[:, 0] = 0.2 * env * np.sin(2 * np.pi * 0.6 * u)
    positions = positions + root[:, None, :]
    streams, ideal, installs, clocks = {}, {}, {}, {}
    for i, (node, (j, lever)) in enumerate(binding(hand, layout).items()):
        mount = Rotation.from_rotvec(rng.normal(0, 0.04, 3) if perturb else np.zeros(3)).as_matrix()
        actual_lever = lever + (rng.normal(0, 0.008, 3) if perturb else 0)
        r = world[:, j] @ mount
        p = positions[:, j] + np.einsum("tij,j->ti", world[:, j], actual_lever)
        acc, gyr = rigid_imu(r, p, 1 / rate)
        if perturb:
            acc += rng.normal(0, 0.04, acc.shape) + rng.normal(0, 0.03, 3) + 0.01 * np.sin(t[:, None])
            gyr += rng.normal(0, 0.003, gyr.shape) + rng.normal(0, 0.002, 3)
        valid = np.ones(len(t), bool)
        valid[:2] = valid[-2:] = False
        if perturb:
            valid[rate + 30 : rate + 33] = False
        alimit, glimit = (10.5, 1.5) if perturb else (156.9, 34.9)
        saturation = (np.abs(acc) >= alimit).any(1) | (np.abs(gyr) >= glimit).any(1)
        acc, gyr = np.clip(acc, -alimit, alimit), np.clip(gyr, -glimit, glimit)
        offset, drift = (i * 0.003, 40 + i * 10) if perturb else (0, 0)
        device = (1 + drift * 1e-6) * t + offset
        if perturb:
            device += rng.normal(0, 0.00002, len(t))
        # Anchors are declared independent clock measurements, not body motion peaks.
        anchor_error = 0.0001 if perturb else 0
        anchors = [[float(device[k] + anchor_error), float(t[k])] for k in (0, len(t) - 1)]
        streams[node] = {
            "timestamp_device_s": device,
            "timestamp_session_s": t.copy(),
            "accel_specific_force_sensor_mps2": acc,
            "gyro_sensor_radps": gyr,
            "valid_mask": valid,
            "saturation_mask": saturation,
            "sample_rate_hz": rate,
            "sync_anchors": anchors,
            "nominal_R_BS": np.eye(3).tolist(),
            "nominal_r_BS": lever.tolist(),
        }
        ideal[node] = r
        installs[node] = {"R_BS": mount.tolist(), "r_BS": actual_lever.tolist()}
        clocks[node] = {"offset_s": offset, "drift_ppm": drift}
    metadata = {
        "schema_version": "0.2",
        "data_mode": "demo_synthetic",
        "data_nature": "synthetic_analytic",
        "validation": "synthetic_demo_only",
        "group_id": f"original_{group:03d}",
        "dominant_hand": hand,
        "kind": kind,
        "speed": speed,
        "layout": layout,
        "calibration_end_s": 1.0,
        "source_rate_hz": rate,
        "model_rate_hz": rate,
        "units": {"accel": "m/s^2", "gyro": "rad/s", "time": "s"},
        "anthropometry": {"mode": "default", "source": "original engineering defaults"},
        "perturbation": "exploratory_not_hardware_calibrated" if perturb else "clean",
    }
    truth = EvaluationTruth(
        local,
        offsets,
        ideal,
        {
            "actual_installation": installs,
            "clock_truth": clocks,
            "root_trajectory_reference_m": root,
            "scale": scale,
        },
    )
    return InferenceSession(streams, metadata).validate(), truth
