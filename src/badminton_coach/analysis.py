from dataclasses import asdict

import numpy as np

from .schemas import EvidenceArtifact


def direct_evidence(session):
    session.validate()
    evidence = []
    quality = {}
    for node, s in session.streams.items():
        valid = s["valid_mask"] & ~s["saturation_mask"]
        t = s["timestamp_session_s"]
        valid &= t >= session.metadata["calibration_end_s"]
        quality[node] = {
            "valid_fraction": float(s["valid_mask"].mean()),
            "saturation_fraction": float(s["saturation_mask"].mean()),
        }
        if not valid.any():
            continue
        values = [
            ("angular_speed_peak", np.linalg.norm(s["gyro_sensor_radps"][valid], axis=1).max(), "rad/s"),
            (
                "specific_force_rms",
                np.sqrt(np.mean(np.sum(s["accel_specific_force_sensor_mps2"][valid] ** 2, axis=1))),
                "m/s^2",
            ),
        ]
        for metric, value, unit in values:
            evidence.append(
                asdict(
                    EvidenceArtifact(
                        f"{node}:{metric}",
                        metric,
                        float(value),
                        unit,
                        [float(t[valid][0]), float(t[valid][-1])],
                        f"raw:{session.metadata['group_id']}:{node}",
                    )
                )
            )
    return {
        "schema_version": "0.2",
        "evidence": evidence,
        "signal_quality": quality,
        "pose_uncertainty": "uncalibrated",
        "evidence_support": "descriptive_inertial_only",
        "reference_applicability": "no_reference",
        "motion_type": "unknown",
        "diagnoses": [],
    }


def pose_angular_peak(rotations, joint, rate, mask):
    from scipy.spatial.transform import Rotation

    delta = rotations[:-1, joint].swapaxes(-1, -2) @ rotations[1:, joint]
    speed = Rotation.from_matrix(delta).magnitude() * rate
    usable = mask[:-1] & mask[1:]
    return float(speed[usable].max()) if usable.any() else None
