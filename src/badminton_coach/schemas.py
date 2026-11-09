from dataclasses import dataclass, field

import numpy as np

NODES = ("pelvis", "left_shank", "right_shank", "racket_forearm")
VERSION = "0.2"


@dataclass
class InferenceSession:
    streams: dict
    metadata: dict
    schema_version: str = VERSION
    racket: None = None
    racket_present: bool = False

    def validate(self):
        if self.metadata.get("units") != {"accel": "m/s^2", "gyro": "rad/s", "time": "s"}:
            raise ValueError("wrong units")
        if self.metadata.get("dominant_hand") not in ("left", "right"):
            raise ValueError("dominant_hand required")
        if self.racket is not None or self.racket_present:
            raise ValueError("racket unsupported in phase I")
        for key, s in self.streams.items():
            if key not in NODES:
                raise ValueError("unknown node")
            n = len(s["timestamp_device_s"])
            if n < 3 or np.any(np.diff(s["timestamp_device_s"]) <= 0):
                raise ValueError("timestamps must increase")
            for f in ("accel_specific_force_sensor_mps2", "gyro_sensor_radps"):
                if s[f].shape != (n, 3) or not np.isfinite(s[f]).all():
                    raise ValueError("invalid signal shape/values")
        return self


@dataclass
class EvaluationTruth:
    local_rotations: np.ndarray
    offsets: np.ndarray
    ideal_sensor_orientations: dict
    metadata: dict = field(default_factory=dict)


@dataclass
class PoseArtifact:
    local_rotations: np.ndarray
    joints_pelvis_m: np.ndarray
    valid_mask: np.ndarray
    metadata: dict


@dataclass
class EvidenceArtifact:
    evidence_id: str
    metric_id: str
    value: float
    unit: str
    time_range_s: list
    source_artifact_id: str
    pipeline_version: str = "0.2.0"
