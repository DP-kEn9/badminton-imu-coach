import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from . import __version__
from .agent import generate_report, markdown
from .analysis import direct_evidence, pose_angular_peak
from .assets import file_hash
from .body import NAMES, OFFSETS, SKELETON_ID, fk, geodesic, rotation_mean
from .fusion import features, fuse, oracle_features
from .models import BiGRU, predict_windows, windows
from .schemas import NODES, PoseArtifact
from .simulation import generate
from .visualization import replay


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def load_config(path):
    c = yaml.safe_load(Path(path).read_text(encoding="utf-8-sig"))
    if c["data_mode"] != "demo_synthetic":
        raise ValueError(
            "licensed_motion: load through assets.load_licensed; training integration not validated"
        )
    if c["orientation_mode"] not in ("fused_orientation", "oracle_orientation"):
        raise ValueError("unsupported orientation mode")
    if c["fusion_backend"] != "vqf_offline":
        raise ValueError("unsupported fusion backend")
    return c


def split_groups(count):
    if count < 12:
        raise ValueError("at least 12 independent source groups")
    # Blocks are distinct original generators; every block includes all six motion kinds.
    return {
        "train": list(range(count - 6)),
        "dev": list(range(count - 6, count - 3)),
        "test": list(range(count - 3, count)),
    }


def prepare(c):
    split = split_groups(c["groups"])
    entries = {}
    for subset, ids in split.items():
        entries[subset] = []
        for group in ids:
            s, gt = generate(group, c["sample_rate_hz"], c["duration_s"])
            x = (
                features(s, fuse(s))
                if c["orientation_mode"] == "fused_orientation"
                else oracle_features(s, gt)
            )
            mask = s.streams["pelvis"]["valid_mask"] & (
                s.streams["pelvis"]["timestamp_session_s"] >= c["calibration_s"]
            )
            entries[subset].append((s, gt, x, mask))
    return entries, split


def train(c, path, device="cpu"):
    if device != "cpu":
        raise ValueError("first version validated on CPU only")
    start = time.perf_counter()
    torch.set_num_threads(2)
    torch.manual_seed(c["seed"])
    np.random.seed(c["seed"])
    entries, split = prepare(c)
    fit = np.concatenate([e[2] for e in entries["train"]])
    mean, std = fit.mean(0), fit.std(0).clip(0.05)
    truth = np.concatenate([e[1].local_rotations[e[3]] for e in entries["train"]])
    baseline = np.stack([rotation_mean(truth[:, j]) for j in range(21)])
    model = BiGRU(hidden=c["hidden"], layers=c["layers"])
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    opt = torch.optim.AdamW(model.parameters(), lr=c["lr"])
    data = []
    for _, gt, x, mask in entries["train"]:
        for st in windows(len(x), c["window"], c["stride"]):
            length = min(c["window"], len(x) - st)
            xx = np.pad((x[st : st + length] - mean) / std, ((0, c["window"] - length), (0, 0)), mode="edge")
            yy = np.pad(
                gt.local_rotations[st : st + length],
                ((0, c["window"] - length), (0, 0), (0, 0), (0, 0)),
                mode="edge",
            )
            mm = np.pad(mask[st : st + length], (0, c["window"] - length))
            data.append((xx, yy, mm))
    loss_history, steps = [], 0
    for _ in range(c["epochs"]):
        order = np.random.permutation(len(data))
        for b in range(0, len(order), c["batch"]):
            batch = [data[i] for i in order[b : b + c["batch"]]]
            xx = torch.tensor(np.stack([d[0] for d in batch]), dtype=torch.float32)
            yy = torch.tensor(np.stack([d[1] for d in batch]), dtype=torch.float32)
            mm = torch.tensor(np.stack([d[2] for d in batch]), dtype=torch.bool)
            pred = model(xx)
            # Frobenius/chordal SO(3) loss is dimensionless; positions are in m.
            rot = ((pred - yy) ** 2).mean((-1, -2, -3))
            pos = ((fk(pred)[1] - fk(yy)[1]) ** 2).mean((-1, -2))
            loss = (rot[mm] + pos[mm]).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_history.append(float(loss.detach()))
            steps += 1
    changed = any(not torch.equal(before[n], p) for n, p in model.named_parameters())
    if not changed:
        raise RuntimeError("parameters did not update")
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "state_dict": model.state_dict(),
        "config": c,
        "mean": torch.tensor(mean),
        "std": torch.tensor(std),
        "baseline": torch.tensor(baseline),
        "skeleton": SKELETON_ID,
        "nodes": list(NODES),
        "orientation_mode": c["orientation_mode"],
        "schema_version": "0.2",
        "split": split,
        "normalization_source": "training groups only",
        "pipeline_version": __version__,
    }
    torch.save(checkpoint, path / "checkpoint.pt")
    summary = {
        "steps": steps,
        "parameter_update_verified": changed,
        "seconds": time.perf_counter() - start,
        "device": device,
        "loss_first": loss_history[0],
        "loss_last": loss_history[-1],
        "loss_units": {"rotation": "dimensionless SO(3) chordal", "FK": "m^2"},
        "split": split,
        "config": c,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "memory": "unavailable: OS hardware query denied",
        "acceptance": {"engineering": "smoke_only", "research": "not_validated", "product": "not_validated"},
    }
    write_json(path / "training.json", summary)
    return summary


def load_model(path):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    if ck["skeleton"] != SKELETON_ID or ck["nodes"] != list(NODES):
        raise ValueError("checkpoint contract mismatch")
    model = BiGRU(hidden=ck["config"]["hidden"], layers=ck["config"]["layers"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck


def predict(session, checkpoint):
    model, ck = load_model(checkpoint)
    if ck["orientation_mode"] != "fused_orientation":
        raise ValueError("oracle checkpoints are benchmark-only")
    x = features(session, fuse(session))
    local = predict_windows(
        model, (x - ck["mean"].numpy()) / ck["std"].numpy(), ck["config"]["window"], ck["config"]["stride"]
    )
    mask = np.logical_and.reduce([s["valid_mask"] & ~s["saturation_mask"] for s in session.streams.values()])
    mask &= (
        next(iter(session.streams.values()))["timestamp_session_s"] >= session.metadata["calibration_end_s"]
    )
    return PoseArtifact(
        local,
        fk(local)[1],
        mask,
        {
            "schema_version": "0.2",
            "root_translation_reference_m": None,
            "anthropometry": session.metadata["anthropometry"],
            "mode": "fused_orientation",
            "unestimated": ["hand/toe terminal rotations", "fingers"],
            "checkpoint_sha256": file_hash(checkpoint),
        },
    )


class BenchmarkRunner:
    @staticmethod
    def metrics(local, gt, mask, known=False):
        positions = fk(local, gt.offsets if known else OFFSETS)[1]
        truth = gt.local_rotations.copy()
        truth[:, 0] = np.eye(3)
        target = fk(truth, gt.offsets)[1]
        errors = np.linalg.norm(positions - target, axis=-1)[mask]
        angle = geodesic(local, truth)[mask]
        return {
            "mpjpe_m": float(errors[:, 1:].mean()),
            "local_rotation_error_rad": float(angle[:, 1:].mean()),
            "per_joint_m": dict(zip(NAMES, errors.mean(0).tolist())),
            "left_m": float(errors[:, [5, 6, 7, 8, 13, 14, 15, 16]].mean()),
            "right_m": float(errors[:, [9, 10, 11, 12, 17, 18, 19, 20]].mean()),
            "observed_segment_joints_m": float(errors[:, [14, 18, 6, 10]].mean()),
            "unobserved_joints_m": float(
                errors[:, [1, 2, 3, 4, 5, 7, 8, 9, 11, 12, 13, 15, 16, 17, 19, 20]].mean()
            ),
            "root_direction_error": "unavailable: arbitrary initial heading and root not predicted",
            "floor_penetration": "unavailable: no reliable root height or floor",
        }


def evaluate(run_dir):
    path = Path(run_dir)
    model, ck = load_model(path / "checkpoint.pt")
    c = ck["config"]
    experiments = []
    for group in ck["split"]["test"]:
        for condition in ("clean", "proximal", "perturbed"):
            s, gt = generate(
                group,
                c["sample_rate_hz"],
                c["duration_s"],
                layout="proximal" if condition == "proximal" else "distal",
                perturb=condition == "perturbed",
            )
            pose = predict(s, path / "checkpoint.pt")
            metrics = BenchmarkRunner.metrics(pose.local_rotations, gt, pose.valid_mask)
            baseline = np.broadcast_to(ck["baseline"].numpy(), pose.local_rotations.shape)
            x = oracle_features(s, gt)
            oracle = predict_windows(
                model, (x - ck["mean"].numpy()) / ck["std"].numpy(), c["window"], c["stride"]
            )
            known = BenchmarkRunner.metrics(pose.local_rotations, gt, pose.valid_mask, True)
            same = "right" if s.metadata["dominant_hand"] == "right" else "left"
            row = {
                "group_id": s.metadata["group_id"],
                "condition": condition,
                "speed": s.metadata["speed"],
                "speed_bin": "fast" if s.metadata["speed"] >= 0.9 else "slow",
                "dominant_hand": same,
                "prediction": metrics,
                "training_mean": BenchmarkRunner.metrics(baseline, gt, pose.valid_mask),
                "oracle_cross_mode": BenchmarkRunner.metrics(oracle, gt, pose.valid_mask),
                "known_anthropometry_research": known,
                "racket_side_m": metrics[same + "_m"],
                "nonracket_side_m": metrics[("left" if same == "right" else "right") + "_m"],
            }
            evidence = direct_evidence(s)
            peak = next(
                e["value"]
                for e in evidence["evidence"]
                if e["evidence_id"] == "racket_forearm:angular_speed_peak"
            )
            global_r = fk(pose.local_rotations)[0]
            row["E6_same_forearm_angular_speed_peak_radps"] = {
                "direct": peak,
                "reconstructed": pose_angular_peak(
                    global_r, 10 if same == "right" else 6, c["sample_rate_hz"], pose.valid_mask
                ),
                "independent_validation": False,
                "root_rotation_caveat": "pose relative to pelvis; comparable here only for fixed-root synthetic test",
            }
            experiments.append(row)
            if condition == "clean" and group == ck["split"]["test"][0]:
                truth_r = gt.local_rotations.copy()
                truth_r[:, 0] = np.eye(3)
                replay(
                    path / "evaluation_replay.html",
                    pose.joints_pelvis_m,
                    c["sample_rate_hz"],
                    fk(truth_r, gt.offsets)[1],
                )
    result = {
        "type": "EvaluationReport",
        "scope": "synthetic_demo_only",
        "research": "not_validated",
        "aggregation": "per original sequence; frames/windows not independent samples",
        "E1": "matched fused BiGRU vs train-only SO(3) mean",
        "E2": "oracle inputs on fused checkpoint: cross-mode only; matched oracle training deferred",
        "E3": "distal vs proximal cross-layout",
        "E4": "clean vs noise/bias/install/drop/saturation/clock compound stress",
        "E5": "default vs known body FK; known lengths are benchmark-only",
        "E6": "forearm angular speed; common source, no independent validation",
        "thresholds": "no research pass threshold established; engineering invariants only",
        "sequences": experiments,
    }
    write_json(path / "evaluation.json", result)
    lines = [
        "# 独立研发评测",
        "",
        result["scope"],
        "",
        "| 原始序列 | 条件 | 模型 MPJPE mm | 均值姿态 mm | 旋转误差 degree |",
        "|---|---|---:|---:|---:|",
    ]
    for row in experiments:
        lines.append(
            f"| {row['group_id']} | {row['condition']} | {row['prediction']['mpjpe_m'] * 1000:.2f} | {row['training_mean']['mpjpe_m'] * 1000:.2f} | {np.rad2deg(row['prediction']['local_rotation_error_rad']):.2f} |"
        )
    lines += [
        "",
        result["E2"],
        "研究/产品：not_validated。仅 3 个独立测试序列，不能支持真实人体或羽毛球算法结论。",
    ]
    (path / "evaluation.md").write_text("\n".join(lines), encoding="utf-8")
    return result


def report(run_dir, backend="template"):
    path = Path(run_dir)
    bundle = json.loads((path / "evidence.json").read_text(encoding="utf-8"))
    result = generate_report(bundle, backend, path / "llm_record.json")
    write_json(path / "coach_report.json", result)
    (path / "coach_report.md").write_text(markdown(result), encoding="utf-8")
    return result


def demo(c, output, device="cpu"):
    path = Path(output)
    if path.exists():
        raise FileExistsError("output exists: choose a new run ID")
    summary = train(c, path, device)
    s, gt = generate(c["groups"] - 3, c["sample_rate_hz"], c["duration_s"])
    pose = predict(s, path / "checkpoint.pt")
    processed = fuse(s)
    # Separate namespaces and files; inference metadata never points at evaluation truth.
    for layer in ("raw", "processed", "evaluation_truth"):
        (path / layer).mkdir()
    for node, stream in s.streams.items():
        np.savez_compressed(
            path / "raw" / f"{node}.npz", **{k: v for k, v in stream.items() if isinstance(v, np.ndarray)}
        )
        p = processed[node]
        np.savez_compressed(
            path / "processed" / f"{node}.npz", **{k: v for k, v in p.items() if isinstance(v, np.ndarray)}
        )
        write_json(
            path / "processed" / f"{node}.json", {k: v for k, v in p.items() if not isinstance(v, np.ndarray)}
        )
    write_json(path / "session.json", s.metadata)
    write_json(
        path / "raw" / "metadata.json",
        {
            node: {k: v for k, v in st.items() if not isinstance(v, np.ndarray)}
            for node, st in s.streams.items()
        },
    )
    np.savez_compressed(
        path / "evaluation_truth" / "truth.npz", local_rotations=gt.local_rotations, offsets=gt.offsets
    )
    np.savez_compressed(
        path / "prediction.npz",
        local_rotations=pose.local_rotations,
        joints_pelvis_m=pose.joints_pelvis_m,
        valid_mask=pose.valid_mask,
    )
    write_json(path / "prediction.json", pose.metadata)
    replay(path / "replay.html", pose.joints_pelvis_m, c["sample_rate_hz"])
    write_json(path / "evidence.json", direct_evidence(s))
    report(path)
    evaluate(path)
    write_json(
        path / "provenance.json",
        {
            "pipeline_version": __version__,
            "seed": c["seed"],
            "config": c,
            "manifest_sha256": file_hash("configs/license_manifest.json"),
            "checkpoint_sha256": file_hash(path / "checkpoint.pt"),
            "data": "original analytic generator",
            "units": "m,s,rad/s,m/s^2",
            "device": device,
        },
    )
    return summary
