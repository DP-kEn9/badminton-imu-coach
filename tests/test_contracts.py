import copy
import json
from dataclasses import asdict

import numpy as np
import pytest
import torch
import yaml
from scipy.spatial.transform import Rotation

from badminton_coach.agent import digest, generate_report, verify_selection
from badminton_coach.analysis import direct_evidence
from badminton_coach.assets import file_hash, load_licensed
from badminton_coach.body import NAMES, OFFSETS, PARENTS, fk, from6, to6
from badminton_coach.fusion import features, fuse, synchronize
from badminton_coach.models import BiGRU, predict_windows, windows
from badminton_coach.pipeline import demo, load_model, predict, split_groups, train
from badminton_coach.simulation import binding, generate, rigid_imu


def test_fk_bone_lengths_and_bindings():
    r = Rotation.random(42, random_state=4).as_matrix().reshape(2, 21, 3, 3)
    _, p = fk(r)
    for j in range(1, 21):
        np.testing.assert_allclose(
            np.linalg.norm(p[:, j] - p[:, PARENTS[j]], axis=-1), np.linalg.norm(OFFSETS[j]), atol=1e-12
        )
    assert binding()["racket_forearm"][0] == 10
    assert binding("left")["racket_forearm"][0] == 6
    assert binding()["left_shank"][0] == 14


@pytest.mark.parametrize("case", ["static", "translation", "rotation"])
def test_physical_imu(case):
    t = np.arange(400) / 100
    r = np.tile(np.eye(3), (len(t), 1, 1))
    p = np.zeros((len(t), 3))
    expected = np.tile([0, 0, 9.80665], (len(t), 1))
    if case == "translation":
        p[:, 0] = 0.5 * 2 * t * t
        expected[:, 0] = 2
    if case == "rotation":
        r = Rotation.from_rotvec(np.column_stack([t * 0, t * 0, 0.4 * t])).as_matrix()
    a, g = rigid_imu(r, p, 0.01)
    np.testing.assert_allclose(a, expected, atol=1e-8)
    np.testing.assert_allclose(g, np.tile([0, 0, 0.4 if case == "rotation" else 0], (len(t), 1)), atol=1e-10)


def test_same_rigid_two_sites():
    t = np.arange(500) / 100
    omega, alpha = 0.6, 0.15
    angle = omega * t + 0.5 * alpha * t * t
    r = Rotation.from_rotvec(np.column_stack([t * 0, t * 0, angle])).as_matrix()
    d = np.array([0.3, 0, 0])
    p2 = np.einsum("tij,j->ti", r, d)
    a1, g1 = rigid_imu(r, np.zeros_like(p2), 0.01)
    a2, g2 = rigid_imu(r, p2, 0.01)
    w = np.column_stack([t * 0, t * 0, omega + alpha * t])
    al = np.tile([0, 0, alpha], (len(t), 1))
    expected = np.cross(al, p2) + np.cross(w, np.cross(w, p2))
    actual = np.einsum("tij,tj->ti", r, a2 - a1)
    np.testing.assert_allclose(actual[3:-3], expected[3:-3], atol=2e-5)
    np.testing.assert_allclose(g1, g2)


@pytest.mark.parametrize("x", [np.zeros(6), np.ones(6), np.array([1, 0, 0, 2, 0, 0])])
def test_degenerate6_is_so3(x):
    r = from6(torch.tensor(x, dtype=torch.float32)).numpy()
    np.testing.assert_allclose(r.T @ r, np.eye(3), atol=1e-6)
    assert np.linalg.det(r) == pytest.approx(1, abs=1e-6)


def test_rotation6_roundtrip():
    r = Rotation.random(10, random_state=5).as_matrix()
    np.testing.assert_allclose(from6(torch.tensor(to6(r))).numpy(), r, atol=1e-12)


def test_clock_sync_and_gaps():
    s, _ = generate(4, perturb=True)
    stream = s.streams["pelvis"]
    assert synchronize(stream)["sync_metadata"]["interpolated_frames"] > 0
    stream["valid_mask"][120:140] = False
    with pytest.raises(ValueError, match="long gap"):
        synchronize(stream)


def test_saturation_and_mount_perturbations():
    s, gt = generate(4, perturb=True)
    assert any(np.any(v["saturation_mask"]) for v in s.streams.values())
    assert gt.metadata["actual_installation"]["pelvis"]["R_BS"] != np.eye(3).tolist()


def test_fusion_static_and_no_truth_access():
    s, gt = generate(0)
    a = fuse(s)
    gt.local_rotations[:] = np.nan
    gt.ideal_sensor_orientations.clear()
    b = fuse(s)
    for n in s.streams:
        np.testing.assert_allclose(a[n]["rotation"], np.tile(np.eye(3), (400, 1, 1)), atol=1e-5)
        np.testing.assert_array_equal(a[n]["rotation"], b[n]["rotation"])
    assert "truth" not in asdict(s)


def test_fusion_continuous_heading_and_quat_sign():
    s, _ = generate(2, kind="rotation")
    p = fuse(s)["pelvis"]
    r = p["rotation"]
    observed = np.unwrap(Rotation.from_matrix(r).as_euler("xyz")[:, 2])
    t = s.streams["pelvis"]["timestamp_session_s"]
    assert observed[-3] > 0.9
    q = p["orientation_sensor_to_reference_wxyz"]
    assert np.min(np.sum(q[:-1] * q[1:], axis=-1)) > 0
    assert np.max(np.abs(np.diff(observed))) < 0.03
    assert t[-1] > 3


def test_bad_calibration():
    s, _ = generate(0)
    s.streams["pelvis"]["gyro_sensor_radps"][:90] = 2
    with pytest.raises(ValueError, match="calibration"):
        fuse(s)


def test_missing_node_direct_available():
    s, _ = generate(4)
    del s.streams["left_shank"]
    assert len(direct_evidence(s)["evidence"]) == 6
    with pytest.raises(ValueError, match="four nodes"):
        features(s, fuse(s))


def test_wrong_units():
    s, _ = generate(0)
    s.metadata["units"]["gyro"] = "degree/s"
    with pytest.raises(ValueError, match="units"):
        s.validate()


def test_no_racket_copy():
    s, _ = generate(0)
    assert s.racket is None and not s.racket_present
    assert len(s.streams) == 4


def test_split_group_no_leak():
    split = split_groups(18)
    assert not set(split["train"]) & set(split["test"])
    assert not set(split["train"]) & set(split["dev"])
    assert len(set(sum(split.values(), []))) == 18


@pytest.mark.parametrize("length", [1, 51, 200, 201, 350, 401])
def test_windows_edges_short_so3(length):
    torch.set_num_threads(2)
    model = BiGRU(hidden=4, layers=1).eval()
    result = predict_windows(model, np.zeros((length, 72), dtype="float32"))
    assert result.shape == (length, 21, 3, 3)
    assert windows(length, 200, 100)[-1] + 200 >= length
    np.testing.assert_allclose(
        result.swapaxes(-1, -2) @ result, np.broadcast_to(np.eye(3), result.shape), atol=1e-5
    )
    assert np.min(np.linalg.det(result)) > 0.9999


def test_agent_no_labels_and_replay(tmp_path, monkeypatch):
    s, _ = generate(4)
    bundle = direct_evidence(s)
    file = tmp_path / "record.json"
    r = generate_report(bundle, "mock", file)
    replay = generate_report(bundle, "replay", file)
    assert r["observations"] == replay["observations"]
    assert r["motion_type"] == "unknown" and r["diagnoses"] == []
    monkeypatch.setenv("LLM_ALLOW_EXTERNAL", "false")
    monkeypatch.setenv("LLM_API_KEY", "secret-test-never-write")
    assert generate_report(bundle, "live", file)["backend"] == "template_fallback"
    assert "secret-test-never-write" not in file.read_text()


@pytest.mark.parametrize(
    "problem", ["bad_id", "diagnosis", "numeric", "conflict", "wrong_unit", "replay_missing"]
)
def test_agent_rejects_untrusted(problem, tmp_path):
    s, _ = generate(4)
    bundle = direct_evidence(s)
    if problem in ("bad_id", "diagnosis", "numeric"):
        response = {
            "observations": [
                {
                    "evidence_id": "invented"
                    if problem == "bad_id"
                    else bundle["evidence"][0]["evidence_id"],
                    "observation_type": "diagnosis" if problem == "diagnosis" else "measurement",
                }
            ]
        }
        if problem == "numeric":
            response["observations"][0]["value"] = 99
        with pytest.raises(ValueError):
            verify_selection(response, bundle)
    elif problem == "replay_missing":
        assert generate_report(bundle, "replay", tmp_path / "absent")["backend"] == "template_fallback"
    else:
        if problem == "conflict":
            bundle["evidence"].append(copy.deepcopy(bundle["evidence"][0]))
        else:
            bundle["evidence"][0]["unit"] = "mph"
        assert generate_report(bundle)["status"] == "evidence_rejected"


def test_license_real_loader_fixture(tmp_path):
    np.savez(
        tmp_path / "motion.npz",
        joint_names=np.array(NAMES),
        timestamp_s=np.arange(10) / 100,
        local_rotations_wxyz=np.tile([1.0, 0, 0, 0], (10, 21, 1)),
    )
    model = {
        "id": "analytic_skeleton_v1",
        "parents": PARENTS,
        "offsets_m": OFFSETS.tolist(),
        "shape_dimension": 0,
    }
    (tmp_path / "model.json").write_text(json.dumps(model))
    assets = []
    for name, category, fmt in [
        ("motion.npz", "motion", "analytic_npz_v1"),
        ("model.json", "body_model", "analytic_skeleton_v1"),
    ]:
        assets.append(
            {
                "asset_id": category,
                "category": category,
                "path": name,
                "sha256": file_hash(tmp_path / name),
                "authorization_status": "approved",
                "allowed_processing": ["local_research"],
                "format": fmt,
            }
        )
    m = tmp_path / "manifest.json"
    m.write_text(json.dumps({"assets": assets}))
    loaded = load_licensed(m, "motion", "body_model")
    assert loaded["truth"].local_rotations.shape == (10, 21, 3, 3)
    assets[0]["authorization_status"] = "pending"
    m.write_text(json.dumps({"assets": assets}))
    with pytest.raises(PermissionError):
        load_licensed(m, "motion", "body_model")
    assets[0]["authorization_status"] = "approved"
    assets[0]["format"] = "SMPL_unknown"
    m.write_text(json.dumps({"assets": assets}))
    with pytest.raises(ValueError, match="unsupported"):
        load_licensed(m, "motion", "body_model")


def test_training_update_reload_and_truth_isolation(tmp_path):
    c = yaml.safe_load(open("configs/smoke.yaml", encoding="utf-8"))
    c.update(epochs=1, hidden=8, layers=1)
    result = train(c, tmp_path)
    assert result["parameter_update_verified"] and result["steps"] > 0
    model, ck = load_model(tmp_path / "checkpoint.pt")
    s, gt = generate(15)
    p = predict(s, tmp_path / "checkpoint.pt")
    before = digest(direct_evidence(s))
    gt.local_rotations[:] = np.nan
    gt.offsets[:] = 10
    q = predict(s, tmp_path / "checkpoint.pt")
    np.testing.assert_array_equal(p.local_rotations, q.local_rotations)
    assert before == digest(direct_evidence(s))
    assert ck["normalization_source"] == "training groups only"
    assert model.training is False


def test_cpu_end_to_end(tmp_path):
    c = yaml.safe_load(open("configs/smoke.yaml", encoding="utf-8"))
    c.update(epochs=1, hidden=8, layers=1)
    target = tmp_path / "run"
    demo(c, target)
    for file in (
        "checkpoint.pt",
        "replay.html",
        "evaluation_replay.html",
        "prediction.npz",
        "coach_report.json",
        "evaluation.json",
    ):
        assert (target / file).stat().st_size > 0
    assert (
        "truth"
        not in (target / "replay.html")
        .read_text(encoding="utf-8")
        .split("const D=")[1]
        .split("; const c=")[0]
    )
    with pytest.raises(FileExistsError):
        demo(c, target)
