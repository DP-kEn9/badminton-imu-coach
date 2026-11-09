import argparse
import importlib.metadata
import json
import platform
from pathlib import Path

from .pipeline import demo, evaluate, load_config, report, train


def main():
    p = argparse.ArgumentParser(description="四身体 IMU 离线解析研究原型")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor")
    for command in ("demo", "train"):
        s = sub.add_parser(command)
        s.add_argument("--config", default="configs/smoke.yaml")
        s.add_argument("--device", default="cpu")
        s.add_argument("--output")
    for command in ("evaluate", "report"):
        s = sub.add_parser(command)
        s.add_argument("--run-dir", required=True)
        if command == "report":
            s.add_argument("--backend", choices=["template", "mock", "replay", "live"], default="template")
    a = p.parse_args()
    if a.command == "doctor":
        result = {
            "python": platform.python_version(),
            "platform": platform.system(),
            "versions": {
                n: importlib.metadata.version(n) for n in ["numpy", "scipy", "torch", "vqf", "pytest", "ruff"]
            },
            "data_mode": "demo_synthetic",
            "hardware": "CPU; GPU not required",
            "research": "not_validated",
        }
    elif a.command in ("demo", "train"):
        c = load_config(a.config)
        target = a.output or c["output"]
        if a.command == "train" and Path(target).exists():
            raise FileExistsError("output exists; choose a new run ID with --output")
        result = demo(c, target, a.device) if a.command == "demo" else train(c, target, a.device)
    elif a.command == "evaluate":
        result = evaluate(a.run_dir)
    else:
        result = report(a.run_dir, a.backend)
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
