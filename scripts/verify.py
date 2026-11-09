"""Run bounded local smoke checks and retain real exit codes; no external requests."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path(__file__).resolve().parents[1]
os.chdir(root)
temporary = root / ".tmp"
temporary.mkdir(exist_ok=True)
os.environ["TEMP"] = os.environ["TMP"] = str(temporary)
run_id = str(time.time_ns())
commands = [
    ["-m", "badminton_coach.cli", "doctor"],
    ["-m", "pytest", "-q", "--basetemp", f".tmp/pytest-{run_id}"],
    ["-m", "ruff", "check", "."],
    ["-m", "badminton_coach.cli", "evaluate", "--run-dir", "artifacts/smoke-v02"],
    ["-m", "badminton_coach.cli", "report", "--run-dir", "artifacts/smoke-v02", "--backend", "template"],
]
results = []
for args in commands:
    start = time.perf_counter()
    result = subprocess.run(
        [sys.executable, *args], capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    results.append(
        {
            "command": "python " + " ".join(args),
            "exit_code": result.returncode,
            "seconds": time.perf_counter() - start,
            "test_summary": result.stdout.strip().splitlines()[-1] if args[1] == "pytest" else None,
        }
    )
    print(results[-1])
Path("artifacts/verification.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
sys.exit(any(r["exit_code"] != 0 for r in results))
