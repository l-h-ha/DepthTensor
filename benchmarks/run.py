"""Run CPU benchmarks and print Markdown."""

import argparse
from itertools import permutations
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys

FRAMEWORKS = {
    "depthtensor": "DepthTensor",
    "torch": "PyTorch",
    "tensorflow": "TensorFlow",
}
WORKER = Path(__file__).with_name("workloads.py").resolve()

def positive_integer(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")

    return number

def cpu_model():
    if sys.platform == "win32":
        import winreg

        try:
            path = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, path) as key:
                return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    elif sys.platform.startswith("linux"):
        try:
            cpuinfo = Path("/proc/cpuinfo").read_text(encoding="utf-8")
            for line in cpuinfo.splitlines():
                key, _, value = line.partition(":")
                if key.strip() in {"model name", "Hardware"} and value.strip():
                    return value.strip()
        except OSError:
            pass

    return platform.processor() or platform.machine()

def check_result(result, framework, samples):
    if not isinstance(result, dict) or result.get("framework") != framework:
        raise RuntimeError(f"Expected benchmark results from {framework}.")

    for field in ("version", "numpy"):
        value = result.get(field)
        if not isinstance(value, str) or not value.strip():
            raise RuntimeError(f"{framework} has no valid {field} version.")

    rows = result.get("rows")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"{framework} returned invalid benchmark rows.")

    timings = {}
    for row in rows:
        if not isinstance(row, dict):
            raise RuntimeError(f"{framework} returned an invalid benchmark row.")

        case = (row.get("workload"), row.get("shape"))
        if not all(isinstance(value, str) and value.strip() for value in case):
            raise RuntimeError(f"Expected a workload name and shape from {framework}.")

        if case in timings:
            raise RuntimeError(f"{framework} returned duplicate timings for {case}.")

        repeats = row.get("repeats")
        values = row.get("samples_us")
        median = row.get("median_us")

        if type(repeats) is not int or repeats < 1:
            raise RuntimeError(f"{framework} returned an invalid repeat count for {case}.")

        if not isinstance(values, list) or len(values) != samples:
            raise RuntimeError(f"{framework} returned the wrong sample count for {case}.")
        if not all(type(value) in (int, float) and 0 < value < math.inf for value in values):
            raise RuntimeError(f"{framework} returned invalid sample timings for {case}.")

        if (
            type(median) not in (int, float)
            or not 0 < median < math.inf
            or not math.isclose(median, statistics.median(values), rel_tol=1e-6, abs_tol=0.001)
        ):
            raise RuntimeError(f"{framework} returned an invalid median for {case}.")

        timings[case] = median

    return result["version"], result["numpy"], timings

def run_framework(framework, args):
    env = os.environ.copy()
    for variable in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "BLIS_NUM_THREADS", "TF_NUM_INTRAOP_THREADS",
    ):
        env[variable] = str(args.threads)

    env.update(
        CUDA_VISIBLE_DEVICES="-1",
        OMP_DYNAMIC="FALSE",
        TF_NUM_INTEROP_THREADS="1",
        TF_CPP_MIN_LOG_LEVEL="2",
    )

    command = [
        sys.executable, "-I", "-B", str(WORKER), framework,
        "--threads", str(args.threads),
        "--warmups", str(args.warmups),
        "--samples", str(args.samples),
    ]

    try:
        completed = subprocess.run(
            command, env=env, capture_output=True, encoding="utf-8",
            errors="replace", timeout=300, check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"{framework} timed out after 300 seconds.") from exc

    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"{framework} exited with code {completed.returncode}.\n{detail}")

    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{framework} returned invalid benchmark JSON.") from exc

    return check_result(result, framework, args.samples)

def run_benchmarks(args):
    orders = tuple(permutations(FRAMEWORKS))
    runs = []

    for index in range(args.runs):
        print(f"Run {index + 1}/{args.runs}", file=sys.stderr, flush=True)

        results = {}
        for framework in orders[index % len(orders)]:
            results[framework] = run_framework(framework, args)

        runs.append(results)

    return runs

def render_report(runs):
    first = runs[0]
    _, numpy, cases = first["depthtensor"]

    for results in runs:
        for framework, (version, numpy_version, timings) in results.items():
            if numpy_version != numpy:
                raise RuntimeError(f"{framework} used a different NumPy version.")

            if version != first[framework][0]:
                raise RuntimeError(f"{framework} changed versions between runs.")

            if timings.keys() != cases.keys():
                raise RuntimeError(f"{framework} ran different benchmark cases.")

    versions = ", ".join(
        f"{label} {first[framework][0]}" for framework, label in FRAMEWORKS.items()
    )
    os_version = platform.version() if sys.platform == "win32" else platform.release()

    report = [
        f"{cpu_model()}, {platform.system()} {os_version}, Python {platform.python_version()}.",
        f"{versions}, NumPy {numpy}.",
        "",
        "Rounded median (min–max) in µs/call.",
        "",
        "| Workload | Input shapes | DepthTensor | PyTorch | TensorFlow |",
        "| --- | --- | ---: | ---: | ---: |",
    ]

    for workload, shape in cases:
        cells = []
        for framework in FRAMEWORKS:
            times = [results[framework][2][workload, shape] for results in runs]
            cells.append(f"{statistics.median(times):.0f} ({min(times):.0f}–{max(times):.0f})")

        row = " | ".join(cells)
        report.append(f"| {workload} | `{shape}` | {row} |")

    return "\n".join(report)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=positive_integer, default=1, help="compute threads per framework")
    parser.add_argument("--warmups", type=positive_integer, default=5, help="warmup calls per workload")
    parser.add_argument("--samples", type=positive_integer, default=7, help="timed samples per workload")
    parser.add_argument("--runs", type=positive_integer, default=6, help="independent runs per framework")
    args = parser.parse_args()

    try:
        report = render_report(run_benchmarks(args))
    except (RuntimeError, OSError, UnicodeError) as exc:
        parser.exit(1, f"error: {exc}\n")

    sys.stdout.reconfigure(encoding="utf-8")
    print(report)

if __name__ == "__main__":
    main()
