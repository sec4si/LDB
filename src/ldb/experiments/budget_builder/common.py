import hashlib
import json
import math
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def validate_spec(spec):
    for key in [
        "weight_bytes",
        "memory_bytes",
        "context_tokens",
        "max_new_tokens",
        "candidate_limit",
        "timeout_seconds",
    ]:
        if type(spec[key]) is not int or spec[key] <= 0:
            raise ValueError(f"Expected positive integer: {key}")
    if spec["max_new_tokens"] < 2:
        raise ValueError("At least two generation steps required for decode measurement")
    if (
        type(spec["concurrency"]) is not int
        or spec["concurrency"] != 1
        or spec["context_tokens"] <= spec["max_new_tokens"]
    ):
        raise ValueError("v1 requires concurrency=1 and context larger than generation budget")
    for key in ["min_task_accuracy", "max_task_drop", "memory_headroom"]:
        value = spec[key]
        if type(value) not in (float, int) or not math.isfinite(value):
            raise ValueError(f"Invalid bounded threshold: {key}")
        if not 0 <= value <= 1 or (key == "memory_headroom" and value == 1):
            raise ValueError(f"Invalid bounded threshold: {key}")
    for key in ["max_delta_nll", "max_delta_nll_upper", "min_decode_tps"]:
        if type(spec[key]) not in (float, int) or not math.isfinite(spec[key]) or spec[key] < 0:
            raise ValueError(f"Invalid threshold: {key}")
    return spec


def freeze(root, paths, name):
    write(Path(root) / name, {str(p): sha(Path(root) / p) for p in paths})


def verify(root, name):
    for path, expected in read(Path(root) / name).items():
        if sha(Path(root) / path) != expected:
            raise ValueError(f"Frozen input changed: {path}")
