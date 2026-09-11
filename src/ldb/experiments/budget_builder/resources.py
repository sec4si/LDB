"""Sample physical process footprint; the watchdog is not an OS hard memory cap."""

import ctypes
import platform
import subprocess
import time

from .common import read, write


class Usage(ctypes.Structure):
    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [
        (name, ctypes.c_uint64)
        for name in [
            "user",
            "system",
            "idle_wakeups",
            "interrupt_wakeups",
            "pageins",
            "wired",
            "resident",
            "footprint",
            "start",
            "exit",
        ]
    ]


def footprint(pid):
    if platform.system() != "Darwin":
        raise RuntimeError(
            "v1 physical-footprint monitor is validated only on macOS; unsupported platforms fail closed"
        )
    lib = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
    value = Usage()
    if lib.proc_pid_rusage(pid, 0, ctypes.byref(value)) != 0:
        raise ProcessLookupError(pid)
    return value.footprint


def run_worker(command, root, output, limit, timeout, environment):
    peak, samples, reason = 0, 0, None
    started = time.monotonic()
    with output.with_suffix(".log").open("w") as log:
        process = subprocess.Popen(
            command,
            cwd=root,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            while process.poll() is None:
                try:
                    peak = max(peak, footprint(process.pid))
                    samples += 1
                except ProcessLookupError:
                    if process.poll() is None:
                        raise
                if limit and peak > limit:
                    reason = "memory_watchdog_exceeded"
                    process.kill()
                if time.monotonic() - started > timeout:
                    reason = "timeout"
                    process.kill()
                time.sleep(0.02)
            code = process.wait()
        except BaseException:
            process.kill()
            process.wait()
            raise
    record = {
        "returncode": code,
        "reason": reason,
        "peak_physical_footprint_bytes": peak,
        "samples": samples,
        "sampling_seconds": 0.02,
        "seconds": time.monotonic() - started,
        "semantics": "macOS proc_pid_rusage physical footprint, sampled from process start through exit; transient peaks can be missed; not an OS-enforced cap",
    }
    if code == 0 and output.exists() and samples:
        record["result"] = read(output)
    write(output.with_suffix(".process.json"), record)
    return record
