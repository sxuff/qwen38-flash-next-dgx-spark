#!/usr/bin/env python3
"""Own only this recipe container. Never stop another service or change system swap."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
GIB = 1024**3
IMAGE = "qwen38-flash-next-tensorfold:609ca419"
LABEL = "tensorfold-exl3-405"
MINIMUM_PRELOAD_AVAILABLE = 96220600952


def sample(disk_root):
    values = {line.split(":")[0]: int(line.split()[1])*1024 for line in Path("/proc/meminfo").read_text().splitlines() if len(line.split()) >= 2 and line.split()[1].isdigit()}
    return dict(mem_available=values["MemAvailable"], swap_used=values["SwapTotal"]-values["SwapFree"], disk_free=shutil.disk_usage(disk_root).free)


def breach(observed, baseline, container_swap=0):
    if observed["mem_available"] < 7*GIB:
        return "MemAvailable below 7 GiB"
    if observed["swap_used"]-baseline["swap_used"] > GIB:
        return "Host swap growth above 1 GiB"
    if observed["disk_free"] < 20*GIB:
        return "Free disk below 20 GiB"
    if container_swap:
        return "Owned container swap is nonzero"
    return None


def validate_artifacts(model, tower):
    for manifest_name, dest in [("exl3-405.json", model), ("vision-bf16.json", tower.parent)]:
        m = json.loads((ROOT / "manifests" / manifest_name).read_text())
        for item in m["files"]:
            p = dest / item["path"]
            if not p.is_file() or p.is_symlink() or p.stat().st_size != item["bytes"]:
                raise ValueError("Missing or wrong-sized verified artifact: " + item["path"])
    if tower.name != "vision_tower_bf16.safetensors":
        raise ValueError("Expected the manifest's BF16 tower filename")


def docker_command(model, tower, cache, name, port, cidfile):
    return ["docker", "run", "--rm", "--name", name, "--cidfile", str(cidfile),
            "--label", "qwen.recipe=" + LABEL, "--gpus", "all", "--memory", "112g",
            "--memory-swap", "112g", "--shm-size", "2g", "--read-only",
            "--tmpfs", "/tmp:rw,noexec,nosuid,size=1g", "--user", f"{os.getuid()}:{os.getgid()}",
            "-p", f"127.0.0.1:{port}:8001", "-v", f"{model}:/model:ro",
            "-v", f"{tower}:/vision/vision_tower_bf16.safetensors:ro", "-v", f"{cache}:/cache:rw",
            "-e", "HOME=/cache/home", "-e", "TORCH_EXTENSIONS_DIR=/cache/torch-extensions",
            "-e", "TRITON_CACHE_DIR=/cache/triton", "-e", "MAX_JOBS=1",
            "-e", "TENSORFOLD_MEMORY_RESERVE_GIB=7", "-e", "TENSORFOLD_NO_UPDATE_CHECK=1",
            IMAGE, "--model", "/model", "--vision-weights", "/vision/vision_tower_bf16.safetensors",
            "--host", "0.0.0.0", "--port", "8001"]


def owned_info(cid):
    p = subprocess.run(["docker", "inspect", cid], text=True, capture_output=True)
    if p.returncode:
        raise RuntimeError("Cannot inspect owned container")
    info = json.loads(p.stdout)[0]
    if info["Id"] != cid or info["Config"]["Labels"].get("qwen.recipe") != LABEL:
        raise RuntimeError("Container ownership mismatch")
    return info


def owned_swap(info):
    pid = int(info["State"]["Pid"])
    if not pid:
        return 0
    paths = Path(f"/proc/{pid}/cgroup").read_text().splitlines()
    cgroup = next(line.split(":", 2)[2] for line in paths if line.startswith("0::"))
    return int((Path("/sys/fs/cgroup") / cgroup.lstrip("/") / "memory.swap.current").read_text())


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("model_root", type=Path)
    p.add_argument("vision_weights", type=Path)
    p.add_argument("--port", type=int, default=8001)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    if not 1 <= a.port <= 65535:
        p.error("port outside 1..65535")
    model, tower = a.model_root.expanduser().resolve(), a.vision_weights.expanduser().resolve()
    name = "qwen38-flash-next-tensorfold"
    cache = (ROOT / "runtime-cache").resolve()
    if a.dry_run:
        print(json.dumps(docker_command(model, tower, cache, name, a.port, cache / "container.cid")))
        return 0
    validate_artifacts(model, tower)
    baseline = sample(model)
    if baseline["mem_available"] < MINIMUM_PRELOAD_AVAILABLE:
        raise SystemExit("Insufficient preload headroom. Another model may still be resident; no service was stopped.")
    reason = breach(baseline, baseline)
    if reason:
        raise SystemExit(reason)
    existing = subprocess.run(["docker", "ps", "-a", "--filter", "name=^/"+name+"$", "--format", "{{.ID}}"], capture_output=True, text=True, check=True)
    if existing.stdout.strip():
        raise SystemExit("Recipe container already exists. Refusing to replace or kill it.")
    image = json.loads(subprocess.check_output(["docker", "image", "inspect", IMAGE], text=True))[0]
    if image["Config"].get("Labels", {}).get("qwen.runtime.revision") != "609ca419abecebdc5a059498a613680bd3aa847f":
        raise SystemExit("Runtime image pin mismatch")
    cache.mkdir(exist_ok=True)
    for folder in ["home", "torch-extensions", "triton"]:
        (cache / folder).mkdir(exist_ok=True)
    process = None
    cid = None
    halted = False
    def stop(_signum, _frame):
        nonlocal halted
        halted = True
    for sig in [signal.SIGTERM, signal.SIGINT]:
        signal.signal(sig, stop)
    with tempfile.TemporaryDirectory(prefix="launch-", dir=cache) as launch:
        cidfile = Path(launch) / "container.cid"
        try:
            process = subprocess.Popen(docker_command(model, tower, cache, name, a.port, cidfile))
            deadline = time.monotonic()+1200
            ready = False
            while process.poll() is None and not halted:
                observed = sample(model)
                if cidfile.exists() and cidfile.read_text().strip():
                    cid = cidfile.read_text().strip()
                    info = owned_info(cid)
                    swap = owned_swap(info)
                else:
                    swap = 0
                reason = breach(observed, baseline, swap)
                if reason:
                    print(json.dumps(dict(safety_abort=reason, baseline=baseline, observed=observed, container_swap=swap)), flush=True)
                    raise RuntimeError(reason)
                if not ready:
                    try:
                        with urllib.request.urlopen(f"http://127.0.0.1:{a.port}/health", timeout=2) as response:
                            health = json.load(response)
                        ready = health.get("context_length") == 262144
                    except (OSError, ValueError):
                        pass
                    if not ready and time.monotonic() > deadline:
                        raise RuntimeError("Startup readiness timeout")
                # A healthy idle service has no idle deadline.
                time.sleep(2)
            if halted:
                return 0
            return process.returncode
        finally:
            if cid is None and cidfile.exists():
                cid = cidfile.read_text().strip() or None
            if cid:
                try:
                    owned_info(cid)
                    subprocess.run(["docker", "kill", cid], stdout=subprocess.DEVNULL, check=True)
                except RuntimeError:
                    if process is not None and process.poll() is None:
                        raise
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


if __name__ == "__main__":
    raise SystemExit(main())
