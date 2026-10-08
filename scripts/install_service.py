#!/usr/bin/env python3
"""Verify pinned artifacts and install only the TensorFold user unit; never start it implicitly."""
import argparse
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
UNIT = "qwen38-flash-next-tensorfold.service"


def systemd_quote(value):
    value = str(value)
    if any(c in value for c in "\n\r\x00"):
        raise ValueError("Invalid path")
    return '"' + value.replace("%", "%%").replace("$", "$$").replace("\\", "\\\\").replace('"', '\\"') + '"'


def render(model, tower):
    command = " ".join(systemd_quote(x) for x in ["/usr/bin/python3", ROOT / "scripts/supervise.py", model, tower])
    return f"""[Unit]
Description=Qwen3.8 Flash-Next TensorFold EXL3 4.05
After=docker.service

[Service]
Type=simple
ExecStart={command}
Restart=no
TimeoutStopSec=30
MemoryMax=512M
MemorySwapMax=0

[Install]
WantedBy=default.target
"""


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("model_root", type=Path)
    p.add_argument("vision_weights", type=Path)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    model, tower = a.model_root.expanduser().resolve(), a.vision_weights.expanduser().resolve()
    if tower.name != "vision_tower_bf16.safetensors":
        p.error("Expected manifest BF16 tower filename")
    unit = render(model, tower)
    if a.dry_run:
        print(unit, end="")
        return 0
    for manifest, destination in [("exl3-405.json", model), ("vision-bf16.json", tower.parent)]:
        subprocess.run(["python3", str(ROOT / "scripts/download_model.py"), "--manifest", str(ROOT / "manifests" / manifest), "--destination", str(destination), "--verify-only"], check=True)
    destination = Path.home() / ".config/systemd/user" / UNIT
    if destination.exists():
        raise SystemExit("Unit already exists. Refusing to overwrite an existing deployment.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x") as handle:
        handle.write(unit)
    destination.chmod(0o644)
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
    print("Installed " + UNIT + ". Not started.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
