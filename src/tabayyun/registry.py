"""Run registry: every experiment run is appended to results/registry.jsonl
with the git commit, config and metrics, so every number in the paper can be
traced to a commit and a command."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def git_state() -> dict:
    def run(*a):
        try:
            return subprocess.check_output(["git", *a], cwd=ROOT, stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return "unknown"
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


def env_state() -> dict:
    try:
        import torch
        dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        tv = torch.__version__
    except Exception:
        dev, tv = "unknown", "unknown"
    return {"python": sys.version.split()[0], "torch": tv, "device": dev, "host": platform.node()}


def config_hash(cfg: dict) -> str:
    return hashlib.sha1(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:10]


def log_run(experiment: str, cfg: dict, metrics: dict, out: str | Path = "results/registry.jsonl",
            smoke: bool = False) -> dict:
    rec = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "experiment": experiment,
        "smoke": smoke,
        "config_hash": config_hash(cfg),
        "git": git_state(),
        "env": env_state(),
        "config": cfg,
        "metrics": metrics,
    }
    p = ROOT / out
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as fh:
        fh.write(json.dumps(rec, default=str) + "\n")
    return rec
