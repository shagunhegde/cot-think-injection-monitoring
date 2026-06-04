#!/usr/bin/env python
"""Provision a RunPod GPU pod and run the full experiment pipeline remotely.

Usage:
    python scripts/run_remote.py experiments/example_h2.yaml [options]

Options:
    --gpu {RTX3090,T4,A100,L4}   GPU type to request (default: RTX3090)
    --ssh-key PATH               SSH private key (default: ~/.ssh/id_rsa)
    --stage STAGE                Stage or phase to run (default: all_datagen, then run_pipeline)
    --shard I --total-shards N   Shard subject/clean_baseline across N pods
    --dry-run                    Estimate cost + exit (zero spend, no pod provisioned)
    --terminate                  Terminate the pod after the job completes
    --pod-id POD_ID              Reuse an existing (running) pod instead of provisioning

Environment (read from .env or shell):
    RUNPOD_API_KEY       required for provisioning
    OPENROUTER_API_KEY   pushed to the pod's .env
    HF_TOKEN             pushed to the pod's .env (optional)

Examples:
    # Single-GPU full run:
    python scripts/run_remote.py experiments/example_h2.yaml --terminate

    # 4-GPU parallel subject inference:
    for i in 0 1 2 3; do
        python scripts/run_remote.py experiments/example_full.yaml \\
            --stage subject --shard $i --total-shards 4 &
    done
    wait
    # Then run analysis on any one pod (or locally):
    python scripts/run_remote.py experiments/example_full.yaml --stage all_analysis

    # Reuse an already-running pod:
    python scripts/run_remote.py experiments/example_h2.yaml --pod-id abc123xyz
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Load .env if present (before anything that reads env vars).
try:
    from dotenv import load_dotenv
    load_dotenv(Path(".env"))
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(message)s")

# Ensure local src/ is importable even without install.
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.infra.remote import (
    provision_pod, wait_for_ready, push_repo, setup_pod,
    run_stage_remote, pull_results, terminate_pod,
)
from src.pipeline.stages import load_run_config


def parse_args():
    p = argparse.ArgumentParser(description="Run the CoT-injection harness on a RunPod GPU")
    p.add_argument("config", help="Path to experiments/*.yaml run config")
    p.add_argument("--gpu", default="RTX3090",
                   choices=["RTX3090", "T4", "A100", "L4"],
                   help="GPU type to request (default: RTX3090)")
    p.add_argument("--ssh-key", default="~/.ssh/id_rsa")
    p.add_argument("--stage", default=None,
                   help="Stage or phase to run (default: runs datagen then analysis)")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--total-shards", type=int, default=1)
    p.add_argument("--dry-run", action="store_true",
                   help="Estimate cost + print matrix size; no pod, no spend")
    p.add_argument("--terminate", action="store_true",
                   help="Terminate the pod after the job")
    p.add_argument("--pod-id", default=None,
                   help="Reuse an existing running pod (skips provisioning)")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_run_config(args.config)
    run_name = cfg.get("run_name", "run")

    if args.dry_run:
        # Pure local estimate — no GPU required.
        from src.infra.paths import resolve_root
        from src.pipeline.stages import dry_run_report
        paths = resolve_root(mount=False, create=True)
        dry_run_report(cfg, paths)
        return

    # ── Provision or reuse pod ────────────────────────────────────────────────
    pod_id = args.pod_id
    pod = None

    if pod_id:
        # Reconnect to an existing pod.
        from src.infra.remote import _runpod_client
        rp = _runpod_client()
        pod = rp.get_pod(pod_id)
        pod["_ssh_port"] = next(
            (p["publicPort"] for p in (pod.get("runtime") or {}).get("ports", [])
             if p.get("privatePort") == 22),
            None,
        )
        pod["_ssh_host"] = "connect.runpod.io"
        if not pod["_ssh_port"]:
            raise RuntimeError(f"Pod {pod_id} has no public SSH port. Is it running?")
        print(f"Reusing pod {pod_id} (ssh:{pod['_ssh_port']})")
    else:
        remote_block = cfg.get("remote", {}) if isinstance(cfg.get("remote"), dict) else {}
        gpu = remote_block.get("gpu", args.gpu)
        volume_mount = remote_block.get("volume_mount", "/workspace/cache")
        pod = provision_pod(
            gpu_type=gpu,
            name=f"cotim-{run_name}",
            volume_mount=volume_mount,
        )
        pod_id = pod["id"]
        pod = wait_for_ready(pod_id, timeout_s=remote_block.get("wait_timeout_s", 300))
        push_repo(pod, ssh_key=args.ssh_key)
        setup_pod(
            pod,
            ssh_key=args.ssh_key,
            env_vars={
                "OPENROUTER_API_KEY": os.environ.get("OPENROUTER_API_KEY", ""),
                "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
            },
            volume_mount=volume_mount,
        )

    # ── Build the remote command(s) ───────────────────────────────────────────
    shard_flags = (
        f"--shard {args.shard} --total-shards {args.total_shards}"
        if args.total_shards > 1 else ""
    )

    if args.stage:
        # Single explicit stage.
        if args.stage in ("faithfulness", "monitor", "clean_fpr", "metrics", "plots",
                          "all_analysis"):
            script = f"bash scripts/run_pipeline.sh {args.config} {shard_flags}"
        else:
            script = f"bash scripts/run_datagen.sh {args.config} {args.stage} {shard_flags}"
        rc = run_stage_remote(pod, script, ssh_key=args.ssh_key)
        if rc != 0:
            sys.exit(rc)
    else:
        # Default: datagen then analysis (full run).
        for script in [
            f"bash scripts/run_datagen.sh {args.config} all_datagen {shard_flags}",
            f"bash scripts/run_pipeline.sh {args.config}",
        ]:
            rc = run_stage_remote(pod, script, ssh_key=args.ssh_key)
            if rc != 0:
                print(f"Remote command failed (exit {rc}): {script}", file=sys.stderr)
                if args.terminate:
                    terminate_pod(pod_id)
                sys.exit(rc)

    # ── Pull results ──────────────────────────────────────────────────────────
    try:
        pull_results(pod, run_name, ssh_key=args.ssh_key)
        print(f"Results pulled to results/{run_name}/")
    except Exception as e:
        print(f"Warning: could not pull results: {e}", file=sys.stderr)

    if args.terminate:
        terminate_pod(pod_id)
        print("Pod terminated.")


if __name__ == "__main__":
    main()
