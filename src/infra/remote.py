"""RunPod remote orchestration — provision, push, run, stream, pull.

This module is used by scripts/run_remote.py and can also be called directly.
It requires the `runpod` and `paramiko` packages (the `remote` extras in pyproject.toml)
and RUNPOD_API_KEY in the environment (or .env file).

Workflow:
  1. provision_pod()     — request a GPU pod (or find an existing one by name)
  2. wait_for_ready()    — poll until SSH is up (up to `timeout_s`)
  3. push_repo()         — git bundle + rsync to the pod's workspace
  4. setup_pod()         — uv sync, write .env, set COTIM_ROOT to a persistent volume
  5. run_stage_remote()  — SSH-exec the stage script; stream stdout/stderr back
  6. pull_results()      — rsync results/<run_name>/ back to the local machine
  7. terminate_pod()     — optional; stops the pod after the job

All SSH interactions go through paramiko so no local `ssh` binary is required.
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)

# ── GPU templates available on RunPod ─────────────────────────────────────────
GPU_TEMPLATES = {
    "T4":   "NVIDIA GeForce RTX 3090",  # rough match — adjust in config
    "A100": "NVIDIA A100-SXM4-40GB",
    "RTX3090": "NVIDIA GeForce RTX 3090",
    "L4":   "NVIDIA L4",
}

DEFAULT_IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
DEFAULT_DISK_GB = 50   # container disk
DEFAULT_VOLUME_GB = 100  # persistent volume for workspace/cache


# ── RunPod client helpers ─────────────────────────────────────────────────────

def _runpod_client():
    try:
        import runpod
    except ImportError:
        raise ImportError(
            "runpod SDK required: pip install runpod  "
            "(or install the '[remote]' extra)."
        )
    api_key = os.environ.get("RUNPOD_API_KEY", "")
    if not api_key:
        raise EnvironmentError("RUNPOD_API_KEY not set.")
    runpod.api_key = api_key
    return runpod


def provision_pod(
    gpu_type: str = "RTX3090",
    *,
    name: str = "cotim",
    image: str = DEFAULT_IMAGE,
    disk_gb: int = DEFAULT_DISK_GB,
    volume_gb: int = DEFAULT_VOLUME_GB,
    volume_mount: str = "/workspace/cache",
) -> dict:
    """Request a RunPod GPU pod. Returns the pod dict with id and connection info."""
    rp = _runpod_client()
    gpu_name = GPU_TEMPLATES.get(gpu_type, gpu_type)
    log.info("Provisioning RunPod pod: %s (%s)  image=%s", name, gpu_name, image)

    # Find a GPU type ID that matches the requested name.
    gpus = rp.get_gpus()
    gpu_id = next(
        (g["id"] for g in gpus if gpu_name.lower() in g.get("displayName", "").lower()),
        None,
    )
    if gpu_id is None:
        available = [g["displayName"] for g in gpus[:10]]
        raise ValueError(
            f"GPU type {gpu_name!r} not found on RunPod. "
            f"Available (first 10): {available}"
        )

    pod = rp.create_pod(
        name=name,
        image_name=image,
        gpu_type_id=gpu_id,
        container_disk_in_gb=disk_gb,
        volume_in_gb=volume_gb,
        volume_mount_path=volume_mount,
        ports="22/tcp",
        env={"COTIM_ROOT": volume_mount},
    )
    log.info("Pod created: id=%s", pod.get("id"))
    return pod


def wait_for_ready(pod_id: str, *, timeout_s: int = 300, poll_s: int = 10) -> dict:
    """Poll until the pod is running and has SSH info. Returns updated pod dict."""
    rp = _runpod_client()
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        pod = rp.get_pod(pod_id)
        status = (pod.get("desiredStatus") or "").lower()
        runtime = pod.get("runtime") or {}
        ports = runtime.get("ports") or []
        ssh_port = next(
            (p.get("publicPort") for p in ports if p.get("privatePort") == 22),
            None,
        )
        if status == "running" and ssh_port:
            log.info("Pod ready: %s (ssh port %s)", pod_id, ssh_port)
            pod["_ssh_port"] = ssh_port
            pod["_ssh_host"] = "connect.runpod.io"
            return pod
        log.info("Waiting for pod %s (status=%s) ...", pod_id, status)
        time.sleep(poll_s)
    raise TimeoutError(f"Pod {pod_id} not ready after {timeout_s}s.")


def _ssh(host: str, port: int, key_path: str):
    """Return an open paramiko SSHClient."""
    try:
        import paramiko
    except ImportError:
        raise ImportError(
            "paramiko required: pip install paramiko  "
            "(or install the '[remote]' extra)."
        )
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host, port=port, username="root",
                   key_filename=key_path, timeout=30)
    return client


def push_repo(
    pod: dict,
    *,
    repo_dir: str = ".",
    ssh_key: str = "~/.ssh/id_rsa",
) -> None:
    """Rsync the local repo to /workspace/cotim on the pod (excludes workspace/cache)."""
    host = pod["_ssh_host"]
    port = pod["_ssh_port"]
    key = str(Path(ssh_key).expanduser())
    remote = f"root@{host}:/workspace/cotim/"
    cmd = [
        "rsync", "-az", "--progress",
        "-e", f"ssh -p {port} -i {key} -o StrictHostKeyChecking=no",
        "--exclude=workspace", "--exclude=.venv", "--exclude=*.egg-info",
        "--exclude=__pycache__", "--exclude=.git",
        f"{Path(repo_dir).resolve()}/", remote,
    ]
    log.info("Pushing repo to pod ...")
    subprocess.run(cmd, check=True)


def setup_pod(
    pod: dict,
    *,
    ssh_key: str = "~/.ssh/id_rsa",
    env_vars: Optional[dict[str, str]] = None,
    volume_mount: str = "/workspace/cache",
) -> None:
    """Install deps, write .env, confirm GPU.

    env_vars should include OPENROUTER_API_KEY (and optionally HF_TOKEN).
    """
    key = str(Path(ssh_key).expanduser())
    ssh = _ssh(pod["_ssh_host"], pod["_ssh_port"], key)
    try:
        env_lines = "\n".join(f'{k}="{v}"' for k, v in (env_vars or {}).items())
        env_lines += f"\nCOTIM_ROOT={volume_mount}\n"

        cmds = [
            "cd /workspace/cotim && pip install -q uv",
            "cd /workspace/cotim && uv sync --extra gpu --extra openrouter --extra dev",
            f"cat > /workspace/cotim/.env << 'EOF'\n{env_lines}\nEOF",
            "nvidia-smi | head -4",
        ]
        for cmd in cmds:
            log.info("$ %s", cmd[:80])
            _, stdout, stderr = ssh.exec_command(cmd, get_pty=True)
            out = stdout.read().decode("utf-8", errors="replace")
            err = stderr.read().decode("utf-8", errors="replace")
            if out.strip():
                print(out, end="")
            if err.strip():
                print("[stderr]", err[:400], file=sys.stderr)
    finally:
        ssh.close()


def run_stage_remote(
    pod: dict,
    script_and_args: str,
    *,
    ssh_key: str = "~/.ssh/id_rsa",
) -> int:
    """SSH into the pod, run a script, stream output live. Returns the exit code."""
    key = str(Path(ssh_key).expanduser())
    ssh = _ssh(pod["_ssh_host"], pod["_ssh_port"], key)
    try:
        full_cmd = f"cd /workspace/cotim && source .env && {script_and_args}"
        log.info("Remote: %s", full_cmd[:120])
        _, stdout, stderr = ssh.exec_command(full_cmd, get_pty=True)
        for line in iter(stdout.readline, ""):
            print(line, end="", flush=True)
        exit_code = stdout.channel.recv_exit_status()
        if exit_code != 0:
            err = stderr.read().decode("utf-8", errors="replace")
            if err.strip():
                print("[stderr]", err[:1000], file=sys.stderr)
        return exit_code
    finally:
        ssh.close()


def pull_results(
    pod: dict,
    run_name: str,
    *,
    local_results_dir: str = "results",
    ssh_key: str = "~/.ssh/id_rsa",
) -> None:
    """Rsync results/<run_name>/ from the pod back to the local machine."""
    host = pod["_ssh_host"]
    port = pod["_ssh_port"]
    key = str(Path(ssh_key).expanduser())
    remote = f"root@{host}:/workspace/cotim/results/{run_name}/"
    local = str(Path(local_results_dir) / run_name / "")
    Path(local).mkdir(parents=True, exist_ok=True)
    cmd = [
        "rsync", "-az", "--progress",
        "-e", f"ssh -p {port} -i {key} -o StrictHostKeyChecking=no",
        remote, local,
    ]
    log.info("Pulling results/%s from pod ...", run_name)
    subprocess.run(cmd, check=True)


def terminate_pod(pod_id: str) -> None:
    rp = _runpod_client()
    rp.terminate_pod(pod_id)
    log.info("Pod %s terminated.", pod_id)
