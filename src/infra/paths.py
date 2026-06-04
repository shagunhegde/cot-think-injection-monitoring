"""Environment-aware path resolution — the ONLY environment-aware code in the repo (§6.5.1).

`resolve_root()` returns a `Paths` dataclass pointing at the durable workspace:
  - On Colab: mount Drive once, root = /gdrive/MyDrive/cot-injection-monitoring
  - Locally:  root = $COTIM_ROOT, else ./workspace

Switching local <-> Colab changes nothing but this root. HF_HOME / HF_DATASETS_CACHE
are pointed at the workspace so model weights + datasets cache durably and survive
Colab restarts.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path

# Colab root is overridable for tests via COTIM_COLAB_ROOT (so we never need to mount
# a real Drive in CI / on this Mac).
_DEFAULT_COLAB_ROOT = "/gdrive/MyDrive/cot-injection-monitoring"
_DRIVE_MOUNT = "/gdrive"


@dataclass(frozen=True)
class Paths:
    """Durable workspace layout (§6.5.2). All artifacts live under these dirs."""

    root: Path
    data: Path
    baselines: Path
    prefills: Path
    cache: Path
    runs: Path
    figures: Path
    ledger: Path
    hf_cache: Path
    hf_home: Path

    def all_dirs(self) -> list[Path]:
        return [getattr(self, f.name) for f in fields(self)]


def is_colab() -> bool:
    """True iff running inside Google Colab. Patchable in tests."""
    try:
        import google.colab  # noqa: F401

        return True
    except ImportError:
        return False


def _mount_drive() -> None:  # pragma: no cover - only runs on real Colab
    from google.colab import drive

    drive.mount(_DRIVE_MOUNT)


def colab_root() -> Path:
    return Path(os.environ.get("COTIM_COLAB_ROOT", _DEFAULT_COLAB_ROOT))


def local_root() -> Path:
    return Path(os.environ.get("COTIM_ROOT", "./workspace")).resolve()


def build_paths(root: Path, *, create: bool = True, set_hf_env: bool = True) -> Paths:
    """Construct the Paths layout under `root`, creating dirs and wiring HF caches."""
    root = Path(root)
    paths = Paths(
        root=root,
        data=root / "data",
        baselines=root / "baselines",
        prefills=root / "prefills",
        cache=root / "cache",
        runs=root / "runs",
        figures=root / "figures",
        ledger=root / "ledger",
        hf_cache=root / "data" / "hf_cache",
        hf_home=root / "data" / "hf_home",
    )
    if create:
        for d in paths.all_dirs():
            d.mkdir(parents=True, exist_ok=True)
    if set_hf_env:
        # setdefault: respect an explicit override from the environment.
        os.environ.setdefault("HF_HOME", str(paths.hf_home))
        os.environ.setdefault("HF_DATASETS_CACHE", str(paths.hf_cache))
    return paths


def resolve_root(*, mount: bool = True, create: bool = True) -> Paths:
    """Resolve the workspace root for the current environment and return Paths."""
    if is_colab():
        if mount:
            _mount_drive()
        root = colab_root()
    else:
        root = local_root()
    return build_paths(root, create=create)
