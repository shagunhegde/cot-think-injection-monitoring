"""Tests for resolve_root: local and mocked-Colab paths."""
import os
import importlib
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from src.infra import paths as paths_mod
from src.infra.paths import build_paths, local_root, colab_root


def test_local_root_uses_cotim_root_env(tmp_path, monkeypatch):
    monkeypatch.setenv("COTIM_ROOT", str(tmp_path))
    assert local_root() == tmp_path


def test_local_root_default_is_workspace(monkeypatch):
    monkeypatch.delenv("COTIM_ROOT", raising=False)
    r = local_root()
    assert r.name == "workspace"


def test_colab_root_uses_env(monkeypatch):
    monkeypatch.setenv("COTIM_COLAB_ROOT", "/fake/drive/root")
    assert colab_root() == Path("/fake/drive/root")


def test_build_paths_creates_dirs(tmp_path):
    p = build_paths(tmp_path, create=True, set_hf_env=False)
    for d in p.all_dirs():
        assert d.exists(), f"missing dir: {d}"


def test_build_paths_no_create_skips_dirs(tmp_path):
    subdir = tmp_path / "fresh_root"
    p = build_paths(subdir, create=False, set_hf_env=False)
    # The root itself shouldn't be created by build_paths when create=False.
    assert not subdir.exists()


def test_build_paths_sets_hf_env(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("HF_DATASETS_CACHE", raising=False)
    p = build_paths(tmp_path, create=True, set_hf_env=True)
    assert os.environ.get("HF_HOME") == str(p.hf_home)
    assert os.environ.get("HF_DATASETS_CACHE") == str(p.hf_cache)


def test_resolve_root_local(tmp_path, monkeypatch):
    monkeypatch.setenv("COTIM_ROOT", str(tmp_path / "ws"))
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("HF_DATASETS_CACHE", raising=False)

    with mock.patch.object(paths_mod, "is_colab", return_value=False):
        p = paths_mod.resolve_root(mount=False, create=True)

    assert p.root == (tmp_path / "ws").resolve()
    assert p.root.exists()


def test_resolve_root_mocked_colab(tmp_path, monkeypatch):
    monkeypatch.setenv("COTIM_COLAB_ROOT", str(tmp_path / "drive_root"))
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("HF_DATASETS_CACHE", raising=False)

    with mock.patch.object(paths_mod, "is_colab", return_value=True), \
         mock.patch.object(paths_mod, "_mount_drive"):
        p = paths_mod.resolve_root(mount=True, create=True)

    assert p.root == tmp_path / "drive_root"
    assert p.root.exists()
