"""Происхождение прогона: хеш коммита и версии ключевых библиотек.

Пишется в каждый чекпойнт (см. mayak.lit.RUN_KEY). Сбой git (нет репозитория,
нет git в PATH) не роняет обучение: поле становится None, и это видно в записи.
"""
from __future__ import annotations

import os
import platform
import subprocess
from datetime import datetime, timezone
from functools import lru_cache
from importlib import metadata

KEY_LIBRARIES = ("torch", "pytorch-lightning", "numpy", "scipy", "pandas", "onnx", "onnxruntime",
                 "hydra-core", "omegaconf")
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _git(*args, cwd=REPO_ROOT):
    try:
        out = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


@lru_cache(maxsize=1)
def git_info():
    """{commit, dirty}: dirty - есть незакоммиченные изменения отслеживаемых файлов."""
    commit = _git("rev-parse", "HEAD")
    if commit is None:
        return dict(commit=None, dirty=None)
    status = _git("status", "--porcelain", "--untracked-files=no")
    return dict(commit=commit, dirty=None if status is None else bool(status))


def _version(dist):
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None


@lru_cache(maxsize=1)
def library_versions():
    out = {"python": platform.python_version(), "mayak": _version("mayak")}
    out.update({lib: _version(lib) for lib in KEY_LIBRARIES})
    return out


def provenance():
    """Запись о происхождении: коммит, версии, момент создания (UTC)."""
    return dict(git=dict(git_info()), versions=dict(library_versions()),
                created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"))


__all__ = ["git_info", "library_versions", "provenance"]
