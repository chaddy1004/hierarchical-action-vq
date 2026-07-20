"""
havq/utils/paths.py
==================
Canonical path conventions shared across stages. Defining these ONCE keeps the
feature-cache writer (preprocessing) and readers (training / tokenization) in
lockstep: if the naming scheme lives in two places it will eventually drift and
training will silently read the wrong cache.
"""

from __future__ import annotations


def feature_subdir(model: str, n_clip_frame: int, stride: int) -> str:
    """Subdirectory (under `paths.features_dir`) holding one feature cache.

    Encodes the settings that determine the cache's content so multiple
    backbones / temporal scales coexist. The preprocessing config supplies
    these as flat keys; the main config supplies them under `feature:` — both
    call this so the produced and queried paths are identical.
    """
    return f"{model}_clip{n_clip_frame}_stride{stride}"
