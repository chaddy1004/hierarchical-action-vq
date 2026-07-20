"""
havq/data/preprocessing/extractors/base_extractor.py
===================================================
Interface every clip feature extractor implements, so the backbone (V-JEPA,
Omnivore, LaVILA, ...) is swappable behind one factory in Preprocessor.
"""

from __future__ import annotations

import abc

import numpy as np


class BaseVideoExtractor(abc.ABC):
    """Maps one clip (T x H x W x C uint8) to one pooled embedding (D,)."""

    @abc.abstractmethod
    def extract_feature(self, video_clip: np.ndarray) -> np.ndarray:
        raise NotImplementedError
