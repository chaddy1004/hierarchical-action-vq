"""
havq/data/preprocessing/extractors/vjepa_extractor.py
====================================================
V-JEPA 2 clip feature extractor. Copied and adapted from stepsegmenter's
VJepaExtractor; this repo stays code-independent.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from transformers import AutoModel, AutoVideoProcessor

from havq.data.preprocessing.extractors.base_extractor import BaseVideoExtractor

logger = logging.getLogger(__name__)


class VJepaExtractor(BaseVideoExtractor):
    """V-JEPA2 feature extractor using HuggingFace AutoModel."""

    def __init__(self, model_path: str, device: str = "cuda"):
        self.device = device
        self.processor = AutoVideoProcessor.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(device).eval()
        logger.info(f"V-JEPA extractor loaded from {model_path}")

    def extract_feature(self, video: np.ndarray) -> np.ndarray:
        """
        Args:
            video: T x H x W x C uint8 numpy array
        Returns:
            mean-pooled clip feature as numpy array, shape (D,)
        """
        video_tensor = torch.from_numpy(video).permute(0, 3, 1, 2)  # T x C x H x W
        inputs = self.processor(video_tensor, return_tensors="pt")["pixel_values_videos"]
        inputs = inputs.to(self.device)
        del video_tensor
        with torch.inference_mode():
            features = self.model.get_vision_features(inputs)
        del inputs
        emb = features.cpu().numpy()
        del features
        if emb.ndim > 1:
            emb = emb.mean(axis=tuple(range(emb.ndim - 1)))
        return emb
