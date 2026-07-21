"""
havq/data/preprocessing/extractors/lapa_extractor.py
====================================================
LAPA latent-action extractor: a PAIR of frames -> discrete action codes.

This deliberately does NOT implement BaseVideoExtractor. That interface maps one
clip (T x H x W x C) to one pooled embedding (D,); LAPA maps two frames to
`code_seq_len` discrete codebook indices. Different arity, different output type
-- forcing it behind the same abstract method would only obscure that.

What it replaces: stages (1) + (3) of CODE_MAP.md at once. V-JEPA gave continuous
clip features that we then had to train an NSVQ on to discretize; LAPA's LAQ is a
latent-action tokenizer that is ALREADY trained (on Open X-Embodiment robot
video), so frames go straight to discrete action tokens with no training step.

The model is the vendored LAQ inverse-dynamics model (havq/model/laq/). For one
pair it: patch-embeds both frames, jointly encodes them (spatial then temporal
attention), down-projects and spatially downsamples each to a `code_seq_len`-cell
grid, SUBTRACTS them, and snaps each cell to its nearest of `codebook_size`
codebook entries. The subtraction is why a code describes the transition (the
action) rather than the scene. Frame order matters: first = before, second =
after.

    extract(first, second) -> (codes, embedding)
        codes:     (B, code_seq_len) int64, values in [0, codebook_size)
        embedding: (B, code_seq_len * quant_dim) float32, the PRE-quantization
                   delta -- continuous, one vector per pair
    pack_symbols(codes, codebook_size) -> (B,) int64, one symbol per pair

Both come out of the SAME forward pass; `embedding` is simply read off one step
before the nearest-neighbour snap, which upstream's inference() discards.

`embedding` is the payload for this project. LAPA's own quantization is a
12-bit bottleneck (code_seq_len=4 cells x log2(codebook_size)=3 bits), sized so a
robot policy can emit actions autoregressively -- not for finding structure. The
downstream alphabet is built by clustering `embedding` instead, at a K chosen for
BPE rather than inherited from LAPA.

`pack_symbols` is kept for reference and diagnostics: it reads the per-cell codes
as digits of a base-`codebook_size` number, collapsing the cell grid into ONE
integer (8^4 = 4096 possible words with the openx checkpoint). Nothing downstream
requires it.

Math mirrors LatentActionQuantization.inference + NSVQ.inference exactly, but
re-does the (few-line) difference and nearest-neighbour lookup inline so codes and
embedding come back from a single encode pass -- upstream's inference() would
decode pixels we never look at. Adapted from ~/dev/latent-action/tools/
latent-action-analysis/idm.py (copied, not imported: each repo stands alone).
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from einops import pack
from PIL import Image
from torchvision import transforms as T

from havq.model.laq import LatentActionQuantization

logger = logging.getLogger(__name__)


# Architecture is fixed BY THE CHECKPOINT, not free to choose: these are the
# openx LAQ values (LAPA/README.md, and laq/inference_sthv2.py's own model
# construction). Wrong values -> state_dict shape mismatch at load.
LAPA_ARCH = {
    "dim": 1024,
    "quant_dim": 32,
    "codebook_size": 8,
    "image_size": 256,
    "patch_size": 32,
    "spatial_depth": 8,
    "temporal_depth": 8,
    "dim_head": 64,
    "heads": 16,
    "code_seq_len": 4,
}

# Exactly LAPA's own inference transform (laq/inference_sthv2.py). Frames arriving
# from PyAV are already rgb24, so the RGB conversion upstream does is a no-op here
# and is dropped; resize to 256x256 and scale to [0,1] are what the checkpoint saw.
FRAME_TRANSFORM = T.Compose([
    T.Resize((LAPA_ARCH["image_size"], LAPA_ARCH["image_size"])),
    T.ToTensor(),
])


def pack_symbols(codes: np.ndarray, codebook_size: int) -> np.ndarray:
    """(B, code_seq_len) per-cell codes -> (B,) one integer symbol per pair.

    Reads the cells as digits of a base-`codebook_size` number, so distinct cell
    combinations always map to distinct symbols and the mapping is invertible.
    Cell ORDER is kept fixed (the grid's raster order) -- two pairs get the same
    symbol only if every cell agrees.
    """
    n_cells = codes.shape[1]
    place_values = codebook_size ** np.arange(n_cells - 1, -1, -1)
    return (codes * place_values).sum(axis=1).astype(np.int64)


class LapaExtractor:
    """Frozen LAPA LAQ tokenizer. The checkpoint is loaded lazily on first use."""

    def __init__(self, checkpoint: str, device: str = "cuda"):
        self.checkpoint = checkpoint
        self.device = device
        self._model = None

    @property
    def codebook_size(self) -> int:
        return LAPA_ARCH["codebook_size"]

    @property
    def code_seq_len(self) -> int:
        return LAPA_ARCH["code_seq_len"]

    @property
    def vocab_size(self) -> int:
        """Number of distinct symbols pack_symbols can produce."""
        return self.codebook_size ** self.code_seq_len

    @property
    def model(self) -> LatentActionQuantization:
        if self._model is None:
            self._model = self.load_model()
        return self._model

    def load_model(self) -> LatentActionQuantization:
        """Build the LAQ architecture and load the checkpoint onto self.device.

        NOTE: LatentActionQuantization hardcodes device='cuda' when it constructs
        its NSVQ (havq/model/laq/latent_action_quantization.py), so construction
        itself needs a visible CUDA device regardless of self.device. Only that
        one line would need patching to run on CPU; we have GPUs, so it stands.
        """
        model = LatentActionQuantization(**LAPA_ARCH)

        state = torch.load(self.checkpoint, map_location="cpu")
        # checkpoint was saved from a DDP-wrapped model
        state = {k.replace("module.", ""): v for k, v in state.items()}
        # the class overrides load_state_dict to force strict=False, so a shape or
        # naming mismatch would load SILENTLY -- check what it skipped, loudly.
        missing, unexpected = model.load_state_dict(state)
        if missing or unexpected:
            raise RuntimeError(
                f"LAQ checkpoint does not match the architecture: "
                f"{len(missing)} missing / {len(unexpected)} unexpected keys "
                f"(first missing: {missing[:3]}, first unexpected: {unexpected[:3]})"
            )

        model = model.to(self.device).eval()
        model.vq.device = torch.device(self.device)
        logger.info(f"LAPA LAQ loaded from {self.checkpoint} onto {self.device}")
        return model

    def preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """One H x W x 3 uint8 frame -> (3, 256, 256) float tensor on the device.

        Kept separate from extract_codes on purpose: with back-to-back pairing
        every frame is the 'after' of one pair and the 'before' of the next, so
        the caller transforms each frame ONCE and reuses the tensor for both.
        """
        return FRAME_TRANSFORM(Image.fromarray(frame)).to(self.device)

    @property
    def embedding_dim(self) -> int:
        """Width of the flattened pre-quantization delta."""
        return LAPA_ARCH["code_seq_len"] * LAPA_ARCH["quant_dim"]

    @torch.no_grad()
    def extract(self, first: torch.Tensor, second: torch.Tensor):
        """Batched frame pairs -> (codes, embedding).

        first, second: (B, 3, 256, 256) float tensors, already on the device
        (the output of preprocess_frame, stacked). first = before, second = after.

        codes:     (B, code_seq_len) int64
        embedding: (B, code_seq_len * quant_dim) float32, pre-quantization
        """
        model = self.model
        # LAQ reads a pair as a 2-frame video: (B, 3, 2, 256, 256), channel before
        # time -- the nn.Conv3d convention its Phenaki/CViViT ancestor was built on.
        video = torch.stack([first, second], dim=2)

        first_frame, rest_frames = video[:, :, :1], video[:, :, 1:]
        tokens = torch.cat(
            (
                model.to_patch_emb_first_frame(first_frame),
                model.to_patch_emb_first_frame(rest_frames),
            ),
            dim=1,
        )

        first_tokens, last_tokens = model.encode(tokens)
        first_tokens, _ = pack([first_tokens], "b * d")
        last_tokens, _ = pack([last_tokens], "b * d")

        batch_size = first_tokens.shape[0]
        vq = model.vq
        encoded_first = vq.encode(first_tokens, batch_size)
        encoded_last = vq.encode(last_tokens, batch_size)
        # the action: difference of the two encoded, downsampled frames
        diff = (encoded_last - encoded_first).reshape(-1, vq.embedding_dim)

        # nearest codebook entry per cell, via the expanded squared distance
        codebooks = vq.codebooks
        distances = (
            diff.pow(2).sum(1, keepdim=True)
            - 2 * diff @ codebooks.t()
            + codebooks.t().pow(2).sum(0, keepdim=True)
        )
        indices = distances.argmin(1)

        codes = indices.reshape(batch_size, model.code_seq_len).cpu().numpy().astype(np.int64)
        # flatten the cell grid so one pair is one vector; cell order is the same
        # raster order pack_symbols uses, so cell c occupies dims [c*quant_dim,
        # (c+1)*quant_dim) and the two representations stay aligned.
        embedding = diff.reshape(batch_size, self.embedding_dim).cpu().numpy().astype(np.float32)
        return codes, embedding
