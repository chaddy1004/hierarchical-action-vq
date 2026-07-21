"""
havq/model/laq/
===============
Vendored copy of LAPA's LAQ inverse-dynamics model (the pretrained latent-action
tokenizer we use instead of training our own NSVQ).

Provenance: LatentActionPretraining/LAPA, `laq/laq_model/`, at commit
6a2dfb9877f8f5d45acd8ea4f91cb502f1d9c9b3 (via the fork github.com/jleem99/LAPA,
checked out at ~/dev/latent-action/external/LAPA). Copied rather than imported
across repos, per STYLE.md: each repo stands alone.

Only the three INFERENCE files are vendored -- attention.py, nsvq.py,
latent_action_quantization.py -- and they are byte-identical to upstream except
for the two `from laq_model.` -> `from havq.model.laq.` import lines in
latent_action_quantization.py, so diffing against upstream stays trivial.

Upstream's own `laq_model/__init__.py` also imports `laq_trainer`, which drags in
accelerate / ema-pytorch / wandb / transformers (and, in LAPA's own environment,
tensorflow -- which segfaults when co-loaded with torch on a GPU node). Dropping
the trainer, which we never call, cuts the dependency chain to torch + einops +
beartype and removes that failure mode entirely.

The checkpoint (`laq_openx.pt`, 1.4 GB) is NOT in this repo; its path is a config
value (`checkpoint` in the lapa preprocessing configs). Download it with:

    wget https://huggingface.co/latent-action-pretraining/LAPA-7B-openx/resolve/main/laq_openx.pt

Its architecture hyperparameters are fixed by the checkpoint, not free to choose
-- see havq/data/preprocessing/extractors/lapa_extractor.py.
"""

from havq.model.laq.latent_action_quantization import LatentActionQuantization

__all__ = ["LatentActionQuantization"]
