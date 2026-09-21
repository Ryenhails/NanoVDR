"""Learned per-token weighting for ColNanoVDR query towers.

The tower is a ColBERT-style stack with one extra piece. A linear projection maps
backbone hidden states to the teacher's token width and each token is L2-normalised,
which stock ``Dense`` and ``Normalize`` already do. On top of that, a weight head
reads the *pre-projection* hidden states and produces one logit per token; a softmax
over the scoring tokens turns those into weights that sum to one.

The weights are folded into the vectors rather than carried alongside them::

    v_i <- w_i * normalize(W h_i)

so a plain MaxSim consumer reproduces ``sum_i w_i max_d <s_i, d>`` with no
weight-aware scoring code. The emitted vectors are therefore *not* unit length,
which is intended. Pair with ``similarity_fn_name="meanmaxsim"`` to match how the
tower was evaluated.

This module also writes the scoring mask into ``attention_mask``: real tokens minus
the tokenizer's special tokens, which is the mask the softmax normalises over.
"""

from __future__ import annotations

import os
from typing import Any

import torch
from torch import nn

from sentence_transformers.base.modules import Module


class ColNanoVDRWeighting(Module):
    config_keys: list[str] = [
        "hidden_dim",
        "hidden_name",
        "mv_name",
        "learn_weights",
        "special_token_ids",
    ]
    config_file_name: str = "config.json"

    def __init__(
        self,
        hidden_dim: int,
        hidden_name: str = "token_embeddings",
        mv_name: str = "mv_embeddings",
        learn_weights: bool = True,
        special_token_ids: list[int] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.hidden_name = hidden_name
        self.mv_name = mv_name
        self.learn_weights = bool(learn_weights)
        self.special_token_ids = list(special_token_ids or [])
        self.weight_head = nn.Linear(self.hidden_dim, 1) if self.learn_weights else None

    def _scoring_mask(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        mask = features["attention_mask"].bool()
        ids = features.get("input_ids")
        if ids is not None:
            for sid in self.special_token_ids:
                mask = mask & (ids != sid)
        return mask

    def forward(self, features: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        mv = features[self.mv_name]
        mask = self._scoring_mask(features)

        if self.weight_head is not None:
            hidden = features[self.hidden_name]
            logits = self.weight_head(hidden).squeeze(-1)
            logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
            w = torch.softmax(logits, dim=1)
            mv = mv * w.unsqueeze(-1)

        mv = mv * mask.unsqueeze(-1).to(mv.dtype)
        features["token_embeddings"] = mv
        features["attention_mask"] = mask.to(features["attention_mask"].dtype)
        return features

    def get_config_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.config_keys}

    def save(self, output_path: str, *args, safe_serialization: bool = True, **kwargs) -> None:
        self.save_config(output_path)
        if self.weight_head is not None:
            self.save_torch_weights(output_path, safe_serialization=safe_serialization)

    @classmethod
    def load(cls, model_name_or_path: str, subfolder: str = "", **kwargs) -> "ColNanoVDRWeighting":
        hub_kwargs = {k: kwargs.get(k) for k in
                      ("token", "cache_folder", "revision", "local_files_only")}
        config = cls.load_config(model_name_or_path=model_name_or_path, subfolder=subfolder,
                                 **hub_kwargs)
        module = cls(**config)
        if module.weight_head is not None:
            module = module.load_torch_weights(
                model_name_or_path=model_name_or_path, subfolder=subfolder, model=module,
                **hub_kwargs)
        return module
