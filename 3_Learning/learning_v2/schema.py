"""Versioned graph/checkpoint contracts and leakage boundary."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch_geometric.data import Data


FEATURE_SCHEMA_VERSION = 2
CHECKPOINT_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class FeatureContract:
    schema_version: int
    graph_source: str
    feature_dim: int
    num_tags: int
    rendered_visual_feature_version: int
    feature_variant: str = "full"
    live_ax_feature_version: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "FeatureContract":
        return cls(**value)

    @classmethod
    def from_data(cls, data: Data, graph_source: str) -> "FeatureContract":
        if data.x.dim() != 2:
            raise ValueError(f"Expected x=[nodes, features], got {tuple(data.x.shape)}")
        if data.tag_indices.numel() != data.x.shape[0]:
            raise ValueError("tag_indices and x contain different node counts")
        return cls(
            schema_version=FEATURE_SCHEMA_VERSION,
            graph_source=graph_source,
            feature_dim=int(data.x.shape[1]),
            num_tags=max(116, int(data.tag_indices.max().item()) + 1 if data.tag_indices.numel() else 116),
            rendered_visual_feature_version=int(getattr(data, "rendered_visual_feature_version", 0)),
            feature_variant=str(getattr(data, "feature_variant", "full")),
            live_ax_feature_version=int(getattr(data, "live_ax_feature_version", 0)),
        )

    def validate(self, data: Data) -> None:
        actual_source = getattr(data, "graph_source", self.graph_source)
        if actual_source != self.graph_source:
            raise ValueError(f"Graph source mismatch: checkpoint={self.graph_source}, data={actual_source}")
        if data.x.dim() != 2 or int(data.x.shape[1]) != self.feature_dim:
            raise ValueError(f"Feature width mismatch: checkpoint={self.feature_dim}, data={tuple(data.x.shape)}")
        if data.tag_indices.numel() != data.x.shape[0]:
            raise ValueError("tag_indices and x contain different node counts")
        if data.tag_indices.numel() and int(data.tag_indices.max().item()) >= self.num_tags:
            raise ValueError("Graph contains a tag index outside the checkpoint vocabulary")
        actual_visual_version = int(getattr(data, "rendered_visual_feature_version", 0))
        if actual_visual_version != self.rendered_visual_feature_version:
            raise ValueError(
                "Rendered feature version mismatch: "
                f"checkpoint={self.rendered_visual_feature_version}, data={actual_visual_version}"
            )
        actual_variant = str(getattr(data, "feature_variant", "full"))
        if actual_variant != self.feature_variant:
            raise ValueError(f"Feature variant mismatch: checkpoint={self.feature_variant}, data={actual_variant}")
        actual_live_ax = int(getattr(data, "live_ax_feature_version", 0))
        if actual_live_ax != self.live_ax_feature_version:
            raise ValueError(f"Live AX feature version mismatch: checkpoint={self.live_ax_feature_version}, data={actual_live_ax}")


def inference_fingerprint(data: Data) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the only tensors a model may consume."""
    return data.x, data.edge_index, data.tag_indices
