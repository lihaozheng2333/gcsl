"""Multi-subspace adaptation of the CLIP vision encoder."""

import torch
from torch import nn
from transformers import CLIPVisionModel

from .layers import GCRLinear


class GCSLDetector(nn.Module):
    def __init__(self, backbone="openai/clip-vit-large-patch14", ranks=(4, 2, 1), nst=True):
        super().__init__()
        self.backbone_name = backbone
        self.ranks = tuple(ranks)
        self.nst = nst
        self.backbone = CLIPVisionModel.from_pretrained(backbone)
        self.backbone.requires_grad_(False)
        # Replace only Q/K/V/O projections in visual Transformer self-attention.
        for layer in self.backbone.vision_model.encoder.layers:
            for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
                original = getattr(layer.self_attn, name)
                setattr(layer.self_attn, name, GCRLinear(original, self.ranks))
        self.head = nn.Linear(self.backbone.config.hidden_size, 2)
        self.prefix = len(self.ranks)

    def adapted_layers(self):
        return [m for m in self.backbone.modules() if isinstance(m, GCRLinear)]

    def set_prefix(self, prefix):
        for layer in self.adapted_layers():
            layer.set_prefix(prefix)
        self.prefix = prefix

    def sample_prefix(self):
        # Sample once per batch and share the prefix across all projections.
        prefix = int(torch.randint(1, len(self.ranks) + 1, ()).item()) if self.nst else len(self.ranks)
        self.set_prefix(prefix)
        return prefix

    def forward(self, images):
        # Always use the full prefix during inference.
        if not self.training:
            self.set_prefix(len(self.ranks))
        features = self.backbone(pixel_values=images).pooler_output
        return self.head(features)

    def orthogonality_loss(self):
        # Compute spectral-basis regularization outside mixed precision.
        with torch.autocast(device_type=self.head.weight.device.type, enabled=False):
            return torch.stack([m.orthogonality_loss() for m in self.adapted_layers()]).mean()

    def adapter_state_dict(self):
        trainable = {name for name, param in self.named_parameters() if param.requires_grad}
        return {name: value.detach().cpu() for name, value in self.state_dict().items() if name in trainable}

    def load_adapter(self, state):
        expected = set(self.adapter_state_dict())
        if set(state) != expected:
            raise ValueError("Adapter parameter keys do not match; check the backbone and rank configuration")
        self.load_state_dict(state, strict=False)
