"""Capacity-controlled binary DDP: the diffusion mechanism is unchanged."""

import torch
from torch import nn
from .model import SharedFrozenModel
from .ddp_model import DDPHead, PROTOCOL as FULL_PROTOCOL

PROTOCOL = dict(
    FULL_PROTOCOL,
    version='ddp-binary-mini-v1',
    width=128,
    layers=3,
    attention_heads=4,
    ffn=512,
    time_embedding='learned sinusoidal 16 -> 512 -> 512; input cosine logSNR',
    neck='FPN128 GN32 no activation, bilinear merge4levels stride4 -> Conv1x1 GN32',
    head_parameters=2117146,
)


class DDPModel(SharedFrozenModel):
    experiment_protocol = PROTOCOL

    def __init__(self, backbone_path, head_names=('ddp',), seed=42, backbone=None):
        assert list(head_names) == ['ddp']
        super().__init__(backbone_path, [], seed, backbone)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.heads = nn.ModuleDict({'ddp': DDPHead(width=128, layers=3, heads=4, time_dim=512)})
        assert sum(p.numel() for p in self.heads.parameters()) == 2117146
