"""DDP binary diffusion head on DINOv3 ViT-B/16 with two arms.

head_only : ViT-B fully frozen (eval, no graph). Trainable: pyramid adapter + DDP head.
llrd_dp   : same, plus the last 4 transformer blocks (9-12) trainable with layer-wise LR decay
            (0.65 per block going down) and drop path 0.1 inside those blocks only.

The adapter and the DDP head are constructed under the same seed BEFORE any arm-specific
configuration, so both arms start from byte-identical adapter/head weights.
"""
import torch
from torch import nn

from u0_multihead.ddp_model import DDPHead, DDPModel

TAPS = (3, 6, 9, 12)            # 1-indexed transformer blocks whose outputs feed the pyramid
TUNED_BLOCKS = 4
BACKBONE_LR = 2e-5              # LR of the top block (block 12) in llrd_dp, same as the refiner arm
LAYER_DECAY = 0.65
DROP_PATH = 0.1
HEAD_LR = 3e-4                  # DDP12 head LR
WEIGHT_DECAY = 0.01             # AdamW weight decay used by the recall707 runner (DDP12 uses the same preset)
ARMS = ('head_only', 'llrd_dp', 'convnext')
HIDDEN = 768


def _norm(c):
    return nn.GroupNorm(32, c)


class PyramidAdapter(nn.Module):
    """ViT tokens at stride 16 -> the 4 maps (128, 256, 512, 1024 ch, stride 4/8/16/32) DDPHead expects.

    Each level comes from its OWN transformer tap (ViTDet-style resampling, multi-tap):
      stride 4  <- tap block 3  : two stride-2 transposed convs
      stride 8  <- tap block 6  : one stride-2 transposed conv
      stride 16 <- tap block 9  : 1x1 conv
      stride 32 <- tap block 12 : 2x2 max-pool + 1x1 conv
    GroupNorm everywhere (fp16 safe, batch-size independent)."""
    def __init__(self, hidden=HIDDEN):
        super().__init__()
        self.s4 = nn.Sequential(nn.ConvTranspose2d(hidden, 256, 2, 2), _norm(256), nn.GELU(),
                                nn.ConvTranspose2d(256, 128, 2, 2), _norm(128))
        self.s8 = nn.Sequential(nn.ConvTranspose2d(hidden, 256, 2, 2), _norm(256))
        self.s16 = nn.Sequential(nn.Conv2d(hidden, 512, 1, bias=False), _norm(512))
        self.s32 = nn.Sequential(nn.MaxPool2d(2, 2), nn.Conv2d(hidden, 1024, 1, bias=False), _norm(1024))

    def forward(self, taps):
        if len(taps) != 4:
            raise ValueError('expected 4 taps (blocks 3, 6, 9, 12)')
        t3, t6, t9, t12 = taps
        return [self.s4(t3), self.s8(t6), self.s16(t9), self.s32(t12)]


def _block_list(vit):
    stacks = [m for m in vit.modules() if isinstance(m, nn.ModuleList) and len(m) == 12]
    assert len(stacks) == 1, 'could not find the 12-block transformer stack'
    return stacks[0]


class DDPViTModel(nn.Module):
    def __init__(self, backbone_path, arm, seed=42, drop_path=DROP_PATH):
        super().__init__()
        if arm not in ('head_only', 'llrd_dp'):
            raise ValueError(arm)
        from transformers import DINOv3ViTModel
        self.arm = arm
        self.drop_path = drop_path if arm == 'llrd_dp' else 0.
        extra = dict(drop_path_rate=self.drop_path) if self.drop_path > 0 else {}
        vit = DINOv3ViTModel.from_pretrained(str(backbone_path), local_files_only=True,
                                             attn_implementation='sdpa', **extra)
        c = vit.config
        assert c.hidden_size == HIDDEN and c.patch_size == 16 and c.num_hidden_layers == 12
        self.n_prefix = 1 + int(c.num_register_tokens)
        self.backbone = vit.requires_grad_(False).eval()
        if arm == 'llrd_dp':
            for block in self.blocks[12 - TUNED_BLOCKS:]:
                block.requires_grad_(True)
        # Adapter + head: identical init in both arms (same seed, built before arm-specific work above is used).
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.adapter = PyramidAdapter()
            self.heads = nn.ModuleDict({'ddp': DDPHead()})
        self._taps = {}
        for index in TAPS:
            self.blocks[index - 1].register_forward_hook(self._make_hook(index))

    @property
    def blocks(self):
        return _block_list(self.backbone)

    def _make_hook(self, index):
        def hook(_module, _inputs, output):
            self._taps[index] = output[0] if isinstance(output, (tuple, list)) else output
        return hook

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        if mode and self.arm == 'llrd_dp' and self.drop_path > 0:
            for block in self.blocks[12 - TUNED_BLOCKS:]:
                block.train(True)       # drop path only where weights train
        return self

    def tuned_blocks(self):
        return list(self.blocks[12 - TUNED_BLOCKS:]) if self.arm == 'llrd_dp' else []

    def features(self, x):
        """x: (B,3,H,W) ImageNet-normalised. Returns the 4 pyramid maps for DDPHead."""
        h, w = x.shape[-2:]
        gh, gw = h // 16, w // 16
        self._taps.clear()
        # Backbone in FP32 (no autocast). The frozen prefix builds no autograd graph because no input or
        # parameter there requires grad; the tuned blocks (llrd_dp) do.
        with torch.autocast(x.device.type, enabled=False):
            self.backbone(pixel_values=x.float())
            taps = []
            for index in TAPS:
                tokens = self.backbone.norm(self._taps[index])[:, -gh * gw:]
                taps.append(tokens.transpose(1, 2).reshape(x.shape[0], HIDDEN, gh, gw).contiguous())
        self._taps.clear()
        return self.adapter(taps)

    def forward(self, image, target):
        """Training objective logits (random diffusion time + noise inside DDPHead)."""
        return self.heads['ddp'].training_logits(self.features(image), image, target)

    # ---- optimisation -------------------------------------------------------------------------------
    def param_groups(self, head_lr=HEAD_LR):
        groups = [dict(params=list(self.adapter.parameters()) + list(self.heads.parameters()),
                       lr=head_lr, weight_decay=WEIGHT_DECAY, name='adapter_head')]
        for k, block in enumerate(reversed(self.tuned_blocks())):     # k=0 is block 12
            groups.append(dict(params=list(block.parameters()), lr=BACKBONE_LR * LAYER_DECAY ** k,
                               weight_decay=WEIGHT_DECAY, name=f'backbone_block{12 - k}'))
        return groups

    def trainable_state(self):
        """Everything that changes during training, with stable names (adapter, head, tuned blocks)."""
        state = {f'adapter.{k}': v.detach().cpu().clone() for k, v in self.adapter.state_dict().items()}
        state.update({f'heads.{k}': v.detach().cpu().clone() for k, v in self.heads.state_dict().items()})
        for i, block in zip(range(12 - TUNED_BLOCKS + 1, 13), self.tuned_blocks()):
            state.update({f'block{i}.{k}': v.detach().cpu().clone() for k, v in block.state_dict().items()})
        return state

    def load_trainable_state(self, state):
        expected = set(self.trainable_state())
        if set(state) != expected:
            raise KeyError(f'missing={sorted(expected - set(state))[:5]} unexpected={sorted(set(state) - expected)[:5]}')
        self.adapter.load_state_dict({k[8:]: v for k, v in state.items() if k.startswith('adapter.')}, strict=True)
        self.heads.load_state_dict({k[6:]: v for k, v in state.items() if k.startswith('heads.')}, strict=True)
        for i, block in zip(range(12 - TUNED_BLOCKS + 1, 13), self.tuned_blocks()):
            prefix = f'block{i}.'
            block.load_state_dict({k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}, strict=True)


def build_optimizer(model, total_updates, warmup=100, head_lr=HEAD_LR):
    from u0_multihead.model import lr_factor
    opt = torch.optim.AdamW(model.param_groups(head_lr))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda n: lr_factor(n, total_updates, warmup))
    return opt, sched


def trainable_parameters(model):
    return [p for p in model.parameters() if p.requires_grad]


class DDPConvNextModel(DDPModel):
    """The ORIGINAL DDP12 model: frozen DINOv3 ConvNeXt-B (4 native feature maps) + DDP head (11,564,106 params).

    Exposes the same training interface as DDPViTModel so the whole runner (data, D4 augmentation, loss, optimizer
    family, scoring) is shared. Only the head trains. arm = 'convnext'."""
    arm = 'convnext'

    def __init__(self, backbone_path, arm='convnext', seed=42, drop_path=0.0):
        assert arm == 'convnext', arm
        super().__init__(backbone_path, ('ddp',), seed)

    def tuned_blocks(self):
        return []

    def forward(self, image, target):
        return self.heads['ddp'].training_logits(self.features(image), image, target)

    def param_groups(self, head_lr=HEAD_LR):
        return [dict(params=list(self.heads.parameters()), lr=head_lr, weight_decay=WEIGHT_DECAY, name='head')]

    def trainable_state(self):
        return {f'heads.{k}': v.detach().cpu().clone() for k, v in self.heads.state_dict().items()}

    def load_trainable_state(self, state):
        expected = set(self.trainable_state())
        if set(state) != expected:
            raise KeyError(f'missing={sorted(expected - set(state))[:5]} unexpected={sorted(set(state) - expected)[:5]}')
        self.heads.load_state_dict({k[6:]: v for k, v in state.items()}, strict=True)


def make_model(backbone_path, arm, seed=42):
    if arm == 'convnext':
        return DDPConvNextModel(backbone_path, arm, seed)
    return DDPViTModel(backbone_path, arm, seed=seed)
