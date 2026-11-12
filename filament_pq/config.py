"""Every fixed constant of the inference pipeline in one place.

None of these values was tuned on the test set. Thresholds marked "fixed" were set before validation and never
changed; the post-processing rule (linear resample, mask threshold .5, D4 TTA x8) was chosen by 5-fold cross-validation
on the 707 labelled training images (see docs/RESULTS.md).
"""

from dataclasses import dataclass, field

IMAGE_SIZE = 2048  # every GONG H-alpha image of MAGFiLO is 2048 x 2048 grayscale
N_FOLDS = 5  # models are trained 5 times, each time holding out one date-disjoint fold


@dataclass(frozen=True)
class ProposalConfig:
    """Stage 1: high-recall filament probability map from a frozen DINOv3 ViT-S+/16 and a small trained head."""

    tile: int = 768  # tile side in pixels
    tile_starts: tuple = (0, 640, 1280)  # 3 x 3 overlapping tiles cover the 2048 image (128 px overlap)
    window_floor: float = 0.05  # Hann blending window, floored so tile borders still count
    mean: tuple = (0.485, 0.456, 0.406)  # ImageNet normalisation (DINOv3 pre-training)
    std: tuple = (0.229, 0.224, 0.225)
    tile_batch: int = 9


@dataclass(frozen=True)
class ComponentConfig:
    """Stage 2: connected components of the low-threshold proposal map, scored by a LightGBM classifier."""

    threshold: float = 0.05  # fixed: favours recall (pixel recall ~.91 at this level)
    min_area: int = 8  # pixels; smaller blobs are noise
    ml_threshold: float = 0.5  # fixed: keep a component when the mean classifier score >= .5
    crop_pad: int = 128  # context around the component bounding box


@dataclass(frozen=True)
class RefinerConfig:
    """Stage 3: prompt-conditioned instance refiner (DINOv3 ViT-B/16, last 4 blocks fine-tuned, recurrent decoder)."""

    side: int = 512  # crops are letterboxed into 512 x 512
    rounds: int = 3  # recurrent refinement rounds; round 3 is used
    tta: int = 8  # dihedral group D4: 4 rotations x optional flip
    batch: int = 16


@dataclass(frozen=True)
class PostprocessConfig:
    """Post-processing: per-crop masks back to the 2048 image, then de-duplication into exclusive instances."""

    presence_threshold: float = 0.5  # crop kept when the refiner's presence (round 3) >= .5
    mask_threshold: float = 0.5  # on the probability resampled (bilinear) to native resolution
    min_area: int = 128  # per-candidate minimum area (pixels)
    nms_iou: float = 0.5  # drop a candidate overlapping an accepted instance with IoU > .5
    final_area: int = 128  # minimum area after removing pixels already taken


@dataclass(frozen=True)
class PipelineConfig:
    proposal: ProposalConfig = field(default_factory=ProposalConfig)
    components: ComponentConfig = field(default_factory=ComponentConfig)
    refiner: RefinerConfig = field(default_factory=RefinerConfig)
    postprocess: PostprocessConfig = field(default_factory=PostprocessConfig)
    folds: tuple = tuple(range(N_FOLDS))  # all 5 fold models are averaged at test time
