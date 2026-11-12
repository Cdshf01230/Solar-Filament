"""Solar filament instance segmentation for the Solar Filament Segmentation Challenge 2026 (MAGFiLO, GONG H-alpha).

Pipeline: proposal map (DINOv3 ViT-S+ head, 5-fold mean) -> components + LightGBM filter -> prompt-conditioned
recurrent refiner (DINOv3 ViT-B, 5-fold mean, D4 TTA) -> native-resolution post-processing -> pixel-disjoint instances.
See docs/PIPELINE.md.
"""

__version__ = '1.0.0'
