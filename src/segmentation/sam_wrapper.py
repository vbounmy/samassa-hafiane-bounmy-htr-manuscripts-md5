"""Thin wrapper around Meta's Segment-Anything model (SAM ViT-B).

Loading SAM is expensive (~358 MB checkpoint, ~2 GB of RAM after init).
Doing it once per script is fine; doing it once per page is not. This
module therefore exposes a small class ``SamSegmenter`` that loads the
model in its constructor and offers a fast ``segment()`` method.

We downscale input images by a configurable factor before inference,
because SAM's automatic-mask-generator is O(N²) in the number of pixels.
At full e-NDP resolution (~3000x4000), one page can take >15 min on CPU;
at 1/4 resolution it drops to ~30 s while keeping enough detail to
distinguish text blocks from margins.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

ModelType = Literal["vit_b", "vit_l", "vit_h"]


class SamSegmenter:
    """Load Segment-Anything once and apply it to many pages.

    Args:
        checkpoint_path: Path to the SAM weights (``sam_vit_b_01ec64.pth``
            for the ViT-B variant we use in this project).
        model_type: Which SAM variant the checkpoint corresponds to.
            Must match the checkpoint filename: ``"vit_b"``, ``"vit_l"``
            or ``"vit_h"``.
        device: Torch device string. ``"cpu"`` is supported but ~30x slower
            than a T4 GPU; pass ``"cuda"`` on Colab.
        downscale: Integer factor by which the input image side is divided
            before inference. A value of 4 brings a 3000-px-wide page down
            to 750 px, dropping per-page latency from ~15 min to ~30 s
            on CPU while preserving the structure relevant to layout
            analysis. The output masks are upscaled back to the original
            resolution by :meth:`segment`.
        points_per_side: Number of grid points the automatic generator
            samples per image side. Lower values trade recall for speed.
    """

    def __init__(
        self,
        checkpoint_path: Path,
        *,
        model_type: ModelType = "vit_b",
        device: str = "cpu",
        downscale: int = 4,
        points_per_side: int = 16,
    ) -> None:
        self.downscale = downscale
        self.device = device
        sam = sam_model_registry[model_type](checkpoint=str(checkpoint_path))
        sam.to(device=device)
        self._generator = SamAutomaticMaskGenerator(
            sam,
            points_per_side=points_per_side,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            min_mask_region_area=2000,  # filters tiny artefacts
        )

    def segment(self, image_bgr: np.ndarray) -> list[dict]:
        """Run SAM on one page and return masks at full resolution.

        Args:
            image_bgr: ``(H, W, 3)`` BGR image as read by OpenCV.

        Returns:
            List of mask dicts (one per detected region). Each dict
            contains:

            - ``segmentation``: boolean ``(H, W)`` array at the **original**
              resolution, True inside the region.
            - ``bbox``: list ``[x, y, w, h]`` in original-resolution pixels.
            - ``area``: integer pixel count at original resolution.
            - ``predicted_iou``: SAM's own confidence score in [0, 1].
            - ``stability_score``: SAM's stability score in [0, 1].

            Masks are sorted from largest to smallest area.
        """
        orig_h, orig_w = image_bgr.shape[:2]

        # Downscale for inference, then convert BGR→RGB (SAM expects RGB).
        small = cv2.resize(
            image_bgr,
            (orig_w // self.downscale, orig_h // self.downscale),
            interpolation=cv2.INTER_AREA,
        )
        small_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        # SAM inference (no_grad keeps RAM usage in check on CPU).
        with torch.no_grad():
            raw_masks = self._generator.generate(small_rgb)

        # Upscale each mask back to the original resolution.
        rescaled: list[dict] = []
        for mask in raw_masks:
            small_seg = mask["segmentation"].astype(np.uint8)
            full_seg = cv2.resize(
                small_seg, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
            ).astype(bool)
            x_s, y_s, w_s, h_s = mask["bbox"]
            rescaled.append({
                "segmentation": full_seg,
                "bbox": [
                    x_s * self.downscale,
                    y_s * self.downscale,
                    w_s * self.downscale,
                    h_s * self.downscale,
                ],
                "area": int(full_seg.sum()),
                "predicted_iou": float(mask["predicted_iou"]),
                "stability_score": float(mask["stability_score"]),
            })

        rescaled.sort(key=lambda m: -m["area"])
        return rescaled
