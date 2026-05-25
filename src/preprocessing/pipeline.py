"""Image preprocessing pipeline for e-NDP manuscript pages (Étape 2.1).

Three transformations are applied in sequence, each tackling a distinct
challenge of historical document images:

1. ``deskew`` — corrects the small rotation introduced by digitization.
2. ``apply_clahe`` — enhances local contrast (Contrast-Limited Adaptive
   Histogram Equalization), which compensates for uneven lighting and faded
   ink without amplifying noise the way a global histogram stretch would.
3. ``binarize_sauvola`` — converts the image to black-on-white using a
   locally adaptive threshold (Sauvola 1997), the de-facto baseline for
   historical document binarization. It tolerates background staining and
   parchment yellowing far better than Otsu's global thresholding.

The project brief also asks for "gestion des zones enluminées". e-NDP being
a corpus of administrative registers without illumination, this step is
documented in the report as not applicable, and our pipeline instead
verifies behaviour on the three e-NDP zone types: block, liste, numérotation.

All functions are pure: they take a NumPy array in, return a new NumPy
array out, and never mutate their input. The pipeline can therefore be
reconfigured (e.g. for ablation studies).
"""

from __future__ import annotations

import cv2
import numpy as np
from skimage.filters import threshold_sauvola


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert an RGB image to single-channel grayscale.

    Args:
        image: Input image as a ``(H, W, 3)`` BGR/RGB array, or already
            a ``(H, W)`` grayscale array (returned as-is).

    Returns:
        Single-channel ``(H, W)`` grayscale image, dtype uint8.
    """
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def estimate_skew_angle(image: np.ndarray, max_angle: float = 10.0) -> float:
    """Estimate the rotation angle of a document page in degrees.

    The image is binarized with Otsu and a minimum-area bounding rectangle
    is fit around the foreground pixels. The angle of that rectangle is
    a robust estimate for small skew angles typical of book scans.

    Args:
        image: Grayscale image as a ``(H, W)`` uint8 array.
        max_angle: Clamp value in degrees. Estimates larger than this
            in absolute value are returned as 0 because they almost always
            reflect detection failures (e.g. portrait/landscape ambiguity).

    Returns:
        Estimated skew angle in degrees. Positive means the page is
        rotated counter-clockwise.
    """
    _, binary = cv2.threshold(
        image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )
    coords = np.column_stack(np.where(binary > 0))
    if coords.size == 0:
        return 0.0
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle
    if abs(angle) > max_angle:
        return 0.0
    return float(angle)


def deskew(image: np.ndarray, max_angle: float = 10.0) -> np.ndarray:
    """Rotate a page image to compensate for digitization skew.

    Args:
        image: Grayscale ``(H, W)`` uint8 array.
        max_angle: Clamp value passed to :func:`estimate_skew_angle`.

    Returns:
        Deskewed image, same shape and dtype as the input.
    """
    angle = estimate_skew_angle(image, max_angle=max_angle)
    if abs(angle) < 0.05:  # below 1/20 degree, rotation is wasted compute
        return image
    h, w = image.shape[:2]
    rot_mat = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(
        image, rot_mat, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Enhance local contrast with CLAHE.

    Args:
        image: Grayscale ``(H, W)`` uint8 array.
        clip_limit: Clipping threshold of the local histogram. Higher
            values yield stronger contrast at the cost of noise amplification.
            2.0 is the OpenCV default and works well on parchment scans.
        tile_grid_size: Number of tiles in (rows, cols). Smaller tiles =
            more local adaptation. ``(8, 8)`` is the standard starting point.

    Returns:
        Contrast-enhanced grayscale image, same shape and dtype.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)


def binarize_sauvola(
    image: np.ndarray,
    window_size: int = 25,
    k: float = 0.2,
) -> np.ndarray:
    """Convert an image to black-and-white using Sauvola's local thresholding.

    Sauvola adapts the threshold to the local mean and standard deviation
    of pixel intensities in a sliding window, which handles uneven backgrounds
    far better than a global threshold.

    Args:
        image: Grayscale ``(H, W)`` uint8 array.
        window_size: Side of the square sliding window in pixels (must be
            odd). 25 is a sane default for ~3000 px wide scans.
        k: Sauvola sensitivity parameter, in [0, 1]. Higher values produce
            thinner strokes; 0.2 is the original recommendation.

    Returns:
        Binarized image as ``(H, W)`` uint8 array where ink is 0 and
        background is 255.
    """
    if window_size % 2 == 0:
        window_size += 1
    threshold = threshold_sauvola(image, window_size=window_size, k=k)
    binary = (image > threshold).astype(np.uint8) * 255
    return binary


def preprocess_page(
    image: np.ndarray,
    *,
    do_deskew: bool = True,
    clahe_clip_limit: float = 2.0,
    sauvola_window: int = 25,
    sauvola_k: float = 0.2,
) -> np.ndarray:
    """Apply the full preprocessing pipeline to a manuscript page image.

    The pipeline is grayscale → optional deskew → CLAHE → Sauvola.

    Args:
        image: RGB or grayscale ``(H, W[, 3])`` uint8 array.
        do_deskew: Whether to apply the deskew step. Disabling it is useful
            for ablation studies and for pages with known correct orientation.
        clahe_clip_limit: See :func:`apply_clahe`.
        sauvola_window: See :func:`binarize_sauvola`.
        sauvola_k: See :func:`binarize_sauvola`.

    Returns:
        Binarized ``(H, W)`` uint8 array (0 = ink, 255 = background).
    """
    gray = to_grayscale(image)
    if do_deskew:
        gray = deskew(gray)
    contrast_enhanced = apply_clahe(gray, clip_limit=clahe_clip_limit)
    binary = binarize_sauvola(
        contrast_enhanced,
        window_size=sauvola_window,
        k=sauvola_k,
    )
    return binary
