"""Shared image and mask helpers for the CTC viewer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtGui import QColor, QImage, QPixmap


def qimage_size(path: Path) -> tuple[int, int] | None:
    image = QImage(str(path))
    if image.isNull():
        return None
    return image.width(), image.height()


def mask_to_pixmap(mask: np.ndarray, color: QColor, opacity: float) -> QPixmap:
    if mask.ndim == 3:
        mask = mask.max(axis=0)
    mask = np.asarray(mask)
    if mask.dtype != np.uint8:
        mask = np.clip(mask, 0, 255).astype(np.uint8)

    height, width = mask.shape[:2]
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[:, :, 0] = color.red()
    rgba[:, :, 1] = color.green()
    rgba[:, :, 2] = color.blue()
    rgba[:, :, 3] = np.clip(mask.astype(np.float32) * opacity, 0, 255).astype(np.uint8)
    image = QImage(rgba.data, width, height, rgba.strides[0], QImage.Format.Format_RGBA8888).copy()
    return QPixmap.fromImage(image)


def union_mask(masks: np.ndarray | None) -> np.ndarray | None:
    if masks is None or masks.size == 0:
        return None
    arr = np.asarray(masks)
    if arr.ndim == 2:
        return arr.astype(np.uint8)
    if arr.ndim == 3:
        return arr.max(axis=0).astype(np.uint8)
    return None
