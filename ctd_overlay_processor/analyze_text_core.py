#!/usr/bin/env python3
"""獨立分析 CTD mask 的字芯黑白與 need_inpaint 判斷。

此腳本只讀取資料，不修改 measure.json。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


REPAIR_EXPAND_PX = 3
SAMPLE_RING_PX = 3
MIN_SAMPLE_PIXELS = 12
MIN_DIRECTIONAL_SAMPLE_PIXELS = 24
SOLID_P90_P10_MAX = 12
SOLID_PEAK_RATIO_MIN = 0.62
SOLID_CLOSE_DELTA_MAX = 10
SOLID_CLOSE_RATIO_MIN = 0.72
SOLID_P95_DELTA_MAX = 16
DIRECTIONAL_SOLID_P90_P10_MAX = 8
DIRECTIONAL_SOLID_CLOSE_RATIO_MIN = 0.82
DIRECTIONAL_SOLID_P95_DELTA_MAX = 12
DIRECTIONAL_FALLBACK_MAX_FULL_SPREAD = 28
DIRECTIONAL_FALLBACK_MIN_FULL_CLOSE_RATIO = 0.45
DIRECTIONAL_FILL_AGREEMENT_MAX = 10
MIN_DIRECTIONAL_AGREEMENT_COUNT = 2
WHITE_DOMINANT_MIN = 242
WHITE_PEAK_RATIO_MIN = 0.68
WHITE_FULL_PEAK_RATIO_MIN = 0.18
WHITE_CLOSE_DELTA_MAX = 10
WHITE_CLOSE_RATIO_MIN = 0.78
DIRECTIONAL_WHITE_CLOSE_RATIO_MIN = 0.84
WHITE_P95_DELTA_MAX = 18


@dataclass
class SolidQuality:
    is_solid: bool
    score: float
    fill_bgr: tuple[int, int, int]
    max_spread: int
    min_peak_ratio: float
    close_ratio: float
    p95_delta: int
    white_close_ratio: float
    white_p95_delta: int
    sample_pixels: int
    mode: str


def _clamp_xyxy(xyxy: list[Any], width: int, height: int) -> tuple[int, int, int, int] | None:
    if not isinstance(xyxy, list) or len(xyxy) != 4:
        return None
    x1, y1, x2, y2 = [int(round(float(value))) for value in xyxy]
    x1 = max(0, min(x1, width))
    y1 = max(0, min(y1, height))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _kernel(radius: int) -> np.ndarray:
    size = radius * 2 + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _direction_mask(shape: tuple[int, int], repair_area: np.ndarray, direction: str) -> np.ndarray:
    ys, xs = np.where(repair_area > 0)
    mask = np.zeros(shape, dtype=np.uint8)
    if xs.size == 0:
        return mask
    left, right = int(xs.min()), int(xs.max()) + 1
    top, bottom = int(ys.min()), int(ys.max()) + 1
    if direction == 'top':
        mask[:top, left:right] = 255
    elif direction == 'bottom':
        mask[bottom:, left:right] = 255
    elif direction == 'left':
        mask[top:bottom, :left] = 255
    elif direction == 'right':
        mask[top:bottom, right:] = 255
    return mask


def _hist_channel(values: np.ndarray) -> tuple[int, float, int]:
    if values.size == 0:
        return 255, 0.0, 0
    hist = np.bincount(values.astype(np.uint8), minlength=256)
    total = int(hist.sum())
    peak_value = int(hist.argmax())
    peak_ratio = float(hist[peak_value] / total) if total else 0.0
    cdf = np.cumsum(hist)
    p10 = int(np.searchsorted(cdf, total * 0.10, side='left'))
    p90 = int(np.searchsorted(cdf, total * 0.90, side='left'))
    return p90 - p10, peak_ratio, peak_value


def _dominant_channel(values: np.ndarray) -> int:
    if values.size == 0:
        return 0
    hist = np.bincount(values.astype(np.uint8), minlength=256)
    return int(hist.argmax())


def _quality_from_sample(color_img: np.ndarray, sample_mask: np.ndarray, mode: str) -> SolidQuality:
    active = sample_mask > 0
    sample_pixels = int(np.count_nonzero(active))
    if sample_pixels < MIN_SAMPLE_PIXELS:
        return SolidQuality(False, -1.0, (0, 0, 0), 255, 0.0, 0.0, 255, 0.0, 255, sample_pixels, mode)

    samples = color_img[active]
    b_values = samples[:, 0]
    g_values = samples[:, 1]
    r_values = samples[:, 2]
    b_spread, b_peak_ratio, b_peak = _hist_channel(b_values)
    g_spread, g_peak_ratio, g_peak = _hist_channel(g_values)
    r_spread, r_peak_ratio, r_peak = _hist_channel(r_values)
    max_spread = int(max(b_spread, g_spread, r_spread))
    min_peak_ratio = float(min(b_peak_ratio, g_peak_ratio, r_peak_ratio))
    fill_bgr = (
        _dominant_channel(b_values),
        _dominant_channel(g_values),
        _dominant_channel(r_values),
    )
    deltas = np.max(
        np.abs(samples.astype(np.int16) - np.array(fill_bgr, dtype=np.int16)),
        axis=1,
    )
    close_ratio = float(np.count_nonzero(deltas <= SOLID_CLOSE_DELTA_MAX) / sample_pixels)
    p95_delta = int(np.percentile(deltas, 95))
    white_deltas = np.max(255 - samples.astype(np.int16), axis=1)
    white_close_ratio = float(np.count_nonzero(white_deltas <= WHITE_CLOSE_DELTA_MAX) / sample_pixels)
    white_p95_delta = int(np.percentile(white_deltas, 95))
    spread_limit = SOLID_P90_P10_MAX
    close_ratio_limit = SOLID_CLOSE_RATIO_MIN
    p95_delta_limit = SOLID_P95_DELTA_MAX
    white_close_ratio_limit = WHITE_CLOSE_RATIO_MIN
    white_peak_ratio_limit = WHITE_FULL_PEAK_RATIO_MIN
    if mode != 'full':
        spread_limit = DIRECTIONAL_SOLID_P90_P10_MAX
        close_ratio_limit = DIRECTIONAL_SOLID_CLOSE_RATIO_MIN
        p95_delta_limit = DIRECTIONAL_SOLID_P95_DELTA_MAX
        white_close_ratio_limit = DIRECTIONAL_WHITE_CLOSE_RATIO_MIN
        white_peak_ratio_limit = WHITE_PEAK_RATIO_MIN
    strict_solid = (
        sample_pixels >= (MIN_DIRECTIONAL_SAMPLE_PIXELS if mode != 'full' else MIN_SAMPLE_PIXELS)
        and max_spread <= spread_limit
        and min_peak_ratio >= SOLID_PEAK_RATIO_MIN
        and close_ratio >= close_ratio_limit
        and p95_delta <= p95_delta_limit
    )
    white_dominant = (
        sample_pixels >= (MIN_DIRECTIONAL_SAMPLE_PIXELS if mode != 'full' else MIN_SAMPLE_PIXELS)
        and b_peak >= WHITE_DOMINANT_MIN
        and g_peak >= WHITE_DOMINANT_MIN
        and r_peak >= WHITE_DOMINANT_MIN
        and min_peak_ratio >= white_peak_ratio_limit
        and white_close_ratio >= white_close_ratio_limit
        and white_p95_delta <= WHITE_P95_DELTA_MAX
    )
    is_solid = strict_solid or white_dominant
    score = (
        min_peak_ratio * 1000.0
        + close_ratio * 400.0
        + white_close_ratio * 120.0
        - max_spread * 4.0
        - p95_delta * 6.0
        + min(sample_pixels, 1000) * 0.001
    )
    if white_dominant:
        score += 50.0
    if strict_solid:
        score += 100.0
    return SolidQuality(
        is_solid,
        score,
        fill_bgr,
        max_spread,
        min_peak_ratio,
        close_ratio,
        p95_delta,
        white_close_ratio,
        white_p95_delta,
        sample_pixels,
        mode,
    )


def _best_quality(color_img: np.ndarray, repair_area: np.ndarray, sample_ring: np.ndarray) -> SolidQuality:
    full = _quality_from_sample(color_img, sample_ring, 'full')
    if full.is_solid:
        return full
    if (
        full.max_spread > DIRECTIONAL_FALLBACK_MAX_FULL_SPREAD
        and full.close_ratio < DIRECTIONAL_FALLBACK_MIN_FULL_CLOSE_RATIO
    ):
        return full

    directionals = []
    for direction in ('top', 'bottom', 'left', 'right'):
        directional = cv2.bitwise_and(sample_ring, _direction_mask(sample_ring.shape, repair_area, direction))
        directionals.append(_quality_from_sample(color_img, directional, direction))

    solid_directionals = [
        item for item in directionals
        if item.is_solid and item.sample_pixels >= MIN_DIRECTIONAL_SAMPLE_PIXELS
    ]
    if len(solid_directionals) < MIN_DIRECTIONAL_AGREEMENT_COUNT:
        return full
    fill_values = np.array([item.fill_bgr for item in solid_directionals], dtype=np.int16)
    fill_disagreement = int(np.max(fill_values.max(axis=0) - fill_values.min(axis=0)))
    if fill_disagreement > DIRECTIONAL_FILL_AGREEMENT_MAX:
        return full
    return max(solid_directionals, key=lambda item: item.score)


def _binary_text_color(image_bgr: np.ndarray, text_mask: np.ndarray) -> tuple[int, int, int] | None:
    text_mask = np.where(text_mask > 0, 255, 0).astype(np.uint8)
    if int(text_mask.sum()) == 0:
        return None
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    mask_pixels = int(np.count_nonzero(text_mask))

    closed = cv2.morphologyEx(text_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    dist = cv2.distanceTransform(closed, cv2.DIST_L2, 3)
    max_dist = float(dist.max())
    core_mask = np.zeros_like(text_mask)
    if max_dist > 0:
        core_threshold = max(1.2, max_dist * 0.42)
        core_mask = np.where(dist >= core_threshold, 255, 0).astype(np.uint8)
        core_mask = cv2.bitwise_and(core_mask, text_mask)

    core_pixels = int(np.count_nonzero(core_mask))
    use_mask = core_mask if core_pixels >= max(20, int(mask_pixels * 0.08)) else text_mask
    values = gray[use_mask == 255]
    if values.size == 0:
        return None

    dark_ratio = float(np.count_nonzero(values < 96) / values.size)
    bright_ratio = float(np.count_nonzero(values > 200) / values.size)
    p20 = float(np.percentile(values, 20))
    p80 = float(np.percentile(values, 80))
    if dark_ratio >= 0.18 or p20 < 96:
        return (0, 0, 0)
    if bright_ratio >= 0.18 or p80 > 200:
        return (255, 255, 255)

    fallback_gray = gray[use_mask == 255]
    return (255, 255, 255) if float(np.median(fallback_gray)) >= 128 else (0, 0, 0)


def calculate_block_colors(image_bgr: np.ndarray, mask: np.ndarray, xyxy: list[Any]) -> dict[str, Any] | None:
    height, width = image_bgr.shape[:2]
    clamped = _clamp_xyxy(xyxy, width, height)
    if clamped is None:
        return None
    x1, y1, x2, y2 = clamped
    local_text = np.zeros(mask.shape[:2], dtype=np.uint8)
    local_text[y1:y2, x1:x2] = np.where(mask[y1:y2, x1:x2] > 0, 255, 0).astype(np.uint8)
    if not np.any(local_text):
        return None

    repair_area = cv2.dilate(local_text, _kernel(REPAIR_EXPAND_PX), iterations=1)
    expanded = cv2.dilate(repair_area, _kernel(SAMPLE_RING_PX), iterations=1)
    sample_ring = cv2.bitwise_and(expanded, cv2.bitwise_not(repair_area))
    text_mask = np.where(mask > 0, 255, 0).astype(np.uint8)
    sample_ring = cv2.bitwise_and(sample_ring, cv2.bitwise_not(text_mask))
    quality = _best_quality(image_bgr, repair_area, sample_ring)

    fg_color = _binary_text_color(image_bgr[y1:y2, x1:x2], local_text[y1:y2, x1:x2])
    if fg_color is None:
        return None

    return {
        'fg_color_rgb': list(fg_color),
        'text_color': 'white' if fg_color == (255, 255, 255) else 'black',
        'need_inpaint': not quality.is_solid,
    }


def read_image(path: Path, mode: int) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, mode)
    if image is None:
        raise FileNotFoundError(f'無法讀取：{path}')
    return image


def page_key_candidates(page: str) -> list[str]:
    path = Path(page)
    candidates = [page]
    if path.suffix:
        candidates.append(path.name)
    else:
        for suffix in ('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'):
            candidates.append(f'{page}{suffix}')
    result = []
    for item in candidates:
        if item not in result:
            result.append(item)
    return result


def resolve_page_name(image_dir: Path, page: str) -> str:
    for candidate in page_key_candidates(page):
        if (image_dir / candidate).is_file():
            return candidate
    raise FileNotFoundError(f'找不到頁面圖片：{page}')


def analyze_page(image_dir: Path, page_name: str) -> list[dict]:
    ctd_dir = image_dir / 'ctd'
    measure_path = ctd_dir / 'measure.json'
    mask_path = ctd_dir / 'progressing' / 'mask' / f'{Path(page_name).stem}.png'
    if not measure_path.is_file():
        raise FileNotFoundError(f'找不到 measure.json：{measure_path}')
    if not mask_path.is_file():
        raise FileNotFoundError(f'找不到 CTD mask：{mask_path}')

    image = read_image(image_dir / page_name, cv2.IMREAD_COLOR)
    mask = read_image(mask_path, cv2.IMREAD_GRAYSCALE)
    measure = json.loads(measure_path.read_text(encoding='utf-8'))
    items = measure.get('pages', {}).get(page_name, [])
    result = []
    for fallback_index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        xyxy = item.get('xyxy_pixel')
        color_info = calculate_block_colors(image, mask, xyxy)
        source_index = item.get('source_block_index', fallback_index)
        row = {
            'source_block_index': source_index,
            'xyxy_pixel': xyxy,
            'font_size': item.get('font_size'),
            'text_color': None if color_info is None else color_info.get('text_color'),
            'fg_color_rgb': None if color_info is None else color_info.get('fg_color_rgb'),
            'need_inpaint': None if color_info is None else color_info.get('need_inpaint'),
        }
        result.append(row)
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='只讀分析 CTD mask 的字芯文字顏色與 need_inpaint。')
    parser.add_argument('image_dir', help='包含原圖和 ctd/ 的圖片資料夾。')
    parser.add_argument('--page', required=True, help='頁面檔名或頁碼，例如 45 或 45.jpg。')
    parser.add_argument('--json', action='store_true', help='用 JSON 輸出。')
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    image_dir = Path(args.image_dir).expanduser().resolve()
    page_name = resolve_page_name(image_dir, args.page)
    rows = analyze_page(image_dir, page_name)
    if args.json:
        print(json.dumps({'page': page_name, 'items': rows}, ensure_ascii=False, indent=2))
        return

    print(f'頁面：{page_name}')
    for row in rows:
        print(
            f'區塊 {row["source_block_index"]}: '
            f'文字={row["text_color"] or "-"} '
            f'需要修復={row["need_inpaint"]} '
            f'字級={row["font_size"]} '
            f'框={row["xyxy_pixel"]}'
        )


if __name__ == '__main__':
    main()
