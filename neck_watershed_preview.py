#!/usr/bin/env python3
"""Draw watershed neck-cut previews for shared speech-bubble detections."""

from __future__ import annotations

import argparse
import json
import math
import os
import os.path as osp
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from utils.io_utils import imread


ALIGN_ENGINE_DIR = Path(__file__).resolve().parent / '建立对齐方框'
if str(ALIGN_ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(ALIGN_ENGINE_DIR))

from layout_core import get_best_component_mask, smooth_mask  # noqa: E402


IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff')
ALIGN_MASK_DILATE_SIZE = 9
ALIGN_MANUAL_DIFF_THRESHOLD = 24
ALIGN_MANUAL_PROTECT_DILATE_SIZE = 5

COLORS = [
    (235, 73, 73),
    (30, 158, 230),
    (245, 160, 0),
    (30, 185, 95),
    (172, 86, 230),
    (0, 180, 168),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Generate neck/watershed previews from ctd/measure.debug.json.'
    )
    parser.add_argument('image_dir', help='Folder passed to new_detect_folder.py.')
    parser.add_argument(
        '--debug-json',
        help='Path to measure.debug.json. Defaults to <image_dir>/ctd/measure.debug.json.',
    )
    parser.add_argument(
        '--mask-dir',
        help='Path to text mask folder. Defaults to <image_dir>/ctd/progressing/mask.',
    )
    parser.add_argument(
        '--deal-overlap-dir',
        help='Path to deal_overlap folder. Defaults to <image_dir>/ctd/progressing/align/deal_overlap.',
    )
    parser.add_argument(
        '--out-dir',
        help='Output folder. Defaults to <image_dir>/ctd/neck_watershed_preview.',
    )
    parser.add_argument(
        '--iou-threshold',
        type=float,
        default=0.92,
        help='Group items whose outer/raw outer IoU is at least this value.',
    )
    parser.add_argument(
        '--seed-dilate',
        type=int,
        default=15,
        help='Dilate text boxes before using them as watershed markers.',
    )
    parser.add_argument(
        '--neck-ratio-threshold',
        type=float,
        default=0.62,
        help='Guide is marked accepted when neck width / smaller lobe width is under this ratio.',
    )
    parser.add_argument(
        '--include-contained',
        action='store_true',
        help='Also group boxes almost fully contained by another outer box. Useful for leak debugging.',
    )
    return parser.parse_args()


def rect_to_xyxy(rect: dict[str, Any] | None) -> list[int] | None:
    if not rect:
        return None
    return [
        int(round(rect['left'])),
        int(round(rect['top'])),
        int(round(rect['left'] + rect['width'])),
        int(round(rect['top'] + rect['height'])),
    ]


def rect_area(box: list[int] | None) -> int:
    if box is None:
        return 0
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def rect_intersection(a: list[int] | None, b: list[int] | None) -> int:
    if a is None or b is None:
        return 0
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    return max(0, x2 - x1) * max(0, y2 - y1)


def rect_iou(a: list[int] | None, b: list[int] | None) -> float:
    intersection = rect_intersection(a, b)
    if intersection <= 0:
        return 0.0
    union = rect_area(a) + rect_area(b) - intersection
    return intersection / union if union else 0.0


def rect_min_cover_ratio(a: list[int] | None, b: list[int] | None) -> float:
    intersection = rect_intersection(a, b)
    smaller = min(rect_area(a), rect_area(b))
    return intersection / smaller if smaller else 0.0


def box_center(box: list[int]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def union_box(boxes: list[list[int]]) -> list[int]:
    return [
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    ]


def clamp_box(box: list[int], width: int, height: int) -> list[int]:
    return [
        max(0, min(width, int(round(box[0])))),
        max(0, min(height, int(round(box[1])))),
        max(0, min(width, int(round(box[2])))),
        max(0, min(height, int(round(box[3])))),
    ]


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, index: int) -> int:
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def union(self, a: int, b: int) -> None:
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a != root_b:
            self.parent[root_b] = root_a


def should_group(
    a: dict[str, Any],
    b: dict[str, Any],
    iou_threshold: float,
    include_contained: bool,
) -> bool:
    if rect_iou(a['outer'], b['outer']) >= iou_threshold:
        return True
    if rect_iou(a['raw_outer'], b['raw_outer']) >= iou_threshold:
        return True
    if not include_contained:
        return False
    return (
        rect_min_cover_ratio(a['outer'], b['outer']) >= 0.98
        or rect_min_cover_ratio(a['raw_outer'], b['raw_outer']) >= 0.98
    )


def collect_shared_groups(
    page_items: list[dict[str, Any]],
    iou_threshold: float,
    include_contained: bool,
) -> list[list[dict[str, Any]]]:
    entries = []
    for item in page_items:
        debug = item.get('layout_debug', {})
        outer = rect_to_xyxy(debug.get('outer_rect'))
        raw_outer = rect_to_xyxy(debug.get('raw_outer_rect'))
        old_box = debug.get('old_xyxy_pixel') or item.get('final_xyxy_pixel')
        if outer is None or old_box is None:
            continue
        entries.append({
            'source_block_index': item.get('source_block_index'),
            'outer': outer,
            'raw_outer': raw_outer,
            'old': [int(round(value)) for value in old_box],
            'final': [int(round(value)) for value in item.get('final_xyxy_pixel', old_box)],
            'seed_point': debug.get('seed_point'),
            'accepted': bool(item.get('accepted')),
            'method': item.get('method'),
        })

    uf = UnionFind(len(entries))
    for index, item_a in enumerate(entries):
        for other_index, item_b in enumerate(entries[index + 1:], index + 1):
            if should_group(item_a, item_b, iou_threshold, include_contained):
                uf.union(index, other_index)

    grouped: dict[int, list[dict[str, Any]]] = {}
    for index, entry in enumerate(entries):
        grouped.setdefault(uf.find(index), []).append(entry)

    groups = [group for group in grouped.values() if len(group) >= 2]
    groups.sort(key=lambda group: (
        min(entry['outer'][1] for entry in group),
        min(entry['outer'][0] for entry in group),
    ))
    return groups


def resolve_image_path(image_dir: Path, page_name: str) -> Path:
    direct = image_dir / page_name
    if direct.exists():
        return direct
    stem = Path(page_name).stem
    for ext in IMAGE_EXTS:
        candidate = image_dir / f'{stem}{ext}'
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f'Cannot find source image for {page_name} in {image_dir}')


def deal_overlap_path_for_page(deal_overlap_dir: Path, page_name: str) -> Path | None:
    stem = Path(page_name).stem
    for ext in IMAGE_EXTS:
        candidate = deal_overlap_dir / f'{stem}{ext}'
        if candidate.exists():
            return candidate
    candidate = deal_overlap_dir / f'{stem}.png'
    return candidate if candidate.exists() else None


def mask_path_for_page(mask_dir: Path, page_name: str) -> Path:
    return mask_dir / f'{Path(page_name).stem}.png'


def mask_to_binary(mask: np.ndarray) -> np.ndarray:
    if len(mask.shape) == 3:
        gray = cv2.cvtColor(mask[:, :, :3], cv2.COLOR_BGR2GRAY)
    else:
        gray = mask
    return gray > 0


def manual_edit_mask(base_img: np.ndarray | None, calc_img: np.ndarray) -> np.ndarray | None:
    if base_img is None or base_img.shape[:2] != calc_img.shape[:2]:
        return None
    base = base_img[:, :, :3] if len(base_img.shape) == 3 else cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)
    calc = calc_img[:, :, :3] if len(calc_img.shape) == 3 else cv2.cvtColor(calc_img, cv2.COLOR_GRAY2BGR)
    diff = cv2.absdiff(base, calc)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    manual = (diff_gray > ALIGN_MANUAL_DIFF_THRESHOLD).astype(np.uint8) * 255
    if cv2.countNonZero(manual) == 0:
        return manual
    kernel = np.ones((ALIGN_MANUAL_PROTECT_DILATE_SIZE, ALIGN_MANUAL_PROTECT_DILATE_SIZE), dtype=np.uint8)
    return cv2.dilate(manual, kernel, iterations=1)


def prepare_align_gray(calc_img: np.ndarray, mask: np.ndarray, base_img: np.ndarray | None) -> np.ndarray:
    gray = cv2.cvtColor(calc_img, cv2.COLOR_BGR2GRAY) if len(calc_img.shape) == 3 else calc_img.copy()
    text_mask = mask_to_binary(mask).astype(np.uint8) * 255
    kernel = np.ones((ALIGN_MASK_DILATE_SIZE, ALIGN_MASK_DILATE_SIZE), dtype=np.uint8)
    text_mask = cv2.dilate(text_mask, kernel, iterations=1)
    manual_mask = manual_edit_mask(base_img, calc_img)
    if manual_mask is not None:
        text_mask[manual_mask > 0] = 0
    inpainted = cv2.inpaint(gray, text_mask, 5, cv2.INPAINT_TELEA)
    if manual_mask is not None:
        inpainted[manual_mask > 0] = gray[manual_mask > 0]
    return inpainted


def item_to_core_item(entry: dict[str, Any], image_width: int, image_height: int) -> dict[str, Any]:
    old_box = entry['old']
    cx, cy = box_center(old_box)
    return {
        'xyxy_pixel': old_box,
        'center_normalized': [
            round(cx / image_width, 4),
            round(cy / image_height, 4),
        ],
    }


def bubble_mask_from_group(gray: np.ndarray, group: list[dict[str, Any]]) -> np.ndarray | None:
    height, width = gray.shape[:2]
    masks = []
    for entry in group:
        best = get_best_component_mask(gray, item_to_core_item(entry, width, height))
        if best is None:
            continue
        _, smoothed, _, _, smooth_area = best
        if smooth_area > 0:
            masks.append(smoothed)

    if not masks:
        return None

    # Same shared bubble normally gives identical masks. Union keeps the preview stable
    # when seeds land on slightly different but still connected parts.
    merged = np.zeros_like(masks[0], dtype=np.uint8)
    for mask in masks:
        merged[mask > 0] = 255

    merged = smooth_mask(merged, 6)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    merged = cv2.morphologyEx(merged, cv2.MORPH_CLOSE, kernel, iterations=1)
    return merged


def marker_from_box(
    bubble_mask: np.ndarray,
    box: list[int],
    label: int,
    seed_dilate: int,
    markers: np.ndarray,
) -> None:
    height, width = bubble_mask.shape[:2]
    x1, y1, x2, y2 = clamp_box(box, width, height)
    if x2 <= x1 or y2 <= y1:
        return

    box_mask = np.zeros_like(bubble_mask, dtype=np.uint8)
    box_mask[y1:y2, x1:x2] = 255
    if seed_dilate > 0:
        ksize = int(seed_dilate) * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        box_mask = cv2.dilate(box_mask, kernel, iterations=1)
    box_mask[bubble_mask == 0] = 0

    if cv2.countNonZero(box_mask) == 0:
        cx, cy = box_center(box)
        cx = max(0, min(width - 1, int(round(cx))))
        cy = max(0, min(height - 1, int(round(cy))))
        if bubble_mask[cy, cx] == 0:
            ys, xs = np.where(bubble_mask > 0)
            if len(xs) == 0:
                return
            nearest = int(np.argmin((xs - cx) ** 2 + (ys - cy) ** 2))
            cx, cy = int(xs[nearest]), int(ys[nearest])
        cv2.circle(box_mask, (cx, cy), max(3, seed_dilate), 255, -1)
        box_mask[bubble_mask == 0] = 0

    markers[box_mask > 0] = label


def watershed_group(
    bubble_mask: np.ndarray,
    group: list[dict[str, Any]],
    seed_dilate: int,
) -> dict[str, Any] | None:
    if len(group) < 2:
        return None

    dist = cv2.distanceTransform((bubble_mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
    if float(dist.max()) <= 0:
        return None

    markers = np.zeros(bubble_mask.shape[:2], dtype=np.int32)
    for label, entry in enumerate(group, start=1):
        marker_from_box(bubble_mask, entry['old'], label, seed_dilate, markers)

    present_labels = sorted(int(label) for label in np.unique(markers) if label > 0)
    if len(present_labels) < 2:
        return None

    # Watershed expects high ridges to be basins after negation. Restrict all pixels
    # outside the bubble to background marker 0.
    max_dist = float(dist.max())
    relief = np.zeros((*bubble_mask.shape[:2], 3), dtype=np.uint8)
    relief_gray = np.clip((1.0 - dist / max_dist) * 255, 0, 255).astype(np.uint8)
    relief[:] = cv2.cvtColor(relief_gray, cv2.COLOR_GRAY2BGR)
    markers_ws = markers.copy()
    cv2.watershed(relief, markers_ws)
    markers_ws[bubble_mask == 0] = 0

    lobe_radii_by_label = {}
    for label in present_labels:
        values = dist[markers_ws == label]
        if values.size:
            lobe_radii_by_label[label] = float(np.percentile(values, 95))
    lobe_radii = list(lobe_radii_by_label.values())
    smaller_lobe_radius = min(lobe_radii) if lobe_radii else 0.0
    guides = neck_guides_from_group(
        bubble_mask=bubble_mask,
        dist=dist,
        group=group,
        lobe_radii_by_label=lobe_radii_by_label,
    )
    if not guides:
        return None

    neck_ratio = max(guide['neck_ratio'] for guide in guides)
    neck_width = max(guide['neck_width'] for guide in guides)

    return {
        'markers': markers,
        'labels': markers_ws,
        'distance': dist,
        'guides': guides,
        'neck_width': neck_width,
        'smaller_lobe_radius': smaller_lobe_radius,
        'smaller_lobe_width': smaller_lobe_radius * 2.0,
        'neck_ratio': neck_ratio,
    }


def mst_pair_indices(group: list[dict[str, Any]]) -> list[tuple[int, int]]:
    if len(group) <= 1:
        return []
    if len(group) == 2:
        return [(0, 1)]

    edges = []
    for index, entry in enumerate(group):
        cx, cy = box_center(entry['old'])
        for other_index, other in enumerate(group[index + 1:], index + 1):
            ox, oy = box_center(other['old'])
            distance = math.hypot(cx - ox, cy - oy)
            edges.append((distance, index, other_index))
    edges.sort()

    uf = UnionFind(len(group))
    pairs = []
    for _, index, other_index in edges:
        if uf.find(index) == uf.find(other_index):
            continue
        uf.union(index, other_index)
        pairs.append((index, other_index))
        if len(pairs) == len(group) - 1:
            break
    return pairs


def cross_section_at(
    bubble_mask: np.ndarray,
    center: tuple[float, float],
    perp: tuple[float, float],
    max_steps: int,
) -> dict[str, Any] | None:
    height, width = bubble_mask.shape[:2]
    cx, cy = center
    x = int(round(cx))
    y = int(round(cy))
    if x < 0 or x >= width or y < 0 or y >= height or bubble_mask[y, x] == 0:
        return None

    px, py = perp

    def trace(sign: float) -> tuple[int, int]:
        last_x, last_y = x, y
        for step in range(1, max_steps + 1):
            tx = int(round(cx + px * step * sign))
            ty = int(round(cy + py * step * sign))
            if tx < 0 or tx >= width or ty < 0 or ty >= height or bubble_mask[ty, tx] == 0:
                break
            last_x, last_y = tx, ty
        return last_x, last_y

    start = trace(-1.0)
    end = trace(1.0)
    length = math.hypot(end[0] - start[0], end[1] - start[1])
    if length <= 1:
        return None
    return {
        'start': start,
        'end': end,
        'length': float(length),
        'center': (x, y),
    }


def scan_neck_guide(
    bubble_mask: np.ndarray,
    dist: np.ndarray,
    entry_a: dict[str, Any],
    entry_b: dict[str, Any],
    lobe_width: float,
    group: list[dict[str, Any]],
) -> dict[str, Any] | None:
    ax, ay = box_center(entry_a['old'])
    bx, by = box_center(entry_b['old'])
    seed_dx = bx - ax
    seed_dy = by - ay
    seed_len = math.hypot(seed_dx, seed_dy)
    if seed_len <= 1:
        return None

    defect_guide = convex_defect_neck_guide(
        bubble_mask=bubble_mask,
        dist=dist,
        entry_a=entry_a,
        entry_b=entry_b,
        lobe_width=lobe_width,
    )
    if defect_guide is not None:
        return defect_guide

    sample_count = max(36, min(260, int(seed_len * 1.2)))
    max_steps = int(max(bubble_mask.shape[:2]))
    candidates = []
    min_radius = max(3.0, lobe_width * 0.035)

    for sample_index in range(1, sample_count):
        t = sample_index / sample_count
        if t < 0.08 or t > 0.92:
            continue
        cx = ax + (bx - ax) * t
        cy = ay + (by - ay) * t
        center_x = int(round(cx))
        center_y = int(round(cy))
        if (
            center_x < 0
            or center_x >= bubble_mask.shape[1]
            or center_y < 0
            or center_y >= bubble_mask.shape[0]
            or bubble_mask[center_y, center_x] == 0
        ):
            continue
        center_radius = float(dist[center_y, center_x])
        if center_radius < min_radius:
            continue

        for angle in range(0, 180, 4):
            radians = math.radians(angle)
            direction = (math.cos(radians), math.sin(radians))
            normal = (-direction[1], direction[0])
            side_a = (ax - cx) * normal[0] + (ay - cy) * normal[1]
            side_b = (bx - cx) * normal[0] + (by - cy) * normal[1]
            if side_a * side_b >= 0:
                continue
            if min(abs(side_a), abs(side_b)) < 8:
                continue

            section = cross_section_at(bubble_mask, (cx, cy), direction, max_steps)
            if section is None:
                continue

            sx, sy = section['center']
            radius = float(dist[sy, sx])
            score = section['length'] + radius * 0.18 + abs(t - 0.5) * 8.0
            section.update({
                't': t,
                'angle': angle,
                'radius': radius,
                'score': float(score),
            })
            candidates.append(section)

    if not candidates:
        return None

    best = min(candidates, key=lambda item: item['score'])
    neck_width = float(best['length'])
    neck_ratio = neck_width / lobe_width if lobe_width > 0 else float('inf')
    return {
        'source_block_indices': [
            entry_a['source_block_index'],
            entry_b['source_block_index'],
        ],
        'start': [int(best['start'][0]), int(best['start'][1])],
        'end': [int(best['end'][0]), int(best['end'][1])],
        'center': [int(best['center'][0]), int(best['center'][1])],
        'neck_width': neck_width,
        'neck_ratio': neck_ratio,
        'distance_radius': float(best['radius']),
        'angle': int(best['angle']),
        'method': 'width_scan',
    }


def contour_defect_points(bubble_mask: np.ndarray) -> list[dict[str, Any]]:
    contours, _ = cv2.findContours(bubble_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 4:
        return []

    hull = cv2.convexHull(contour, returnPoints=False)
    if hull is None or len(hull) < 4:
        return []

    try:
        defects = cv2.convexityDefects(contour, hull)
    except cv2.error:
        return []
    if defects is None:
        return []

    x, y, w, h = cv2.boundingRect(contour)
    depth_threshold = max(6.0, min(w, h) * 0.025)
    raw_points = []
    for defect in defects[:, 0, :]:
        start_index, end_index, far_index, depth_raw = [int(value) for value in defect]
        depth = depth_raw / 256.0
        if depth < depth_threshold:
            continue
        far = contour[far_index][0]
        start = contour[start_index][0]
        end = contour[end_index][0]
        raw_points.append({
            'point': [int(far[0]), int(far[1])],
            'start': [int(start[0]), int(start[1])],
            'end': [int(end[0]), int(end[1])],
            'depth': float(depth),
        })

    raw_points.sort(key=lambda item: item['depth'], reverse=True)
    deduped = []
    for candidate in raw_points:
        px, py = candidate['point']
        duplicate = False
        for existing in deduped:
            ex, ey = existing['point']
            if math.hypot(px - ex, py - ey) < 10:
                duplicate = True
                break
        if not duplicate:
            deduped.append(candidate)
    return deduped[:24]


def signed_line_side(point: tuple[float, float], start: tuple[float, float], end: tuple[float, float]) -> float:
    return (end[0] - start[0]) * (point[1] - start[1]) - (end[1] - start[1]) * (point[0] - start[0])


def line_inside_ratio(mask: np.ndarray, start: tuple[int, int], end: tuple[int, int]) -> float:
    length = max(1.0, math.hypot(end[0] - start[0], end[1] - start[1]))
    sample_count = max(8, min(400, int(length)))
    inside = 0
    height, width = mask.shape[:2]
    for index in range(sample_count + 1):
        t = index / sample_count
        x = int(round(start[0] + (end[0] - start[0]) * t))
        y = int(round(start[1] + (end[1] - start[1]) * t))
        if 0 <= x < width and 0 <= y < height and mask[y, x] > 0:
            inside += 1
    return inside / (sample_count + 1)


def projection_t(point: tuple[float, float], start: tuple[float, float], end: tuple[float, float]) -> float:
    vx = end[0] - start[0]
    vy = end[1] - start[1]
    denom = vx * vx + vy * vy
    if denom <= 0:
        return 0.5
    return ((point[0] - start[0]) * vx + (point[1] - start[1]) * vy) / denom


def convex_defect_neck_guide(
    bubble_mask: np.ndarray,
    dist: np.ndarray,
    entry_a: dict[str, Any],
    entry_b: dict[str, Any],
    lobe_width: float,
) -> dict[str, Any] | None:
    defects = contour_defect_points(bubble_mask)
    if len(defects) < 2:
        return None

    seed_a = box_center(entry_a['old'])
    seed_b = box_center(entry_b['old'])
    lobe_width = max(1.0, lobe_width)
    candidates = []

    for index, defect_a in enumerate(defects):
        point_a = tuple(defect_a['point'])
        for defect_b in defects[index + 1:]:
            point_b = tuple(defect_b['point'])
            length = math.hypot(point_b[0] - point_a[0], point_b[1] - point_a[1])
            if length < 4:
                continue

            side_a = signed_line_side(seed_a, point_a, point_b)
            side_b = signed_line_side(seed_b, point_a, point_b)
            norm_side = min(abs(side_a), abs(side_b)) / length
            min_seed_separation = max(6.0, lobe_width * 0.06)
            if side_a * side_b >= 0 or norm_side < min_seed_separation:
                continue

            midpoint = ((point_a[0] + point_b[0]) / 2.0, (point_a[1] + point_b[1]) / 2.0)
            t = projection_t(midpoint, seed_a, seed_b)
            if t < 0.03 or t > 0.97:
                continue

            inside_ratio = line_inside_ratio(bubble_mask, point_a, point_b)
            if inside_ratio < 0.45:
                continue

            mx = max(0, min(dist.shape[1] - 1, int(round(midpoint[0]))))
            my = max(0, min(dist.shape[0] - 1, int(round(midpoint[1]))))
            midpoint_radius = float(dist[my, mx])
            depth_score = min((defect_a['depth'] + defect_b['depth']) / lobe_width, 1.5)
            length_ratio = length / lobe_width
            radius_ratio = midpoint_radius / lobe_width
            score = (
                length_ratio
                + radius_ratio * 0.45
                + abs(t - 0.5) * 0.45
                + (1.0 - inside_ratio) * 0.35
                - depth_score * 0.42
                - min(norm_side / lobe_width, 1.0) * 0.22
            )
            candidates.append({
                'start': [int(point_a[0]), int(point_a[1])],
                'end': [int(point_b[0]), int(point_b[1])],
                'center': [int(round(midpoint[0])), int(round(midpoint[1]))],
                'neck_width': float(length),
                'neck_ratio': float(length_ratio),
                'distance_radius': midpoint_radius,
                'depths': [float(defect_a['depth']), float(defect_b['depth'])],
                'inside_ratio': float(inside_ratio),
                'score': float(score),
                'method': 'convex_defect',
            })

    if not candidates:
        return None

    best = min(candidates, key=lambda item: item['score'])
    best['source_block_indices'] = [
        entry_a['source_block_index'],
        entry_b['source_block_index'],
    ]
    return best


def neck_guides_from_group(
    bubble_mask: np.ndarray,
    dist: np.ndarray,
    group: list[dict[str, Any]],
    lobe_radii_by_label: dict[int, float],
) -> list[dict[str, Any]]:
    guides = []
    for index, other_index in mst_pair_indices(group):
        label_a = index + 1
        label_b = other_index + 1
        radius_a = lobe_radii_by_label.get(label_a, 0.0)
        radius_b = lobe_radii_by_label.get(label_b, 0.0)
        lobe_width = min(radius_a, radius_b) * 2.0
        guide = scan_neck_guide(
            bubble_mask=bubble_mask,
            dist=dist,
            entry_a=group[index],
            entry_b=group[other_index],
            lobe_width=lobe_width,
            group=group,
        )
        if guide is not None:
            guide['smaller_lobe_width'] = float(lobe_width)
            guides.append(guide)
    return guides


def draw_label(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    fill: tuple[int, int, int, int],
    font: ImageFont.ImageFont,
) -> None:
    x, y = xy
    bbox = draw.textbbox((x, y), text, font=font)
    pad = 3
    draw.rounded_rectangle(
        (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad),
        radius=4,
        fill=(255, 255, 255, 230),
        outline=fill,
        width=1,
    )
    draw.text((x, y), text, fill=fill, font=font)


def draw_mask_overlay(
    overlay: Image.Image,
    mask: np.ndarray,
    color: tuple[int, int, int],
    alpha: int,
) -> None:
    active = mask > 0
    if not np.any(active):
        return
    layer = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
    layer[active] = (*color, alpha)
    overlay.alpha_composite(Image.fromarray(layer, mode='RGBA'))


def draw_guide_line(
    draw: ImageDraw.ImageDraw,
    guide: dict[str, Any],
    color: tuple[int, int, int, int],
    font: ImageFont.ImageFont,
) -> None:
    start = tuple(guide['start'])
    end = tuple(guide['end'])
    center = tuple(guide['center'])
    draw.line((*start, *end), fill=color, width=7)
    for x, y in (start, end, center):
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color, outline=(255, 255, 255, 255), width=2)
    ratio = guide.get('neck_ratio')
    label = f'guide r={ratio:.2f}' if isinstance(ratio, float) and math.isfinite(ratio) else 'guide'
    draw_label(draw, (center[0] + 7, center[1] + 7), label, color, font)


def draw_page_preview(
    image_dir: Path,
    mask_dir: Path,
    deal_overlap_dir: Path,
    out_dir: Path,
    page_name: str,
    page_items: list[dict[str, Any]],
    iou_threshold: float,
    include_contained: bool,
    seed_dilate: int,
    neck_ratio_threshold: float,
) -> dict[str, Any]:
    image_path = resolve_image_path(image_dir, page_name)
    mask_path = mask_path_for_page(mask_dir, page_name)
    overlap_path = deal_overlap_path_for_page(deal_overlap_dir, page_name)

    base_img = imread(str(image_path))
    calc_img = imread(str(overlap_path)) if overlap_path is not None else base_img
    text_mask = imread(str(mask_path))
    if base_img is None:
        raise FileNotFoundError(f'Cannot read image: {image_path}')
    if calc_img is None:
        raise FileNotFoundError(f'Cannot read calculation image: {overlap_path}')
    if text_mask is None:
        raise FileNotFoundError(f'Cannot read mask: {mask_path}')

    gray = prepare_align_gray(calc_img, text_mask, base_img=base_img)
    image = Image.open(image_path).convert('RGBA')
    width, height = image.size
    overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, 'RGBA')
    font = ImageFont.load_default()

    groups = collect_shared_groups(page_items, iou_threshold, include_contained)
    summaries = []

    for group_index, group in enumerate(groups, start=1):
        color = COLORS[(group_index - 1) % len(COLORS)]
        line = (*color, 245)
        bubble_mask = bubble_mask_from_group(gray, group)
        outer = clamp_box(union_box([entry['outer'] for entry in group]), width, height)
        if bubble_mask is not None:
            draw_mask_overlay(overlay, bubble_mask, color, 38)

        draw.rectangle(outer, outline=line, width=4)

        result = watershed_group(bubble_mask, group, seed_dilate) if bubble_mask is not None else None
        status = 'no-watershed'
        ratio = None
        if result is not None:
            ratio = float(result['neck_ratio'])
            status = 'neck' if ratio <= neck_ratio_threshold else 'weak-neck'
            labels = result['labels']
            for label, entry in enumerate(group, start=1):
                label_color = COLORS[(label - 1) % len(COLORS)]
                draw_mask_overlay(overlay, (labels == label).astype(np.uint8) * 255, label_color, 34)
            for guide in result['guides']:
                draw_guide_line(draw, guide, line, font)

        title = f'G{group_index} {status}'
        if ratio is not None and math.isfinite(ratio):
            title += f' r={ratio:.2f}'
        draw_label(draw, (outer[0] + 6, max(0, outer[1] - 19)), title, line, font)

        ordered = sorted(group, key=lambda entry: (entry['old'][1], entry['old'][0]))
        for item_index, entry in enumerate(ordered):
            item_color = COLORS[item_index % len(COLORS)]
            item_line = (*item_color, 255)
            old_box = clamp_box(entry['old'], width, height)
            draw.rectangle(old_box, fill=(*item_color, 30), outline=item_line, width=4)
            center_x, center_y = box_center(old_box)
            draw.ellipse(
                (center_x - 5, center_y - 5, center_x + 5, center_y + 5),
                fill=item_line,
                outline=(255, 255, 255, 255),
                width=2,
            )
            draw_label(
                draw,
                (old_box[0] + 2, max(0, old_box[1] - 16)),
                f'id {entry["source_block_index"]}',
                item_line,
                font,
            )

        summaries.append({
            'group': group_index,
            'source_block_indices': [entry['source_block_index'] for entry in ordered],
            'outer_union_xyxy': outer,
            'status': status,
            'neck_ratio': round(ratio, 4) if ratio is not None and math.isfinite(ratio) else None,
            'neck_width': round(float(result['neck_width']), 2) if result is not None else None,
            'smaller_lobe_width': round(float(result['smaller_lobe_width']), 2) if result is not None else None,
            'guides': [
                {
                    'source_block_indices': guide['source_block_indices'],
                    'start': guide['start'],
                    'end': guide['end'],
                    'center': guide['center'],
                    'method': guide.get('method'),
                    'neck_width': round(float(guide['neck_width']), 2),
                    'neck_ratio': round(float(guide['neck_ratio']), 4) if math.isfinite(float(guide['neck_ratio'])) else None,
                }
                for guide in result['guides']
            ] if result is not None else [],
        })

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'{Path(page_name).stem}.neck_watershed.png'
    Image.alpha_composite(image, overlay).convert('RGB').save(out_path, quality=95)
    return {
        'preview': str(out_path),
        'used_deal_overlap': str(overlap_path) if overlap_path is not None else None,
        'groups': summaries,
    }


def main() -> None:
    args = parse_args()
    image_dir = Path(args.image_dir).expanduser().resolve()
    debug_path = Path(args.debug_json).expanduser().resolve() if args.debug_json else image_dir / 'ctd' / 'measure.debug.json'
    mask_dir = Path(args.mask_dir).expanduser().resolve() if args.mask_dir else image_dir / 'ctd' / 'progressing' / 'mask'
    deal_overlap_dir = Path(args.deal_overlap_dir).expanduser().resolve() if args.deal_overlap_dir else image_dir / 'ctd' / 'progressing' / 'align' / 'deal_overlap'
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else image_dir / 'ctd' / 'neck_watershed_preview'

    data = json.loads(debug_path.read_text(encoding='utf-8'))
    align_pages = data.get('align', {}).get('transMap', {})
    if not align_pages:
        raise ValueError(f'No align.transMap pages found in {debug_path}')

    summary = {}
    for page_name, page_items in align_pages.items():
        summary[page_name] = draw_page_preview(
            image_dir=image_dir,
            mask_dir=mask_dir,
            deal_overlap_dir=deal_overlap_dir,
            out_dir=out_dir,
            page_name=page_name,
            page_items=page_items,
            iou_threshold=args.iou_threshold,
            include_contained=args.include_contained,
            seed_dilate=args.seed_dilate,
            neck_ratio_threshold=args.neck_ratio_threshold,
        )

    summary_path = out_dir / 'neck_watershed_summary.json'
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
