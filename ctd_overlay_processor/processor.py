#!/usr/bin/env python3
"""讀取 new_detect_folder.py 輸出，提供即時疊圖資料。

本模組刻意不 import、不修改 new_detect_folder.py。它只讀取圖片資料夾與
ctd/ 內已存在的輸出；如果尚未產生 ctd 資料，也允許先載入原圖列表。
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .analyze_text_core import enrich_measure_items
except Exception:
    try:
        from analyze_text_core import enrich_measure_items
    except Exception:
        def enrich_measure_items(image_dir, page_name, items):
            return 0


IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tif', '.tiff')
CTD_MEASURE_JSON = 'measure.json'


@dataclass(slots=True)
class BoxOverlay:
    """One text block with the fields usually needed by a GUI overlay."""

    source_block_index: int
    xyxy_pixel: tuple[int, int, int, int]
    center_normalized: tuple[float, float] | None = None
    orientation: str = 'vertical'
    font_size: float | None = None
    font_size_method: str | None = None
    text_color: str | None = None
    text_has_stroke: bool | None = None
    need_inpaint: bool | None = None
    accepted: bool | None = None
    method: str | None = None
    error_route: bool = False
    error_reason: str | None = None
    block_xyxy_pixel: tuple[int, int, int, int] | None = None
    raw_align: dict[str, Any] = field(default_factory=dict)
    raw_measure: dict[str, Any] = field(default_factory=dict)
    measure_item_index: int | None = None

    @property
    def width(self) -> int:
        return max(0, self.xyxy_pixel[2] - self.xyxy_pixel[0])

    @property
    def height(self) -> int:
        return max(0, self.xyxy_pixel[3] - self.xyxy_pixel[1])

    @property
    def center_pixel(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.xyxy_pixel
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    @property
    def direction_suffix(self) -> str:
        return 'H' if self.orientation == 'horizontal' else 'V'

    @property
    def needs_stroke(self) -> bool:
        return self.text_has_stroke is True or self.need_inpaint is True

    @property
    def font_label(self) -> str:
        if self.font_size is None:
            parts = [self.direction_suffix]
        else:
            parts = [f'{int(round(self.font_size))}{self.direction_suffix}']
        if self.text_color == 'black':
            parts.append('黑')
        elif self.text_color == 'white':
            parts.append('白')
        if self.needs_stroke:
            parts.append('描邊')
        return ','.join(parts)


@dataclass(slots=True)
class LineOverlay:
    source_block_index: int | None
    polygon: tuple[tuple[float, float], ...]
    xyxy_pixel: tuple[int, int, int, int]
    font_width_px: float | None = None
    orientation: str = 'vertical'
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AlignMasks:
    smoothed_masks: np.ndarray
    outer_body_masks: np.ndarray
    accepted: np.ndarray

    @property
    def count(self) -> int:
        return int(max(len(self.smoothed_masks), len(self.outer_body_masks), len(self.accepted)))


@dataclass(slots=True)
class PageOverlay:
    page_name: str
    image_path: Path
    mask_path: Path | None
    align_mask_path: Path | None
    boxes: list[BoxOverlay]
    lines: list[LineOverlay]
    align_masks: AlignMasks | None
    char_boxes: list[dict[str, Any]]

    @property
    def stem(self) -> str:
        return Path(self.page_name).stem


def load_json(path: Path, required: bool = True) -> dict[str, Any]:
    if not path.is_file():
        if required:
            raise FileNotFoundError(f'找不到 JSON：{path}')
        return {}
    with path.open('r', encoding='utf8') as f:
        return json.load(f)


def resolve_image_path(image_dir: Path, page_name: str) -> Path:
    direct = image_dir / page_name
    if direct.is_file():
        return direct

    stem = Path(page_name).stem
    for ext in IMAGE_EXTENSIONS:
        candidate = image_dir / f'{stem}{ext}'
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f'找不到原圖：{page_name}')


def natural_sort_key(name: str) -> list[tuple[int, str | int]]:
    parts = re.split(r'(\d+)', name)
    return [
        (0, int(part)) if part.isdigit() else (1, part.lower())
        for part in parts
        if part
    ]


def find_source_images(image_dir: Path) -> list[str]:
    if not image_dir.is_dir():
        return []
    return sorted(
        [
            path.name
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ],
        key=natural_sort_key,
    )


def xyxy_from_item(item: dict[str, Any]) -> tuple[int, int, int, int] | None:
    for key in ('xyxy_pixel', 'final_xyxy_pixel', 'new_xyxy_pixel'):
        value = item.get(key)
        if isinstance(value, list) and len(value) == 4:
            return tuple(int(round(float(v))) for v in value)

    if all(key in item for key in ('x', 'y', 'w', 'h')):
        x = int(round(float(item.get('x') or 0)))
        y = int(round(float(item.get('y') or 0)))
        w = int(round(float(item.get('w') or 0)))
        h = int(round(float(item.get('h') or 0)))
        return x, y, x + w, y + h
    return None


def normalized_center_from_xyxy(xyxy: tuple[int, int, int, int], image_size: tuple[int, int] | None) -> tuple[float, float] | None:
    if image_size is None:
        return None
    width, height = image_size
    if width <= 0 or height <= 0:
        return None
    x1, y1, x2, y2 = xyxy
    return round(((x1 + x2) / 2.0) / width, 4), round(((y1 + y2) / 2.0) / height, 4)


def tuple_center(value: Any) -> tuple[float, float] | None:
    if isinstance(value, list) and len(value) == 2:
        return float(value[0]), float(value[1])
    return None


def line_box_from_item(item: dict[str, Any]) -> tuple[int, int, int, int] | None:
    polygon = item.get('polygon')
    if isinstance(polygon, list) and len(polygon) >= 4:
        arr = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
        x1, y1 = arr.min(axis=0)
        x2, y2 = arr.max(axis=0)
        return int(x1), int(y1), int(x2), int(y2)
    return xyxy_from_item(item)


def polygon_from_item(item: dict[str, Any]) -> tuple[tuple[float, float], ...]:
    polygon = item.get('polygon')
    if isinstance(polygon, list) and len(polygon) >= 4:
        arr = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
        return tuple((float(x), float(y)) for x, y in arr)
    xyxy = xyxy_from_item(item)
    if xyxy is None:
        return tuple()
    x1, y1, x2, y2 = xyxy
    return ((x1, y1), (x2, y1), (x2, y2), (x1, y2))


def line_orientation(line: dict[str, Any]) -> str:
    poly = polygon_from_item(line)
    if len(poly) >= 4:
        pts = np.asarray(poly[:4], dtype=np.float64)
        lengths = [float(np.linalg.norm(pts[(i + 1) % 4] - pts[i])) for i in range(4)]
        pair_a = (lengths[0] + lengths[2]) / 2.0
        pair_b = (lengths[1] + lengths[3]) / 2.0
        index = 0 if pair_a >= pair_b else 1
        vec = pts[(index + 1) % 4] - pts[index]
        return 'horizontal' if abs(vec[0]) >= abs(vec[1]) else 'vertical'

    xyxy = line_box_from_item(line)
    if xyxy is None:
        return 'vertical'
    x1, y1, x2, y2 = xyxy
    return 'horizontal' if (x2 - x1) >= (y2 - y1) else 'vertical'


def source_index_from_item(item: dict[str, Any], fallback: int) -> int:
    try:
        return int(item.get('source_block_index', fallback))
    except (TypeError, ValueError):
        return fallback


def load_align_masks(path: Path, expected_count: int | None = None) -> AlignMasks | None:
    if not path.is_file():
        return None

    with np.load(path) as data:
        smoothed = np.asarray(data.get('smoothed_masks', np.zeros((0, 0, 0), dtype=np.uint8)))
        outer = np.asarray(data.get('outer_body_masks', np.zeros((0, 0, 0), dtype=np.uint8)))
        accepted = np.asarray(data.get('accepted', np.zeros((0,), dtype=np.bool_)), dtype=np.bool_)

    if expected_count is not None and accepted.size == 0:
        accepted = np.zeros((expected_count,), dtype=np.bool_)
    return AlignMasks(smoothed_masks=smoothed, outer_body_masks=outer, accepted=accepted)


def _ocr_characters_by_box(measure_ocr: dict[str, Any], page_name: str) -> dict[tuple[int, int, int], dict[str, Any]]:
    result = {}
    for fallback_index, block in enumerate((measure_ocr.get('pages') or {}).get(page_name, []) or []):
        if not isinstance(block, dict):
            continue
        source_index = source_index_from_item(block, fallback_index)
        fit_by_position = {
            (int(fit.get('line_index', 0)), int(fit.get('character_index', 0))): fit
            for fit in (block.get('font_fit') or {}).get('character_results', []) or []
            if isinstance(fit, dict)
        }
        for character in block.get('ocr_characters', []) or []:
            if not isinstance(character, dict):
                continue
            line_index = int(character.get('line_index', 0))
            character_index = int(character.get('character_index', 0))
            item = dict(character)
            fit = fit_by_position.get((line_index, character_index))
            if isinstance(fit, dict):
                item['font_filter_accepted'] = fit.get('accepted') is True
            result[(source_index, line_index, character_index)] = item
    return result


def load_char_boxes(
    measure_debug: dict[str, Any],
    page_name: str,
    measure_ocr: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    ocr_by_box = _ocr_characters_by_box(measure_ocr or {}, page_name)
    processed_sources = {key[0] for key in ocr_by_box}
    font_debug_pages = measure_debug.get('font_size', {})
    for block_debug in font_debug_pages.get(page_name, []):
        source_index = source_index_from_item(block_debug, 0)
        boxes_by_line: dict[int, list[dict[str, Any]]] = {}
        for char_box in block_debug.get('char_boxes', []) or []:
            if isinstance(char_box, dict):
                boxes_by_line.setdefault(int(char_box.get('line_index', 0)), []).append(char_box)
        orientation = str((block_debug.get('font_size_debug') or {}).get('orientation') or 'vertical')
        for line_index, boxes in boxes_by_line.items():
            def sort_key(char_box: dict[str, Any]) -> tuple[float, float]:
                bbox = char_box.get('bbox') or [0, 0, 0, 0]
                x1, y1, x2, y2 = [float(value) for value in bbox]
                return ((x1 + x2) / 2, (y1 + y2) / 2) if orientation == 'horizontal' else ((y1 + y2) / 2, (x1 + x2) / 2)

            for character_index, char_box in enumerate(sorted(boxes, key=sort_key)):
                if not isinstance(char_box, dict):
                    continue
                item = dict(char_box)
                item['source_block_index'] = source_index
                item['character_index'] = character_index
                ocr_item = ocr_by_box.get((source_index, line_index, character_index))
                if isinstance(ocr_item, dict):
                    if source_index in processed_sources and ocr_item.get('status') != 'accepted':
                        continue
                    for key in (
                        'ocr_text', 'ocr_probability', 'status', 'selected_pad',
                        'font_filter_accepted',
                    ):
                        if ocr_item.get(key) is not None:
                            item[key] = ocr_item[key]
                result.append(item)
    return result


def normalize_measure_map(measure: dict[str, Any]) -> None:
    pages = measure.get('pages') or {}
    for items in pages.values():
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            if item.get('text_has_stroke') is None and item.get('has_outline') is not None:
                item['text_has_stroke'] = bool(item.get('has_outline'))
            item.pop('has_outline', None)


class CtdOverlayProcessor:
    """Read a folder produced by new_detect_folder.py and build overlay state."""

    def __init__(self, image_dir: str | os.PathLike[str]):
        self.image_dir = Path(image_dir).expanduser().resolve()
        self.ctd_dir = self.image_dir / 'ctd'
        self.progressing_dir = self.ctd_dir / 'progressing'
        self.mask_dir = self.progressing_dir / 'mask'
        self.align_mask_dir = self.progressing_dir / 'align' / 'masks'

        self.block_map_path = self.progressing_dir / 'block_map.json'
        self.line_trans_map_path = self.progressing_dir / 'line_trans_map.json'
        self.aligned_box_map_path = self.progressing_dir / 'aligned_box_map.json'
        self.ctd_measure_path = self.ctd_dir / CTD_MEASURE_JSON
        self.measure_path = self.ctd_measure_path
        self.measure_debug_path = self.ctd_dir / 'measure.debug.json'
        self.measure_ocr_path = self.ctd_dir / 'measure_ocr.json'

        self.block_map = load_json(self.block_map_path, required=False)
        self.line_trans_map = load_json(self.line_trans_map_path, required=False)
        self.aligned_box_map = load_json(self.aligned_box_map_path, required=False)
        self.measure = load_json(self.measure_path, required=False)
        normalize_measure_map(self.measure)
        self.measure_debug = load_json(self.measure_debug_path, required=False)
        self.measure_ocr = load_json(self.measure_ocr_path, required=False)
        self.source_images = find_source_images(self.image_dir)

    def page_names(self) -> list[str]:
        names = set()
        names.update((self.measure.get('pages') or {}).keys())
        names.update((self.aligned_box_map.get('transMap') or {}).keys())
        names.update((self.block_map.get('blockMap') or {}).keys())
        names.update(self.source_images)
        return sorted(names, key=natural_sort_key)

    def missing_required_data(self) -> list[Path]:
        paths = [
            self.block_map_path,
            self.aligned_box_map_path,
            self.ctd_measure_path,
        ]
        return [path for path in paths if not path.is_file()]

    def overlay_data_issues(self) -> list[str]:
        issues = [f'缺少檔案：{path.name}' for path in self.missing_required_data()]
        if issues:
            return issues

        if self.source_images and not (self.block_map.get('blockMap') or {}):
            issues.append('block_map.json 沒有任何頁面資料')
        if self.source_images and not (self.aligned_box_map.get('transMap') or {}):
            issues.append('aligned_box_map.json 沒有任何頁面資料')
        if self.source_images and not (self.measure.get('pages') or {}):
            issues.append(f'{CTD_MEASURE_JSON} 沒有任何頁面資料')
        return issues

    def has_overlay_data(self) -> bool:
        return not self.overlay_data_issues()

    def load_page(self, page_name: str, image_size: tuple[int, int] | None = None) -> PageOverlay:
        image_path = resolve_image_path(self.image_dir, page_name)
        stem = Path(page_name).stem
        mask_path = self.mask_dir / f'{stem}.png'
        align_mask_path = self.align_mask_dir / f'{stem}.npz'

        block_items = self.block_map.get('blockMap', {}).get(page_name, [])
        align_items = self.aligned_box_map.get('transMap', {}).get(page_name, [])
        measure_items = self.measure.get('pages', {}).get(page_name, [])
        line_items = self.line_trans_map.get('transMap', {}).get(page_name, [])
        if (
            isinstance(measure_items, list)
            and mask_path.is_file()
            and any(
                isinstance(item, dict)
                and (
                    item.get('text_color') is None
                    or (item.get('text_has_stroke') is None and item.get('has_outline') is None)
                    or item.get('need_inpaint') is None
                )
                for item in measure_items
            )
        ):
            try:
                enrich_measure_items(self.image_dir, page_name, measure_items)
            except Exception:
                pass

        blocks_by_source = {
            source_index_from_item(item, index): xyxy_from_item(item)
            for index, item in enumerate(block_items)
            if isinstance(item, dict)
        }
        align_by_source = {
            source_index_from_item(item, index): item
            for index, item in enumerate(align_items)
            if isinstance(item, dict)
        }

        boxes = []
        source_measure_items = measure_items if measure_items else align_items
        for index, measure_item in enumerate(source_measure_items):
            if not isinstance(measure_item, dict):
                continue
            source_index = source_index_from_item(measure_item, index)
            align_item = align_by_source.get(source_index, {})
            xyxy = xyxy_from_item(measure_item) or xyxy_from_item(align_item)
            if xyxy is None:
                continue
            center = (
                tuple_center(measure_item.get('center_normalized'))
                or tuple_center(align_item.get('new_center_normalized'))
                or tuple_center(align_item.get('final_center_normalized'))
                or normalized_center_from_xyxy(xyxy, image_size)
            )
            boxes.append(
                BoxOverlay(
                    source_block_index=source_index,
                    xyxy_pixel=xyxy,
                    center_normalized=center,
                    orientation=str(measure_item.get('orientation') or 'vertical'),
                    font_size=(
                        float(measure_item['font_size'])
                        if measure_item.get('font_size') is not None
                        else None
                    ),
                    font_size_method=measure_item.get('font_size_method'),
                    text_color=measure_item.get('text_color'),
                    text_has_stroke=(
                        bool(measure_item.get('text_has_stroke', measure_item.get('has_outline')))
                        if measure_item.get('text_has_stroke') is not None or measure_item.get('has_outline') is not None
                        else None
                    ),
                    need_inpaint=(
                        bool(measure_item['need_inpaint'])
                        if measure_item.get('need_inpaint') is not None
                        else None
                    ),
                    accepted=align_item.get('accepted'),
                    method=align_item.get('method'),
                    error_route=bool(align_item.get('error_route')),
                    error_reason=align_item.get('error_reason'),
                    block_xyxy_pixel=blocks_by_source.get(source_index),
                    raw_align=dict(align_item),
                    raw_measure=dict(measure_item),
                    measure_item_index=index if measure_items else None,
                )
            )

        lines = []
        for index, item in enumerate(line_items):
            if not isinstance(item, dict):
                continue
            xyxy = line_box_from_item(item)
            polygon = polygon_from_item(item)
            if xyxy is None or not polygon:
                continue
            lines.append(
                LineOverlay(
                    source_block_index=(
                        int(item['source_block_index'])
                        if item.get('source_block_index') is not None
                        else None
                    ),
                    polygon=polygon,
                    xyxy_pixel=xyxy,
                    font_width_px=(
                        float(item['font_width_px'])
                        if item.get('font_width_px') is not None
                        else None
                    ),
                    orientation=line_orientation(item),
                    raw=dict(item),
                )
            )

        align_masks = load_align_masks(align_mask_path, expected_count=len(align_items))
        return PageOverlay(
            page_name=page_name,
            image_path=image_path,
            mask_path=mask_path if mask_path.is_file() else None,
            align_mask_path=align_mask_path if align_mask_path.is_file() else None,
            boxes=boxes,
            lines=lines,
            align_masks=align_masks,
            char_boxes=load_char_boxes(self.measure_debug, page_name, self.measure_ocr),
        )

    def summary(self) -> dict[str, Any]:
        pages = self.page_names()
        box_count = sum(len(self.measure.get('pages', {}).get(page, [])) for page in pages)
        line_count = sum(len(self.line_trans_map.get('transMap', {}).get(page, [])) for page in pages)
        return {
            'image_dir': str(self.image_dir),
            'ctd_dir': str(self.ctd_dir),
            'pages': len(pages),
            'boxes': box_count,
            'lines': line_count,
            'source_images': len(self.source_images),
            'has_overlay_data': self.has_overlay_data(),
            'missing_required_data': [str(path) for path in self.missing_required_data()],
            'overlay_data_issues': self.overlay_data_issues(),
            'has_measure_debug': self.measure_debug_path.is_file(),
            'has_line_trans_map': self.line_trans_map_path.is_file(),
            'has_align_masks_dir': self.align_mask_dir.is_dir(),
            'measure_path': str(self.measure_path),
        }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='讀取 ctd JSON/NPZ 輸出，列出可用疊圖資料摘要。',
    )
    parser.add_argument('image_dir', help='包含原圖和 ctd/ 的圖片資料夾。')
    parser.add_argument('--page', default=None, help='指定要檢查的頁面檔名。')
    parser.add_argument('--json', action='store_true', help='用 JSON 格式輸出。')
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    processor = CtdOverlayProcessor(args.image_dir)
    if args.page:
        page = processor.load_page(args.page)
        data = {
            'page': page.page_name,
            'image_path': str(page.image_path),
            'mask_path': str(page.mask_path) if page.mask_path else None,
            'align_mask_path': str(page.align_mask_path) if page.align_mask_path else None,
            'boxes': len(page.boxes),
            'lines': len(page.lines),
            'char_boxes': len(page.char_boxes),
            'align_masks': page.align_masks.count if page.align_masks else 0,
        }
    else:
        data = processor.summary()

    if args.json:
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return

    for key, value in data.items():
        print(f'{key}: {value}')


if __name__ == '__main__':
    main()
