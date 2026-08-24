"""Estimate paragraph font size from OCR boxes and cached font ink metrics."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FONT_PATH = PROJECT_ROOT / 'assets' / 'fonts' / 'NotoSansCJKjp-Medium.otf'
DEFAULT_METRICS_PATH = PROJECT_ROOT / 'assets' / 'fonts' / 'NotoSansCJKjp-Medium.ink-metrics.json'
METRICS_SCHEMA_VERSION = 1


def ocr_characters(text: object) -> list[str]:
    normalized = unicodedata.normalize('NFC', str(text or ''))
    return [char for char in normalized if not char.isspace()]


def _box_size(box: dict[str, Any]) -> tuple[float, float]:
    width = float(box.get('width') or 0)
    height = float(box.get('height') or 0)
    if width > 0 and height > 0:
        return width, height
    bbox = box.get('bbox')
    if isinstance(bbox, list) and len(bbox) == 4:
        return max(0.0, float(bbox[2]) - float(bbox[0])), max(0.0, float(bbox[3]) - float(bbox[1]))
    return 0.0, 0.0


@lru_cache(maxsize=4)
def load_font_ink_metrics(metrics_path: str | Path = DEFAULT_METRICS_PATH) -> dict[str, Any]:
    path = Path(metrics_path)
    if not path.is_file():
        raise FileNotFoundError(f'找不到字型墨跡緩存：{path}')
    data = json.loads(path.read_text(encoding='utf-8'))
    if data.get('schema_version') != METRICS_SCHEMA_VERSION:
        raise RuntimeError(f'不支援的字型墨跡緩存版本：{data.get("schema_version")}')
    metrics = data.get('metrics') or {}
    if metrics.get('units') != 'reference_pixel_ratio':
        raise RuntimeError(f'不支援的字型墨跡緩存單位：{metrics.get("units")}')
    if not isinstance(data.get('glyphs'), dict):
        raise RuntimeError(f'字型墨跡緩存缺少 glyphs：{path}')
    return data


@lru_cache(maxsize=4)
def _sha256_file(path_text: str) -> str:
    digest = hashlib.sha256()
    with Path(path_text).open('rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def validate_font_ink_metrics(
    font_path: str | Path = DEFAULT_FONT_PATH,
    metrics_path: str | Path = DEFAULT_METRICS_PATH,
) -> dict[str, Any]:
    font = Path(font_path)
    if not font.is_file():
        raise FileNotFoundError(f'找不到字體：{font}')
    metrics = load_font_ink_metrics(metrics_path)
    expected_hash = str((metrics.get('font') or {}).get('sha256') or '')
    if expected_hash and _sha256_file(str(font)) != expected_hash:
        raise RuntimeError(f'字型墨跡緩存與字體不匹配，請重新生成：{metrics_path}')
    return metrics


def character_ink_ratio(
    character: str,
    metrics_path: str | Path = DEFAULT_METRICS_PATH,
) -> tuple[float, float] | None:
    values = (load_font_ink_metrics(metrics_path).get('glyphs') or {}).get(character)
    if not isinstance(values, list) or len(values) != 2:
        return None
    width_ratio = float(values[0])
    height_ratio = float(values[1])
    return (width_ratio, height_ratio) if width_ratio > 0 and height_ratio > 0 else None


def character_ink_size(
    character: str,
    pixel_size: float,
    metrics_path: str | Path = DEFAULT_METRICS_PATH,
) -> tuple[float, float] | None:
    """Scale one cached glyph ink ratio to a requested font size."""
    ratios = character_ink_ratio(character, metrics_path)
    if ratios is None or pixel_size <= 0:
        return None
    return ratios[0] * float(pixel_size), ratios[1] * float(pixel_size)


def _relative_error(predicted: float, target: float) -> float:
    if predicted <= 0 or target <= 0:
        return math.inf
    return abs(predicted - target) / target


def _round_positive(value: float) -> int:
    return max(1, int(math.floor(float(value) + 0.5)))


def fit_character_pixel_size(
    character: str,
    box: dict[str, Any],
    orientation: str,
    metrics_path: str | Path = DEFAULT_METRICS_PATH,
) -> dict[str, Any] | None:
    """Infer one character's font size directly from cached glyph ratios."""
    target_width, target_height = _box_size(box)
    ratios = character_ink_ratio(character, metrics_path)
    if target_width <= 0 or target_height <= 0 or ratios is None:
        return None
    width_ratio, height_ratio = ratios
    width_size = target_width / width_ratio
    height_size = target_height / height_ratio
    if orientation == 'horizontal':
        width_weight, height_weight = 0.25, 0.75
    else:
        width_weight, height_weight = 0.75, 0.25
    estimated_size = width_size * width_weight + height_size * height_weight
    pixel_size = _round_positive(estimated_size)
    rendered_width = width_ratio * pixel_size
    rendered_height = height_ratio * pixel_size
    width_error = _relative_error(rendered_width, target_width)
    height_error = _relative_error(rendered_height, target_height)
    error = width_error * width_weight + height_error * height_weight
    return {
        'character': character,
        'estimated_pixel_size': round(float(estimated_size), 3),
        'pixel_size': pixel_size,
        'error': round(float(error), 4),
        'target_width': round(target_width, 2),
        'target_height': round(target_height, 2),
        'rendered_width': round(float(rendered_width), 2),
        'rendered_height': round(float(rendered_height), 2),
        'font_width_ratio': width_ratio,
        'font_height_ratio': height_ratio,
        'bbox': box.get('bbox'),
    }


def _median_absolute_deviation(values: list[float]) -> tuple[float, float]:
    median = float(statistics.median(values))
    mad = float(statistics.median(abs(value - median) for value in values))
    return median, mad


def fit_ocr_item(
    item: dict[str, Any],
    *,
    font_path: str | Path = DEFAULT_FONT_PATH,
    metrics_path: str | Path = DEFAULT_METRICS_PATH,
    maximum_fit_error: float = 0.45,
    minimum_reliable_characters: int = 3,
) -> dict[str, Any]:
    validate_font_ink_metrics(font_path, metrics_path)
    orientation = str(item.get('orientation') or 'vertical')
    character_results = []
    accepted_fits = []
    accepted_boxes = []
    for character_item in item.get('ocr_characters', []) or []:
        if not isinstance(character_item, dict):
            continue
        characters = ocr_characters(character_item.get('ocr_text'))
        result = {
            'line_index': character_item.get('line_index'),
            'character_index': character_item.get('character_index'),
            'ocr_text': character_item.get('ocr_text') or '',
            'ocr_probability': float(character_item.get('ocr_probability') or 0),
            'bbox': character_item.get('bbox'),
            'accepted': False,
        }
        if character_item.get('status') != 'accepted':
            result['reason'] = str(character_item.get('status') or 'ocr_rejected')
        elif len(characters) != 1:
            result['reason'] = 'not_single_character'
        else:
            fit = fit_character_pixel_size(
                characters[0],
                character_item,
                orientation,
                metrics_path,
            )
            if fit is None:
                result['reason'] = 'font_metric_unavailable'
            else:
                result.update(fit)
                result['accepted'] = float(fit['error']) <= maximum_fit_error
                if result['accepted']:
                    accepted_fits.append(result)
                    accepted_boxes.append(character_item)
                else:
                    result['reason'] = 'fit_error_too_large'
        character_results.append(result)

    sizes = [float(fit['estimated_pixel_size']) for fit in accepted_fits]
    if not sizes:
        status = 'no_reliable_characters'
        robust_fits: list[dict[str, Any]] = []
        median = None
        mad = None
    else:
        median, mad = _median_absolute_deviation(sizes)
        tolerance = max(2.0, mad * 3.0)
        robust_fits = [
            fit
            for fit in accepted_fits
            if abs(float(fit['estimated_pixel_size']) - median) <= tolerance
        ]
        robust_ids = {id(fit) for fit in robust_fits}
        for fit in accepted_fits:
            if id(fit) not in robust_ids:
                fit['accepted'] = False
                fit['reason'] = 'font_size_outlier'
        status = 'ready' if len(robust_fits) >= minimum_reliable_characters else 'too_few_reliable_characters'

    robust_sizes = [float(fit['estimated_pixel_size']) for fit in robust_fits]
    suggested = _round_positive(float(statistics.median(robust_sizes))) if status == 'ready' else None
    robust_positions = {
        (fit.get('line_index'), fit.get('character_index'))
        for fit in robust_fits
    }
    filtered_boxes = [
        dict(box)
        for box in accepted_boxes
        if (box.get('line_index'), box.get('character_index')) in robust_positions
    ]
    return {
        'status': status,
        'font_path': str(Path(font_path)),
        'metrics_path': str(Path(metrics_path)),
        'original_font_size': float(item.get('font_size') or 0),
        'suggested_font_size': suggested,
        'accepted_character_count': len(robust_fits),
        'rejected_character_count': max(0, len(character_results) - len(robust_fits)),
        'total_fitted_character_count': len(accepted_fits),
        'minimum_reliable_characters': minimum_reliable_characters,
        'median_before_outlier_filter': round(float(median), 2) if median is not None else None,
        'mad': round(float(mad), 2) if mad is not None else None,
        'character_results': character_results,
        'filtered_char_boxes': filtered_boxes,
    }


def calibrate_ocr_output(
    output: dict[str, Any],
    font_path: str | Path = DEFAULT_FONT_PATH,
    metrics_path: str | Path = DEFAULT_METRICS_PATH,
) -> int:
    metrics = validate_font_ink_metrics(font_path, metrics_path)
    ready = 0
    for items in (output.get('pages') or {}).values():
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            fit = fit_ocr_item(item, font_path=font_path, metrics_path=metrics_path)
            item['font_fit'] = fit
            if fit.get('status') == 'ready':
                ready += 1
    output['font_calibration'] = {
        'method': 'mit48_cached_font_ink_ratio',
        'font_path': str(Path(font_path)),
        'metrics_path': str(Path(metrics_path)),
        'font_sha256': (metrics.get('font') or {}).get('sha256'),
        'glyph_count': (metrics.get('counts') or {}).get('glyph_count'),
        'rounding': 'nearest_integer_half_up',
    }
    return ready
