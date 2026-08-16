"""Filter OCR character boxes and estimate block font size from geometry."""

from __future__ import annotations

import math
import unicodedata
from typing import Any


def ocr_characters(text: object) -> list[str]:
    normalized = unicodedata.normalize('NFC', str(text or ''))
    return [char for char in normalized if not char.isspace()]


def _percentile(values: list[float], percentile: float) -> float:
    sorted_values = sorted(float(value) for value in values if float(value) > 0)
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * percentile / 100.0
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _dominant_values(values: list[float]) -> list[float]:
    valid = [float(value) for value in values if float(value) > 0]
    if not valid:
        return []
    threshold = max(1.0, _percentile(valid, 75) * 0.55)
    selected = [value for value in valid if value >= threshold]
    return selected or valid


def _box_size(box: dict[str, Any]) -> tuple[float, float]:
    width = float(box.get('width') or 0)
    height = float(box.get('height') or 0)
    if width > 0 and height > 0:
        return width, height
    bbox = box.get('bbox')
    if isinstance(bbox, list) and len(bbox) == 4:
        return max(0.0, float(bbox[2]) - float(bbox[0])), max(0.0, float(bbox[3]) - float(bbox[1]))
    return 0.0, 0.0


def _reliable_square_limit(width: float, height: float) -> float:
    return max(5.0, max(width, height) * 0.16)


def estimate_font_size_from_boxes(
    boxes: list[dict[str, Any]],
    orientation: str,
) -> tuple[float | None, dict[str, Any]]:
    sizes = [_box_size(box) for box in boxes]
    widths = [width for width, _ in sizes if width > 0]
    heights = [height for _, height in sizes if height > 0]
    if not widths or not heights:
        return None, {'method': 'filtered_char_box_geometry', 'accepted': False, 'reason': 'no_valid_boxes'}

    primary_values = _dominant_values(heights if orientation == 'horizontal' else widths)
    secondary_values = _dominant_values(widths if orientation == 'horizontal' else heights)
    if not primary_values or not secondary_values:
        return None, {'method': 'filtered_char_box_geometry', 'accepted': False, 'reason': 'no_dominant_values'}

    primary_percentile = 100 if orientation == 'horizontal' else 60
    secondary_percentile = 75 if orientation == 'horizontal' else 100
    primary_size = _percentile(primary_values, primary_percentile)
    secondary_limit = max(primary_size * 1.6, primary_size + 8.0)
    secondary_filtered = []
    for box in boxes:
        width, height = _box_size(box)
        primary = height if orientation == 'horizontal' else width
        secondary = width if orientation == 'horizontal' else height
        if primary <= 0 or secondary <= 0 or secondary > secondary_limit:
            continue
        if len(boxes) <= 4 and secondary > primary_size + 5.0 and abs(height - width) > _reliable_square_limit(width, height):
            continue
        secondary_filtered.append(secondary)
    if not secondary_filtered:
        secondary_filtered = secondary_values
    secondary_size = _percentile(secondary_filtered, secondary_percentile)
    font_size = max(primary_size, secondary_size)

    reliable_min_size = max(1.0, primary_size * 0.55)
    required_square_count = 1 if len(boxes) <= 4 else 4
    reliable_square_sizes = []
    for box in boxes:
        width, height = _box_size(box)
        candidate = max(width, height)
        if (
            width > 0
            and height > 0
            and candidate >= reliable_min_size
            and abs(height - width) <= _reliable_square_limit(width, height)
        ):
            reliable_square_sizes.append(candidate)
    method = 'filtered_char_box_geometry'
    reliable_square_size = None
    if len(reliable_square_sizes) >= required_square_count:
        reliable_square_size = max(reliable_square_sizes)
        font_size = reliable_square_size
        method = 'filtered_char_box_reliable_square'

    return float(font_size), {
        'method': method,
        'accepted': True,
        'orientation': orientation,
        'char_count': len(boxes),
        'widths': widths,
        'heights': heights,
        'primary_dimension': 'H' if orientation == 'horizontal' else 'W',
        'secondary_dimension': 'W' if orientation == 'horizontal' else 'H',
        'primary_percentile': primary_percentile,
        'secondary_percentile': secondary_percentile,
        'primary_size': primary_size,
        'secondary_size': secondary_size,
        'secondary_filtered': secondary_filtered,
        'reliable_square_required_count': required_square_count,
        'reliable_square_sizes': reliable_square_sizes,
        'reliable_square_size': reliable_square_size,
        'font_size': float(font_size),
    }


def fit_ocr_item(item: dict[str, Any], *, minimum_reliable_characters: int = 3) -> dict[str, Any]:
    valid_boxes = []
    character_results = []
    for character_item in item.get('ocr_characters', []) or []:
        if not isinstance(character_item, dict):
            continue
        text = ocr_characters(character_item.get('ocr_text'))
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
        elif len(text) != 1:
            result['reason'] = 'not_single_character'
        else:
            result['accepted'] = True
            valid_boxes.append(character_item)
        character_results.append(result)

    orientation = str(item.get('orientation') or 'vertical')
    font_size, geometry = estimate_font_size_from_boxes(valid_boxes, orientation)
    if font_size is None:
        status = 'no_reliable_characters'
    elif len(valid_boxes) < minimum_reliable_characters:
        status = 'too_few_reliable_characters'
    else:
        status = 'ready'

    return {
        'status': status,
        'original_font_size': float(item.get('font_size') or 0),
        'suggested_font_size': round(float(font_size), 1) if font_size is not None else None,
        'accepted_character_count': len(valid_boxes),
        'rejected_character_count': max(0, len(character_results) - len(valid_boxes)),
        'minimum_reliable_characters': minimum_reliable_characters,
        'geometry': geometry,
        'character_results': character_results,
        'filtered_char_boxes': [dict(box) for box in valid_boxes],
    }


def calibrate_ocr_output(output: dict[str, Any]) -> int:
    ready = 0
    for items in (output.get('pages') or {}).values():
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            fit = fit_ocr_item(item)
            item['font_fit'] = fit
            if fit.get('status') == 'ready':
                ready += 1
    output['font_calibration'] = {
        'method': 'mit48_filtered_char_box_geometry',
        'uses_qfontmetrics': False,
    }
    return ready
