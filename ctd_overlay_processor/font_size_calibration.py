"""Fit detected character ink boxes to a concrete Qt font."""

from __future__ import annotations

import math
import statistics
import unicodedata
from pathlib import Path
from typing import Any

from PySide6.QtGui import QFont, QFontDatabase, QFontMetricsF


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FONT_PATH = PROJECT_ROOT / 'assets' / 'fonts' / 'NotoSansCJKjp-Medium.otf'


def load_application_font(font_path: str | Path = DEFAULT_FONT_PATH) -> tuple[int, str]:
    path = Path(font_path)
    if not path.is_file():
        raise FileNotFoundError(f'找不到字體：{path}')
    font_id = QFontDatabase.addApplicationFont(str(path))
    if font_id < 0:
        raise RuntimeError(f'Qt 無法載入字體：{path}')
    families = QFontDatabase.applicationFontFamilies(font_id)
    if not families:
        raise RuntimeError(f'字體沒有可用 family：{path}')
    return font_id, families[0]


def ocr_characters(text: object) -> list[str]:
    normalized = unicodedata.normalize('NFC', str(text or ''))
    return [char for char in normalized if not char.isspace()]


def character_is_fit_candidate(char: str) -> bool:
    if not char:
        return False
    codepoint = ord(char[0])
    return (
        0x3040 <= codepoint <= 0x30FF
        or 0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
    )


def _box_size(box: dict[str, Any]) -> tuple[float, float]:
    width = float(box.get('width') or 0)
    height = float(box.get('height') or 0)
    if width > 0 and height > 0:
        return width, height
    bbox = box.get('bbox')
    if isinstance(bbox, list) and len(bbox) == 4:
        return max(0.0, float(bbox[2]) - float(bbox[0])), max(0.0, float(bbox[3]) - float(bbox[1]))
    return 0.0, 0.0


def _relative_error(predicted: float, target: float) -> float:
    if predicted <= 0 or target <= 0:
        return math.inf
    return abs(predicted - target) / target


def fit_character_pixel_size(
    char: str,
    box: dict[str, Any],
    font_family: str,
    orientation: str,
    initial_size: float,
) -> dict[str, Any] | None:
    target_w, target_h = _box_size(box)
    if target_w <= 0 or target_h <= 0 or not character_is_fit_candidate(char):
        return None

    minimum = max(4, int(math.floor(initial_size * 0.55)))
    maximum = min(999, max(minimum, int(math.ceil(initial_size * 1.70))))
    font = QFont(font_family)
    font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
    best = None
    for pixel_size in range(minimum, maximum + 1):
        font.setPixelSize(pixel_size)
        metrics = QFontMetricsF(font)
        if not metrics.inFontUcs4(ord(char)):
            return None
        rect = metrics.tightBoundingRect(char)
        predicted_w = float(rect.width())
        predicted_h = float(rect.height())
        width_error = _relative_error(predicted_w, target_w)
        height_error = _relative_error(predicted_h, target_h)
        if orientation == 'horizontal':
            error = height_error * 0.75 + width_error * 0.25
        else:
            error = width_error * 0.75 + height_error * 0.25
        candidate = (error, abs(pixel_size - initial_size), pixel_size, predicted_w, predicted_h)
        if best is None or candidate < best:
            best = candidate
    if best is None or not math.isfinite(best[0]):
        return None
    return {
        'character': char,
        'pixel_size': int(best[2]),
        'error': round(float(best[0]), 4),
        'target_width': round(target_w, 2),
        'target_height': round(target_h, 2),
        'rendered_width': round(float(best[3]), 2),
        'rendered_height': round(float(best[4]), 2),
        'bbox': box.get('bbox'),
    }


def _median_absolute_deviation(values: list[float]) -> tuple[float, float]:
    median = float(statistics.median(values))
    mad = float(statistics.median(abs(value - median) for value in values))
    return median, mad


def fit_ocr_item(
    item: dict[str, Any],
    font_family: str,
    *,
    maximum_fit_error: float = 0.45,
    minimum_reliable_characters: int = 3,
    maximum_size_change_ratio: float = 0.35,
    minimum_ocr_probability: float = 0.60,
) -> dict[str, Any]:
    initial_size = max(1.0, float(item.get('font_size') or 1))
    character_results = []
    accepted_fits = []
    for character_item in item.get('ocr_characters', []) or []:
        if not isinstance(character_item, dict):
            continue
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
            character_results.append(result)
            continue
        if result['ocr_probability'] < minimum_ocr_probability:
            result['reason'] = 'ocr_probability_too_low'
            character_results.append(result)
            continue
        chars = ocr_characters(character_item.get('ocr_text'))
        if len(chars) != 1 or not character_is_fit_candidate(chars[0]):
            result['reason'] = 'unsupported_or_non_japanese_text'
            character_results.append(result)
            continue
        orientation = str(character_item.get('orientation') or item.get('orientation') or 'vertical')
        fit = fit_character_pixel_size(chars[0], character_item, font_family, orientation, initial_size)
        if fit is None:
            result['reason'] = 'font_fit_unavailable'
            character_results.append(result)
            continue
        result.update(fit)
        result['accepted'] = fit['error'] <= maximum_fit_error
        if not result['accepted']:
            result['reason'] = 'fit_error_too_large'
        else:
            accepted_fits.append(result)
        character_results.append(result)

    sizes = [float(fit['pixel_size']) for fit in accepted_fits]
    if not sizes:
        return {
            'status': 'no_reliable_characters',
            'font_family': font_family,
            'original_font_size': initial_size,
            'suggested_font_size': None,
            'accepted_character_count': 0,
            'character_results': character_results,
        }

    median, mad = _median_absolute_deviation(sizes)
    tolerance = max(2.0, mad * 3.0)
    robust_fits = [fit for fit in accepted_fits if abs(float(fit['pixel_size']) - median) <= tolerance]
    robust_sizes = [float(fit['pixel_size']) for fit in robust_fits]
    if len(robust_fits) < minimum_reliable_characters:
        return {
            'status': 'too_few_reliable_characters',
            'font_family': font_family,
            'original_font_size': initial_size,
            'suggested_font_size': None,
            'accepted_character_count': len(robust_fits),
            'total_fitted_character_count': len(accepted_fits),
            'minimum_reliable_characters': minimum_reliable_characters,
            'median_before_outlier_filter': round(median, 2),
            'mad': round(mad, 2),
            'character_results': character_results,
        }
    suggested = max(1, int(math.floor(float(statistics.median(robust_sizes)) + 0.5)))
    size_change_ratio = abs(float(suggested) - initial_size) / initial_size
    if size_change_ratio > maximum_size_change_ratio:
        return {
            'status': 'suggestion_too_far_from_detected',
            'font_family': font_family,
            'original_font_size': initial_size,
            'suggested_font_size': None,
            'rejected_suggested_font_size': suggested,
            'size_change_ratio': round(size_change_ratio, 4),
            'maximum_size_change_ratio': maximum_size_change_ratio,
            'accepted_character_count': len(robust_fits),
            'total_fitted_character_count': len(accepted_fits),
            'median_before_outlier_filter': round(median, 2),
            'mad': round(mad, 2),
            'character_results': character_results,
        }
    return {
        'status': 'ready',
        'font_family': font_family,
        'original_font_size': initial_size,
        'suggested_font_size': suggested,
        'size_change_ratio': round(size_change_ratio, 4),
        'accepted_character_count': len(robust_fits),
        'total_fitted_character_count': len(accepted_fits),
        'median_before_outlier_filter': round(median, 2),
        'mad': round(mad, 2),
        'character_results': character_results,
    }


def calibrate_ocr_output(output: dict[str, Any], font_family: str) -> int:
    ready = 0
    for items in (output.get('pages') or {}).values():
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            fit = fit_ocr_item(item, font_family)
            item['font_fit'] = fit
            if fit.get('status') == 'ready':
                ready += 1
    output['font_calibration'] = {'font_family': font_family}
    return ready
