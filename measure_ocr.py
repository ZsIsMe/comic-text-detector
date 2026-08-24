#!/usr/bin/env python3
"""Run mit48px CTC OCR independently for character boxes in measure.json."""

from __future__ import annotations

import argparse
import json
import os.path as osp
import time
import unicodedata
from pathlib import Path
from typing import Any

import cv2

from ctd_overlay_processor.font_size_calibration import calibrate_ocr_output
from ctd_overlay_processor.mit48px_ocr import (
    DEFAULT_ALPHABET_PATH,
    DEFAULT_IMPLEMENTATION_PATH,
    DEFAULT_MODEL_PATH,
    Mit48pxCtcOcr,
    prepare_character_crop,
)


def _load_json(path: str | Path) -> dict:
    with open(path, 'r', encoding='utf8') as file:
        return json.load(file)


def _write_json(path: str | Path, data: dict) -> None:
    with open(path, 'w', encoding='utf8') as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def _resolve_input_paths(path1: str, path2: str | None, measure_json: str | None) -> tuple[str, str]:
    if measure_json is not None:
        return measure_json, path1
    if path2 is None:
        return osp.join(path1, 'ctd', 'measure.json'), path1
    return path1, path2


def _image_path_for_page(image_dir: str | Path, page_name: str) -> Path:
    image_dir = Path(image_dir)
    path = image_dir / page_name
    if path.is_file():
        return path
    stem = Path(page_name).stem
    for extension in ('.png', '.jpg', '.jpeg', '.bmp', '.webp'):
        candidate = image_dir / f'{stem}{extension}'
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f'找不到原圖：{page_name}')


def _source_index(item: dict[str, Any], fallback: int) -> int:
    try:
        return int(item.get('source_block_index', fallback))
    except (TypeError, ValueError):
        return fallback


def _debug_items_by_source(measure_debug: dict, page_name: str) -> dict[int, dict]:
    result = {}
    for fallback, item in enumerate((measure_debug.get('font_size') or {}).get(page_name, []) or []):
        if isinstance(item, dict):
            result[_source_index(item, fallback)] = item
    return result


def _ordered_char_boxes(char_boxes: list[dict], orientation: str) -> list[dict]:
    def key(box: dict) -> tuple[float, float]:
        bbox = box.get('bbox') or [0, 0, 0, 0]
        x1, y1, x2, y2 = [float(value) for value in bbox]
        if orientation == 'horizontal':
            return (x1 + x2) / 2.0, (y1 + y2) / 2.0
        return (y1 + y2) / 2.0, (x1 + x2) / 2.0

    return sorted((dict(box) for box in char_boxes if isinstance(box, dict)), key=key)


def _character_tasks_for_page(
    items: list[dict],
    measure_debug: dict,
    page_name: str,
    source_block_filter: int | None,
) -> tuple[list[dict], list[dict]]:
    debug_by_source = _debug_items_by_source(measure_debug, page_name)
    output_items = []
    tasks = []
    for item_index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        source_index = _source_index(item, item_index)
        if source_block_filter is not None and source_index != source_block_filter:
            continue
        output_item = dict(item)
        output_item['measure_item_index'] = item_index
        output_item['ocr_characters'] = []
        output_index = len(output_items)
        output_items.append(output_item)

        grouped: dict[int, list[dict]] = {}
        for char_box in debug_by_source.get(source_index, {}).get('char_boxes', []) or []:
            if not isinstance(char_box, dict):
                continue
            try:
                line_index = int(char_box.get('line_index', 0))
            except (TypeError, ValueError):
                line_index = 0
            grouped.setdefault(line_index, []).append(char_box)

        orientation = str(item.get('orientation') or 'vertical')
        for line_index, boxes in sorted(grouped.items()):
            for character_index, box in enumerate(_ordered_char_boxes(boxes, orientation)):
                bbox = box.get('bbox')
                if not isinstance(bbox, list) or len(bbox) != 4:
                    continue
                tasks.append({
                    'output_index': output_index,
                    'source_block_index': source_index,
                    'line_index': line_index,
                    'character_index': character_index,
                    'orientation': orientation,
                    'box': box,
                })
    return output_items, tasks


def _clean_single_character(text: object) -> str | None:
    normalized = unicodedata.normalize('NFC', str(text or '')).strip()
    characters = [character for character in normalized if not character.isspace()]
    return characters[0] if len(characters) == 1 else None


def _choose_variant(variants: list[dict], minimum_probability: float) -> dict:
    candidates = []
    for variant in variants:
        character = _clean_single_character(variant.get('text'))
        if character is None:
            continue
        candidates.append((float(variant.get('probability') or 0), character, variant))
    if not candidates:
        return {'status': 'not_single_character', 'ocr_text': '', 'ocr_probability': 0.0}
    probability, character, variant = max(candidates, key=lambda candidate: candidate[0])
    if probability < minimum_probability:
        return {
            'status': 'low_confidence',
            'ocr_text': character,
            'ocr_probability': round(probability, 6),
            'selected_pad': variant.get('pad'),
        }
    agreeing = sum(
        1 for _, candidate_character, _ in candidates
        if candidate_character == character
    )
    return {
        'status': 'accepted',
        'ocr_text': character,
        'ocr_probability': round(probability, 6),
        'selected_pad': variant.get('pad'),
        'agreeing_variant_count': agreeing,
    }


def _iter_pages(measure: dict, page_filter: str | None) -> list[tuple[str, list[dict]]]:
    pages = measure.get('pages') or {}
    if page_filter is not None:
        return [(page_filter, pages.get(page_filter, []))]
    return list(pages.items())


def apply_calibrated_font_sizes(
    measure: dict,
    output: dict,
    *,
    even_font_size: bool = False,
) -> int:
    changed = 0
    measure_pages = measure.get('pages') or {}
    for page_name, output_items in (output.get('pages') or {}).items():
        measure_items = measure_pages.get(page_name)
        if not isinstance(output_items, list) or not isinstance(measure_items, list):
            continue
        for output_item in output_items:
            if not isinstance(output_item, dict):
                continue
            fit = output_item.get('font_fit') or {}
            suggested = fit.get('suggested_font_size')
            item_index = output_item.get('measure_item_index')
            if fit.get('status') != 'ready' or not isinstance(suggested, int) or not isinstance(item_index, int):
                continue
            if not (0 <= item_index < len(measure_items)) or not isinstance(measure_items[item_index], dict):
                continue
            font_size = max(1, min(999, int(suggested)))
            if even_font_size and font_size % 2:
                font_size += 1
            measure_item = measure_items[item_index]
            old_size = measure_item.get('font_size')
            measure_item.setdefault('font_size_detected', old_size)
            measure_item['font_size'] = font_size
            measure_item['font_size_method'] = 'mit48_cached_font_ink_ratio'
            fit['applied_font_size'] = font_size
            try:
                unchanged = int(round(float(old_size))) == font_size
            except (TypeError, ValueError):
                unchanged = False
            if not unchanged:
                changed += 1
    return changed


def run(
    measure_path: str,
    image_dir: str,
    output_path: str | None,
    model_path: str,
    alphabet_path: str,
    implementation_path: str,
    device: str,
    pads: list[int],
    minimum_probability: float,
    page: str | None,
    measure_debug_path: str | None,
    source_block_index: int | None,
    limit_pages: int | None,
    limit_items: int | None,
    batch_size: int,
    save_crops: str | None,
    dry_run: bool,
) -> str:
    measure = _load_json(measure_path)
    measure_debug_path = measure_debug_path or osp.join(osp.dirname(measure_path), 'measure.debug.json')
    measure_debug = _load_json(measure_debug_path) if osp.isfile(measure_debug_path) else {}
    output_path = output_path or osp.join(osp.dirname(measure_path), 'measure_ocr.json')
    pages = _iter_pages(measure, page)
    if limit_pages is not None:
        pages = pages[:limit_pages]

    save_crops_path = Path(save_crops) if save_crops else None
    if save_crops_path is not None:
        save_crops_path.mkdir(parents=True, exist_ok=True)
    ocr = None if dry_run else Mit48pxCtcOcr(
        device,
        model_path=model_path,
        alphabet_path=alphabet_path,
        implementation_path=implementation_path,
    )
    output = {
        'ocr_engine': 'mit48px_ctc_character',
        'model_path': str(model_path),
        'alphabet_path': str(alphabet_path),
        'device': ocr.device if ocr is not None else device,
        'pads': pads,
        'minimum_probability': minimum_probability,
        'pages': {},
    }

    start_time = time.perf_counter()
    total_pages = len(pages)
    for page_index, (page_name, items) in enumerate(pages, start=1):
        page_start = time.perf_counter()
        image_bgr = cv2.imread(str(_image_path_for_page(image_dir, page_name)), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f'無法讀取原圖：{page_name}')
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        output_items, tasks = _character_tasks_for_page(
            items, measure_debug, page_name, source_block_index,
        )
        if limit_items is not None:
            output_items = output_items[:limit_items]
            allowed = set(range(len(output_items)))
            tasks = [task for task in tasks if task['output_index'] in allowed]
        print(
            f'[{page_index}/{total_pages}] OCR {page_name}: '
            f'{len(output_items)} boxes, {len(tasks)} characters',
            flush=True,
        )

        crops = []
        crop_owners = []
        for task_index, task in enumerate(tasks):
            bbox = task['box']['bbox']
            for pad in pads:
                crop = prepare_character_crop(image_rgb, bbox, pad)
                crops.append(crop)
                crop_owners.append((task_index, pad))
                if save_crops_path is not None:
                    name = (
                        f'{Path(page_name).stem}-b{task["source_block_index"]:03d}'
                        f'-l{task["line_index"]:02d}-c{task["character_index"]:02d}-p{pad}.png'
                    )
                    cv2.imwrite(str(save_crops_path / name), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))

        variants_by_task: list[list[dict]] = [[] for _ in tasks]
        if dry_run:
            recognized = [{'text': '', 'probability': 0.0, 'character_probabilities': []} for _ in crops]
        else:
            recognized = ocr.recognize_batch(crops, batch_size=batch_size) if ocr is not None else []
        for (task_index, pad), result in zip(crop_owners, recognized):
            variants_by_task[task_index].append({**result, 'pad': pad})

        for task, variants in zip(tasks, variants_by_task):
            selection = _choose_variant(variants, minimum_probability)
            output_items[task['output_index']]['ocr_characters'].append({
                'line_index': task['line_index'],
                'character_index': task['character_index'],
                'orientation': task['orientation'],
                **task['box'],
                'ocr_variants': variants,
                **selection,
            })
        for output_item in output_items:
            characters = output_item['ocr_characters']
            characters.sort(key=lambda item: (int(item['line_index']), int(item['character_index'])))
            lines: dict[int, list[str]] = {}
            for character in characters:
                lines.setdefault(int(character['line_index']), []).append(
                    str(character.get('ocr_text') or '□'),
                )
            output_item['ocr_text'] = '\n'.join(''.join(line) for _, line in sorted(lines.items()))
        output['pages'][page_name] = output_items
        print(
            f'[{page_index}/{total_pages}] 完成 {page_name} '
            f'({time.perf_counter() - page_start:.1f}s)',
            flush=True,
        )

    _write_json(output_path, output)
    print(f'總耗時：{time.perf_counter() - start_time:.1f}s', flush=True)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description='Run mit48px CTC OCR independently for each character box.')
    parser.add_argument('path1', help='Image folder, or measure.json in the old two-argument form')
    parser.add_argument('path2', nargs='?', default=None, help='Image folder for old two-argument form')
    parser.add_argument('--measure-json', default=None)
    parser.add_argument('--output', default=None)
    parser.add_argument('--model', default=str(DEFAULT_MODEL_PATH))
    parser.add_argument('--alphabet', default=str(DEFAULT_ALPHABET_PATH))
    parser.add_argument('--implementation', default=str(DEFAULT_IMPLEMENTATION_PATH))
    parser.add_argument('--device', default='cpu', choices=['cpu', 'mps', 'cuda'])
    parser.add_argument('--pads', default='4,8', help='Comma-separated character crop padding values')
    parser.add_argument('--minimum-probability', type=float, default=0.3)
    parser.add_argument('--page', default=None)
    parser.add_argument('--measure-debug', default=None)
    parser.add_argument('--source-block-index', type=int, default=None)
    parser.add_argument('--limit-pages', type=int, default=None)
    parser.add_argument('--limit-items', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--save-crops', default=None)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--calibrate-font-sizes', action='store_true')
    parser.add_argument('--apply-font-sizes', action='store_true')
    parser.add_argument('--even-font-size', action='store_true')
    args = parser.parse_args()
    measure_path, image_dir = _resolve_input_paths(args.path1, args.path2, args.measure_json)
    pads = sorted({max(0, int(value.strip())) for value in args.pads.split(',') if value.strip()})
    if not pads:
        pads = [4]
    output = run(
        measure_path=measure_path,
        image_dir=image_dir,
        output_path=args.output,
        model_path=args.model,
        alphabet_path=args.alphabet,
        implementation_path=args.implementation,
        device=args.device,
        pads=pads,
        minimum_probability=max(0.0, min(1.0, args.minimum_probability)),
        page=args.page,
        measure_debug_path=args.measure_debug,
        source_block_index=args.source_block_index,
        limit_pages=args.limit_pages,
        limit_items=args.limit_items,
        batch_size=max(1, args.batch_size),
        save_crops=args.save_crops,
        dry_run=args.dry_run,
    )
    if args.calibrate_font_sizes or args.apply_font_sizes:
        calibrated = _load_json(output)
        ready_count = calibrate_ocr_output(calibrated)
        changed_count = 0
        if args.apply_font_sizes:
            measure = _load_json(measure_path)
            changed_count = apply_calibrated_font_sizes(
                measure,
                calibrated,
                even_font_size=args.even_font_size,
            )
            _write_json(measure_path, measure)
        _write_json(output, calibrated)
        print(f'字級校準：{ready_count} 個可靠區塊，更新 measure.json {changed_count} 個區塊。', flush=True)
    print(f'輸出：{output}', flush=True)


if __name__ == '__main__':
    main()
