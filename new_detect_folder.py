#!/usr/bin/env python3
"""精簡版資料夾偵測腳本，輸出 block map、line-trans、mask 和 center 對齊結果。"""

import argparse
import json
import os
import os.path as osp
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from detect_folder import (
    CENTER_DIR,
    DEAL_OVERLAP_DIR,
    LINE_TRANS_BOX_DIR,
    NumpyEncoder,
    REFINEMASK_ANNOTATION,
    TextDetector,
    _align_block_boxes,
    _clean_aligned_items,
    _draw_aligned_boxes,
    _draw_line_width_measurements,
    _ensure_deal_overlap_image,
    _final_boxes_overlap,
    _find_component_boxes,
    _shrink_line_polygons,
    find_all_imgs,
    imread,
    imwrite,
)
from detect_folder import (
    SHRINK_PERCENTILE_HIGH,
    SHRINK_PERCENTILE_LOW,
    SHRINK_PERCENTILE_PADDING,
)


CTD_DIR = 'ctd'
PROGRESSING_DIR = 'progressing'
MASK_DIR = 'mask'
ALIGN_DIR = 'align'
MEASURE_PREVIEW_DIR = 'measure_preview'
BLOCK_MAP_JSON = 'block_map.json'
LINE_TRANS_MAP_JSON = 'line_trans_map.json'
ALIGNED_BOX_MAP_JSON = 'aligned_box_map.json'
MEASURE_JSON = 'measure.json'
MEASURE_DEBUG_JSON = 'measure.debug.json'


def _ensure_dirs(ctd_dir: str) -> dict[str, str]:
    progressing_dir = osp.join(ctd_dir, PROGRESSING_DIR)
    paths = {
        'ctd': ctd_dir,
        'progressing': progressing_dir,
        'mask': osp.join(progressing_dir, MASK_DIR),
        'line_trans_box': osp.join(progressing_dir, LINE_TRANS_BOX_DIR),
        'align': osp.join(progressing_dir, ALIGN_DIR),
        'center': osp.join(progressing_dir, ALIGN_DIR, CENTER_DIR),
        'deal_overlap': osp.join(progressing_dir, ALIGN_DIR, DEAL_OVERLAP_DIR),
        'measure_preview': osp.join(ctd_dir, MEASURE_PREVIEW_DIR),
    }
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    return paths


def _block_item_from_xyxy(box: list[int], img_w: int, img_h: int, index: int) -> dict:
    x1, y1, x2, y2 = [int(v) for v in box]
    return {
        'xyxy_pixel': [x1, y1, x2, y2],
        'center_normalized': [
            round(((x1 + x2) / 2) / img_w, 4),
            round(((y1 + y2) / 2) / img_h, 4),
        ],
        'source_block_index': index,
    }


def _block_boxes_from_items(items: list[dict]) -> list[list[int]]:
    boxes = []
    for item in items:
        xyxy = item.get('xyxy_pixel')
        if isinstance(xyxy, list) and len(xyxy) == 4:
            boxes.append([int(round(v)) for v in xyxy])
    return boxes


def _load_json(path: str) -> dict:
    with open(path, 'r', encoding='utf8') as f:
        return json.load(f)


def _write_json(path: str, data: dict) -> None:
    with open(path, 'w', encoding='utf8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)


def _image_path_for_page(img_dir: str, page_name: str) -> str:
    path = osp.join(img_dir, page_name)
    if osp.isfile(path):
        return path
    stem = Path(page_name).stem
    for ext in ('.png', '.jpg', '.jpeg', '.bmp'):
        candidate = osp.join(img_dir, f'{stem}{ext}')
        if osp.isfile(candidate):
            return candidate
    raise FileNotFoundError(f'找不到原圖：{page_name}')


def _mask_path_for_page(paths: dict[str, str], page_name: str) -> str:
    return osp.join(paths['mask'], f'{Path(page_name).stem}.png')


def _deal_overlap_path_for_page(paths: dict[str, str], page_name: str) -> str:
    return osp.join(paths['deal_overlap'], f'{Path(page_name).stem}.png')


def _xyxy_from_align_item(item: dict) -> list[int]:
    if item.get('accepted') is False:
        xyxy = item.get('final_xyxy_pixel') or item.get('new_xyxy_pixel')
    else:
        xyxy = item.get('new_xyxy_pixel') or item.get('final_xyxy_pixel')
    if isinstance(xyxy, list) and len(xyxy) == 4:
        return [int(round(v)) for v in xyxy]

    x = int(round(item.get('x', 0)))
    y = int(round(item.get('y', 0)))
    w = int(round(item.get('w', 0)))
    h = int(round(item.get('h', 0)))
    return [x, y, x + w, y + h]


def _center_from_align_item(
    item: dict,
    xyxy: list[int],
    img_w: int,
    img_h: int,
) -> list[float]:
    if item.get('accepted') is False:
        center = item.get('final_center_normalized')
    else:
        center = item.get('new_center_normalized') or item.get('final_center_normalized')
    if isinstance(center, list) and len(center) == 2:
        return [round(float(center[0]), 4), round(float(center[1]), 4)]
    return _center_normalized_from_xyxy(xyxy, img_w, img_h)


def _center_normalized_from_xyxy(
    xyxy: list[int],
    img_w: int,
    img_h: int,
) -> list[float]:
    x1, y1, x2, y2 = xyxy
    return [
        round(((x1 + x2) / 2) / img_w, 4),
        round(((y1 + y2) / 2) / img_h, 4),
    ]


def _line_box_from_item(item: dict) -> list[int] | None:
    poly = item.get('polygon')
    if isinstance(poly, list) and len(poly) >= 4:
        arr = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
        x1, y1 = arr.min(axis=0)
        x2, y2 = arr.max(axis=0)
        return [int(x1), int(y1), int(x2), int(y2)]
    if all(key in item for key in ('x', 'y', 'w', 'h')):
        x = int(round(item['x']))
        y = int(round(item['y']))
        w = int(round(item['w']))
        h = int(round(item['h']))
        return [x, y, x + w, y + h]
    return None


def _line_center(item: dict) -> tuple[float, float] | None:
    poly = item.get('polygon')
    if isinstance(poly, list) and len(poly) >= 4:
        arr = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
        center = arr.mean(axis=0)
        return float(center[0]), float(center[1])
    box = _line_box_from_item(item)
    if box is None:
        return None
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2


def _line_width(item: dict) -> float:
    value = item.get('font_width_px')
    if isinstance(value, (int, float)):
        return float(value)

    poly = item.get('polygon')
    if isinstance(poly, list) and len(poly) >= 4:
        pts = np.asarray(poly, dtype=np.float64).reshape(4, 2)
        lengths = [
            float(np.linalg.norm(pts[(i + 1) % 4] - pts[i]))
            for i in range(4)
        ]
        return min((lengths[0] + lengths[2]) / 2.0, (lengths[1] + lengths[3]) / 2.0)

    box = _line_box_from_item(item)
    if box is None:
        return 0.0
    x1, y1, x2, y2 = box
    return float(min(max(0, x2 - x1), max(0, y2 - y1)))


def _line_orientation(item: dict) -> str:
    poly = item.get('polygon')
    if isinstance(poly, list) and len(poly) >= 4:
        pts = np.asarray(poly, dtype=np.float64).reshape(4, 2)
        lengths = [
            float(np.linalg.norm(pts[(i + 1) % 4] - pts[i]))
            for i in range(4)
        ]
        pair_a = (lengths[0] + lengths[2]) / 2.0
        pair_b = (lengths[1] + lengths[3]) / 2.0
        long_indices = (0, 2) if pair_a >= pair_b else (1, 3)
        vec = pts[(long_indices[0] + 1) % 4] - pts[long_indices[0]]
        return 'horizontal' if abs(vec[0]) >= abs(vec[1]) else 'vertical'

    box = _line_box_from_item(item)
    if box is None:
        return 'vertical'
    x1, y1, x2, y2 = box
    return 'horizontal' if (x2 - x1) >= (y2 - y1) else 'vertical'


def _overlap_area(a: list[int], b: list[int]) -> int:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    return max(0, x2 - x1) * max(0, y2 - y1)


def _line_groups_by_block(
    block_items: list[dict],
    line_items: list[dict],
) -> dict[int, list[dict]]:
    groups: dict[int, list[dict]] = {}
    block_boxes = []
    for item in block_items:
        xyxy = item.get('xyxy_pixel')
        if not isinstance(xyxy, list) or len(xyxy) != 4:
            continue
        source_index = int(item.get('source_block_index', len(block_boxes)))
        block_boxes.append((source_index, [int(round(v)) for v in xyxy]))

    for line in line_items:
        center = _line_center(line)
        line_box = _line_box_from_item(line)
        best_index = None
        if center is not None:
            cx, cy = center
            for source_index, box in block_boxes:
                if box[0] <= cx <= box[2] and box[1] <= cy <= box[3]:
                    best_index = source_index
                    break

        if best_index is None and line_box is not None:
            best_area = 0
            for source_index, box in block_boxes:
                area = _overlap_area(line_box, box)
                if area > best_area:
                    best_area = area
                    best_index = source_index

        if best_index is not None:
            groups.setdefault(best_index, []).append(line)
    return groups


def _lower_median(values: list[float]) -> float:
    sorted_values = sorted(values)
    return sorted_values[(len(sorted_values) - 1) // 2]


def _orientation_from_lines(lines: list[dict]) -> str:
    if not lines:
        return 'vertical'
    counts = {'horizontal': 0, 'vertical': 0}
    areas = {'horizontal': 0.0, 'vertical': 0.0}
    for line in lines:
        orientation = _line_orientation(line)
        counts[orientation] += 1
        areas[orientation] += float(line.get('area') or 0)
    if counts['horizontal'] == counts['vertical']:
        return 'horizontal' if areas['horizontal'] > areas['vertical'] else 'vertical'
    return 'horizontal' if counts['horizontal'] > counts['vertical'] else 'vertical'


def _build_measure_maps(
    img_dir: str,
    block_map: dict,
    line_trans_map: dict,
    aligned_box_map: dict,
) -> tuple[dict, dict]:
    pages = {}
    block_pages = block_map.get('blockMap', {})
    line_pages = line_trans_map.get('transMap', {})
    align_pages = aligned_box_map.get('transMap', {})

    for page_name, align_items in align_pages.items():
        img = imread(_image_path_for_page(img_dir, page_name))
        img_h, img_w = img.shape[:2]
        block_items = block_pages.get(page_name, [])
        line_items = line_pages.get(page_name, [])
        line_groups = _line_groups_by_block(block_items, line_items)

        page_items = []
        for item in align_items:
            source_index = int(item.get('source_block_index', len(page_items)))
            xyxy = _xyxy_from_align_item(item)
            center = _center_from_align_item(item, xyxy, img_w, img_h)

            matched_lines = line_groups.get(source_index, [])
            widths = [_line_width(line) for line in matched_lines]
            widths = [value for value in widths if value > 0]
            if widths:
                font_size = _lower_median(widths)
            else:
                x1, y1, x2, y2 = xyxy
                font_size = float(min(max(0, x2 - x1), max(0, y2 - y1)))

            page_items.append({
                'source_block_index': source_index,
                'xyxy_pixel': xyxy,
                'center_normalized': center,
                'orientation': _orientation_from_lines(matched_lines),
                'font_size': round(float(font_size), 1),
            })
        pages[page_name] = page_items

    measure = {'pages': pages}
    measure_debug = {
        'pages': pages,
        'block': block_map,
        'line': line_trans_map,
        'align': aligned_box_map,
    }
    return measure, measure_debug


def _align_item_by_source_index(items: list[dict]) -> dict[int, dict]:
    result = {}
    for fallback_index, item in enumerate(items):
        source_index = int(item.get('source_block_index', fallback_index))
        result[source_index] = item
    return result


def _label_box_for_measure_item(measure_item: dict) -> list[int]:
    xyxy = measure_item.get('xyxy_pixel')
    if isinstance(xyxy, list) and len(xyxy) == 4:
        return [int(round(v)) for v in xyxy]
    return [0, 0, 0, 0]


def _single_text_size(text: str, font_scale: float) -> tuple[int, int, int]:
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, 1)
    return text_w, text_h + baseline, baseline


def _fit_label_origin(
    box: list[int],
    canvas_shape: tuple[int, int] | tuple[int, int, int],
    text_w: int,
    text_h: int,
) -> tuple[int, int]:
    img_h, img_w = canvas_shape[:2]
    x1, y1, x2, y2 = box
    pad = 5
    # Prefer outside bottom-right; if clipped, try other outside corners.
    candidates = [
        (x2 + pad, y2 + pad),
        (x2 + pad, y1 - text_h - pad),
        (x1 - text_w - pad, y2 + pad),
        (x1 - text_w - pad, y1 - text_h - pad),
    ]
    for x, y in candidates:
        if 1 <= x and x + text_w <= img_w - 1 and 1 <= y and y + text_h <= img_h - 1:
            return int(round(x)), int(round(y))

    x = max(1, min(int(round(x2 + pad)), img_w - text_w - 1))
    y = max(1, min(int(round(y2 + pad)), img_h - text_h - 1))
    return x, y


def _draw_black_text(
    canvas: np.ndarray,
    text: str,
    origin: tuple[int, int],
    font_scale: float,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, text, origin, font, font_scale, (0, 0, 0), 1, cv2.LINE_AA)


def _draw_measure_block_labels(
    canvas: np.ndarray,
    measure_items: list[dict],
    align_items: list[dict],
) -> np.ndarray:
    if not measure_items:
        return canvas

    font_scale = max(0.36, min(0.52, canvas.shape[1] / 1700))
    for item in measure_items:
        orientation = str(item.get('orientation', 'vertical'))
        font_size = item.get('font_size', 0)
        suffix = 'H' if orientation == 'horizontal' else 'V'
        text = f'{int(round(float(font_size)))}{suffix}'
        text_w, text_h, baseline = _single_text_size(text, font_scale)
        box = _label_box_for_measure_item(item)
        x, y = _fit_label_origin(box, canvas.shape, text_w, text_h)
        pad = 3
        cv2.rectangle(
            canvas,
            (x - pad, y - pad),
            (x + text_w + pad, y + text_h + pad),
            (255, 255, 255),
            -1,
        )
        _draw_black_text(canvas, text, (x, y + text_h - baseline), font_scale)
    return canvas


def _center_marker_color(item: dict) -> tuple[int, int, int]:
    debug = item.get('layout_debug', {})
    if not debug.get('accepted'):
        return (0, 0, 0)
    center_mode = debug.get('resolved_center_mode', debug.get('center_mode'))
    if center_mode == 'outer':
        return (255, 0, 0)
    if center_mode == 'inner':
        return (0, 165, 255)
    if center_mode == 'average':
        return (0, 255, 0)
    return (0, 255, 0)


def _center_xy_for_align_item(item: dict) -> tuple[int, int]:
    xyxy = item.get('final_xyxy_pixel') or item.get('new_xyxy_pixel')
    if isinstance(xyxy, list) and len(xyxy) == 4:
        x1, y1, x2, y2 = [float(v) for v in xyxy]
        return int(round((x1 + x2) / 2)), int(round((y1 + y2) / 2))

    x = float(item.get('x', 0))
    y = float(item.get('y', 0))
    w = float(item.get('w', 0))
    h = float(item.get('h', 0))
    return int(round(x + w / 2)), int(round(y + h / 2))


def _draw_measure_center_blocks(
    canvas: np.ndarray,
    measure_items: list[dict],
    align_items: list[dict],
) -> np.ndarray:
    align_by_source = _align_item_by_source_index(align_items)
    height, width = canvas.shape[:2]
    for measure_item in measure_items:
        source_index = int(measure_item.get('source_block_index', -1))
        align_item = align_by_source.get(source_index)
        if align_item is None:
            continue
        cx, cy = _center_xy_for_align_item(align_item)
        side = max(4, int(round(float(measure_item.get('font_size') or 0))))
        half = side / 2
        x1 = max(0, int(round(cx - half)))
        y1 = max(0, int(round(cy - half)))
        x2 = min(width - 1, int(round(cx + half)))
        y2 = min(height - 1, int(round(cy + half)))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), _center_marker_color(align_item), -1)
    return canvas


def _write_measure_previews(
    paths: dict[str, str],
    line_trans_map: dict,
    aligned_box_map: dict,
    measure_map: dict,
) -> None:
    line_pages = line_trans_map.get('transMap', {})
    align_pages = aligned_box_map.get('transMap', {})
    measure_pages = measure_map.get('pages', {})

    for page_name, measure_items in tqdm(measure_pages.items(), desc='measure preview'):
        center_path = osp.join(paths['center'], f'{Path(page_name).stem}.png')
        if not osp.isfile(center_path):
            raise FileNotFoundError(f'找不到 center 預覽圖：{center_path}')

        center_img = imread(center_path)
        canvas = _draw_line_width_measurements(center_img, line_pages.get(page_name, []))
        canvas = _draw_measure_center_blocks(
            canvas,
            measure_items,
            align_pages.get(page_name, []),
        )
        canvas = _draw_measure_block_labels(
            canvas,
            measure_items,
            align_pages.get(page_name, []),
        )
        imwrite(osp.join(paths['measure_preview'], f'{Path(page_name).stem}.png'), canvas)


def _align_pages(
    img_dir: str,
    paths: dict[str, str],
    block_map: dict,
) -> tuple[dict, dict]:
    aligned_map = {}
    summary = {
        'pages': 0,
        'boxes': 0,
        'accepted': 0,
        'using_deal_overlap': 0,
        'overlap_pages': 0,
        'copied': 0,
    }

    pages = block_map.get('blockMap', {})
    for page_name, block_items in tqdm(pages.items(), desc='重定位'):
        img_path = _image_path_for_page(img_dir, page_name)
        mask_path = _mask_path_for_page(paths, page_name)
        if not osp.isfile(mask_path):
            raise FileNotFoundError(f'找不到 mask：{mask_path}')

        img = imread(img_path)
        mask = imread(mask_path)
        overlap_path = _deal_overlap_path_for_page(paths, page_name)
        calc_img = imread(overlap_path) if osp.exists(overlap_path) else img
        if osp.exists(overlap_path):
            summary['using_deal_overlap'] += 1

        block_boxes = _block_boxes_from_items(block_items)
        aligned_items = _align_block_boxes(calc_img, mask, block_boxes, base_img=img)
        clean_items = _clean_aligned_items(aligned_items)
        aligned_map[page_name] = clean_items

        if _final_boxes_overlap(aligned_items):
            summary['overlap_pages'] += 1
            helper_status = _ensure_deal_overlap_image(img_path, overlap_path)
            if helper_status == 'copied':
                summary['copied'] += 1

        center_img = _draw_aligned_boxes(img, aligned_items)
        imwrite(osp.join(paths['center'], f'{Path(page_name).stem}.png'), center_img)

        summary['pages'] += 1
        summary['boxes'] += len(clean_items)
        summary['accepted'] += sum(1 for item in clean_items if item.get('accepted'))

    return {'transMap': aligned_map}, summary


def _detect_pages(
    img_dir: str,
    model_path: str,
    device: str | None,
    paths: dict[str, str],
) -> tuple[dict, dict]:
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    imglist = find_all_imgs(img_dir, abs_path=True)
    if not imglist:
        print(f'資料夾內沒有可處理的圖片：{img_dir}')
        return {'blockMap': {}}, {'transMap': {}}

    print(f'資料夾：{img_dir}')
    print(f'輸出：{paths["ctd"]}')
    print(f'模型：{model_path}')
    print(f'裝置：{device}')
    print(f'圖片數量：{len(imglist)}')

    detector = TextDetector(model_path=model_path, input_size=1024, device=device, act='leaky')
    block_map = {}
    line_trans_map = {}

    for img_path in tqdm(imglist, desc='偵測中'):
        imgname = osp.basename(img_path)
        page_key = imgname
        imname = Path(imgname).stem
        img = imread(img_path)
        im_h, im_w = img.shape[:2]

        _, mask_refined, blk_list = detector(
            img,
            refine_mode=REFINEMASK_ANNOTATION,
            keep_undetected_mask=True,
        )

        polys = []
        block_boxes = []
        for blk in blk_list:
            polys += blk.lines
            block_boxes.append([int(x) for x in blk.xyxy])

        block_map[page_key] = [
            _block_item_from_xyxy(box, im_w, im_h, index)
            for index, box in enumerate(block_boxes)
        ]

        component_boxes = _find_component_boxes(mask_refined)
        percentile_items = _shrink_line_polygons(
            mask_refined,
            polys,
            percentile_low=SHRINK_PERCENTILE_LOW,
            percentile_high=SHRINK_PERCENTILE_HIGH,
            padding=SHRINK_PERCENTILE_PADDING,
            method='percentile',
        )
        line_trans_items = _shrink_line_polygons(
            mask_refined,
            polys,
            padding=SHRINK_PERCENTILE_PADDING,
            component_boxes=component_boxes,
            method='line_trans_component',
            fallback_items=percentile_items,
        )
        line_trans_map[page_key] = line_trans_items

        imwrite(osp.join(paths['mask'], f'{imname}.png'), mask_refined)
        line_trans_img = _draw_line_width_measurements(img, line_trans_items)
        imwrite(osp.join(paths['line_trans_box'], f'{imname}.png'), line_trans_img)

    return {'blockMap': block_map}, {'transMap': line_trans_map}


def run(
    img_dir: str,
    model_path: str,
    device: str | None = None,
    only_align: bool = False,
) -> int:
    img_dir = osp.abspath(img_dir)
    if not osp.isdir(img_dir):
        raise FileNotFoundError(f'找不到資料夾：{img_dir}')

    ctd_dir = osp.join(img_dir, CTD_DIR)
    paths = _ensure_dirs(ctd_dir)
    block_map_path = osp.join(paths['progressing'], BLOCK_MAP_JSON)
    line_trans_map_path = osp.join(paths['progressing'], LINE_TRANS_MAP_JSON)
    aligned_box_map_path = osp.join(paths['progressing'], ALIGNED_BOX_MAP_JSON)
    measure_path = osp.join(ctd_dir, MEASURE_JSON)
    measure_debug_path = osp.join(ctd_dir, MEASURE_DEBUG_JSON)

    if only_align:
        if not osp.isfile(block_map_path):
            raise FileNotFoundError(f'--only-align 需要既有 block map：{block_map_path}')
        if not osp.isfile(line_trans_map_path):
            raise FileNotFoundError(f'--only-align 需要既有 line trans map：{line_trans_map_path}')
        block_map = _load_json(block_map_path)
        line_trans_map = _load_json(line_trans_map_path)
        print(f'只重定位：{img_dir}')
    else:
        model_path = osp.abspath(model_path)
        if not osp.isfile(model_path):
            raise FileNotFoundError(f'找不到模型檔：{model_path}')
        block_map, line_trans_map = _detect_pages(img_dir, model_path, device, paths)
        _write_json(block_map_path, block_map)
        _write_json(line_trans_map_path, line_trans_map)

    aligned_box_map, align_summary = _align_pages(img_dir, paths, block_map)
    _write_json(aligned_box_map_path, aligned_box_map)
    measure_map, measure_debug_map = _build_measure_maps(
        img_dir,
        block_map,
        line_trans_map,
        aligned_box_map,
    )
    _write_json(measure_path, measure_map)
    _write_json(measure_debug_path, measure_debug_map)
    _write_measure_previews(paths, line_trans_map, aligned_box_map, measure_map)

    print('完成。輸出：')
    print(f'  - {block_map_path}')
    if not only_align:
        print(f'  - {line_trans_map_path}')
        print(f'  - {paths["mask"]}/<檔名>.png')
        print(f'  - {paths["line_trans_box"]}/<檔名>.png')
    print(f'  - {aligned_box_map_path}')
    print(f'  - {measure_path}')
    print(f'  - {measure_debug_path}')
    print(f'  - {paths["center"]}/<檔名>.png')
    print(f'  - {paths["measure_preview"]}/<檔名>.png')
    print(f'  - {paths["deal_overlap"]}/<檔名>.png')
    print(f"重定位頁數：{align_summary['pages']}")
    print(f"重定位 block 數：{align_summary['boxes']}")
    print(f"accepted：{align_summary['accepted']}")
    print(f"使用 deal_overlap 計算頁數：{align_summary['using_deal_overlap']}")
    print(f"偵測到重疊頁數：{align_summary['overlap_pages']}")
    print(f"新複製到 deal_overlap：{align_summary['copied']}")
    return align_summary['pages']


def main() -> None:
    default_model = osp.join(osp.dirname(__file__), 'data', 'comictextdetector.pt')
    parser = argparse.ArgumentParser(
        description='精簡版漫畫文字偵測與 center 重定位輸出。',
    )
    parser.add_argument('img_dir', help='輸入圖片資料夾路徑')
    parser.add_argument(
        '--model',
        default=default_model,
        help=f'模型路徑（預設：{default_model}）',
    )
    parser.add_argument(
        '--device',
        choices=['cpu', 'cuda'],
        default=None,
        help='推理裝置（預設：有 GPU 則用 cuda，否則 cpu）',
    )
    parser.add_argument(
        '--only-align',
        action='store_true',
        help='不跑模型，只讀 ctd/block_map.json 和 ctd/mask/ 重新生成對齊結果。',
    )
    args = parser.parse_args()

    run(
        img_dir=args.img_dir,
        model_path=args.model,
        device=args.device,
        only_align=args.only_align,
    )


if __name__ == '__main__':
    main()
