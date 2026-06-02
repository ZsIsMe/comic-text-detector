#!/usr/bin/env python3
"""Raw folder text detection without bubble-center alignment."""

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
    LINE_TRANS_BOX_DIR,
    SHRINK_PERCENTILE_HIGH,
    SHRINK_PERCENTILE_LOW,
    SHRINK_PERCENTILE_PADDING,
    _draw_line_width_measurements,
    _find_component_boxes,
    _shrink_line_polygons,
)
from inference import TextDetector
from utils.io_utils import NumpyEncoder, find_all_imgs, imread, imwrite
from utils.textmask import REFINEMASK_ANNOTATION


CTD_DIR = 'ctd'
PROGRESSING_DIR = 'progressing'
MASK_DIR = 'mask'
ALIGN_DIR = 'align'
ALIGN_MASK_DIR = 'masks'
MEASURE_PREVIEW_DIR = 'measure_preview'
ONLY_TEXT_DIR = 'only_text'
INPAINTED_DIR = 'inpainted'
BLOCK_MAP_JSON = 'block_map.json'
LINE_TRANS_MAP_JSON = 'line_trans_map.json'
ALIGNED_BOX_MAP_JSON = 'aligned_box_map.json'
MEASURE_JSON = 'measure.json'
MEASURE_DEBUG_JSON = 'measure.debug.json'
ONLY_TEXT_COLOR = (255, 0, 255)
ONLY_TEXT_OPACITY = 0.4
INPAINT_RADIUS = 3
INPAINT_MASK_EXPANSION = 5
RAW_BOX_COLOR = (0, 0, 0)
RAW_CENTER_COLOR = (0, 0, 0)


def _ensure_dirs(ctd_dir: str) -> dict[str, str]:
    progressing_dir = osp.join(ctd_dir, PROGRESSING_DIR)
    paths = {
        'ctd': ctd_dir,
        'progressing': progressing_dir,
        'mask': osp.join(progressing_dir, MASK_DIR),
        'line_trans_box': osp.join(progressing_dir, LINE_TRANS_BOX_DIR),
        'align_masks': osp.join(progressing_dir, ALIGN_DIR, ALIGN_MASK_DIR),
        'measure_preview': osp.join(ctd_dir, MEASURE_PREVIEW_DIR),
        'only_text': osp.join(ctd_dir, ONLY_TEXT_DIR),
        'inpainted': osp.join(ctd_dir, INPAINTED_DIR),
    }
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    return paths


def _write_json(path: str, data: dict) -> None:
    with open(path, 'w', encoding='utf8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)


def _image_path_for_page(img_dir: str, page_name: str) -> str:
    path = osp.join(img_dir, page_name)
    if osp.isfile(path):
        return path
    stem = Path(page_name).stem
    for ext in ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff'):
        candidate = osp.join(img_dir, f'{stem}{ext}')
        if osp.isfile(candidate):
            return candidate
    raise FileNotFoundError(f'找不到原圖：{page_name}')


def _mask_path_for_page(paths: dict[str, str], page_name: str) -> str:
    return osp.join(paths['mask'], f'{Path(page_name).stem}.png')


def _align_mask_path_for_page(paths: dict[str, str], page_name: str) -> str:
    return osp.join(paths['align_masks'], f'{Path(page_name).stem}.npz')


def _only_text_path_for_page(paths: dict[str, str], page_name: str) -> str:
    return osp.join(paths['only_text'], f'{Path(page_name).stem}.png')


def _inpainted_path_for_page(paths: dict[str, str], page_name: str) -> str:
    return osp.join(paths['inpainted'], f'{Path(page_name).stem}.png')


def _xyxy_area(box: list[int]) -> int:
    return max(0, int(box[2]) - int(box[0])) * max(0, int(box[3]) - int(box[1]))


def _center_normalized_from_xyxy(box: list[int], img_w: int, img_h: int) -> list[float]:
    x1, y1, x2, y2 = box
    return [
        round(((x1 + x2) / 2) / img_w, 4),
        round(((y1 + y2) / 2) / img_h, 4),
    ]


def _block_item_from_xyxy(box: list[int], img_w: int, img_h: int, index: int) -> dict:
    x1, y1, x2, y2 = [int(v) for v in box]
    xyxy = [x1, y1, x2, y2]
    return {
        'xyxy_pixel': xyxy,
        'center_normalized': _center_normalized_from_xyxy(xyxy, img_w, img_h),
        'source_block_index': index,
    }


def _raw_align_item_from_block(item: dict, img_w: int, img_h: int) -> dict:
    xyxy = [int(round(v)) for v in item['xyxy_pixel']]
    x1, y1, x2, y2 = xyxy
    center = _center_normalized_from_xyxy(xyxy, img_w, img_h)
    cx = round((x1 + x2) / 2, 2)
    cy = round((y1 + y2) / 2, 2)
    source_index = int(item.get('source_block_index', 0))
    return {
        'x': x1,
        'y': y1,
        'w': max(0, x2 - x1),
        'h': max(0, y2 - y1),
        'area': _xyxy_area(xyxy),
        'method': 'raw_detector_box',
        'accepted': False,
        'source_block_index': source_index,
        'new_xyxy_pixel': xyxy,
        'new_center_normalized': center,
        'final_xyxy_pixel': xyxy,
        'final_center_normalized': center,
        'layout_debug': {
            'processed': False,
            'accepted': False,
            'skip_reason': 'raw_detector_box',
            'old_xyxy_pixel': xyxy,
            'old_center_pixel': [cx, cy],
            'final_center_pixel': [cx, cy],
            'calculation_method': 'raw_detector_box',
        },
    }


def _raw_aligned_map_from_blocks(
    block_map: dict,
    image_sizes: dict[str, tuple[int, int]],
) -> dict:
    result = {}
    for page_name, block_items in block_map.get('blockMap', {}).items():
        img_h, img_w = image_sizes[page_name]
        result[page_name] = [
            _raw_align_item_from_block(item, img_w, img_h)
            for item in block_items
        ]
    return {'transMap': result}


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
            xyxy = [int(round(v)) for v in item['final_xyxy_pixel']]
            center = _center_normalized_from_xyxy(xyxy, img_w, img_h)

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
        'mode': 'raw_detector_box',
    }
    return measure, measure_debug


def _write_align_masks(path: str, count: int, image_shape: tuple[int, int]) -> None:
    height, width = image_shape
    os.makedirs(osp.dirname(path), exist_ok=True)
    np.savez_compressed(
        path,
        smoothed_masks=np.zeros((count, height, width), dtype=np.uint8),
        outer_body_masks=np.zeros((count, height, width), dtype=np.uint8),
        accepted=np.zeros((count,), dtype=np.bool_),
    )


def _draw_raw_boxes(img: np.ndarray, align_items: list[dict]) -> np.ndarray:
    canvas = img[:, :, :3].copy() if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    line_width = max(2, min(canvas.shape[:2]) // 400)
    for item in align_items:
        x1, y1, x2, y2 = [int(round(v)) for v in item['final_xyxy_pixel']]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), RAW_BOX_COLOR, line_width)
    return canvas


def _draw_measure_center_blocks(
    canvas: np.ndarray,
    measure_items: list[dict],
) -> np.ndarray:
    height, width = canvas.shape[:2]
    for item in measure_items:
        cx_norm, cy_norm = item.get('center_normalized', [0, 0])
        cx = float(cx_norm) * width
        cy = float(cy_norm) * height
        side = max(4, int(round(float(item.get('font_size') or 0))))
        half = side / 2
        x1 = max(0, int(round(cx - half)))
        y1 = max(0, int(round(cy - half)))
        x2 = min(width - 1, int(round(cx + half)))
        y2 = min(height - 1, int(round(cy + half)))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), RAW_CENTER_COLOR, -1)
    return canvas


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


def _draw_measure_block_labels(
    canvas: np.ndarray,
    measure_items: list[dict],
) -> np.ndarray:
    font_scale = max(0.36, min(0.52, canvas.shape[1] / 1700))
    for item in measure_items:
        suffix = 'H' if item.get('orientation') == 'horizontal' else 'V'
        text = f'{int(round(float(item.get("font_size", 0))))}{suffix}'
        text_w, text_h, baseline = _single_text_size(text, font_scale)
        box = item.get('xyxy_pixel') or [0, 0, 0, 0]
        x, y = _fit_label_origin([int(round(v)) for v in box], canvas.shape, text_w, text_h)
        pad = 3
        cv2.rectangle(
            canvas,
            (x - pad, y - pad),
            (x + text_w + pad, y + text_h + pad),
            (255, 255, 255),
            -1,
        )
        cv2.putText(
            canvas,
            text,
            (x, y + text_h - baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return canvas


def _write_measure_previews(
    img_dir: str,
    paths: dict[str, str],
    line_trans_map: dict,
    aligned_box_map: dict,
    measure_map: dict,
) -> None:
    line_pages = line_trans_map.get('transMap', {})
    align_pages = aligned_box_map.get('transMap', {})
    measure_pages = measure_map.get('pages', {})
    for page_name, measure_items in tqdm(measure_pages.items(), desc='measure preview'):
        img = imread(_image_path_for_page(img_dir, page_name))
        canvas = _draw_raw_boxes(img, align_pages.get(page_name, []))
        canvas = _draw_line_width_measurements(canvas, line_pages.get(page_name, []))
        canvas = _draw_measure_center_blocks(canvas, measure_items)
        canvas = _draw_measure_block_labels(canvas, measure_items)
        imwrite(osp.join(paths['measure_preview'], f'{Path(page_name).stem}.png'), canvas)


def _parse_color_string(color_str: str) -> tuple[int, int, int]:
    value = color_str.strip()
    if value.startswith('#'):
        value = value[1:]
    if len(value) == 6 and ',' not in value:
        try:
            return tuple(int(value[index:index + 2], 16) for index in (0, 2, 4))
        except ValueError:
            pass
    parts = value.split(',')
    if len(parts) == 3:
        try:
            rgb = tuple(int(part.strip()) for part in parts)
            if all(0 <= channel <= 255 for channel in rgb):
                return rgb
        except ValueError:
            pass
    raise ValueError(f'顏色格式錯誤：{color_str}，請使用 "#ff00ff" 或 "255,0,255"')


def _alpha_from_mask(mask: np.ndarray, opacity: float) -> np.ndarray:
    mask_gray = cv2.cvtColor(mask[:, :, :3], cv2.COLOR_BGR2GRAY) if len(mask.shape) == 3 else mask
    return np.clip(mask_gray.astype(np.float32) * opacity, 0, 255).astype(np.uint8)


def _dilate_mask(mask: np.ndarray, size: int) -> np.ndarray:
    if size <= 1:
        return mask
    if size % 2 == 0:
        size += 1
    kernel = np.ones((size, size), dtype=np.uint8)
    return cv2.dilate(mask, kernel, iterations=1)


def _block_zone_from_items(
    block_items: list[dict],
    image_shape: tuple[int, int],
    padding: int = 0,
) -> np.ndarray:
    height, width = image_shape[:2]
    zone = np.zeros((height, width), dtype=np.uint8)
    pad = max(0, int(padding))
    for item in block_items:
        x1, y1, x2, y2 = [int(round(v)) for v in item.get('xyxy_pixel', [0, 0, 0, 0])]
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(width, x2 + pad)
        y2 = min(height, y2 + pad)
        if x2 > x1 and y2 > y1:
            zone[y1:y2, x1:x2] = 255
    return zone


def _write_only_text_images(
    paths: dict[str, str],
    page_names: list[str],
    color_rgb: tuple[int, int, int],
    opacity: float,
) -> None:
    for page_name in tqdm(page_names, desc='only text'):
        mask = imread(_mask_path_for_page(paths, page_name), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f'無法讀取文字 mask：{_mask_path_for_page(paths, page_name)}')
        alpha = _alpha_from_mask(mask, opacity)
        output = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
        text_zone = alpha > 0
        output[text_zone, 0] = color_rgb[2]
        output[text_zone, 1] = color_rgb[1]
        output[text_zone, 2] = color_rgb[0]
        output[:, :, 3] = alpha
        imwrite(_only_text_path_for_page(paths, page_name), output)


def _write_inpainted_images(
    img_dir: str,
    paths: dict[str, str],
    page_names: list[str],
    block_map: dict,
    radius: int,
    mask_expansion: int,
) -> None:
    block_pages = block_map.get('blockMap', {})
    for page_name in tqdm(page_names, desc='inpainted'):
        img = imread(_image_path_for_page(img_dir, page_name))
        mask = imread(_mask_path_for_page(paths, page_name), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f'無法讀取文字 mask：{_mask_path_for_page(paths, page_name)}')
        text_mask = np.where(mask > 0, 255, 0).astype(np.uint8)
        block_zone = _block_zone_from_items(
            block_pages.get(page_name, []),
            mask.shape[:2],
            padding=mask_expansion,
        )
        inpaint_mask = cv2.bitwise_and(text_mask, block_zone)
        inpaint_mask = _dilate_mask(inpaint_mask, mask_expansion)
        color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img[:, :, :3]
        inpainted_bgr = cv2.inpaint(color_img, inpaint_mask, radius, cv2.INPAINT_TELEA)
        inpainted = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2BGRA)
        inpainted[:, :, 3] = inpaint_mask
        inpainted[inpaint_mask == 0, :3] = 0
        imwrite(_inpainted_path_for_page(paths, page_name), inpainted)


def _detect_pages(
    img_dir: str,
    model_path: str,
    device: str | None,
    paths: dict[str, str],
    save_line_trans_preview: bool,
) -> tuple[dict, dict, dict[str, tuple[int, int]]]:
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    imglist = find_all_imgs(img_dir, abs_path=True)
    if not imglist:
        print(f'資料夾內沒有可處理的圖片：{img_dir}')
        return {'blockMap': {}}, {'transMap': {}}, {}

    print(f'資料夾：{img_dir}')
    print(f'輸出：{paths["ctd"]}')
    print(f'模型：{model_path}')
    print(f'裝置：{device}')
    print(f'圖片數量：{len(imglist)}')

    detector = TextDetector(model_path=model_path, input_size=1024, device=device, act='leaky')
    block_map = {}
    line_trans_map = {}
    image_sizes = {}

    for img_path in tqdm(imglist, desc='偵測中'):
        imgname = osp.basename(img_path)
        page_key = imgname
        imname = Path(imgname).stem
        img = imread(img_path)
        im_h, im_w = img.shape[:2]
        image_sizes[page_key] = (im_h, im_w)

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
        if save_line_trans_preview:
            line_trans_img = _draw_line_width_measurements(img, line_trans_items)
            imwrite(osp.join(paths['line_trans_box'], f'{imname}.png'), line_trans_img)

    return {'blockMap': block_map}, {'transMap': line_trans_map}, image_sizes


def run(
    img_dir: str,
    model_path: str,
    device: str | None,
    save_line_trans_preview: bool,
    only_text_color: tuple[int, int, int],
    only_text_opacity: float,
    inpaint_radius: int,
    inpaint_mask_expansion: int,
) -> int:
    img_dir = osp.abspath(img_dir)
    if not osp.isdir(img_dir):
        raise FileNotFoundError(f'找不到資料夾：{img_dir}')
    model_path = osp.abspath(model_path)
    if not osp.isfile(model_path):
        raise FileNotFoundError(f'找不到模型檔：{model_path}')

    ctd_dir = osp.join(img_dir, CTD_DIR)
    paths = _ensure_dirs(ctd_dir)
    block_map_path = osp.join(paths['progressing'], BLOCK_MAP_JSON)
    line_trans_map_path = osp.join(paths['progressing'], LINE_TRANS_MAP_JSON)
    aligned_box_map_path = osp.join(paths['progressing'], ALIGNED_BOX_MAP_JSON)
    measure_path = osp.join(ctd_dir, MEASURE_JSON)
    measure_debug_path = osp.join(ctd_dir, MEASURE_DEBUG_JSON)

    block_map, line_trans_map, image_sizes = _detect_pages(
        img_dir,
        model_path,
        device,
        paths,
        save_line_trans_preview=save_line_trans_preview,
    )
    aligned_box_map = _raw_aligned_map_from_blocks(block_map, image_sizes)

    _write_json(block_map_path, block_map)
    _write_json(line_trans_map_path, line_trans_map)
    _write_json(aligned_box_map_path, aligned_box_map)
    for page_name, items in aligned_box_map.get('transMap', {}).items():
        _write_align_masks(_align_mask_path_for_page(paths, page_name), len(items), image_sizes[page_name])

    measure_map, measure_debug_map = _build_measure_maps(
        img_dir,
        block_map,
        line_trans_map,
        aligned_box_map,
    )
    _write_json(measure_path, measure_map)
    _write_json(measure_debug_path, measure_debug_map)
    page_names = list(measure_map.get('pages', {}).keys())
    _write_measure_previews(img_dir, paths, line_trans_map, aligned_box_map, measure_map)
    _write_only_text_images(paths, page_names, only_text_color, only_text_opacity)
    _write_inpainted_images(
        img_dir,
        paths,
        page_names,
        block_map,
        inpaint_radius,
        inpaint_mask_expansion,
    )

    page_count = len(page_names)
    block_count = sum(len(items) for items in aligned_box_map.get('transMap', {}).values())
    print('完成。輸出：')
    print(f'  - {block_map_path}')
    print(f'  - {line_trans_map_path}')
    print(f'  - {paths["mask"]}/<檔名>.png')
    if save_line_trans_preview:
        print(f'  - {paths["line_trans_box"]}/<檔名>.png')
    print(f'  - {aligned_box_map_path}')
    print(f'  - {paths["align_masks"]}/<檔名>.npz')
    print(f'  - {measure_path}')
    print(f'  - {measure_debug_path}')
    print(f'  - {paths["measure_preview"]}/<檔名>.png')
    print(f'  - {paths["only_text"]}/<檔名>.png')
    print(f'  - {paths["inpainted"]}/<檔名>.png')
    print(f'raw 頁數：{page_count}')
    print(f'raw block 數：{block_count}')
    return page_count


def main() -> None:
    default_model = osp.join(osp.dirname(__file__), 'data', 'comictextdetector.pt')
    parser = argparse.ArgumentParser(
        description='Raw 漫畫文字偵測輸出，不做氣泡中心重定位。',
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
        '--save-line-trans-preview',
        action='store_true',
        help='額外輸出 ctd/progressing/line-trans-box/<檔名>.png。',
    )
    parser.add_argument(
        '--only-text-color',
        default=','.join(str(value) for value in ONLY_TEXT_COLOR),
        help='純字圖顏色，支援 "#ff00ff" 或 "255,0,255"。',
    )
    parser.add_argument(
        '--only-text-opacity',
        type=float,
        default=ONLY_TEXT_OPACITY,
        help='純字圖透明度，0.0 到 1.0。',
    )
    parser.add_argument(
        '--inpaint-radius',
        type=int,
        default=INPAINT_RADIUS,
        help='inpainted 預覽的周圍顏色填補半徑。',
    )
    parser.add_argument(
        '--inpaint-mask-expansion',
        type=int,
        default=INPAINT_MASK_EXPANSION,
        help='inpainted 預覽先把 mask 向外擴張的像素核大小。',
    )
    args = parser.parse_args()
    only_text_color = _parse_color_string(args.only_text_color)
    run(
        img_dir=args.img_dir,
        model_path=args.model,
        device=args.device,
        save_line_trans_preview=args.save_line_trans_preview,
        only_text_color=only_text_color,
        only_text_opacity=args.only_text_opacity,
        inpaint_radius=args.inpaint_radius,
        inpaint_mask_expansion=args.inpaint_mask_expansion,
    )


if __name__ == '__main__':
    main()
