#!/usr/bin/env python3
"""生成 CTD 疊圖資料，不輸出靜態預覽圖。

這個檔案放在新工具目錄裡，負責調用既有 new_detect_folder.py 的偵測、
對齊與字級量測函式。它不修改 new_detect_folder.py，也不生成
measure_preview、only_text、inpainted 等預覽圖片。
"""

from __future__ import annotations

import argparse
import os
import os.path as osp
import sys
from pathlib import Path


IMAGE_EXTENSIONS = {'.bmp', '.jpg', '.png', '.jpeg'}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
LAYOUT_HELPER_DIR = PROJECT_ROOT / '建立对齐方框'
if LAYOUT_HELPER_DIR.is_dir() and str(LAYOUT_HELPER_DIR) not in sys.path:
    sys.path.insert(0, str(LAYOUT_HELPER_DIR))

from ctd_overlay_processor.analyze_text_core import enrich_measure_map


def patch_numpy_compat() -> None:
    try:
        import numpy as np
    except Exception:
        return

    aliases = {
        'bool8': 'bool_',
        'float_': 'float64',
        'int0': 'intp',
        'unicode_': 'str_',
    }
    for old_name, new_name in aliases.items():
        if not hasattr(np, old_name) and hasattr(np, new_name):
            setattr(np, old_name, getattr(np, new_name))


def safe_find_all_imgs(img_dir: str, abs_path: bool = False) -> list[str]:
    root = Path(img_dir)
    if not root.is_dir():
        return []
    result = []
    for path in sorted(root.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        result.append(str(path) if abs_path else path.name)
    return result


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {'1', 'true', 'yes', 'y', 'on'}:
        return True
    if normalized in {'0', 'false', 'no', 'n', 'off'}:
        return False
    raise argparse.ArgumentTypeError(f'布林值格式錯誤：{value}，請使用 true 或 false')


def load_detection_api():
    patch_numpy_compat()
    try:
        import new_detect_folder as ndf
        from new_detect_folder import (
            ALIGNED_BOX_MAP_JSON,
            BLOCK_MAP_JSON,
            CTD_DIR,
            LINE_TRANS_MAP_JSON,
            MEASURE_DEBUG_JSON,
            MEASURE_JSON,
            INPAINT_MASK_EXPANSION,
            INPAINT_RADIUS,
            ONLY_TEXT_COLOR,
            ONLY_TEXT_OPACITY,
            _align_pages,
            _build_measure_maps,
            _detect_pages,
            _ensure_dirs,
            _load_json,
            _write_inpainted_images,
            _write_only_text_images,
            _write_json,
        )
    except Exception as exc:
        print(
            '無法載入偵測流程。請確認偵測依賴已安裝，'
            f'並且 new_detect_folder.py 位於專案根目錄。\n錯誤：{exc}',
            file=sys.stderr,
        )
        raise SystemExit(1)

    ndf.find_all_imgs = safe_find_all_imgs

    return {
        'ALIGNED_BOX_MAP_JSON': ALIGNED_BOX_MAP_JSON,
        'BLOCK_MAP_JSON': BLOCK_MAP_JSON,
        'CTD_DIR': CTD_DIR,
        'LINE_TRANS_MAP_JSON': LINE_TRANS_MAP_JSON,
        'MEASURE_DEBUG_JSON': MEASURE_DEBUG_JSON,
        'MEASURE_JSON': MEASURE_JSON,
        'INPAINT_MASK_EXPANSION': INPAINT_MASK_EXPANSION,
        'INPAINT_RADIUS': INPAINT_RADIUS,
        'ONLY_TEXT_COLOR': ONLY_TEXT_COLOR,
        'ONLY_TEXT_OPACITY': ONLY_TEXT_OPACITY,
        '_align_pages': _align_pages,
        '_build_measure_maps': _build_measure_maps,
        '_detect_pages': _detect_pages,
        '_ensure_dirs': _ensure_dirs,
        '_load_json': _load_json,
        '_write_inpainted_images': _write_inpainted_images,
        '_write_only_text_images': _write_only_text_images,
        '_write_json': _write_json,
    }


def run_detection(
    image_dir: str,
    model_path: str | None = None,
    device: str | None = None,
    only_align: bool = False,
    need_neck: bool = False,
) -> int:
    os.chdir(PROJECT_ROOT)
    image_dir = osp.abspath(image_dir)
    if not osp.isdir(image_dir):
        raise FileNotFoundError(f'找不到圖片資料夾：{image_dir}')

    api = load_detection_api()
    ctd_dir_name = api['CTD_DIR']
    ensure_dirs = api['_ensure_dirs']
    load_json = api['_load_json']
    write_json = api['_write_json']
    detect_pages = api['_detect_pages']
    align_pages = api['_align_pages']
    build_measure_maps = api['_build_measure_maps']
    write_inpainted_images = api['_write_inpainted_images']
    write_only_text_images = api['_write_only_text_images']

    if model_path is None:
        model_path = str(PROJECT_ROOT / 'data' / 'comictextdetector.pt')
    model_path = osp.abspath(model_path)

    ctd_dir = osp.join(image_dir, ctd_dir_name)
    paths = ensure_dirs(ctd_dir)
    block_map_path = osp.join(paths['progressing'], api['BLOCK_MAP_JSON'])
    line_trans_map_path = osp.join(paths['progressing'], api['LINE_TRANS_MAP_JSON'])
    aligned_box_map_path = osp.join(paths['progressing'], api['ALIGNED_BOX_MAP_JSON'])
    measure_path = osp.join(ctd_dir, api['MEASURE_JSON'])
    measure_debug_path = osp.join(ctd_dir, api['MEASURE_DEBUG_JSON'])

    if only_align:
        if not osp.isfile(block_map_path):
            raise FileNotFoundError(f'只重算對齊需要既有 block map：{block_map_path}')
        if not osp.isfile(line_trans_map_path):
            raise FileNotFoundError(f'只重算對齊需要既有 line trans map：{line_trans_map_path}')
        block_map = load_json(block_map_path)
        line_trans_map = load_json(line_trans_map_path)
        print(f'只重算對齊：{image_dir}', flush=True)
    else:
        if not osp.isfile(model_path):
            raise FileNotFoundError(f'找不到模型檔：{model_path}')
        block_map, line_trans_map = detect_pages(
            image_dir,
            model_path,
            device,
            paths,
            save_line_trans_preview=False,
        )
        if not block_map.get('blockMap') and safe_find_all_imgs(image_dir, abs_path=False):
            raise RuntimeError(
                '偵測流程沒有產生任何頁面資料。'
                '這通常是圖片搜尋或模型初始化異常，已停止寫入以避免覆蓋既有 JSON。'
            )
        write_json(block_map_path, block_map)
        write_json(line_trans_map_path, line_trans_map)

    aligned_box_map, align_summary = align_pages(
        image_dir,
        paths,
        block_map,
        save_center_preview=False,
        need_neck=need_neck,
    )
    write_json(aligned_box_map_path, aligned_box_map)

    measure_map, measure_debug_map = build_measure_maps(
        image_dir,
        paths,
        block_map,
        line_trans_map,
        aligned_box_map,
    )
    enriched_count, enrich_errors = enrich_measure_map(Path(image_dir), measure_map)
    write_json(measure_path, measure_map)
    write_json(measure_debug_path, measure_debug_map)
    page_names = list(measure_map.get('pages', {}).keys())
    write_only_text_images(
        image_dir,
        paths,
        page_names,
        api['ONLY_TEXT_COLOR'],
        api['ONLY_TEXT_OPACITY'],
    )
    write_inpainted_images(
        image_dir,
        paths,
        page_names,
        block_map,
        api['INPAINT_RADIUS'],
        api['INPAINT_MASK_EXPANSION'],
    )

    print('完成。已生成 GUI 所需資料：', flush=True)
    print(f'  - {block_map_path}', flush=True)
    print(f'  - {line_trans_map_path}', flush=True)
    print(f'  - {paths["mask"]}/<檔名>.png', flush=True)
    print(f'  - {aligned_box_map_path}', flush=True)
    print(f'  - {paths["align_masks"]}/<檔名>.npz', flush=True)
    print(f'  - {paths["only_text"]}/<檔名>.png', flush=True)
    print(f'  - {paths["inpainted"]}/<檔名>.png', flush=True)
    print(f'  - {measure_path}', flush=True)
    print(f'  - {measure_debug_path}', flush=True)
    print(f'文字顏色/描邊分析：{enriched_count} 個區塊', flush=True)
    if enrich_errors:
        print('部分頁面無法分析文字顏色/描邊：', flush=True)
        for error in enrich_errors:
            print(f'  - {error}', flush=True)
    print(f'重定位頁數：{align_summary["pages"]}', flush=True)
    print(f'重定位區塊數：{align_summary["boxes"]}', flush=True)
    print(f'採用重定位：{align_summary["accepted"]}', flush=True)
    return int(align_summary['pages'])


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='生成 CTD JSON/NPZ 疊圖資料。')
    parser.add_argument('image_dir', help='包含原圖的圖片資料夾。')
    parser.add_argument(
        '--model',
        default=None,
        help='模型路徑，預設使用專案 data/comictextdetector.pt。',
    )
    parser.add_argument(
        '--device',
        choices=['cpu', 'cuda'],
        default=None,
        help='推理裝置，預設自動選擇。',
    )
    parser.add_argument(
        '--only-align',
        action='store_true',
        help='不跑模型，只用既有 block/line/mask 重算對齊與量測。',
    )
    parser.add_argument(
        '--need-neck',
        type=parse_bool,
        default=False,
        help='是否啟用 neck/deal_overlap 輔助流程，預設 false。',
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_detection(
        image_dir=args.image_dir,
        model_path=args.model,
        device=args.device,
        only_align=args.only_align,
        need_neck=args.need_neck,
    )


if __name__ == '__main__':
    main()
