#!/usr/bin/env python3
"""Helpers for LabelPlus txt -> MEO -> MEO BT generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    from .build_text_rect_update import (
        build_updated_translate,
        collect_unmatched_translate_texts,
    )
    from .lp_to_meo import _default_output_path as default_meo_output_path
    from .lp_to_meo import _filtered_output_path as filtered_meo_output_path
    from .lp_to_meo import convert as convert_lp_to_meo
except ImportError:
    from build_text_rect_update import (
        build_updated_translate,
        collect_unmatched_translate_texts,
    )
    from lp_to_meo import _default_output_path as default_meo_output_path
    from lp_to_meo import _filtered_output_path as filtered_meo_output_path
    from lp_to_meo import convert as convert_lp_to_meo


def bt_output_path(meo_path: Path) -> Path:
    return meo_path.with_name(f'{meo_path.stem}_bt.json')


def build_bt_from_labelplus_txt(
    txt_path: str | Path,
    measure_path: str | Path,
    image_dir: str | Path,
    *,
    filter_group_id: int | None = 1,
    tolerance_px: int = 50,
    font_size_step: int = 1,
    font_size_min: int = 0,
) -> dict[str, Any]:
    txt_path = Path(txt_path).expanduser().resolve()
    measure_path = Path(measure_path).expanduser().resolve()
    image_dir = Path(image_dir).expanduser().resolve()

    if not txt_path.is_file():
        raise FileNotFoundError(f'找不到 LabelPlus txt：{txt_path}')
    if not measure_path.is_file():
        raise FileNotFoundError(f'找不到 ctd/measure.json：{measure_path}')
    if not image_dir.is_dir():
        raise FileNotFoundError(f'找不到圖片資料夾：{image_dir}')

    meo_path = default_meo_output_path(txt_path)
    convert_lp_to_meo(str(txt_path), str(meo_path), filter_group_id=filter_group_id)
    filtered_path = (
        filtered_meo_output_path(meo_path)
        if filter_group_id is not None
        else None
    )

    translate_data = json.loads(meo_path.read_text(encoding='utf-8'))
    measure_data = json.loads(measure_path.read_text(encoding='utf-8'))
    updated = build_updated_translate(
        translate_data,
        measure_data,
        image_dir,
        tolerance_px=tolerance_px,
        font_size_step=font_size_step,
        font_size_min=font_size_min,
    )

    bt_path = bt_output_path(meo_path)
    bt_path.write_text(
        json.dumps(updated, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )

    unmatched = collect_unmatched_translate_texts(
        translate_data,
        measure_data,
        image_dir,
        tolerance_px=tolerance_px,
    )

    return {
        'txt_path': txt_path,
        'measure_path': measure_path,
        'image_dir': image_dir,
        'meo_path': meo_path,
        'filtered_path': filtered_path,
        'bt_path': bt_path,
        'pages': len(updated.get('transMap', {})),
        'labels': sum(len(items) for items in updated.get('transMap', {}).values()),
        'unmatched_pages': len(unmatched),
        'unmatched_labels': sum(len(items) for items in unmatched.values()),
    }
