#!/usr/bin/env python3
"""Generate one normalized ink bounding-box measurement per OCR character."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PySide6.QtGui import QFont, QGuiApplication, QRawFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FONT_PATH = PROJECT_ROOT / 'assets' / 'fonts' / 'NotoSansCJKjp-Medium.otf'
DEFAULT_ALPHABET_PATH = Path(
    '/Users/zhongsheng/Documents/comic_translate/BallonsTranslator/data/alphabet-all-v5.txt',
)
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / 'assets' / 'fonts' / 'NotoSansCJKjp-Medium.ink-metrics.json'
SCHEMA_VERSION = 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def generate_metrics(
    font_path: Path,
    alphabet_path: Path,
    output_path: Path,
    reference_pixel_size: int,
) -> dict:
    if not font_path.is_file():
        raise FileNotFoundError(f'找不到字體：{font_path}')
    if not alphabet_path.is_file():
        raise FileNotFoundError(f'找不到 OCR 字典：{alphabet_path}')
    if reference_pixel_size <= 0:
        raise ValueError('參考像素尺寸必須大於 0')

    application = QGuiApplication.instance() or QGuiApplication(sys.argv[:1])
    raw_font = QRawFont(
        str(font_path),
        float(reference_pixel_size),
        QFont.HintingPreference.PreferNoHinting,
    )
    if not raw_font.isValid():
        raise RuntimeError(f'Qt 無法載入字體：{font_path}')

    alphabet = alphabet_path.read_text(encoding='utf-8').splitlines()
    glyphs: dict[str, list[float]] = {}
    skipped_non_single_character = 0
    skipped_missing_glyph = 0
    skipped_empty_glyph = 0
    for entry in alphabet:
        if len(entry) != 1:
            skipped_non_single_character += 1
            continue
        glyph_indexes = raw_font.glyphIndexesForString(entry)
        if len(glyph_indexes) != 1 or int(glyph_indexes[0]) == 0:
            skipped_missing_glyph += 1
            continue
        rect = raw_font.boundingRect(int(glyph_indexes[0]))
        width = float(rect.width())
        height = float(rect.height())
        if width <= 0 or height <= 0:
            skipped_empty_glyph += 1
            continue
        glyphs[entry] = [
            round(width / reference_pixel_size, 6),
            round(height / reference_pixel_size, 6),
        ]

    data = {
        'schema_version': SCHEMA_VERSION,
        'font': {
            'file_name': font_path.name,
            'sha256': sha256_file(font_path),
        },
        'alphabet': {
            'file_name': alphabet_path.name,
            'sha256': sha256_file(alphabet_path),
            'entry_count': len(alphabet),
            'single_character_entry_count': sum(len(entry) == 1 for entry in alphabet),
        },
        'metrics': {
            'engine': 'Qt QRawFont.boundingRect',
            'hinting': 'PreferNoHinting',
            'reference_pixel_size': reference_pixel_size,
            'units': 'reference_pixel_ratio',
            'fields': ['ink_width', 'ink_height'],
        },
        'counts': {
            'glyph_count': len(glyphs),
            'skipped_non_single_character': skipped_non_single_character,
            'skipped_missing_glyph': skipped_missing_glyph,
            'skipped_empty_glyph': skipped_empty_glyph,
        },
        'glyphs': glyphs,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + '.tmp')
    temporary_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )
    temporary_path.replace(output_path)
    del application
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--font', type=Path, default=DEFAULT_FONT_PATH)
    parser.add_argument('--alphabet', type=Path, default=DEFAULT_ALPHABET_PATH)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument('--reference-pixel-size', type=int, default=1000)
    args = parser.parse_args()
    data = generate_metrics(
        args.font.expanduser().resolve(),
        args.alphabet.expanduser().resolve(),
        args.output.expanduser().resolve(),
        args.reference_pixel_size,
    )
    counts = data['counts']
    print(
        f'已生成 {args.output}：{counts["glyph_count"]} 個 glyph，'
        f'缺字 {counts["skipped_missing_glyph"]} 個，'
        f'空字形 {counts["skipped_empty_glyph"]} 個。',
    )


if __name__ == '__main__':
    main()
