#!/usr/bin/env python3
"""Create icon assets for 塗白."""

from __future__ import annotations

import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
PNG_PATH = ROOT / 'tubai_icon_1024.png'
ICONSET_DIR = ROOT / 'tubai.iconset'
ICNS_PATH = ROOT / 'tubai.icns'


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        '/System/Library/Fonts/STHeiti Medium.ttc',
        '/System/Library/Fonts/Hiragino Sans GB.ttc',
        '/System/Library/Fonts/STHeiti Light.ttc',
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def create_png() -> None:
    size = 1024
    image = Image.new('RGBA', (size, size), (255, 255, 255, 0))
    draw = ImageDraw.Draw(image)

    margin = 88
    radius = 168
    shadow = (0, 0, 0, 28)
    border = (32, 150, 140, 255)
    border_dark = (20, 92, 98, 255)
    fill = (255, 255, 255, 255)
    ink = (21, 35, 42, 255)
    accent = (236, 68, 78, 255)

    shadow_box = (margin + 18, margin + 24, size - margin + 18, size - margin + 24)
    draw.rounded_rectangle(shadow_box, radius=radius, fill=shadow)
    box = (margin, margin, size - margin, size - margin)
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=border, width=28)
    draw.rounded_rectangle((margin + 34, margin + 34, size - margin - 34, size - margin - 34), radius=130, outline=border_dark, width=6)

    font = _font(560)
    text = '塗'
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = (size - text_w) / 2 - bbox[0]
    y = (size - text_h) / 2 - bbox[1] - 20
    draw.text((x + 12, y + 16), text, font=font, fill=(0, 0, 0, 34))
    draw.text((x, y), text, font=font, fill=ink)

    draw.rounded_rectangle((300, 808, 724, 848), radius=20, fill=accent)
    draw.rounded_rectangle((350, 858, 674, 886), radius=14, fill=(255, 216, 90, 255))

    image.save(PNG_PATH)


def create_iconset() -> None:
    ICONSET_DIR.mkdir(exist_ok=True)
    base = Image.open(PNG_PATH)
    sizes = [
        (16, 'icon_16x16.png'),
        (32, 'icon_16x16@2x.png'),
        (32, 'icon_32x32.png'),
        (64, 'icon_32x32@2x.png'),
        (128, 'icon_128x128.png'),
        (256, 'icon_128x128@2x.png'),
        (256, 'icon_256x256.png'),
        (512, 'icon_256x256@2x.png'),
        (512, 'icon_512x512.png'),
        (1024, 'icon_512x512@2x.png'),
    ]
    for px, name in sizes:
        base.resize((px, px), Image.Resampling.LANCZOS).save(ICONSET_DIR / name)


def create_icns() -> None:
    try:
        subprocess.run(['iconutil', '-c', 'icns', str(ICONSET_DIR), '-o', str(ICNS_PATH)], check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass


def main() -> None:
    create_png()
    create_iconset()
    create_icns()


if __name__ == '__main__':
    main()
