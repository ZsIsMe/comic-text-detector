#!/usr/bin/env python3
"""Generate a colored JPEG or transparent only_text PNG from an image pair.

The edited image is expected to have text/effects removed. Pixels whose visual
difference exceeds the threshold form the output mask. By default, that mask is
painted magenta at 40% opacity over the original image and saved as a JPEG.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


DEFAULT_ORIGINAL = Path(
    "/Users/zhongsheng/Documents/comic_data/血刃之花/剩餘/53/002.jpg"
)
DEFAULT_EDITED = Path(
    "/Users/zhongsheng/Documents/comic_data/血刃之花/剩餘/finish/53/002.jpg"
)


def parse_rgb(value: str) -> tuple[int, int, int]:
    value = value.strip().lstrip("#")
    if len(value) == 6 and "," not in value:
        try:
            return tuple(int(value[index : index + 2], 16) for index in (0, 2, 4))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"無效顏色：{value}") from exc

    try:
        channels = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"無效顏色：{value}") from exc
    if len(channels) != 3 or any(channel < 0 or channel > 255 for channel in channels):
        raise argparse.ArgumentTypeError(f"無效顏色：{value}")
    return channels


def read_image(path: Path) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"無法讀取圖片：{path}")
    return image


def write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"PNG 編碼失敗：{path}")
    encoded.tofile(path)


def write_jpeg(path: Path, image: np.ndarray, quality: int = 95) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(
        ".jpg",
        image,
        [cv2.IMWRITE_JPEG_QUALITY, quality],
    )
    if not ok:
        raise RuntimeError(f"JPEG 編碼失敗：{path}")
    encoded.tofile(path)


def build_difference_mask(
    original: np.ndarray,
    edited: np.ndarray,
    threshold: int,
    min_area: int,
) -> np.ndarray:
    if original.shape != edited.shape:
        raise ValueError(
            f"兩張圖尺寸不一致：{original.shape} != {edited.shape}"
        )

    original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    edited_gray = cv2.cvtColor(edited, cv2.COLOR_BGR2GRAY)
    luminance_diff = cv2.absdiff(original_gray, edited_gray)
    channel_diff = cv2.absdiff(original, edited).max(axis=2)
    difference = np.maximum(luminance_diff, channel_diff)

    support = np.where(difference >= threshold, 255, 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(support, 8)
    cleaned = np.zeros_like(support)
    for label in range(1, count):
        x, y, width, height, area = stats[label]
        keep_thin_stroke = area >= 3 and max(width, height) >= 4
        if area >= min_area or keep_thin_stroke:
            cleaned[labels == label] = 255

    # Join one-pixel JPEG/anti-aliasing gaps without noticeably expanding strokes.
    cleaned = cv2.morphologyEx(
        cleaned,
        cv2.MORPH_CLOSE,
        np.ones((2, 2), dtype=np.uint8),
    )

    return cleaned


def generate_only_text(
    original_path: Path,
    edited_path: Path,
    output_path: Path,
    color_rgb: tuple[int, int, int],
    opacity: float,
    threshold: int,
    min_area: int,
) -> tuple[int, int]:
    original = read_image(original_path)
    edited = read_image(edited_path)
    mask = build_difference_mask(original, edited, threshold, min_area)

    output = np.zeros((*mask.shape, 4), dtype=np.uint8)
    active = mask > 0
    output[active, 0] = color_rgb[2]
    output[active, 1] = color_rgb[1]
    output[active, 2] = color_rgb[0]
    output[active, 3] = int(255 * opacity)
    write_png(output_path, output)
    return int(np.count_nonzero(active)), int(output[:, :, 3].max())


def generate_colored(
    original_path: Path,
    edited_path: Path,
    output_path: Path,
    color_rgb: tuple[int, int, int],
    opacity: float,
    threshold: int,
    min_area: int,
) -> int:
    original = read_image(original_path)
    edited = read_image(edited_path)
    mask = build_difference_mask(original, edited, threshold, min_area)
    active = mask > 0

    output = original.copy()
    alpha = int(255 * opacity) / 255.0
    color_bgr = np.array(
        [color_rgb[2], color_rgb[1], color_rgb[0]],
        dtype=np.float32,
    )
    output[active] = (
        color_bgr * alpha
        + original[active].astype(np.float32) * (1.0 - alpha)
    ).astype(np.uint8)
    write_jpeg(output_path, output)
    return int(np.count_nonzero(active))


def batch_generate(
    root: Path,
    finish_root: Path,
    start: int,
    end: int,
    color_rgb: tuple[int, int, int],
    opacity: float,
    threshold: int,
    min_area: int,
    output_kind: str,
) -> tuple[int, int, list[str]]:
    image_suffixes = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
    generated = 0
    unchanged = 0
    failures: list[str] = []

    for chapter in range(start, end + 1):
        source_dir = root / str(chapter)
        edited_dir = finish_root / str(chapter)
        if not source_dir.is_dir() or not edited_dir.is_dir():
            failures.append(f"{chapter}: 原圖或 finish 資料夾不存在")
            continue

        source_images = sorted(
            path
            for path in source_dir.iterdir()
            if path.is_file() and path.suffix.lower() in image_suffixes
        )
        edited_by_stem = {
            path.stem: path
            for path in edited_dir.iterdir()
            if path.is_file() and path.suffix.lower() in image_suffixes
        }
        chapter_generated = 0
        for original_path in source_images:
            edited_path = edited_by_stem.get(original_path.stem)
            if edited_path is None:
                failures.append(f"{chapter}/{original_path.name}: 找不到修改圖")
                continue
            try:
                if output_kind == "colored":
                    output_path = source_dir / "colored" / f"{original_path.stem}.jpg"
                    pixels = generate_colored(
                        original_path,
                        edited_path,
                        output_path,
                        color_rgb,
                        opacity,
                        threshold,
                        min_area,
                    )
                else:
                    output_path = source_dir / "only_text" / f"{original_path.stem}.png"
                    pixels, _ = generate_only_text(
                        original_path,
                        edited_path,
                        output_path,
                        color_rgb,
                        opacity,
                        threshold,
                        min_area,
                    )
            except Exception as exc:
                failures.append(f"{chapter}/{original_path.name}: {exc}")
                continue
            generated += 1
            chapter_generated += 1
            if pixels == 0:
                unchanged += 1
        print(f"{chapter}: {chapter_generated}/{len(source_images)}")

    return generated, unchanged, failures


def main() -> None:
    parser = argparse.ArgumentParser(
        description="對比原圖與修改圖，產生洋紅差異疊圖 colored JPEG。"
    )
    parser.add_argument("--original", type=Path, default=DEFAULT_ORIGINAL)
    parser.add_argument("--edited", type=Path, default=DEFAULT_EDITED)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--color", type=parse_rgb, default=(255, 0, 255))
    parser.add_argument("--opacity", type=float, default=0.4)
    parser.add_argument("--threshold", type=int, default=35)
    parser.add_argument("--min-area", type=int, default=6)
    parser.add_argument("--batch-root", type=Path, default=None)
    parser.add_argument("--finish-root", type=Path, default=None)
    parser.add_argument("--start", type=int, default=52)
    parser.add_argument("--end", type=int, default=77)
    parser.add_argument(
        "--output-kind",
        choices=("colored", "only-text"),
        default="colored",
        help="預設輸出疊在原圖上的 colored JPEG；也可輸出透明 only_text PNG。",
    )
    args = parser.parse_args()

    if not 0.0 <= args.opacity <= 1.0:
        parser.error("--opacity 必須介於 0.0 與 1.0")
    if not 0 <= args.threshold <= 254:
        parser.error("--threshold 必須介於 0 與 254")
    if args.min_area < 1:
        parser.error("--min-area 必須大於 0")

    if args.batch_root is not None:
        finish_root = args.finish_root or args.batch_root / "finish"
        generated, unchanged, failures = batch_generate(
            args.batch_root,
            finish_root,
            args.start,
            args.end,
            args.color,
            args.opacity,
            args.threshold,
            args.min_area,
            args.output_kind,
        )
        print(f"完成：{generated} 張，無有效差異：{unchanged} 張，失敗：{len(failures)} 張")
        for failure in failures:
            print(f"失敗：{failure}")
        if failures:
            raise SystemExit(1)
        return

    if args.output_kind == "colored":
        output = args.output or args.original.parent / "colored" / f"{args.original.stem}.jpg"
        pixels = generate_colored(
            args.original,
            args.edited,
            output,
            args.color,
            args.opacity,
            args.threshold,
            args.min_area,
        )
        print(f"已輸出：{output}")
        print(f"差異像素：{pixels}")
    else:
        output = args.output or args.original.parent / "only_text" / f"{args.original.stem}.png"
        pixels, max_alpha = generate_only_text(
            args.original,
            args.edited,
            output,
            args.color,
            args.opacity,
            args.threshold,
            args.min_area,
        )
        print(f"已輸出：{output}")
        print(f"差異像素：{pixels}，最大 alpha：{max_alpha}")


if __name__ == "__main__":
    main()
