import argparse
import copy
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from layout_core import NumpyEncoder, apply_safety_rules, calculate_layout, normalized_center, rect_area, rects_overlap, to_gray
from preview_draw import draw_item_preview, preview_base_image


def load_image_any(path, unchanged=False):
    flag = cv2.IMREAD_UNCHANGED if unchanged else cv2.IMREAD_COLOR
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, flag)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return img


def page_image_path(inpainted_dir, page_name):
    stem = Path(page_name).stem
    for ext in (".png", ".PNG"):
        candidate = inpainted_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing inpainted PNG for {page_name}: {inpainted_dir / (stem + '.png')}")


def calculation_image_path(inpainted_dir, page_name):
    original_path = page_image_path(inpainted_dir, page_name)
    overlap_path = inpainted_dir / "deal_overlap" / original_path.name
    return overlap_path if overlap_path.exists() else original_path, original_path, overlap_path


def save_png(path, image):
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError(f"Could not encode preview: {path}")
    encoded.tofile(str(path))


def save_page_masks(path, smoothed_masks, outer_body_masks, accepted, image_shape):
    path.parent.mkdir(parents=True, exist_ok=True)
    if smoothed_masks:
        smoothed_array = np.stack(smoothed_masks).astype(np.uint8)
        outer_array = np.stack(outer_body_masks).astype(np.uint8)
    else:
        height, width = image_shape
        smoothed_array = np.zeros((0, height, width), dtype=np.uint8)
        outer_array = np.zeros((0, height, width), dtype=np.uint8)

    np.savez_compressed(
        path,
        smoothed_masks=smoothed_array,
        outer_body_masks=outer_array,
        accepted=np.array(accepted, dtype=np.bool_),
    )


def final_boxes_overlap(items, min_ratio=0.05):
    boxes = [item.get("final_xyxy_pixel") for item in items if item.get("final_xyxy_pixel")]
    for i, box_a in enumerate(boxes):
        for box_b in boxes[i + 1:]:
            if not rects_overlap(box_a, box_b):
                continue
            x1 = max(box_a[0], box_b[0])
            y1 = max(box_a[1], box_b[1])
            x2 = min(box_a[2], box_b[2])
            y2 = min(box_a[3], box_b[3])
            overlap_area = max(0, x2 - x1) * max(0, y2 - y1)
            smaller_area = min(rect_area(box_a), rect_area(box_b))
            if smaller_area > 0 and overlap_area / smaller_area >= min_ratio:
                return True
    return False


def ensure_overlap_helper(original_path, overlap_path):
    if overlap_path.exists():
        return "exists"
    overlap_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(original_path, overlap_path)
    return "copied"


def mark_layout_failed(item, error, img_w, img_h):
    old_xyxy = [int(round(v)) for v in item.get("xyxy_pixel", [0, 0, 0, 0])]
    item["new_xyxy_pixel"] = None
    item["new_center_normalized"] = None
    item["final_xyxy_pixel"] = old_xyxy
    item["final_center_normalized"] = normalized_center(old_xyxy, img_w, img_h)
    item["layout_debug"] = {
        "processed": False,
        "accepted": False,
        "skip_reason": "layout_calculation_failed",
        "error": str(error),
    }
    return item


def process_item(item, gray, center_mode):
    img_h, img_w = gray.shape
    layout = calculate_layout(gray, item, center_mode=center_mode)
    return apply_safety_rules(item, layout, img_w, img_h)


def debug_json_path(output_json):
    return output_json.with_name(f"{output_json.stem}.debug{output_json.suffix}")


def build_clean_output(original_data, debug_data):
    clean_data = copy.deepcopy(original_data)
    original_pages = clean_data.get("pages", {})
    debug_pages = debug_data.get("pages", {})

    for page_name, clean_items in original_pages.items():
        debug_items = debug_pages.get(page_name, [])
        for clean_item, debug_item in zip(clean_items, debug_items):
            final_xyxy = debug_item.get("final_xyxy_pixel")
            final_center = debug_item.get("final_center_normalized")
            if final_xyxy is not None and "xyxy_pixel" in clean_item:
                clean_item["xyxy_pixel"] = final_xyxy
            if final_center is not None and "center_normalized" in clean_item:
                clean_item["center_normalized"] = final_center

    return clean_data


def process_json(json_path, output_json=None, preview_dir=None, mask_dir=None, center_mode="auto"):
    json_path = Path(json_path)
    base_dir = json_path.parent
    inpainted_dir = base_dir / "inpainted"
    output_json = Path(output_json) if output_json else json_path.with_name(f"{json_path.stem}_recentered{json_path.suffix}")
    output_debug_json = debug_json_path(output_json)
    preview_dir = Path(preview_dir) if preview_dir else inpainted_dir / "recentered_preview"
    mask_dir = Path(mask_dir) if mask_dir else inpainted_dir / "recentered_masks"
    preview_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    original_data = copy.deepcopy(data)

    pages = data.get("pages")
    if not isinstance(pages, dict):
        raise ValueError("JSON must contain a pages object")

    total = 0
    failed = 0
    pages_using_overlap = 0
    pages_with_overlap = 0
    overlap_helpers_copied = 0
    total_boxes = sum(len(items) for items in pages.values())
    print(f"Start processing {len(pages)} pages, {total_boxes} text boxes.", flush=True)
    print(f"Overlap helper folder: {inpainted_dir / 'deal_overlap'}", flush=True)
    print(f"Preview folder: {preview_dir}", flush=True)
    print(f"Mask folder: {mask_dir}", flush=True)
    print(f"Center mode: {center_mode}", flush=True)

    for page_index, (page_name, items) in enumerate(pages.items(), start=1):
        page_failed_before = failed
        calc_path, original_path, overlap_path = calculation_image_path(inpainted_dir, page_name)
        using_overlap = calc_path == overlap_path
        if using_overlap:
            pages_using_overlap += 1

        calc_img = load_image_any(calc_path, unchanged=True)
        preview_img = load_image_any(original_path, unchanged=True)
        gray = to_gray(calc_img)
        preview = preview_base_image(preview_img)
        smoothed_masks = []
        outer_body_masks = []
        accepted_flags = []

        for item_index, item in enumerate(items):
            total += 1
            preview_masks = {
                "smoothed_mask": np.zeros(gray.shape, dtype=np.uint8),
                "outer_body_mask": np.zeros(gray.shape, dtype=np.uint8),
            }
            try:
                layout = process_item(item, gray, center_mode)
                preview_masks = layout.pop("_preview_masks", preview_masks)
                item.update(layout)
            except Exception as e:
                failed += 1
                mark_layout_failed(item, e, gray.shape[1], gray.shape[0])
            smoothed_masks.append(preview_masks["smoothed_mask"])
            outer_body_masks.append(preview_masks["outer_body_mask"])
            accepted_flags.append(bool(item.get("layout_debug", {}).get("accepted")))
            draw_item_preview(preview, item, preview_masks, item_index)

        has_overlap = final_boxes_overlap(items)
        helper_status = None
        if has_overlap:
            pages_with_overlap += 1
            helper_status = ensure_overlap_helper(original_path, overlap_path)
            if helper_status == "copied":
                overlap_helpers_copied += 1

        save_png(preview_dir / f"{Path(page_name).stem}.png", preview)
        save_page_masks(mask_dir / f"{Path(page_name).stem}.npz", smoothed_masks, outer_body_masks, accepted_flags, gray.shape)
        page_failed = failed - page_failed_before
        status_parts = ["ok" if page_failed == 0 else f"{page_failed} failed"]
        if using_overlap:
            status_parts.append("using deal_overlap")
        if has_overlap:
            status_parts.append("overlap detected")
            status_parts.append("copied to deal_overlap" if helper_status == "copied" else "deal_overlap exists")
        status = ", ".join(status_parts)
        print(f"[{page_index}/{len(pages)}] {page_name}: {status}", flush=True)

    clean_data = build_clean_output(original_data, data)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(clean_data, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    with output_debug_json.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

    summary = {
        "pages_using_overlap": pages_using_overlap,
        "pages_with_overlap": pages_with_overlap,
        "overlap_helpers_copied": overlap_helpers_copied,
    }
    return output_json, output_debug_json, preview_dir, mask_dir, total, failed, summary


def main():
    parser = argparse.ArgumentParser(description="Recenter text rectangles from text_rect.json using matching inpainted PNGs.")
    parser.add_argument("json_path", help="Path to text_rect.json")
    parser.add_argument("--output-json", default=None, help="Output JSON path. Default: <name>_recentered.json")
    parser.add_argument("--preview-dir", default=None, help="Output preview folder. Default: inpainted/recentered_preview")
    parser.add_argument("--mask-dir", default=None, help="Output npz mask folder. Default: inpainted/recentered_masks")
    parser.add_argument(
        "--center-mode",
        choices=("auto", "outer", "inner", "average"),
        default="auto",
        help="Candidate center strategy: auto, outer, inner, or average. Default: auto",
    )
    args = parser.parse_args()

    output_json, output_debug_json, preview_dir, mask_dir, total, failed, summary = process_json(
        args.json_path,
        args.output_json,
        args.preview_dir,
        args.mask_dir,
        args.center_mode,
    )
    print(f"Processed {total} text boxes, failed {failed}.")
    print(f"Pages using deal_overlap: {summary['pages_using_overlap']}")
    print(f"Pages with overlap: {summary['pages_with_overlap']}")
    print(f"New overlap helper images copied: {summary['overlap_helpers_copied']}")
    print(f"JSON saved: {output_json}")
    print(f"Debug JSON saved: {output_debug_json}")
    print(f"Preview folder: {preview_dir}")
    print(f"Mask folder: {mask_dir}")


if __name__ == "__main__":
    main()
