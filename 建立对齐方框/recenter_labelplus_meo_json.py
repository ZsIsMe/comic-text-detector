import argparse
import copy
import json
from pathlib import Path

import numpy as np

from layout_core import NumpyEncoder, to_gray
from preview_draw import draw_item_preview, preview_base_image
from recenter_text_rects import (
    calculation_image_path,
    debug_json_path,
    ensure_overlap_helper,
    final_boxes_overlap,
    load_image_any,
    mark_layout_failed,
    process_item,
    save_page_masks,
    save_png,
)


def validate_transmap(trans_map):
    if not isinstance(trans_map, dict):
        raise ValueError("JSON must contain a transMap object")

    errors = []
    total = 0
    for page_name, items in trans_map.items():
        if not isinstance(items, list):
            errors.append(f"{page_name}: value must be a list")
            continue
        for item_index, item in enumerate(items, start=1):
            total += 1
            if not isinstance(item, dict):
                errors.append(f"{page_name} item {item_index}: item must be an object")
                continue
            xyxy = item.get("xyxy_pixel")
            if not (isinstance(xyxy, list) and len(xyxy) == 4):
                errors.append(f"{page_name} item {item_index}: missing or invalid xyxy_pixel")
                continue
            try:
                [float(v) for v in xyxy]
            except (TypeError, ValueError):
                errors.append(f"{page_name} item {item_index}: xyxy_pixel must contain numbers")

    if errors:
        preview = "\n".join(errors[:20])
        more = f"\n... and {len(errors) - 20} more" if len(errors) > 20 else ""
        raise ValueError(f"Invalid transMap format:\n{preview}{more}")
    return total


def build_clean_output(original_data, debug_data):
    clean_data = copy.deepcopy(original_data)
    clean_trans_map = clean_data.get("transMap", {})
    debug_trans_map = debug_data.get("transMap", {})

    for page_name, clean_items in clean_trans_map.items():
        debug_items = debug_trans_map.get(page_name, [])
        for clean_item, debug_item in zip(clean_items, debug_items):
            final_xyxy = debug_item.get("final_xyxy_pixel")
            final_center = debug_item.get("final_center_normalized")
            if final_xyxy is not None and "xyxy_pixel" in clean_item:
                clean_item["xyxy_pixel"] = final_xyxy
            if final_center is not None:
                if "x" in clean_item:
                    clean_item["x"] = final_center[0]
                if "y" in clean_item:
                    clean_item["y"] = final_center[1]

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

    trans_map = data.get("transMap")
    total_boxes = validate_transmap(trans_map)

    total = 0
    failed = 0
    pages_using_overlap = 0
    pages_with_overlap = 0
    overlap_helpers_copied = 0
    print(f"Start processing {len(trans_map)} pages, {total_boxes} text boxes from transMap.", flush=True)
    print(f"Overlap helper folder: {inpainted_dir / 'deal_overlap'}", flush=True)
    print(f"Preview folder: {preview_dir}", flush=True)
    print(f"Mask folder: {mask_dir}", flush=True)
    print(f"Center mode: {center_mode}", flush=True)

    for page_index, (page_name, items) in enumerate(trans_map.items(), start=1):
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
        print(f"[{page_index}/{len(trans_map)}] {page_name}: {', '.join(status_parts)}", flush=True)

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
    parser = argparse.ArgumentParser(description="Recenter text rectangles from transMap JSON using matching inpainted PNGs.")
    parser.add_argument("json_path", help="Path to transMap JSON")
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
