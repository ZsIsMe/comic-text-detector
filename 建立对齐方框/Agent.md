# Agent Notes

## Text Rectangle Recentering Workflow

The recentering tool reads a `text_rect.json` file, finds matching PNG files in the sibling `inpainted/` directory, calculates candidate text placement from the inpainted image, writes a new JSON next to the source JSON, and saves preview images in a sibling preview folder.

### Path Rules

- Input is a JSON path, for example `example/text_rect.json`.
- Matching images are loaded from `example/inpainted/`.
- Page names in JSON can be `.jpg`; the script maps them by stem to `.png`, for example `001.jpg` -> `inpainted/001.png`.
- If `inpainted/deal_overlap/<stem>.png` exists, use it for layout calculation instead of `inpainted/<stem>.png`.
- Preview images should always use the original `inpainted/<stem>.png` as the base image, not the `deal_overlap` image.
- Preview images are saved under `inpainted/recentered_preview/` by default.
- Per-page mask debug data is saved as `.npz` under `inpainted/recentered_masks/` by default.
- No `.npz` cache should be used or generated; every run recalculates from the image.

### Candidate Geometry

For each text item:

1. Use the original text rectangle center as the primary seed point.
2. Use flood fill / magic-wand logic on the inpainted PNG to get the containing bubble/region mask.
3. Smooth the mask.
4. Calculate the outer minimum bounding rectangle of the smoothed mask: `outer_rect`.
5. Calculate the largest inner rectangle inside the mask: `inner_rect`.
6. Preserve the original outer rectangle as `raw_outer_rect` for debugging.
7. If `raw_outer_rect` has a strong one-sided tail/asymmetry around `inner_rect`, refine `outer_rect` with mask projection before calculating the candidate.
8. Candidate text center is selected by `--center-mode`.
9. `--center-mode auto` is the default. It detects right-angle corners in the mask; if any are found, it resolves to `average`, otherwise it resolves to `outer`.
10. `--center-mode outer` uses the refined outer rectangle center.
11. `--center-mode inner` uses the largest inner rectangle center.
12. `--center-mode average` uses the average of the refined outer rectangle center and inner rectangle center.
13. Candidate result rectangle is centered on the selected candidate center.
14. Candidate result width is the average of refined outer width and inner width.
15. Candidate result height is the average of refined outer height and inner height.
16. If inner rectangle calculation fails, fall back to the outer rectangle center and outer rectangle size as the candidate.

### Safety Rules

A candidate should only be accepted when all safety checks pass:

- **Overlap with old box:** the candidate result rectangle must overlap its own original `xyxy_pixel` rectangle. If it does not overlap, skip processing for this item.
- **Center inside bubble bbox:** the candidate center must be inside `outer_rect`. If not, skip processing.
- **Large movement:** if movement is greater than `150px`, skip processing to avoid obvious bad jumps.
- **Image bounds:** the un-clamped candidate rectangle must be fully inside the image. If any normalized coordinate would be outside `0..1`, skip processing.
- **Outer/inner area ratio:** if refined `outer_rect` area is greater than `4x` the `inner_rect` area, skip processing.

Skipping means the final text rectangle remains the original rectangle.

### JSON Output Rules

- Write two JSON files.
- The main output, for example `text_rect_recentered.json`, should keep only the original `text_rect.json` schema/fields. It should write final adopted coordinates back into the original fields such as `xyxy_pixel` and `center_normalized`.
- The debug output, for example `text_rect_recentered.debug.json`, should keep the full current debug structure, including fields such as `new_xyxy_pixel`, `new_center_normalized`, `final_xyxy_pixel`, `final_center_normalized`, and `layout_debug`.
- Do not add JSON fields for overlap-helper status, calculation image source, or helper paths. Keep overlap-helper information in terminal output only.

### Overlap Helper Rules

- After processing each page, check final text boxes for overlap.
- If overlap is detected, create `inpainted/deal_overlap/` if needed and copy the original `inpainted/<stem>.png` into it.
- Never overwrite an existing `inpainted/deal_overlap/<stem>.png`; it may contain manual edits.
- On later runs, if a matching `deal_overlap` image exists, use it for calculation.
- Keep terminal logging concise: report page-level status, whether `deal_overlap` was used, whether overlap was detected, and whether a helper image was copied or already existed.

### Preview Rules

- Overlay refined outer-body mask areas for every candidate using at least 20 distinct light colors to distinguish text boxes.
- Keep smoothed mask arrays in `.npz` for data inspection, but do not show smoothed masks in preview images.
- Mask arrays are saved separately in `.npz` files and must not be embedded into JSON.
- If a candidate is accepted, draw three boxes:
  - `outer_rect` as a dashed box.
  - `inner_rect` as a dashed box.
  - final/result text box as a green solid box.
- If a candidate is not accepted, draw only one box:
  - the original/final text box as a black solid box.
- Do not draw text labels such as `inner 1`, `outer 1`, or `result 1` on previews.
- Keep preview logging concise: print page-level progress only, not per-box progress.
