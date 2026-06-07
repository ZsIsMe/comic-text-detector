# Font-size measurement from character boxes

`new_detect_folder.py` estimates paragraph `font_size` from the character boxes generated for `measure_wh_preview`.

## Goal

The value should represent the best rendering font size for the main paragraph text. Ruby/furigana and small side notes should not pull the value down, and merged or badly split characters should not push the value up.

## Character boxes

Character boxes are extracted from each line mask in `detect_folder.py`.

Before splitting along the text direction, the mask pixels are filtered by short-axis lanes:

- The widest / most populated short-axis lane is treated as the main text lane.
- Side lanes around `30-50%` of the main text size are treated as ruby/furigana.
- Ruby lanes are ignored for character boxes and for font-size estimation.

Each kept character box records:

- `W`: horizontal image width of the character ink box.
- `H`: vertical image height of the character ink box.

For vertical Japanese text, `W` is a stable line-width signal, while the upper normal range of `H` is often close to the actual font size. Both are used.

## Paragraph font-size rule

For each paragraph:

1. Collect all character boxes from the matched line-trans boxes.
2. Determine orientation from the matched lines.
3. Select dominant values for each dimension by dropping very small values below `0.55 * P75`.
4. For vertical text:
   - `primary_size = P60(valid W)`
   - `secondary_size = P75(valid H, capped by primary_size * 1.6 or primary_size + 8)`
5. For horizontal text, swap W/H:
   - `primary_size = P60(valid H)`
   - `secondary_size = P75(valid W, capped by primary_size * 1.6 or primary_size + 8)`
6. `font_size = max(primary_size, secondary_size)`.

If no character boxes are available, the old line-width upper-median fallback is used. If that also fails, the aligned paragraph box short side is used.

## Output and preview

`measure.json` stores:

- `font_size`
- `font_size_method`

`measure.debug.json` stores the character boxes and intermediate font-size values under `font_size`.

Previews draw a paragraph font-size marker:

- A square at the paragraph corner sized to the computed `font_size`.
- A label such as `FS35`.

The marker appears in both `ctd/measure_preview/` and `ctd/measure_wh_preview/`.
