# Solid Inpaint Folder

Small folder-level pipeline for generating text masks, solid-background inpaint overlays, and fallback masks.

## Usage

```bash
python solid_inpaint/detect_solid_inpaint_folder.py /path/to/image_folder
```

Optional arguments:

```bash
--model /path/to/comictextdetector.pt
--device cpu
--device cuda
```

## Output

For an input folder `/path/to/images`, files are written to:

```text
/path/to/images/ctd_inpainted/mask/<name>.png
/path/to/images/ctd_inpainted/other_mask/<name>.png
/path/to/images/ctd_inpainted/inpainted/<name>.png
/path/to/images/ctd_inpainted/solid_inpaint_report.json
/path/to/images/ctd_inpainted/preview_report.pdf
```

- `mask`: refined text mask from Comic Text Detector.
- `other_mask`: repair areas whose sampled background is not reliable enough for solid-color filling.
- `inpainted`: transparent full-canvas BGRA overlay containing only automatically filled solid-background regions.
- `preview_report.pdf`: page-by-page review PDF with original, composited preview, mask, and other mask.

## Photoshop PSD Companion

After generating the Python outputs, run this JSX in Photoshop:

```text
solid_inpaint/create_psds_from_outputs.jsx
```

It creates:

```text
/path/to/images/ctd_inpainted/psd/<name>.psd
```

Each PSD contains two layers:

```text
bg
overlay-manual
```

And two alpha channels:

```text
TEXT_CHANNEL
OTHER_CHANNEL
```

The dialog also includes an optional action runner. If enabled, the selected Photoshop Action is executed after channels/layers are created and before the PSD is saved.
