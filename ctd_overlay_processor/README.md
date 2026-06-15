# CTD Overlay Processor

這個目錄是獨立於 `new_detect_folder.py` 的生成/檢視工具。它不修改偵測腳本；需要資料時會調用既有流程生成 GUI 所需的 JSON/NPZ/mask，但不落地生成 `measure_preview` 這類預覽 PNG。

## 目標

- 用 Python 處理器讀取圖片資料夾同級的 `measure.custom.json`，並搭配 `ctd/` 內的 JSON/NPZ。
- 讓 PySide6 程序即時控制方框、mask、字級、line polygon、char boxes 的顯示。
- 保留 `mask` 作為資料來源，但顯示效果由程序動態疊圖。
- 新圖片資料夾沒有 `ctd/` 時，可在 GUI 內按「生成/更新 CTD」建立資料。

## 讀取的資料

```text
<圖片資料夾>/ctd/
  measure.debug.json
  progressing/
    block_map.json
    line_trans_map.json
    aligned_box_map.json
    mask/<檔名>.png
    align/masks/<檔名>.npz
<圖片資料夾>/measure.custom.json
```

## 命令列生成資料

新圖片資料夾還沒有 `ctd/` 時，可以用：

```bash
python ctd_overlay_processor/run_detection.py /path/to/image_folder
```

這會生成：

```text
ctd/progressing/block_map.json
ctd/progressing/line_trans_map.json
ctd/progressing/aligned_box_map.json
ctd/progressing/mask/<檔名>.png
ctd/progressing/align/masks/<檔名>.npz
ctd/measure.json
ctd/measure.debug.json
measure.custom.json
```

`ctd/measure.json` 是偵測流程的原始量測結果；GUI 顯示與編輯只讀寫圖片資料夾同級的 `measure.custom.json`。「生成/更新 CTD」會用最新結果覆蓋 `measure.custom.json`。

它不會生成 `measure_preview/`、`measure_wh_preview/`、`only_text/`、`inpainted/` 等靜態預覽圖。

## 命令列檢查資料

```bash
python -m ctd_overlay_processor.processor /path/to/image_folder
```

檢查單頁：

```bash
python -m ctd_overlay_processor.processor /path/to/image_folder --page 001.png --json
```

## PySide6 檢視器

需要已安裝 `PySide6` 或 `PySide6-Essentials`：

```bash
python -m ctd_overlay_processor.viewer /path/to/image_folder
```

## 雙擊啟動

macOS 可以直接雙擊：

```text
ctd_overlay_processor/launch.command
```

第一次啟動會自動建立：

```text
ctd_overlay_processor/.venv/
```

並安裝本工具和偵測流程需要的依賴。之後會直接啟動 viewer，進入後用「選擇資料夾」選圖片資料夾；如果還沒有 `ctd/`，按「生成/更新 CTD」即可。

介面中的圖層開關會即時重建：

- text mask
- npz smoothed mask
- npz outer body mask
- 原始 block boxes
- aligned boxes
- line polygons
- char boxes
- font labels

目前第一階段可編輯並保存到 `measure.custom.json`。普通修改會先暫存在當前頁；按「保存修改」或 `Command+S` 才寫入，切換頁面前也會先保存當前頁。`Command+Z` 只撤銷當前頁的普通修改，不跨頁撤銷；「字體取偶數」會對全部頁面立即保存，不支持撤銷。

- 單框字體大小
- 單框位置與大小
- 文字黑/白
- 原字描邊
- 需要修復/描邊
- 全部頁面「字體取偶數」
