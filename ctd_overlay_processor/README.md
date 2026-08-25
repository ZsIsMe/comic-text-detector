# CTD Overlay Processor

這個目錄是獨立於 `new_detect_folder.py` 的生成/檢視工具。它不修改偵測腳本；需要資料時會調用既有流程生成 GUI 所需的 JSON/NPZ/mask，但不落地生成 `measure_preview` 這類預覽 PNG。

## 目標

- 生成並只讀顯示 `<圖片資料夾>/ctd/measure.json`。
- 導入 LabelPlus txt，生成 `*_meo.json`、`*_meo.框外.json`、`*_meo_bt.json`。
- 在 GUI 左側編輯 `_meo_bt.json`，右側對照 `ctd/measure.json`。
- 允許左側選中 `_bt` 條目後，點右側 measure 框手動套用框、中心、字級、方向與樣式。

## 讀取的資料

```text
<圖片資料夾>/ctd/
  measure.json
  measure.debug.json
  progressing/
    block_map.json
    line_trans_map.json
    aligned_box_map.json
    mask/<檔名>.png
    align/masks/<檔名>.npz
```

`ctd/measure.json` 是 CTD 偵測流程的量測結果，在 GUI 裡只讀顯示，不再作為人工編輯文件。

## 命令列生成 CTD

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
```

它不會生成 `measure_preview/`、`measure_wh_preview/`、`only_text/`、`inpainted/` 等靜態預覽圖，也不再生成外層 `measure.custom.json`。

## LabelPlus 導入

在 GUI 中按「導入 LabelPlus txt」，選擇例如：

```text
翻譯_0.txt
```

會在 txt 同目錄生成：

```text
翻譯_0_meo.json
翻譯_0_meo.框外.json
翻譯_0_meo_bt.json
```

其中 `翻譯_0_meo_bt.json` 是後續 GUI 編輯文件。生成時會根據 `ctd/measure.json` 嘗試自動匹配，並寫入：

- `match_status: "auto"`：唯一匹配成功。
- `match_status: "duplicate"`：多條文字命中同一 OCR 框，需要人工確認。
- `match_status: "unmatched"`：未命中 OCR 框，需要人工確認。
- `match_status: "manual"`：GUI 手動套用或調整後的條目。

## PySide6 GUI

需要已安裝完整 `PySide6`，HTML 豎排預覽會使用其中的 QtWebEngine：

```bash
python -m ctd_overlay_processor.viewer /path/to/image_folder
```

macOS 可以直接雙擊：

```text
ctd_overlay_processor/launch.command
```

第一次啟動會自動建立：

```text
ctd_overlay_processor/.venv/
```

並安裝本工具和偵測流程需要的依賴。之後會直接啟動 viewer，進入後用「選擇資料夾」選圖片資料夾；如果還沒有 `ctd/`，按「生成/更新 CTD」即可。

## GUI 佈局

- 左側畫面：顯示圖片和目前載入的 `_bt.json`，支持選中、拖動、調整框、改文字和樣式。
- 右側畫面：顯示圖片和 `ctd/measure.json` 疊圖，只讀。
- 側欄列表：顯示當前頁 `_bt` 條目，標出自動、待確認、未匹配、手動狀態。

手動匹配流程：

1. 在左側列表或左側畫面選中一條 `_bt`。
2. 在右側點一個 measure 框。
3. GUI 將右側 measure 的框、中心、字級、方向、顏色/描邊套用到左側 `_bt`。
4. 按「保存 _bt.json」或 `Command+S` 寫入 `_meo_bt.json`。

## 可編輯內容

目前只編輯 `_bt.json`：

- 文本內容
- 字體大小
- 排版方向
- 文字黑/白
- 描邊粗細
- 需要修復/描邊
- 框位置與大小
- 單框方向鍵移動：方向鍵 1px，`Shift` + 方向鍵 10px，`Ctrl` / `Command` + `Shift` + 方向鍵 50px
- `Esc`：取消目前文字框的選取與編輯焦點，不修改文字框內容
- `Ctrl` / `Command` + `Option` / `Alt` + `[` / `]`：逆時針 / 順時針旋轉選取文字框 5°
- 文字框剪貼簿：按「加入持久剪貼簿列表」可將目前選取框加入全域列表。`F1` 僅暫存複製目前選取框，不寫入列表；單擊列表項目也只會暫存複製該項。`F2` 在左側圖片游標位置貼上暫存內容；沒有暫存內容時不執行。剪貼簿歷史會在軟體重開後保留，暫存內容則不保留。
- `Ctrl` / `Command` + `N`：在左側圖片游標位置新增文案；系統剪貼簿有非空白文字時，會自動帶入該文字，否則新增空文案。
- 單框刪除：`Delete` / `Backspace`

點擊「生成/更新 CTD」時會先設定預設字級與 Step（首次預設為 `24.0, 2.0`）。逐字 OCR 會排除不可靠與異常字級，再依可靠結果的範圍建立候選字級，選擇所有可靠字元長寬總誤差最低的候選。`ctd/measure.json` 的 `font_size` 預設保留一位小數。

## 命令列檢查資料

```bash
python -m ctd_overlay_processor.processor /path/to/image_folder
```

檢查單頁：

```bash
python -m ctd_overlay_processor.processor /path/to/image_folder --page 001.png --json
```
