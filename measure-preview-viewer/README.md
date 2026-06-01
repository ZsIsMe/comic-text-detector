# Measure Preview Viewer

本工具用原圖和 `ctd` 內的 JSON，在瀏覽器中即時重建 `measure_preview` 的主要視覺效果。

## 啟動方法

在專案根目錄啟動本地靜態 server：

```bash
cd /Users/zhongsheng/Projects/comic-text-detector
python3 -m http.server 8765 --bind 127.0.0.1
```

然後在瀏覽器打開：

```text
http://127.0.0.1:8765/measure-preview-viewer/index.html
```

關閉服務時，回到啟動 server 的終端按 `Ctrl+C`。

如果當前已經在 `measure-preview-viewer/` 目錄下，也可以這樣啟動：

```bash
python3 -m http.server 8765 --bind 127.0.0.1
```

此時打開：

```text
http://127.0.0.1:8765/index.html
```

## 使用方法

1. 打開上面的 viewer URL。
2. 點擊「選擇圖片資料夾」。
3. 選擇包含原圖和 `ctd/measure.json` 的圖片資料夾。

目標資料夾應該類似：

```text
<圖片資料夾>/
  001.png
  002.png
  ctd/
    measure.json
    measure.debug.json
    measure_ocr.json  # 可選
```

如果瀏覽器不支援資料夾選擇，可以用「手動選檔」選中 `measure.json`、`measure.debug.json`、可選的 `measure_ocr.json`，以及對應原圖。

## 顯示內容

- 原始 block 半透明底色
- `raw_outer_rect`
- `outer_rect`
- `inner_rect`
- final/result box
- line-trans 短邊卡尺
- 中心色塊
- `28V` / `20H` 字號方向標籤
- OCR 文字，可選

## 限制

原始 `center` preview 中 magic wand / flood fill 的逐像素候選區域沒有保存在 `measure.debug.json` 裡，所以不能精確還原。本工具的 `Approx bubble` 圖層只會用 `outer_rect` 或 `raw_outer_rect` 做近似。
