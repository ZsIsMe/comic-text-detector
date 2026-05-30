# new_detect_folder.py 使用說明

`new_detect_folder.py` 是精簡版資料夾偵測與 center 重定位腳本。

它保留模型偵測、文字 mask、line-trans 結果與 center 對齊結果，但不再輸出舊流程中的 YOLO `.txt`、文字行 `.txt`、block-box 視覺化和 line-box 視覺化。

## 執行完整流程

在專案根目錄執行：

```bash
python new_detect_folder.py /Users/zhongsheng/Downloads/xxxx
```

指定 CPU：

```bash
python new_detect_folder.py /Users/zhongsheng/Downloads/xxxx --device cpu
```

指定 CUDA：

```bash
python new_detect_folder.py /Users/zhongsheng/Downloads/xxxx --device cuda
```

指定模型路徑：

```bash
python new_detect_folder.py /Users/zhongsheng/Downloads/xxxx --model /path/to/comictextdetector.pt
```

預設模型路徑為：

```text
data/comictextdetector.pt
```

## 只重跑對齊

如果已經有 `ctd/progressing/block_map.json` 和 `ctd/progressing/mask/`，可以不重跑模型，只重新生成 center 對齊結果：

```bash
python new_detect_folder.py /Users/zhongsheng/Downloads/xxxx --only-align
```

`--only-align` 會讀取：

```text
<圖片資料夾>/ctd/progressing/block_map.json
<圖片資料夾>/ctd/progressing/line_trans_map.json
<圖片資料夾>/ctd/progressing/mask/<檔名>.png
<圖片資料夾>/ctd/progressing/align/deal_overlap/<檔名>.png  # 如果存在
```

並更新：

```text
<圖片資料夾>/ctd/progressing/aligned_box_map.json
<圖片資料夾>/ctd/measure.json
<圖片資料夾>/ctd/measure.debug.json
<圖片資料夾>/ctd/progressing/align/center/<檔名>.png
<圖片資料夾>/ctd/progressing/align/neck/<檔名>.png       # 自動 neck 切線計算圖，如果有 shared 氣泡 guide
<圖片資料夾>/ctd/progressing/align/neck/<檔名>.json      # 每頁 neck guide summary
<圖片資料夾>/ctd/measure_preview/<檔名>.png
<圖片資料夾>/ctd/progressing/align/deal_overlap/<檔名>.png  # 如果偵測到重疊且檔案不存在
```

## 輸出結構

完整流程會生成：

```text
<圖片資料夾>/ctd/
  measure.json
  measure.debug.json

  measure_preview/
    <檔名>.png

  progressing/
    block_map.json
    line_trans_map.json
    aligned_box_map.json

    mask/
      <檔名>.png

    line-trans-box/
      <檔名>.png

    align/
      center/
        <檔名>.png

      neck/
        <檔名>.png
        <檔名>.json

      deal_overlap/
        <檔名>.png
```

不會生成：

```text
<圖片資料夾>/ctd/<檔名>.txt
<圖片資料夾>/ctd/line-<檔名>.txt
<圖片資料夾>/ctd/block-box/
<圖片資料夾>/ctd/line-box/
```

## block_map.json

`ctd/progressing/block_map.json` 保存模型偵測出的原始 block box，是 `--only-align` 的主要輸入。

格式：

```json
{
  "blockMap": {
    "267.png": [
      {
        "xyxy_pixel": [681, 113, 743, 278],
        "center_normalized": [0.8527, 0.1629],
        "source_block_index": 0
      }
    ]
  }
}
```

欄位說明：

- `xyxy_pixel`：原始 block box，像素座標。
- `center_normalized`：原始 block 中心點，正規化座標。
- `source_block_index`：該頁中的 block 索引。

## line_trans_map.json

`ctd/progressing/line_trans_map.json` 保存 line + trans 混合框資料。

這份資料來自原本的 `line-trans-box` 算法，用文字行方向與 mask component 估算更貼近文字像素的文字行框。

視覺化圖輸出到：

```text
ctd/progressing/line-trans-box/<檔名>.png
```

這張圖現在使用原圖作為底圖，不再畫完整文字行方框；每個文字行只在 polygon 最上方的短邊附近畫細卡尺標記，數字表示該文字行短邊寬度（px），可作為字體寬度參考。

## aligned_box_map.json

`ctd/progressing/aligned_box_map.json` 保存 center 重定位後的 block box 結果。

格式：

```json
{
  "transMap": {
    "267.png": [
      {
        "x": 642,
        "y": 110,
        "w": 125,
        "h": 196,
        "area": 24500,
        "method": "outer_center",
        "accepted": true,
        "source_block_index": 0,
        "final_xyxy_pixel": [642, 110, 767, 306],
        "layout_debug": {
          "center_mode": "outer",
          "resolved_center_mode": "outer",
          "outer_rect": {},
          "inner_rect": {},
          "accepted": true,
          "skip_reason": null
        }
      }
    ]
  }
}
```

實際使用結果框時，優先讀：

```text
final_xyxy_pixel
```

也可以使用：

```text
x/y/w/h
```

如果 `accepted=false`，表示該項目沒有採用新候選框，而是回退原始 block box。

## measure.json

`ctd/measure.json` 合併最常用的排版測量資訊，每個 item 對應一個文字 block。

格式：

```json
{
  "pages": {
    "267.png": [
      {
        "source_block_index": 0,
        "xyxy_pixel": [642, 110, 767, 306],
        "center_normalized": [0.8437, 0.1733],
        "orientation": "vertical",
        "font_size": 25.6
      }
    ]
  }
}
```

欄位說明：

- `source_block_index`：對應原始 block 索引。
- `xyxy_pixel`：align 成功時使用 `new_xyxy_pixel`；如果 align 回退，使用 `final_xyxy_pixel`（原始 block）。
- `center_normalized`：align 成功時使用 `new_center_normalized`；如果 align 回退，使用 `final_center_normalized`。
- `orientation`：文字方向，只會是 `horizontal` 或 `vertical`；匹配不到 line 時預設 `vertical`。
- `font_size`：同一 block 內 line-trans 短邊寬度的下中位數；匹配不到 line 時回退為 `min(new_xyxy_width, new_xyxy_height)`。

## measure.debug.json

`ctd/measure.debug.json` 在 `measure.json` 的基礎上附帶原始三份資料，方便追查來源：

```json
{
  "pages": {},
  "block": {},
  "line": {},
  "align": {}
}
```

## measure_preview 預覽圖

預覽圖位於：

```text
ctd/measure_preview/<檔名>.png
```

這張圖合併 `ctd/progressing/align/center/<檔名>.png` 和 `ctd/progressing/line-trans-box/<檔名>.png` 的資訊：

- 保留 center 對齊預覽中的 block 陰影、候選區域、中心與結果框。
- 疊加 line-trans 的短邊卡尺測量標記。
- 在每個 `xyxy_pixel` 框外右下角標出如 `28V` / `20H` 的 block 字體資訊，方便檢查 `measure.json` 是否合理。

## center 預覽圖

預覽圖位於：

```text
ctd/progressing/align/center/<檔名>.png
```

畫面規則：

- 使用原圖作為底圖。
- 原始 block 區域疊淺黑色背景。
- magic wand 找到的候選區域疊隨機淺色底色。
- 成功採用時畫三個框：
  - `outer_rect`：最小外置矩形，藍色虛線。
  - `inner_rect`：最大內置矩形，橘色虛線。
  - final/result box：最終文字框，綠色實線。
- 回退時只畫回退後的 final box。
- 中心十字顏色由最後採用的策略決定：
  - `outer`：藍色。
  - `inner`：橘色。
  - `average`：綠色。
  - 回退：黑色。

## center 重定位邏輯

重定位流程復用 `建立对齐方框` 子工程的核心邏輯：

1. 讀取 `block_map.json` 中的 block box。
2. 讀取 `ctd/progressing/mask/<檔名>.png`。
3. 用原圖和 mask 模擬去字圖。
4. 透過 magic wand / flood fill 找氣泡或可放字區域。
5. 計算：
   - `outer_rect`：最小外置矩形。
   - `inner_rect`：最大內置矩形。
   - final/result box：最後候選文字框。
6. 使用安全規則判斷是否採用新框。
7. 若不採用，回退原始 block box。

安全規則包含：

- 新候選框必須和原始 block 有重疊。
- 新候選框中心點必須仍在原始 block 內；否則視為跑到其他氣泡，回退原始 block。
- 新候選框中心點必須在 magic wand 找到的 `outer_rect` 內。
- 中心移動距離不能超過安全閾值。
- 候選框不能超出圖片。
- `outer_rect` / `inner_rect` 面積比例不能過大。

支援的中心策略與舊工程一致：

- `outer`：使用 `outer_rect` 中心。
- `inner`：使用 `inner_rect` 中心。
- `average`：使用 `outer_rect` 與 `inner_rect` 中心平均。
- `auto`：預設策略；偵測到直角形狀時使用 `average`，否則使用 `outer`。

另外新增規則：

```text
如果某個 block 的 outer_rect 和其他 block 的 outer_rect 有重疊，
則該 block 重新使用 outer 策略計算。
```

被這條規則覆寫的項目會在 JSON 中帶有：

```json
"outer_overlap_center_mode_override": true
```

## deal_overlap

`ctd/progressing/align/deal_overlap/` 用於處理共用氣泡。

優先級：

```text
人工修改過的 deal_overlap > 自動 neck 圖 > 原圖
```

注意：程式會比較原圖與 `deal_overlap` 圖。只有 `deal_overlap` 圖真的和原圖有差異時，才視為人工修改並作為最高優先計算圖。單純由程式複製出來、但尚未修改的 helper 會被忽略，仍會嘗試自動 neck 切割。

流程：

1. 每頁先用原圖，或人工修改過的 `deal_overlap` 圖，做第一次重定位。
2. 如果沒有人工修改過的 `deal_overlap`，程式會檢查 shared outer bubble group。
3. 對找到 guide 的 shared group，程式會生成：

```text
ctd/progressing/align/neck/<檔名>.png
ctd/progressing/align/neck/<檔名>.json
```

4. 如果有生成 `neck/<檔名>.png`，會用這張自動切線圖重新做一次重定位。
5. 如果仍有 final boxes 重疊，或 shared group 沒找到 guide，會建立或保留 `deal_overlap` helper，交給人工處理。

## neck 自動切割

`ctd/progressing/align/neck/` 是自動生成的計算輔助圖，用於處理多文段共用同一氣泡。

`neck/<檔名>.png` 是在原圖上畫入黑色 guide line 的計算圖。它不是 preview 圖，會被下一次重定位用來讓 flood fill 被切開。

`neck/<檔名>.json` 記錄每組 shared bubble 的 guide：

```json
{
  "page": "267.png",
  "generated": true,
  "groups": [
    {
      "group": 1,
      "source_block_indices": [0, 1],
      "status": "neck",
      "neck_ratio": 1.1453,
      "guides": [
        {
          "method": "convex_defect",
          "start": [694, 312],
          "end": [619, 170]
        }
      ]
    }
  ]
}
```

目前正式 align 流程的策略是：

```text
只要 shared group 找得到 guide，就畫進 neck 圖並重跑 align。
neck_ratio 只作為 debug/confidence 資訊，不再阻止切割。
```

如果某組 shared group 沒找到 guide，`status` 會是 `no-neck`，該頁仍會被視為需要人工檢查，並保留或建立 `deal_overlap` helper。

## deal_overlap 人工處理

`ctd/progressing/align/deal_overlap/` 是人工 override。

helper 建立流程：

1. 每頁重定位完成後，檢查 final boxes 是否重疊。
2. 如果仍有重疊，或 shared group 沒找到 neck guide，將原圖複製到：

```text
ctd/progressing/align/deal_overlap/<檔名>.png
```

3. 如果該檔案已存在，不覆蓋，避免覆蓋人工修改。
4. 下次執行完整流程或 `--only-align` 時，如果 `deal_overlap/<檔名>.png` 已被人工修改，會優先用它作為 magic wand 計算圖。
5. `center/` 預覽仍然使用原圖作為底圖。

人工使用方式：

- 打開 `ctd/progressing/align/deal_overlap/<檔名>.png`。
- 在共用氣泡中畫線，把連通區切開。
- 再執行：

```bash
python new_detect_folder.py /Users/zhongsheng/Downloads/xxxx --only-align
```

程式會比較原圖與 `deal_overlap` 圖，保護人工修改區域，避免文字 inpaint 把人工切線擦掉。

人工切線不需要使用指定顏色。

## 常用命令

完整偵測：

```bash
python new_detect_folder.py /Users/zhongsheng/Downloads/xxxx --device cpu
```

只重跑對齊：

```bash
python new_detect_folder.py /Users/zhongsheng/Downloads/xxxx --only-align
```

查看輸出：

```text
/Users/zhongsheng/Downloads/xxxx/ctd/measure.json
/Users/zhongsheng/Downloads/xxxx/ctd/measure_preview/
/Users/zhongsheng/Downloads/xxxx/ctd/progressing/align/center/
```
