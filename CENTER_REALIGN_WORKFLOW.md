# center 重定位流程說明

`detect_folder.py` 目前除了原本的文字偵測輸出，還會基於 block box、文字 mask 和 magic wand 邏輯，重新估算文字框在氣泡中的位置。

## 執行命令

在專案根目錄執行：

```bash
python detect_folder.py /Users/zhongsheng/Downloads/xxxx
```

如果需要指定裝置：

```bash
python detect_folder.py /Users/zhongsheng/Downloads/xxxx --device cpu
python detect_folder.py /Users/zhongsheng/Downloads/xxxx --device cuda
```

如果需要額外輸出模型原始 block JSON：

```bash
python detect_folder.py /Users/zhongsheng/Downloads/xxxx --json
```

## 主要輸出

執行後會在圖片資料夾下生成或更新 `ctd/`：

```text
<圖片資料夾>/ctd/
```

其中包含：

- `<檔名>.txt`：YOLO 格式的文字區塊 block box。
- `line-<檔名>.txt`：文字行四邊形。
- `mask-<檔名>.png`：文字分割 mask。
- `line_trans_map.json`：line + trans 混合後的文字行框資料。
- `aligned_box_map.json`：重定位後的 block box 結果。
- `block-box/<檔名>.png`：block box 視覺化。
- `line-box/<檔名>.png`：line box 視覺化。
- `line-trans-box/<檔名>.png`：line-trans box 視覺化。

另外會在圖片資料夾下生成：

```text
<圖片資料夾>/center/
<圖片資料夾>/deal_overlap/
```

## aligned_box_map.json

`ctd/aligned_box_map.json` 是重定位後的主要資料輸出，格式如下：

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

實際可用的結果框可讀：

- `x/y/w/h`
- 或 `final_xyxy_pixel`

如果 `accepted=false`，代表該項目沒有採用新候選框，而是回退到原始 block box。

## center 預覽圖

`center/<檔名>.png` 用原圖作為底圖，用來人工檢查重定位效果。

畫面規則：

- 原始 block 區域會疊一層淺黑色背景。
- magic wand 找到的候選區域會疊隨機淺色底色。
- 成功採用時會畫三個框：
  - `outer_rect`：最小外置矩形，藍色虛線。
  - `inner_rect`：最大內置矩形，橘色虛線。
  - final/result box：最終文字框，綠色實線。
- 未採用時只畫回退後的 final box，通常是黑色。
- 中心十字顏色由最後採用的策略決定：
  - `outer`：藍色。
  - `inner`：橘色。
  - `average`：綠色。
  - 回退：黑色。

## 重定位邏輯

重定位流程盡量復用 `建立对齐方框` 子工程的核心邏輯：

1. 將模型輸出的 block box 轉成舊工程使用的 `xyxy_pixel` / `center_normalized` 結構。
2. 使用原圖和 `mask-<檔名>.png` 模擬去字圖。
3. 將文字 mask 膨脹後用 `cv2.inpaint()` 補掉文字。
4. 將模擬去字圖傳入舊工程的 `calculate_layout()`。
5. 使用舊工程的 `apply_safety_rules()` 做安全檢查。
6. 使用舊工程的 `draw_item_preview()` 畫預覽框。

目前支援的中心策略與舊工程一致：

- `outer`：使用 `outer_rect` 中心。
- `inner`：使用 `inner_rect` 中心。
- `average`：使用 `outer_rect` 和 `inner_rect` 中心平均。
- `auto`：預設策略；偵測到直角形狀時使用 `average`，否則使用 `outer`。

另外新增一條規則：

```text
如果某個 block 的 outer_rect 和其他 block 的 outer_rect 有重疊，
則該 block 重新使用 outer 策略計算。
```

被這條規則覆寫的項目，會在 JSON 中帶有：

```json
"outer_overlap_center_mode_override": true
```

## deal_overlap

`deal_overlap/` 用於處理共用氣泡或候選框重疊問題。

流程：

1. 每頁重定位完成後，檢查 final boxes 是否重疊。
2. 如果有重疊，將原圖複製到：

```text
<圖片資料夾>/deal_overlap/<檔名>.png
```

3. 如果該檔案已存在，不會覆蓋，避免蓋掉人工修改。
4. 下次執行時，如果 `deal_overlap/<檔名>.png` 存在，會優先用它作為 magic wand 計算圖。
5. `center/` 預覽仍然使用原圖作為底圖。

人工處理方式：

- 在 `deal_overlap/<檔名>.png` 上畫線，把共用氣泡切開。
- 線條需要真正切斷白色連通區。
- 下次執行 `detect_folder.py` 時，程式會使用修改後的 `deal_overlap` 圖重新計算。

為了避免文字 mask 的 inpaint 把人工切線擦掉，程式會比較原圖和 `deal_overlap` 圖：

```text
原圖 vs deal_overlap
-> 找出人工修改區域
-> 從 inpaint mask 中保護這些區域
-> 保留人工切線作為 magic wand 阻隔
```

因此人工切線不需要指定特定顏色。

## 不生成 npz

目前沒有生成 `.npz`。

原因：

- 目前 `.npz` 不參與加速或重複使用。
- 本流程主要依賴現有 `ctd/*.txt`、`mask-*.png` 和原圖即可重算。
- 減少額外輸出，方便人工檢查。

如果未來需要調試 mask 細節，再考慮補充 `.npz` 輸出。
