# neck_watershed_preview.py 使用說明

`neck_watershed_preview.py` 用來測試「多文段共用同一氣泡」時的自動虛擬切割線。

它不會修改 `measure.json`、`measure.debug.json` 或既有偵測結果，只會額外輸出 preview 圖和 summary JSON，方便觀察算法效果。

正式 `new_detect_folder.py` 的 align 流程已經接入同一套 neck guide 算法。每次跑完整流程或 `--only-align` 時，都會對 shared bubble group 嘗試生成：

```text
<圖片資料夾>/ctd/progressing/align/neck/<檔名>.png
<圖片資料夾>/ctd/progressing/align/neck/<檔名>.json
```

正式流程中，只要 shared group 找得到 guide，就會畫入 `align/neck/<檔名>.png` 並重跑 align。`neck_ratio` 只作為 debug/confidence 資訊，不再阻止切割。

## 執行方式

在專案根目錄執行：

```bash
.venv/bin/python neck_watershed_preview.py /Users/zhongsheng/Downloads/new_test
```

指定其他資料夾：

```bash
.venv/bin/python neck_watershed_preview.py /path/to/image_folder
```

腳本預設讀取：

```text
<圖片資料夾>/ctd/measure.debug.json
<圖片資料夾>/ctd/progressing/mask/<檔名>.png
<圖片資料夾>/ctd/progressing/align/deal_overlap/<檔名>.png  # 如果存在
```

輸出到：

```text
<圖片資料夾>/ctd/neck_watershed_preview/
  <檔名>.neck_watershed.png
  neck_watershed_summary.json
```

如果存在 `deal_overlap/<檔名>.png`，腳本會優先用它作為氣泡計算圖；preview 底圖仍是原圖。

注意：preview 腳本只負責觀察效果，會按 `--neck-ratio-threshold` 標示 `neck` 或 `weak-neck`。正式 `new_detect_folder.py` 不使用這個門檻來阻止切割。

## 算法思路

### 1. 找 shared bubble group

腳本先讀取 `measure.debug.json` 裡的：

```text
align.transMap[].layout_debug.outer_rect
align.transMap[].layout_debug.raw_outer_rect
```

如果兩個文字 block 的 `outer_rect` 或 `raw_outer_rect` IoU 高於門檻，預設 `0.92`，就視為同一個 shared bubble group。

這一步的目的不是判斷最終框是否重疊，而是找出「magic wand/flood fill 抓到了同一顆氣泡」的情況。即使最後 safety fallback 回原 block，只要曾經共用同一個 `outer_rect`，都會被抓出來。

### 2. 重建氣泡 mask

對每個 shared group，腳本會重建 align 階段使用的灰度圖：

1. 讀原圖或 `deal_overlap` 圖。
2. 讀 text mask。
3. 對 text mask 膨脹。
4. 用 OpenCV inpaint 去掉文字干擾。
5. 用每個 block 的 seed 重新 flood fill。
6. 合併同一 group 的 flood-fill mask。
7. 做平滑和 close，得到 shared bubble mask。

這裡的重點是讓後續判斷盡量基於氣泡形態，而不是文字位置。

### 3. Watershed 分區

對 shared bubble mask 做 distance transform：

```text
氣泡中心距離邊界遠，distance 值大
氣泡連接窄頸距離邊界近，distance 值小
```

然後用每個文字 block 作為 marker seed，在 `-distance` relief 上做 watershed。

Watershed 的用途是輔助判斷每個文字 block 大概屬於哪個氣泡區域，但目前 preview 的最終 guide line 不直接使用 watershed boundary。因為 watershed boundary 常會沿著整個子區域邊界走，不一定像人工切一刀。

### 4. Convex defect neck guide

實際畫出的 guide line 目前優先使用輪廓凹陷點：

1. 對 shared bubble mask 找最大外輪廓。
2. 計算 convex hull。
3. 用 `cv2.convexityDefects` 找輪廓相對凸包的凹陷點。
4. 選擇可以把兩個 seed 分到兩側的凹陷點對。
5. 用以下條件評分：
   - 凹陷點連線越短越好。
   - 凹陷越深越好。
   - 連線越接近兩個 seed 的中間區域越好。
   - 連線穿過氣泡內部的比例要足夠。
   - 兩個 seed 必須位於切線兩側，而且不能太貼近切線。

這一步符合目前的直覺：氣泡交匯處的凹陷位置很有機會就是切割位置。

### 5. Width scan fallback

如果找不到合適的 convex defect 凹陷點對，腳本會退回寬度掃描：

1. 沿兩個 seed 中心連線取樣。
2. 在每個取樣點嘗試多個角度。
3. 找穿過 bubble mask 的最短橫截面。
4. 把最短橫截面當成 fallback guide line。

這個 fallback 比最初的 projection/gap 更貼近氣泡形態，但優先級低於 convex defect。

## Preview 圖怎麼看

每個 shared group 會畫：

```text
G1/G2/G3     shared bubble group
id N         原始 source_block_index
彩色半透明區域  watershed 分區
粗線 guide    候選虛擬切割線
neck r=...   neck_ratio
```

`neck_ratio` 的含義：

```text
neck_ratio = guide_width / smaller_lobe_width
```

大致判讀：

```text
0.00 - 0.60   明顯窄頸
0.60 - 1.00   有凹陷或連接，但較弱
> 1.00        guide 很寬，可能只是同一大氣泡內多段文字
```

preview 腳本預設 `--neck-ratio-threshold 0.62`。低於這個值會標成 `neck`，高於這個值會標成 `weak-neck`。這只是 preview 標示，不影響正式 align。正式 align 目前是「有 guide 就切」。

## 常用參數

### 指定 debug JSON

```bash
.venv/bin/python neck_watershed_preview.py /path/to/image_folder \
  --debug-json /path/to/measure.debug.json
```

### 指定輸出資料夾

```bash
.venv/bin/python neck_watershed_preview.py /path/to/image_folder \
  --out-dir /tmp/neck_preview
```

### 調整 shared group 判斷

```bash
.venv/bin/python neck_watershed_preview.py /path/to/image_folder \
  --iou-threshold 0.9
```

`--iou-threshold` 越低，越容易把相近但不完全相同的 bubble mask 分到同一組。

### 包含巨大漏框包含小框的情況

```bash
.venv/bin/python neck_watershed_preview.py /path/to/image_folder \
  --include-contained
```

這個模式會把「一個巨大 outer_rect 幾乎包含另一個 outer_rect」也當成 group。它適合 debug 漏框，但容易過度分組，不建議作為日常預設。

### 調整 marker seed 膨脹

```bash
.venv/bin/python neck_watershed_preview.py /path/to/image_folder \
  --seed-dilate 20
```

`--seed-dilate` 只影響 watershed marker，不直接決定 guide line。它會影響彩色分區和 lobe width 估算。

### 調整 neck 門檻

```bash
.venv/bin/python neck_watershed_preview.py /path/to/image_folder \
  --neck-ratio-threshold 0.75
```

門檻越高，越多 group 會被標成 `neck`。

這個參數只影響 `neck_watershed_preview.py` 的標示，不影響 `new_detect_folder.py` 的正式 align 行為。

## Summary JSON

`neck_watershed_summary.json` 會記錄每頁每組的結果：

```json
{
  "ttt.jpg": {
    "preview": ".../ttt.neck_watershed.png",
    "used_deal_overlap": null,
    "groups": [
      {
        "group": 1,
        "source_block_indices": [0, 2],
        "status": "weak-neck",
        "neck_ratio": 1.6721,
        "neck_width": 375.68,
        "smaller_lobe_width": 224.67,
        "guides": [
          {
            "method": "convex_defect",
            "start": [1110, 256],
            "end": [1214, 617],
            "center": [1162, 436]
          }
        ]
      }
    ]
  }
}
```

`method` 目前有兩種：

```text
convex_defect  優先方案，使用輪廓凹陷點對
width_scan     fallback，使用多角度最窄橫截面
```

## 測試建議

推薦流程：

1. 先跑完整偵測：

```bash
.venv/bin/python new_detect_folder.py /path/to/image_folder
```

這一步會自動生成正式計算圖：

```text
<圖片資料夾>/ctd/progressing/align/neck/<檔名>.png
<圖片資料夾>/ctd/progressing/align/neck/<檔名>.json
```

2. 需要觀察算法判斷時，再生成 neck preview：

```bash
.venv/bin/python neck_watershed_preview.py /path/to/image_folder
```

3. 查看：

```text
<圖片資料夾>/ctd/neck_watershed_preview/*.neck_watershed.png
```

4. 如果某頁已經人工修改過 `deal_overlap`，直接重跑 preview。腳本會自動使用該頁的 `deal_overlap` 圖。

5. 如果想觀察漏框或大框包含小框：

```bash
.venv/bin/python neck_watershed_preview.py /path/to/image_folder --include-contained
```

## 目前限制

- preview 腳本不會修改正式 align 結果；正式流程由 `new_detect_folder.py` 生成 `progressing/align/neck/`。
- `weak-neck` 不代表正式流程不切；它只代表 preview 門檻下氣泡形態沒有明顯窄頸。
- 雲朵泡、尾巴很多的氣泡、或人物線條漏進 bubble mask 時，convex defect 可能會產生多個候選凹陷，需要靠 `neck_ratio` 和圖像一起判讀。
- 對「同一顆大氣泡裡自然排了兩段文字」的情況，正式流程目前也會在有 guide 時切割，因為策略是「重疊/shared group 有 guide 就切」。
- 如果 `deal_overlap` 人工切線存在，preview 結果會反映人工切線後的 bubble mask。

## 正式 align 行為

`new_detect_folder.py` 的 align 階段目前行為：

1. 如果 `deal_overlap/<檔名>.png` 存在且已人工修改，使用它作為最高優先計算圖，跳過自動 neck。
2. 如果沒有人工修改過的 `deal_overlap`，先用原圖做第一次 align。
3. 收集 shared outer bubble group。
4. 對每個有 guide 的 shared group，在原圖上畫 guide line，生成 `progressing/align/neck/<檔名>.png`。
5. 如果生成了 neck 圖，用 neck 圖重跑 align。
6. 如果仍有 final overlap，或 shared group 沒有 guide，建立或保留 `deal_overlap` helper 交給人工處理。

優先級：

```text
人工修改過的 deal_overlap > 自動 neck 圖 > 原圖
```
