# neck align 優化筆記

本文記錄 `new_detect_folder.py` 中 neck 自動切割流程的耗時來源與後續優化方向。

## 背景

目前 align 階段已接入 neck 自動切割：

```text
普通 align
→ 找 shared outer bubble group
→ 對 shared group 生成 neck guide
→ 畫入 progressing/align/neck/<檔名>.png
→ 用 neck 圖重跑 align
```

人工修改過的 `deal_overlap` 仍然最高優先：

```text
人工 deal_overlap > 自動 neck > 原圖
```

## 目前主要耗時

neck 流程主要是 CPU 圖像處理，不是模型推理。

主要耗時點：

```text
1. 每頁先跑一次普通 align。
2. neck 階段重新 prepare align gray，包括 text mask dilate 和 inpaint。
3. 每個 shared group 重新 flood fill 建 bubble mask。
4. 對 bubble mask 做 smooth / morphology close。
5. 對 shared mask 做 distanceTransform。
6. 用 markers 跑 watershed。
7. 用 convexityDefects 或 width scan 找 guide。
8. 一旦生成 neck 圖，整頁再跑一次 align。
```

其中 OpenCV 的 `inpaint`、`floodFill`、`distanceTransform`、`watershed` 在目前用法下基本都是 CPU 操作。

## CUDA 是否能加速

直接使用 CUDA 的收益有限。

原因：

```text
- 這段不是 PyTorch 模型推理，主要是 OpenCV 傳統圖像處理。
- cv2.cuda 並不完整覆蓋 inpaint / floodFill / watershed 等操作。
- 即使部分操作能搬到 GPU，也會有 CPU/GPU 資料搬運成本。
- 目前更大的問題是重複計算，而不是單個 kernel 太慢。
```

因此優先方向應該是減少重複工作、縮小處理範圍、加 cache，而不是先做 GPU 化。

## 優化優先級

### P0: page-level gray/inpaint cache

目前普通 align 和 neck 流程都會準備 align gray。

優化方式：

```text
1. 每頁只做一次 prepare_align_gray。
2. 普通 align、neck bubble mask、二次 align 共用同一份 inpainted gray。
3. 如果使用 neck 圖重跑 align，再只針對 neck 圖生成一次對應 gray。
```

預期收益：

```text
高。inpaint 是主要耗時之一。
```

風險：

```text
中低。需要整理 _align_block_boxes / layout_core 的接口，避免每個 block 內部反覆 prepare gray。
```

### P0: outer_rect / raw_outer_rect group cache

同一 shared group 內多個 block 通常得到完全相同的 bubble mask。

優化方式：

```text
1. 用 outer_rect 或 raw_outer_rect 作為 key。
2. 同一 key 的 bubble mask 只 flood fill / smooth 一次。
3. group 內多個 block 共享該 mask。
```

預期收益：

```text
高。可減少重複 flood fill 和 mask 後處理。
```

風險：

```text
低。shared group 本身就是以 outer/raw outer 相似度聚合。
```

### P0: 二次 align 只重跑受影響 block

目前只要某頁生成 neck 圖，就整頁所有 block 重新 align。

優化方式：

```text
1. 記錄 neck guide 涉及的 source_block_indices。
2. 二次 align 只重跑這些 block。
3. 其他 block 沿用第一次 align 結果。
4. 最後合併成 aligned_items。
```

預期收益：

```text
高。很多頁只有 2-4 個 block 需要 neck，但整頁可能有十幾個 block。
```

風險：

```text
中。neck guide 可能改變相鄰 bubble 的 flood fill，理論上受影響範圍可能比 source_block_indices 更大。
```

保守做法：

```text
重跑 source_block_indices，加上與 neck guide bbox 有接近或 overlap 的 block。
```

### P1: cheap prefilter

現在所有 shared group 都會嘗試完整 neck 分析。

優化方式：

```text
在 distanceTransform / watershed 前先做 cheap filter：

- group size 是否 >= 2
- outer_rect 是否完全相同或 IoU 非常高
- old block center 距離是否合理
- old block 是否都位於 outer_rect 內
- shared outer_rect 面積是否過大
- block 之間是否有基本分離方向
```

預期收益：

```text
中。可以跳過明顯不像連體氣泡的 shared group。
```

風險：

```text
中。filter 太嚴會漏掉真正需要切的頁。
```

目前策略是「重疊/shared group 有 guide 就切」，所以 prefilter 應保持保守。

### P1: width_scan 降採樣

`width_scan` 是 fallback，但在大 group 或多 pair group 中會變慢。

優化方式：

```text
1. 降低角度數量，例如 4 度改成 8 度。
2. 降低中心線 sample_count 上限。
3. 只在 convex_defect 失敗且 group size 小時啟用。
4. 對過大的 shared outer_rect 直接跳過 width_scan。
```

預期收益：

```text
中。
```

風險：

```text
中低。fallback 精度會下降，但 convex_defect 仍是優先方案。
```

### P1: neck result cache

如果原圖、mask、block_map、deal_overlap 沒變，可以復用既有 neck 結果。

優化方式：

```text
1. 在 neck json 中記錄 input signature。
2. signature 包含：
   - source image mtime/size 或 hash
   - mask mtime/size 或 hash
   - block items hash
   - deal_overlap modified 狀態與 mtime/hash
   - algorithm version
3. 下次 align 時 signature 相同，直接讀 neck png/json。
```

預期收益：

```text
高，尤其是反覆 --only-align 測試時。
```

風險：

```text
中。cache invalidation 要寫清楚，避免使用過期 neck 圖。
```

### P2: ROI-only operations

現在某些 mask 操作仍在整頁尺度上跑。

優化方式：

```text
1. 對每個 shared group 裁切 outer_rect union 的 ROI。
2. distanceTransform / watershed / convex defects 都在 ROI 中做。
3. 最後把 guide 座標轉回全圖。
```

預期收益：

```text
中到高，取決於頁面尺寸和 shared group 面積。
```

風險：

```text
中。ROI padding 要足夠，否則可能切掉氣泡邊界或尾巴。
```

### P2: 並行頁面處理

align 是逐頁處理，可以做 page-level 並行。

優化方式：

```text
1. 用 multiprocessing / concurrent.futures 按頁並行。
2. 每個 worker 處理一頁 align + neck + preview。
3. 最後合併 aligned_box_map。
```

預期收益：

```text
高，適合多頁資料夾。
```

風險：

```text
中高。需要注意 tqdm、寫檔、OpenCV thread 數量、記憶體佔用。
```

建議先做前面 P0/P1，再考慮並行。

## 建議實施順序

推薦順序：

```text
1. P0 page-level gray/inpaint cache
2. P0 二次 align 只重跑受影響 block
3. P0 outer_rect / raw_outer_rect group cache
4. P1 neck result cache
5. P1 width_scan 降採樣
6. P2 ROI-only operations
7. P2 page-level 並行
```

前三項應該能明顯降低目前 `--only-align` 的耗時，同時不改變算法策略。

## 測試指標

每次優化後建議記錄：

```text
- align 總耗時
- 頁數
- block 數
- shared 頁數
- neck 自動切割頁數
- neck guide 數
- accepted 數
- deal_overlap 使用頁數
- 未修改 deal_overlap 忽略頁數
```

並對比：

```text
- measure.json 是否有非預期變化
- center preview 是否有明顯退化
- align/neck/*.json 是否仍可解釋
```

## 目前結論

CUDA 不是第一優先。當前瓶頸主要是重複 CPU 圖像處理與整頁二次 align。

最值得先做的是：

```text
page-level gray cache
只重跑受 neck 影響的 block
bubble mask/group cache
```

## 2026-06-02 實作進度

測試資料：

```text
/Volumes/zs/comic_data/钱球大联盟篇/9_非单行本/75
圖片數量：21
block 數：162
```

### 已完成：page-level gray/inpaint reuse

`detect_folder.py` 現在支援把同一頁已準備好的 `prepared_gray` 傳入 `_align_block_boxes()` /
`_align_block_box()`。同一次 align 呼叫內不再對每個 block 重複 `_prepare_align_gray()`。

`new_detect_folder.py` 在每頁第一次 align 前準備一次 `calc_gray`，並同時復用給 neck guide 分析。
這不是跨執行 cache，不保存舊結果；每次執行仍會依照當前輸入重新計算。

實測：

```text
原始重定位：21/21 [01:40, 4.80s/it]
第一階段後：21/21 [01:09, 3.31s/it]
收益：約 31 秒，約 31%
```

效果驗證：

```text
流程正常完成
neck shared 頁數：11
neck 自動切割頁數：10
neck guide 數：20
neck 無 guide group 數：1
偵測到重疊頁數：5
```

### 已完成：neck 後只重跑受影響 block

`new_detect_folder.py` 現在生成 neck 圖後不再整頁重跑 align，而是保守地重跑：

```text
guide.source_block_indices
同一個 neck shared group 的所有 block
與 guide bbox 附近或相交的 block
與 affected block 有 outer overlap 連通關係的 block
```

實測：

```text
第二階段後：21/21 [01:07, 3.23s/it]
相對第一階段收益：約 2 秒
```

效果驗證：

```text
aligned_box_map vs 第一階段：changed_items 0
measure.json vs 第一階段：measure_equal True
```

局部重跑覆蓋統計：

```text
generated neck pages：10
affected blocks：42 / 83
```

結論：

```text
第二階段邏輯安全，但在此測試集收益有限。
目前慢頁主要仍卡在 neck guide 生成/分析本身，而不是 neck 後二次 align。
```

### 本次測試中生成 neck 的頁面

```text
9_45.jpeg        guides=2  groups=[[0, 3, 5]]
9_45_副本.jpeg   guides=3  groups=[[0, 3, 6, 5]]
9_46.jpeg        guides=2  groups=[[0, 2, 5, 9]]
9_47.jpeg        guides=1  groups=[[0, 1]]
9_49.jpeg        guides=2  groups=[[0, 1], [4, 5]]
9_55.jpeg        guides=2  groups=[[5, 10, 9]]
9_56.jpeg        guides=1  groups=[[5, 6]]
9_57.jpeg        guides=4  groups=[[2, 4, 6, 7, 10]]
9_58.jpeg        guides=1  groups=[[1, 4]]
9_64.jpeg        guides=2  groups=[[3, 7, 8]]
```

shared group 但沒有 guide：

```text
9_62.jpeg        groups=[[7, 6]]  status=no-neck
```

### 下一步觀察：不該 neck 的頁面也被分析

目前 `collect_shared_groups()` 只要 outer/raw_outer 高 IoU 就會進入完整 neck 分析。
這可能包含一些其實不需要 neck 的情況，例如：

```text
同一個大外框下的普通相鄰文字
人物線條或背景讓 flood fill 外框異常偏大
雲朵泡、尾巴、複雜邊界導致 convex defect 過度產生 guide
block 很近但沒有真正共用連體氣泡
shared group 面積過大，實際不是單一可切 bubble
```

下一階段比繼續壓二次 align 更值得做：

```text
1. 加 per-page/per-group neck profiling，拆出 bubble_mask / distanceTransform / watershed / guide scan 耗時。
2. 加 cheap prefilter，先跳過明顯不該 neck 的 shared group。
3. 在 neck json 中記錄 skipped_reason，方便回看是否誤殺。
4. 對本次生成 neck 的 10 頁逐頁人工看 preview，標記哪些是真正需要 neck、哪些是誤分析。
```
