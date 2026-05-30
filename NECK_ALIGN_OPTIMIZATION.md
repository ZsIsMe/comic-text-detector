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
