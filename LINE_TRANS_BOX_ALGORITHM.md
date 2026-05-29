# line-trans-box 算法說明

`line-trans-box` 是目前用來估算漫畫文字行尺寸的混合算法。它的目標是結合兩類方法的優點：

- `line-box`：保留 comic-text-detector 對文字行、直排、斜排的語義理解。
- `trans/component` 思路：用文字像素連通區域收縮框，讓寬度更接近實際字形。

輸出位置：

```text
ctd/line-trans-box/<檔名>.png
ctd/line_trans_map.json
```

## 為什麼需要它

原始 `line-box` 來自 comic-text-detector 的文字行四邊形，能處理斜向文字，也較容易忽略注音、小碎字等非主文字。但它常會保留一些空白，框寬偏大。

單純從 mask 連通區域得到的 component 框更貼近字形，但容易把一行文字拆碎，也可能受大框污染影響。

`line-trans-box` 的設計原則是：

```text
用 line-box 決定「這是一行文字，以及方向」
用乾淨 component 決定「實際文字像素主要在哪裡」
如果 component 不可靠，就回退到穩定的分位數收縮
```

## 處理流程

對每一個 `line-<檔名>.txt` 裡的文字行四邊形，依序執行：

1. 讀取原始 `line-box` 四邊形。
2. 根據四邊形計算文字行方向軸。
3. 若文字行接近水平或垂直，將方向吸附成正框，避免豎排文字被誤視為斜排。
4. 從 `mask_refined` 產生 component 候選框。
5. 過濾明顯污染的大 component。
6. 若有乾淨 component，僅使用該 component 範圍內的 mask 像素沿 line 方向收縮。
7. 若沒有乾淨 component，使用 line polygon 內 mask 像素的 2% / 98% 分位數收縮作為 fallback。
8. 輸出新的四邊形 polygon，以及兼容舊流程的 `x/y/w/h`。

## 大框污染過濾

component 需要通過以下限制才會參與收縮：

```text
component 寬度 <= line 框寬度 * 1.35
component 高度 <= line 框高度 * 1.35
component 面積 <= line 框面積 * 1.8
component 至少 50% 落入 line 框，或至少覆蓋 line 框 8%
```

這可以避免一個跨越多行或多列的大 component 把文字行框撐大。

## Fallback 策略

如果某條 line 找不到乾淨 component，算法不會丟掉這一行，而是回退到分位數收縮：

```text
line polygon 內 mask 像素
-> 投影到 line 方向軸
-> 取 2% / 98% 分位數
-> 轉回圖片座標
```

因此 `line_trans_map.json` 的框數應該和 `line-*.txt` 的文字行數一致。

## JSON 欄位

`line_trans_map.json` 範例：

```json
{
  "x": 154,
  "y": 719,
  "w": 15,
  "h": 142,
  "area": 433,
  "font_size_proxy_px": 15,
  "axis_snapped": true,
  "method": "line_trans_component",
  "matched_component_count": 1,
  "source_line_index": 5,
  "polygon": [[154, 719], [168, 719], [168, 860], [154, 860]]
}
```

欄位說明：

- `x/y/w/h`：輸出 polygon 的水平外接矩形。
- `area`：參與收縮的文字 mask 像素數。
- `font_size_proxy_px`：目前取 `min(w, h)`，直排文字通常可視為字寬估計。
- `axis_snapped`：是否因接近水平/垂直而被吸附成正框。
- `method`：
  - `line_trans_component`：使用乾淨 component 收縮。
  - `line_trans_component_fallback`：component 不可靠，回退到分位數收縮。
- `matched_component_count`：參與收縮的 component 數量。
- `source_line_index`：對應 `line-<檔名>.txt` 中的第幾行。
- `polygon`：真正的方向框。斜排文字會保留四邊形方向。

## 使用建議

如果目標是「翻譯排版前估算文字行寬度」，優先使用 `line_trans_map.json`。

它比原始 `line-box` 更貼近實際字形，又比純 component 框更穩定，不會因 component 拆碎而失去文字行語義。
