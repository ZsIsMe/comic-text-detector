# Solid Inpaint UI 計劃

## 範圍決策

UI 使用 PySide6。

第一版只做「選資料夾、執行、瀏覽、預覽、重新生成」，不做 mask 編輯。

mask 編輯功能放到第二版，避免第一版同時處理偵測流程、圖像顯示、座標映射、undo/redo 和筆刷邏輯，導致驗證成本過高。

## V1 目標

V1 的目標是讓用戶不需要命令行也能完成：

```text
1. 選擇圖片文件夾
2. 執行偵測與生成
3. 查看進度
4. 瀏覽每張圖的結果
5. 調整 mask / 原圖疊加透明度
6. 預覽 inpainted 合成效果
7. 手動重新生成當前頁或全部頁
```

V1 不包含：

```text
1. 矩形框選加減 mask
2. 圓形筆刷加減 mask
3. undo / redo
4. 直接生成 PSD
5. Photoshop Action 調用
```

PSD 仍使用現有配套腳本：

```text
solid_inpaint/create_psds_from_outputs.jsx
```

## V1 入口

建議新增：

```text
solid_inpaint/solid_inpaint_ui.py
```

命令：

```bash
python solid_inpaint/solid_inpaint_ui.py
```

## V1 主界面

界面分為四個區域：

```text
頂部工具列
  選擇文件夾
  執行 / 載入
  重新生成當前頁
  重新生成全部頁
  打開輸出資料夾

左側文件列表
  顯示圖片文件名
  顯示狀態：未處理 / 已生成 / 有 other_mask / 失敗

中間編輯預覽區
  顯示原圖 + mask 疊加
  slider 控制 mask alpha
  原圖透明度使用 1 - alpha
  支持縮放、平移、適應窗口、100%

右側 inpainted 預覽區
  顯示原圖 + inpainted overlay 合成結果
  支持縮放、平移、適應窗口、100%
```

## V1 狀態模型

UI 需要維護：

```text
current_folder
image_list
current_index
current_image
current_mask
current_other_mask
current_inpainted_overlay
current_report_entry
mask_alpha
zoom_state_left
zoom_state_right
```

## V1 背景任務

耗時任務必須在 worker thread 中執行：

```text
1. 首次偵測整個資料夾
2. 重新生成全部頁
```

UI 主線程只做：

```text
1. 更新 progress bar
2. 更新當前文件名
3. 顯示錯誤摘要
4. 任務完成後刷新文件列表與預覽
```

## V1 文件讀寫

如果輸出不存在：

```text
選擇文件夾
-> 點擊執行
-> 跑 detector
-> 生成 mask / other_mask / inpainted / PDF report
```

如果輸出已存在：

```text
選擇文件夾
-> 自動載入 ctd_inpainted
-> 可直接瀏覽
-> 用戶也可以重新執行
```

## V1 重新生成

V1 沒有編輯功能，所以「重新生成」主要用於：

```text
1. 修改算法參數後重算
2. 外部替換 mask 後重算
3. 輸出丟失或不完整時補算
```

第一版可以先提供：

```text
重新生成當前頁
重新生成全部頁
```

## V2 編輯功能

第二版加入 mask 編輯。

工具：

```text
矩形工具
  左鍵拖拽：添加選區
  右鍵拖拽：去掉選區

圓形筆刷
  左鍵拖 / 點：添加 mask
  右鍵拖 / 點：去掉 mask
  [ / ] 調整半徑
```

快捷鍵：

```text
R：矩形工具
B：筆刷工具
[：縮小筆刷
]：放大筆刷
Ctrl+Z：撤銷
Ctrl+Shift+Z：重做
Space + 拖拽：平移
滾輪：縮放
F：適應窗口
1：100%
```

V2 重新生成：

```text
保存當前 mask
-> 用新 mask 重新計算 inpainted
-> 用新 mask 重新計算 other_mask
-> 更新 preview
```

## 技術注意

坐標映射是 V2 的主要風險點。圖像顯示需要可靠轉換：

```text
screen position
-> viewport position
-> image pixel position
```

V1 先做好縮放和平移，V2 再在同一套坐標系上加編輯，風險更低。
