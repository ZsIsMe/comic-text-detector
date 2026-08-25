#!/usr/bin/env python3
"""ctd JSON/NPZ 即時疊圖檢視器。

檢視器會在記憶體中根據原圖、ctd/measure.json、ctd/progressing/*.json
和 align/masks/*.npz 重建只讀疊圖，不會輸出預覽 PNG。
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np

QT_IMPORT_ERROR: ModuleNotFoundError | None = None
QT_WEBENGINE_AVAILABLE = False
QWebEngineView = None
BtMeasurementDialog = None
try:
    from PySide6.QtCore import QEvent, QPointF, QProcess, QRectF, QSize, QSettings, Qt, QTimer, Signal
    from PySide6.QtGui import QAction, QBrush, QColor, QFont, QImage, QKeyEvent, QKeySequence, QPainter, QPen, QPixmap, QTextCursor
    from PySide6.QtWidgets import (
        QApplication,
        QButtonGroup,
        QCheckBox,
        QComboBox,
        QDialog,
        QDialogButtonBox,
        QDockWidget,
        QFileDialog,
        QFrame,
        QGraphicsItem,
        QGraphicsPixmapItem,
        QGraphicsScene,
        QGraphicsView,
        QHBoxLayout,
        QLabel,
        QListWidget,
        QMainWindow,
        QMessageBox,
        QPlainTextEdit,
        QPushButton,
        QRadioButton,
        QScrollArea,
        QSplitter,
        QDoubleSpinBox,
        QSpinBox,
        QSlider,
        QTableWidget,
        QTableWidgetItem,
        QToolBar,
        QVBoxLayout,
        QWidget,
    )
except ModuleNotFoundError as exc:
    QT_IMPORT_ERROR = exc

if QT_IMPORT_ERROR is None:
    try:
        from PySide6.QtWebEngineWidgets import QWebEngineView as _QWebEngineView

        QWebEngineView = _QWebEngineView
        QT_WEBENGINE_AVAILABLE = True
    except (ImportError, ModuleNotFoundError):
        QT_WEBENGINE_AVAILABLE = False

try:
    from .font_size_calibration import DEFAULT_FONT_SIZE_BASE, DEFAULT_FONT_SIZE_STEP
    from .processor import (
        BoxOverlay,
        CtdOverlayProcessor,
        PageOverlay,
        normalized_center_from_xyxy,
        tuple_center,
        xyxy_from_item,
    )
    from .labelplus_pipeline import build_bt_from_labelplus_txt
    from .measure_view import char_box_label
except ImportError:
    from font_size_calibration import DEFAULT_FONT_SIZE_BASE, DEFAULT_FONT_SIZE_STEP
    from processor import (
        BoxOverlay,
        CtdOverlayProcessor,
        PageOverlay,
        normalized_center_from_xyxy,
        tuple_center,
        xyxy_from_item,
    )
    from labelplus_pipeline import build_bt_from_labelplus_txt
    from measure_view import char_box_label


if QT_IMPORT_ERROR is None:
    try:
        from .bt_measurement_dialog import BtMeasurementDialog
    except ImportError:
        from bt_measurement_dialog import BtMeasurementDialog


    def qimage_size(path: Path) -> tuple[int, int] | None:
        image = QImage(str(path))
        if image.isNull():
            return None
        return image.width(), image.height()


    def mask_to_pixmap(mask: np.ndarray, color: QColor, opacity: float) -> QPixmap:
        if mask.ndim == 3:
            mask = mask.max(axis=0)
        mask = np.asarray(mask)
        if mask.dtype != np.uint8:
            mask = np.clip(mask, 0, 255).astype(np.uint8)

        height, width = mask.shape[:2]
        rgba = np.zeros((height, width, 4), dtype=np.uint8)
        rgba[:, :, 0] = color.red()
        rgba[:, :, 1] = color.green()
        rgba[:, :, 2] = color.blue()
        rgba[:, :, 3] = np.clip(mask.astype(np.float32) * opacity, 0, 255).astype(np.uint8)
        image = QImage(rgba.data, width, height, rgba.strides[0], QImage.Format.Format_RGBA8888).copy()
        return QPixmap.fromImage(image)


    def union_mask(masks: np.ndarray | None) -> np.ndarray | None:
        if masks is None or masks.size == 0:
            return None
        arr = np.asarray(masks)
        if arr.ndim == 2:
            return arr.astype(np.uint8)
        if arr.ndim == 3:
            return arr.max(axis=0).astype(np.uint8)
        return None


    try:
        from .viewer_widgets import BtAnnotationItem, ImageView, NavigatorWidget
    except ImportError:
        from viewer_widgets import BtAnnotationItem, ImageView, NavigatorWidget
    try:
        from .viewer_popovers import BtMatchPopover, BtTextEditPopover, HtmlTextOverlay
    except ImportError:
        from viewer_popovers import BtMatchPopover, BtTextEditPopover, HtmlTextOverlay
    try:
        from .viewer_dialogs import (
            ConfirmPreviewDialog,
            FontSizeRegularizeDialog,
            RangeSlider,
            compact_int_px,
            compact_px,
            even_font_size,
            positive_int,
            show_error_details,
            show_exception_details,
        )
    except ImportError:
        from viewer_dialogs import (
            ConfirmPreviewDialog,
            FontSizeRegularizeDialog,
            RangeSlider,
            compact_int_px,
            compact_px,
            even_font_size,
            positive_int,
            show_error_details,
            show_exception_details,
        )

    class CtdOverlayViewer(QMainWindow):
        DEFAULT_LAYER_CHECKED = frozenset(('show_char_boxes', 'show_font_labels'))

        def __init__(self, image_dir: str | None = None) -> None:
            super().__init__()
            self.setWindowTitle('CTD 疊圖檢視器')
            self.resize(1280, 860)
            self.settings = QSettings('comic-text-detector', 'ctd-overlay-processor')
            startup_image_dir = image_dir or self._last_existing_image_dir()
            self.processor: CtdOverlayProcessor | None = None
            self.page: PageOverlay | None = None
            self.page_names: list[str] = []
            self.current_image_dir: str | None = startup_image_dir
            self.bt_data: dict[str, Any] | None = None
            self.bt_path: Path | None = None
            self.bt_dirty = False
            self.selected_bt_index: int | None = None
            self.selected_bt_indices: set[int] = set()
            self.bt_undo_stack: list[dict[str, object]] = []
            self._bt_cached_page_name: str | None = None
            self._bt_cached_source_image: QImage | None = None
            self._bt_cached_base_image: QImage | None = None
            self._bt_cached_base_show_inpainted: bool | None = None
            self._bt_cached_image_size: tuple[int, int] | None = None
            self._bt_displayed_base_key: tuple[str, bool] | None = None
            self._bt_editor_preview_snapshot: dict[str, object] | None = None
            self._bt_editor_preview_status: str | None = None
            self._bt_editor_preview_render_timer = QTimer(self)
            self._bt_editor_preview_render_timer.setSingleShot(True)
            self._bt_editor_preview_render_timer.setInterval(16)
            self._bt_editor_preview_render_timer.timeout.connect(self._render_bt_editor_preview)
            self._bt_editor_preview_commit_timer = QTimer(self)
            self._bt_editor_preview_commit_timer.setSingleShot(True)
            self._bt_editor_preview_commit_timer.setInterval(220)
            self._bt_editor_preview_commit_timer.timeout.connect(self._commit_bt_editor_preview)
            self.detect_process: QProcess | None = None
            self.detect_output_chunks: list[str] = []
            self.detect_command: list[str] = []
            self.font_calibration_process: QProcess | None = None
            self.font_calibration_output_chunks: list[str] = []
            self.font_calibration_command: list[str] = []
            self.generated_default_font_size = DEFAULT_FONT_SIZE_BASE
            self.generated_font_size_step = DEFAULT_FONT_SIZE_STEP
            self.generated_font_size_method = 'ocr_aligned'
            self.generated_calibration_backup: dict[Path, bytes | None] | None = None
            self.hover_char_box: dict | None = None
            self.selected_box_index: int | None = None
            self.undo_stack: list[dict[str, object]] = []
            self.save_action: QAction | None = None
            self.undo_action: QAction | None = None
            self.prev_page_action: QAction | None = None
            self.next_page_action: QAction | None = None
            self.toggle_right_panel_action: QAction | None = None
            self.increase_font_action: QAction | None = None
            self.decrease_font_action: QAction | None = None
            self.increase_font_10_action: QAction | None = None
            self.decrease_font_10_action: QAction | None = None
            self.rotate_counterclockwise_action: QAction | None = None
            self.rotate_clockwise_action: QAction | None = None
            self.rotate_counterclockwise_5_action: QAction | None = None
            self.rotate_clockwise_5_action: QAction | None = None
            self.copy_box_action: QAction | None = None
            self.copy_to_memory_action: QAction | None = None
            self.paste_from_memory_action: QAction | None = None
            self.clear_bt_selection_action: QAction | None = None
            self.delete_box_action: QAction | None = None
            self.measure_angle_action: QAction | None = None
            self.add_empty_box_action: QAction | None = None
            self.move_box_actions: list[QAction] = []
            self.measure_dirty = False
            self.current_page_row = -1
            self._updating_editor = False
            self._syncing_views = False
            self._fitting_views = False
            self._pending_viewport_source: ImageView | None = None
            self._viewport_sync_timer = QTimer(self)
            self._viewport_sync_timer.setSingleShot(True)
            self._viewport_sync_timer.setInterval(16)
            self._viewport_sync_timer.timeout.connect(self._flush_viewport_sync)
            self._popover_bt_item: dict[str, Any] | None = None
            self._box_drag_mode: str | None = None
            self._box_drag_start: tuple[float, float] | None = None
            self._box_drag_original: tuple[int, int, int, int] | None = None
            self._box_drag_temporary = False
            self._bt_drag_mode: str | None = None
            self._bt_drag_start: tuple[float, float] | None = None
            self._bt_drag_original: tuple[int, int, int, int] | None = None
            self._bt_drag_original_item: dict[str, Any] | None = None
            self._bt_drag_original_items: dict[int, dict[str, Any]] | None = None
            self._bt_drag_original_xyxys: dict[int, tuple[int, int, int, int]] = {}
            self._bt_drag_indices: list[int] = []
            self._bt_drag_temporary = False
            self._bt_drag_active = False
            self._bt_drag_render_timer = QTimer(self)
            self._bt_drag_render_timer.setSingleShot(True)
            self._bt_drag_render_timer.setInterval(16)
            self._bt_drag_render_timer.timeout.connect(self._render_bt_drag_preview)
            self._bt_cursor_image_pos: tuple[float, float] | None = None
            self.show_bt_inpainted = True
            self.bt_clipboard_items = self.load_bt_clipboard_items()
            self.active_bt_clipboard_index: int | None = None
            self.memory_bt_clipboard_item: dict[str, Any] | None = None

            self.bt_view = ImageView()
            self.view = ImageView()
            self.bt_annotation_item = BtAnnotationItem(self)
            self.bt_view.scene().addItem(self.bt_annotation_item)
            self.bt_match_popover = BtMatchPopover(self)
            self.bt_text_popover = BtTextEditPopover(self)
            self.bt_html_overlay = HtmlTextOverlay(self.bt_view) if QT_WEBENGINE_AVAILABLE else None
            self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
            self.right_layer_dock: QDockWidget | None = None
            self._last_right_splitter_width = 520
            self.measure_editor_windows: list[QMainWindow] = []

            self.page_list = QListWidget()
            self.bt_item_list = QListWidget()
            self.bt_clipboard_list = QListWidget()
            self.font_size_table = QTableWidget()
            self.status_label = QLabel('尚未選擇資料夾')
            self.status_label.setWordWrap(True)
            self.font_calibration_progress_label = QLabel('')
            self.font_calibration_progress_label.setWordWrap(True)
            self.font_calibration_progress_label.hide()
            self.show_mask = QCheckBox('文字遮罩')
            self.show_block_boxes = QCheckBox('原始區塊框')
            self.show_align_boxes = QCheckBox('重定位框')
            self.show_line_polygons = QCheckBox('文字行多邊形')
            self.show_char_boxes = QCheckBox('單字框')
            self.show_npz_smoothed = QCheckBox('NPZ 平滑遮罩')
            self.show_npz_outer = QCheckBox('NPZ 外輪廓遮罩')
            self.show_font_labels = QCheckBox('字級標籤')
            self.show_bt_font_labels_check = QCheckBox('_bt 字級標籤')
            self.show_bt_font_labels_check.setToolTip('顯示/隱藏左側 _bt 文字框旁的字級標籤')
            self.show_popover_check = QCheckBox('預覽浮窗')
            self.show_popover_check.setToolTip('顯示/隱藏 _bt 匹配的局部預覽浮窗 (Command+P)')
            self.navigator = NavigatorWidget()
            self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
            self.generate_button = QPushButton('生成/更新 CTD')
            self.edit_measure_button = QPushButton('編輯 measure.json')
            self.font_size_method_group = QButtonGroup(self)
            self.ocr_aligned_font_radio = QRadioButton('OCR 對齊逐字計算')
            self.char_box_font_radio = QRadioButton('單字框計算')
            self.font_size_method_group.addButton(self.ocr_aligned_font_radio)
            self.font_size_method_group.addButton(self.char_box_font_radio)
            self.ocr_aligned_font_radio.setChecked(True)
            self.auto_calibrate_device_combo = QComboBox()
            self.auto_calibrate_device_combo.addItem('字級 OCR：CPU', 'cpu')
            self.auto_calibrate_device_combo.addItem('字級 OCR：MPS（可能回退 CPU）', 'mps')
            self.auto_calibrate_device_combo.addItem('字級 OCR：CUDA', 'cuda')
            self.import_labelplus_button = QPushButton('導入 LabelPlus txt')
            self.open_bt_button = QPushButton('打開 _bt.json')
            self.save_button = QPushButton('保存 _bt.json')
            self.even_font_button = QPushButton('字體取偶數')
            self.regularize_font_button = QPushButton('規整字體大小')
            self.copy_bt_button = QPushButton('複製文字框')
            self.copy_bt_button.setToolTip('複製當前文字框（Command/Ctrl+D）')
            self.copy_to_clipboard_button = QPushButton('加入持久化列表')
            self.copy_to_clipboard_button.setToolTip('將目前選取文字框加入跨專案保留的剪貼簿列表')
            self.delete_clipboard_button = QPushButton('從持久化列表刪除')
            self.measure_preview_button = QPushButton('測量角度（⌘/Ctrl+M）')
            self.measure_preview_button.setToolTip('打開測量角度（Command/Ctrl+M）')
            self.add_empty_bt_button = QPushButton('新建文本框')
            self.add_empty_bt_button.setToolTip('在鼠標位置新增預設空文案（Command/Ctrl+N）')
            self.add_empty_bt_button.setEnabled(False)
            self.char_info_label = QLabel('游標單字框：未選中')
            self.char_info_label.setWordWrap(True)
            self.box_editor_title = QLabel('未選擇文字框')
            self.box_editor_title.setWordWrap(True)
            self.bt_path_label = QLabel('尚未載入 _bt.json')
            self.bt_path_label.setWordWrap(True)
            self.bt_stats_label = QLabel('_bt 統計：未載入')
            self.bt_stats_label.setWordWrap(True)
            self.bt_text_edit = QPlainTextEdit()
            self.bt_text_edit.setPlaceholderText('選中左側 _bt 條目後編輯文字')
            self.bt_text_edit.setMinimumHeight(90)
            self.font_size_spin = QSpinBox()
            self.orientation_button_group = QButtonGroup(self)
            self.orientation_vertical_button = QPushButton('直排')
            self.orientation_horizontal_button = QPushButton('橫排')
            self.orientation_button_group.addButton(self.orientation_vertical_button)
            self.orientation_button_group.addButton(self.orientation_horizontal_button)
            self.rotation_spin = QDoubleSpinBox()
            self.color_button_group = QButtonGroup(self)
            self.color_black_button = QPushButton('黑')
            self.color_white_button = QPushButton('白')
            self.color_button_group.addButton(self.color_black_button)
            self.color_button_group.addButton(self.color_white_button)
            for button in (
                self.orientation_vertical_button,
                self.orientation_horizontal_button,
                self.color_black_button,
                self.color_white_button,
            ):
                button.setCheckable(True)
            self.stroke_weight_spin = QSpinBox()
            self.text_has_stroke_check = QCheckBox('原字描邊')
            self.need_inpaint_check = QCheckBox('需要修復/描邊')

            self._restore_layer_checkbox_states()
            self.show_popover_check.setChecked(self._settings_bool('ui/show_match_popover', True))
            self.show_bt_font_labels_check.setChecked(self._settings_bool('ui/show_bt_font_labels', True))
            self.opacity_slider.setRange(5, 90)
            self.opacity_slider.setValue(35)

            self._build_toolbar()
            self._build_side_panel()
            center_tools = self._build_center_tools_panel()
            self._build_layer_panel()
            self.main_splitter.addWidget(self.bt_view)
            self.main_splitter.addWidget(center_tools)
            self.main_splitter.addWidget(self.view)
            self.main_splitter.setStretchFactor(0, 3)
            self.main_splitter.setStretchFactor(1, 0)
            self.main_splitter.setStretchFactor(2, 3)
            self.main_splitter.setSizes([520, 300, 520])
            self.setCentralWidget(self.main_splitter)
            self.restore_right_panel_state()
            self._connect_signals()

            if startup_image_dir:
                self.load_folder(startup_image_dir)

        def keyPressEvent(self, event: QKeyEvent) -> None:
            if (
                event.key() == Qt.Key.Key_Q
                and not event.modifiers()
                and QApplication.focusWidget() is not self.bt_text_edit
            ):
                self.show_bt_inpainted = not self.show_bt_inpainted
                self.render_bt_page(refit=False)
                state = '顯示' if self.show_bt_inpainted else '隱藏'
                self.status_label.setText(f'左側 inpainted 已{state}。')
                event.accept()
                return
            super().keyPressEvent(event)

        def _clear_bt_image_cache(self) -> None:
            self._bt_cached_page_name = None
            self._bt_cached_source_image = None
            self._bt_cached_base_image = None
            self._bt_cached_base_show_inpainted = None
            self._bt_cached_image_size = None
            self._bt_displayed_base_key = None

        def show_shortcuts_dialog(self) -> None:
            shortcuts_text = '''_bt.json 編輯器快捷鍵

一般
  Command/Ctrl + S              保存 _bt.json
  Command/Ctrl + Z              撤銷上一個修改
  Page Up / Page Down           上一頁 / 下一頁
  Q                             顯示 / 隱藏左側 inpainted 圖像
  Command/Ctrl + P              顯示 / 隱藏預覽浮窗
  Esc                           取消文字框選取與編輯焦點

文字框
  Command/Ctrl + D              複製目前文字框，在同頁新增一份
  F1                            暫存複製目前選取文字框
  F2                            在左側圖片游標位置貼上暫存文字框
  Command/Ctrl + N              在游標位置新增文案（優先帶入系統剪貼簿文字）
  Delete / Backspace            刪除選取文字框
  Command/Ctrl + M              開啟測量角度

位置與樣式（需選取文字框）
  方向鍵                         移動 1px
  Shift + 方向鍵                 移動 10px
  Command/Ctrl + Shift + 方向鍵  移動 50px
  Command/Ctrl + + 或 =          字體 +2
  Command/Ctrl + -               字體 -2
  Command/Ctrl + Option/Alt + + 或 =  字體 +10
  Command/Ctrl + Option/Alt + -       字體 -10
  Command/Ctrl + [ / ]           逆時針 / 順時針旋轉 1°
  Command/Ctrl + Option/Alt + [ / ]  逆時針 / 順時針旋轉 5°

持久剪貼簿列表
  「加入持久剪貼簿列表」按鈕      將目前選取框加入跨專案保存的列表
  單擊列表項目                   暫存複製該項，再以 F2 貼上
'''
            dialog = QDialog(self)
            dialog.setWindowTitle('快捷鍵說明')
            dialog.setMinimumSize(620, 560)
            layout = QVBoxLayout(dialog)
            text_edit = QPlainTextEdit(shortcuts_text)
            text_edit.setReadOnly(True)
            text_edit.setFont(QFont('Menlo', 13))
            layout.addWidget(text_edit)
            close_button = QPushButton('關閉')
            close_button.clicked.connect(dialog.accept)
            layout.addWidget(close_button)
            dialog.exec()

        def _last_existing_image_dir(self) -> str | None:
            value = self.settings.value('last_image_dir', '', str)
            if not value:
                return None
            path = Path(value).expanduser()
            if not path.is_dir():
                return None
            return str(path.resolve())

        def _save_last_image_dir(self, image_dir: str) -> None:
            path = Path(image_dir).expanduser()
            if path.is_dir():
                self.settings.setValue('last_image_dir', str(path.resolve()))

        def _bt_mapping_key(self, image_dir: str | None = None) -> str | None:
            image_dir = image_dir or self.current_image_dir
            if not image_dir:
                return None
            path = Path(image_dir).expanduser()
            try:
                resolved = path.resolve()
            except OSError:
                return None
            digest = hashlib.sha1(str(resolved).encode('utf-8')).hexdigest()
            return f'bt_json_for_folder/{digest}'

        def _save_bt_mapping(self, bt_path: Path, image_dir: str | None = None) -> None:
            key = self._bt_mapping_key(image_dir)
            if key is None:
                return
            self.settings.setValue(key, str(bt_path.expanduser().resolve()))

        def _mapped_bt_path(self, image_dir: str | None = None) -> Path | None:
            key = self._bt_mapping_key(image_dir)
            if key is None:
                return None
            value = self.settings.value(key, '', str)
            if not value:
                return None
            path = Path(value).expanduser()
            if path.is_file():
                return path.resolve()
            self.settings.remove(key)
            return None

        def _autoload_mapped_bt_json(self) -> None:
            path = self._mapped_bt_path()
            if path is None:
                return
            try:
                self.load_bt_json_path(path, remember=False)
                self.status_label.setText(f'{self.status_label.text()}\n已自動載入：{path.name}')
            except Exception as exc:
                key = self._bt_mapping_key()
                if key is not None:
                    self.settings.remove(key)
                show_exception_details(
                    self,
                    '自動打開 _bt.json 失敗',
                    f'已找到此資料夾記錄的 _bt.json，但無法打開：\n{path}',
                    exc,
                )

        def _build_toolbar(self) -> None:
            toolbar = QToolBar('主工具列')
            toolbar.setMovable(False)
            self.addToolBar(toolbar)

            open_action = QAction('選擇資料夾', self)
            open_action.triggered.connect(self.choose_folder)
            toolbar.addAction(open_action)

            generate_action = QAction('生成/更新 CTD', self)
            generate_action.triggered.connect(self.generate_ctd)
            toolbar.addAction(generate_action)

            import_labelplus_action = QAction('導入 LabelPlus txt', self)
            import_labelplus_action.triggered.connect(self.import_labelplus_txt)
            toolbar.addAction(import_labelplus_action)

            open_bt_action = QAction('打開 _bt.json', self)
            open_bt_action.triggered.connect(self.open_bt_json)
            toolbar.addAction(open_bt_action)

            shortcuts_action = QAction('快捷鍵說明', self)
            shortcuts_action.setToolTip('查看 _bt.json 編輯器的全部快捷鍵')
            shortcuts_action.triggered.connect(self.show_shortcuts_dialog)
            toolbar.addAction(shortcuts_action)

            fit_action = QAction('適合視窗', self)
            fit_action.triggered.connect(self.fit_both_views)
            toolbar.addAction(fit_action)

            self.toggle_right_panel_action = QAction('收起右側', self)
            self.toggle_right_panel_action.setCheckable(True)
            self.toggle_right_panel_action.setToolTip('展開/收起右邊圖片和最右功能區')
            self.toggle_right_panel_action.triggered.connect(self.toggle_right_panel)
            toolbar.addAction(self.toggle_right_panel_action)

            toolbar.addWidget(self.show_popover_check)
            toolbar.addWidget(self.show_bt_font_labels_check)
            toggle_popover_action = QAction('切換預覽浮窗', self)
            toggle_popover_action.setShortcuts([QKeySequence('Meta+P'), QKeySequence('Ctrl+P')])
            toggle_popover_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
            toggle_popover_action.triggered.connect(self.toggle_match_popover_enabled)
            self.addAction(toggle_popover_action)

            self.prev_page_action = QAction('上一頁', self)
            self.prev_page_action.setShortcut(QKeySequence(Qt.Key.Key_PageUp))
            self.prev_page_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
            self.prev_page_action.triggered.connect(lambda: self.go_to_relative_page(-1))
            toolbar.addAction(self.prev_page_action)

            self.next_page_action = QAction('下一頁', self)
            self.next_page_action.setShortcut(QKeySequence(Qt.Key.Key_PageDown))
            self.next_page_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
            self.next_page_action.triggered.connect(lambda: self.go_to_relative_page(1))
            toolbar.addAction(self.next_page_action)

            self.save_action = QAction('保存 _bt', self)
            self.save_action.setShortcuts([QKeySequence('Meta+S'), QKeySequence('Ctrl+S')])
            self.save_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
            self.save_action.triggered.connect(self.save_pending_changes)
            self.save_action.setEnabled(False)
            toolbar.addAction(self.save_action)

            self.undo_action = QAction('撤銷', self)
            self.undo_action.setShortcuts([QKeySequence('Meta+Z'), QKeySequence('Ctrl+Z')])
            self.undo_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
            self.undo_action.triggered.connect(self.undo_last_edit)
            self.undo_action.setEnabled(False)
            toolbar.addAction(self.undo_action)

            self.increase_font_action = QAction('字體+2', self)
            self.increase_font_action.setShortcuts([
                QKeySequence('Meta++'),
                QKeySequence('Meta+='),
                QKeySequence('Ctrl++'),
                QKeySequence('Ctrl+='),
            ])
            self.increase_font_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
            self.increase_font_action.triggered.connect(lambda: self.nudge_selected_font_size(2))
            self.increase_font_action.setEnabled(False)
            self.addAction(self.increase_font_action)

            self.decrease_font_action = QAction('字體-2', self)
            self.decrease_font_action.setShortcuts([QKeySequence('Meta+-'), QKeySequence('Ctrl+-')])
            self.decrease_font_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
            self.decrease_font_action.triggered.connect(lambda: self.nudge_selected_font_size(-2))
            self.decrease_font_action.setEnabled(False)
            self.addAction(self.decrease_font_action)

            self.increase_font_10_action = QAction('字體+10', self)
            self.increase_font_10_action.setShortcuts([
                QKeySequence('Meta+Alt++'),
                QKeySequence('Meta+Alt+='),
                QKeySequence('Ctrl+Alt++'),
                QKeySequence('Ctrl+Alt+='),
            ])
            self.increase_font_10_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
            self.increase_font_10_action.triggered.connect(lambda: self.nudge_selected_font_size(10))
            self.increase_font_10_action.setEnabled(False)
            self.addAction(self.increase_font_10_action)

            self.decrease_font_10_action = QAction('字體-10', self)
            self.decrease_font_10_action.setShortcuts([
                QKeySequence('Meta+Alt+-'),
                QKeySequence('Ctrl+Alt+-'),
            ])
            self.decrease_font_10_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
            self.decrease_font_10_action.triggered.connect(lambda: self.nudge_selected_font_size(-10))
            self.decrease_font_10_action.setEnabled(False)
            self.addAction(self.decrease_font_10_action)

            self.rotate_counterclockwise_action = QAction('逆時針+1', self)
            self.rotate_counterclockwise_action.setShortcuts([QKeySequence('Meta+['), QKeySequence('Ctrl+[')])
            self.rotate_counterclockwise_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
            self.rotate_counterclockwise_action.triggered.connect(lambda: self.nudge_selected_rotation(1.0))
            self.rotate_counterclockwise_action.setEnabled(False)
            self.addAction(self.rotate_counterclockwise_action)

            self.rotate_clockwise_action = QAction('順時針-1', self)
            self.rotate_clockwise_action.setShortcuts([QKeySequence('Meta+]'), QKeySequence('Ctrl+]')])
            self.rotate_clockwise_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
            self.rotate_clockwise_action.triggered.connect(lambda: self.nudge_selected_rotation(-1.0))
            self.rotate_clockwise_action.setEnabled(False)
            self.addAction(self.rotate_clockwise_action)

            self.rotate_counterclockwise_5_action = QAction('逆時針+5', self)
            self.rotate_counterclockwise_5_action.setShortcuts([
                QKeySequence('Meta+Alt+['),
                QKeySequence('Ctrl+Alt+['),
            ])
            self.rotate_counterclockwise_5_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
            self.rotate_counterclockwise_5_action.triggered.connect(lambda: self.nudge_selected_rotation(5.0))
            self.rotate_counterclockwise_5_action.setEnabled(False)
            self.addAction(self.rotate_counterclockwise_5_action)

            self.rotate_clockwise_5_action = QAction('順時針-5', self)
            self.rotate_clockwise_5_action.setShortcuts([
                QKeySequence('Meta+Alt+]'),
                QKeySequence('Ctrl+Alt+]'),
            ])
            self.rotate_clockwise_5_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
            self.rotate_clockwise_5_action.triggered.connect(lambda: self.nudge_selected_rotation(-5.0))
            self.rotate_clockwise_5_action.setEnabled(False)
            self.addAction(self.rotate_clockwise_5_action)

            self.copy_box_action = QAction('複製文字框', self)
            self.copy_box_action.setShortcuts([QKeySequence('Meta+D'), QKeySequence('Ctrl+D')])
            self.copy_box_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
            self.copy_box_action.setToolTip('複製當前文字框（Command/Ctrl+D）')
            self.copy_box_action.triggered.connect(self.copy_selected_box)
            self.copy_box_action.setEnabled(False)
            self.addAction(self.copy_box_action)

            self.copy_to_memory_action = QAction('暫存複製文字框', self)
            self.copy_to_memory_action.setShortcut(QKeySequence('F1'))
            self.copy_to_memory_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
            self.copy_to_memory_action.setToolTip('暫存複製目前文字框（F1）')
            self.copy_to_memory_action.triggered.connect(self.copy_selected_box_to_memory)
            self.copy_to_memory_action.setEnabled(False)
            self.addAction(self.copy_to_memory_action)

            self.paste_from_memory_action = QAction('在游標位置暫存貼上文字框', self)
            self.paste_from_memory_action.setShortcut(QKeySequence('F2'))
            self.paste_from_memory_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
            self.paste_from_memory_action.setToolTip('在游標位置貼上暫存文字框（F2）')
            self.paste_from_memory_action.triggered.connect(self.paste_box_from_memory)
            self.paste_from_memory_action.setEnabled(False)
            self.addAction(self.paste_from_memory_action)

            self.clear_bt_selection_action = QAction('取消選取文字框', self)
            self.clear_bt_selection_action.setShortcut(QKeySequence(Qt.Key.Key_Escape))
            self.clear_bt_selection_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
            self.clear_bt_selection_action.setToolTip('取消目前文字框的選取與編輯焦點（Esc）')
            self.clear_bt_selection_action.triggered.connect(self.clear_current_bt_focus)
            self.addAction(self.clear_bt_selection_action)

            self.measure_angle_action = QAction('測量角度', self)
            self.measure_angle_action.setShortcuts([QKeySequence('Meta+M'), QKeySequence('Ctrl+M')])
            self.measure_angle_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
            self.measure_angle_action.setToolTip('打開測量角度（Command/Ctrl+M）')
            self.measure_angle_action.triggered.connect(self.open_bt_measurement_dialog)
            self.measure_angle_action.setEnabled(False)
            self.addAction(self.measure_angle_action)

            self.add_empty_box_action = QAction('新增空文案', self)
            self.add_empty_box_action.setShortcuts([QKeySequence('Meta+N'), QKeySequence('Ctrl+N')])
            self.add_empty_box_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
            self.add_empty_box_action.setToolTip('在鼠標位置新增預設空文案（Command/Ctrl+N）')
            self.add_empty_box_action.triggered.connect(self.add_empty_bt_box)
            self.add_empty_box_action.setEnabled(False)
            self.addAction(self.add_empty_box_action)

            self.delete_box_action = QAction('刪除文字框', self)
            self.delete_box_action.setShortcuts([QKeySequence(Qt.Key.Key_Delete), QKeySequence(Qt.Key.Key_Backspace)])
            self.delete_box_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
            self.delete_box_action.triggered.connect(self.delete_selected_box)
            self.delete_box_action.setEnabled(False)
            self.addAction(self.delete_box_action)

            move_shortcuts = (
                ('左移', Qt.Key.Key_Left, -1, 0, 1),
                ('右移', Qt.Key.Key_Right, 1, 0, 1),
                ('上移', Qt.Key.Key_Up, 0, -1, 1),
                ('下移', Qt.Key.Key_Down, 0, 1, 1),
                ('左移10', Qt.KeyboardModifier.ShiftModifier | Qt.Key.Key_Left, -1, 0, 10),
                ('右移10', Qt.KeyboardModifier.ShiftModifier | Qt.Key.Key_Right, 1, 0, 10),
                ('上移10', Qt.KeyboardModifier.ShiftModifier | Qt.Key.Key_Up, 0, -1, 10),
                ('下移10', Qt.KeyboardModifier.ShiftModifier | Qt.Key.Key_Down, 0, 1, 10),
                ('左移50', Qt.KeyboardModifier.ShiftModifier | Qt.KeyboardModifier.ControlModifier | Qt.Key.Key_Left, -1, 0, 50),
                ('右移50', Qt.KeyboardModifier.ShiftModifier | Qt.KeyboardModifier.ControlModifier | Qt.Key.Key_Right, 1, 0, 50),
                ('上移50', Qt.KeyboardModifier.ShiftModifier | Qt.KeyboardModifier.ControlModifier | Qt.Key.Key_Up, 0, -1, 50),
                ('下移50', Qt.KeyboardModifier.ShiftModifier | Qt.KeyboardModifier.ControlModifier | Qt.Key.Key_Down, 0, 1, 50),
                ('左移50（Mac）', Qt.KeyboardModifier.ShiftModifier | Qt.KeyboardModifier.MetaModifier | Qt.Key.Key_Left, -1, 0, 50),
                ('右移50（Mac）', Qt.KeyboardModifier.ShiftModifier | Qt.KeyboardModifier.MetaModifier | Qt.Key.Key_Right, 1, 0, 50),
                ('上移50（Mac）', Qt.KeyboardModifier.ShiftModifier | Qt.KeyboardModifier.MetaModifier | Qt.Key.Key_Up, 0, -1, 50),
                ('下移50（Mac）', Qt.KeyboardModifier.ShiftModifier | Qt.KeyboardModifier.MetaModifier | Qt.Key.Key_Down, 0, 1, 50),
            )
            for label, key, dx, dy, step in move_shortcuts:
                action = QAction(label, self)
                action.setShortcut(QKeySequence(key))
                action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
                action.triggered.connect(lambda checked=False, dx=dx, dy=dy, step=step: self.nudge_selected_box_position(dx * step, dy * step))
                self.move_box_actions.append(action)
                self.addAction(action)

        def _build_side_panel(self) -> None:
            panel = QWidget()
            layout = QVBoxLayout(panel)
            layout.setContentsMargins(10, 10, 10, 10)
            layout.setSpacing(8)

            layout.addWidget(QLabel('資料狀態'))
            layout.addWidget(self.status_label)
            layout.addWidget(QLabel('頁面'))
            self.page_list.setMinimumHeight(150)
            layout.addWidget(self.page_list, 1)
            layout.addWidget(QLabel('_bt.json'))
            layout.addWidget(self.bt_path_label)
            self.open_bt_button.clicked.connect(self.open_bt_json)
            layout.addWidget(self.open_bt_button)
            layout.addWidget(self.bt_stats_label)
            layout.addWidget(QLabel('當前頁 _bt 條目'))
            self.bt_item_list.setMinimumHeight(260)
            self.bt_item_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
            layout.addWidget(self.bt_item_list, 2)
            layout.addWidget(QLabel('游標資訊'))
            layout.addWidget(self.char_info_label)
            layout.addWidget(QLabel('導航器'))
            layout.addWidget(self.navigator)

            dock = QDockWidget('頁面 / _bt 條目', self)
            dock.setWidget(panel)
            dock.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
            self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)

        def _build_center_tools_panel(self) -> QWidget:
            panel = QWidget()
            panel.setMinimumWidth(280)
            layout = QVBoxLayout(panel)
            layout.setContentsMargins(10, 10, 10, 10)
            layout.setSpacing(8)

            title = QLabel('字級快捷列表')
            title.setToolTip('點擊字級即可套用到目前選中的 _bt 條目。')
            list_font = QFont('Menlo', 13)
            list_font.setBold(True)
            list_font.setStyleHint(QFont.StyleHint.Monospace)
            self.font_size_table.setFont(list_font)
            self.font_size_table.setColumnCount(2)
            self.font_size_table.setHorizontalHeaderLabels(['字級', '數目'])
            self.font_size_table.verticalHeader().setVisible(False)
            self.font_size_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
            self.font_size_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
            self.font_size_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
            self.font_size_table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            self.font_size_table.setAlternatingRowColors(True)
            self.font_size_table.setShowGrid(False)
            self.font_size_table.horizontalHeader().setStretchLastSection(True)
            self.font_size_table.horizontalHeader().setDefaultAlignment(Qt.AlignmentFlag.AlignCenter)
            self.font_size_table.setColumnWidth(0, 86)
            self.font_size_table.setMinimumHeight(340)
            self.font_size_table.setStyleSheet(
                'QTableWidget {'
                '  background: #101214;'
                '  alternate-background-color: #181a1d;'
                '  color: #e9ecef;'
                '  border: 1px solid #33383d;'
                '}'
                'QHeaderView::section {'
                '  background: #2a2e33;'
                '  color: #f4f6f8;'
                '  border: 0;'
                '  border-bottom: 1px solid #454b52;'
                '  padding: 6px 4px;'
                '  font-weight: 700;'
                '}'
                'QTableWidget::item {'
                '  padding: 5px 8px;'
                '}'
            )
            self.even_font_button.clicked.connect(self.preview_even_font_sizes)
            self.even_font_button.hide()
            self.regularize_font_button.clicked.connect(self.open_regularize_font_size_dialog)

            self.font_size_spin.setRange(1, 999)
            self.font_size_spin.setSuffix(' px')
            self.orientation_vertical_button.setChecked(True)
            self.rotation_spin.setRange(-180.0, 180.0)
            self.rotation_spin.setDecimals(2)
            self.rotation_spin.setSingleStep(1.0)
            self.rotation_spin.setSuffix(' deg')
            self.color_black_button.setChecked(True)
            self.stroke_weight_spin.setRange(0, 99)
            self.stroke_weight_spin.setSuffix(' px')

            choice_style = (
                'QPushButton { padding: 4px 8px; min-height: 24px; }'
                'QPushButton:checked { background: #4a5057; border: 1px solid #7c8791; }'
            )
            for button in (
                self.orientation_vertical_button,
                self.orientation_horizontal_button,
                self.color_black_button,
                self.color_white_button,
            ):
                button.setStyleSheet(choice_style)

            typography_row = QWidget()
            typography_layout = QHBoxLayout(typography_row)
            typography_layout.setContentsMargins(0, 0, 0, 0)
            typography_layout.setSpacing(6)
            typography_layout.addWidget(QLabel('字級'))
            typography_layout.addWidget(self.font_size_spin, 1)
            typography_layout.addWidget(QLabel('方向'))
            typography_layout.addWidget(self.orientation_horizontal_button)
            typography_layout.addWidget(self.orientation_vertical_button)

            rotation_row = QWidget()
            rotation_layout = QHBoxLayout(rotation_row)
            rotation_layout.setContentsMargins(0, 0, 0, 0)
            rotation_layout.setSpacing(6)
            rotation_layout.addWidget(QLabel('角度'))
            rotation_layout.addWidget(self.rotation_spin, 1)

            appearance_row = QWidget()
            appearance_layout = QHBoxLayout(appearance_row)
            appearance_layout.setContentsMargins(0, 0, 0, 0)
            appearance_layout.setSpacing(6)
            appearance_layout.addWidget(QLabel('顏色'))
            appearance_layout.addWidget(self.color_black_button)
            appearance_layout.addWidget(self.color_white_button)
            appearance_layout.addWidget(QLabel('描邊'))
            appearance_layout.addWidget(self.stroke_weight_spin, 1)

            appearance_actions_row = QWidget()
            appearance_actions_layout = QHBoxLayout(appearance_actions_row)
            appearance_actions_layout.setContentsMargins(0, 0, 0, 0)
            appearance_actions_layout.setSpacing(8)
            appearance_actions_layout.addWidget(self.text_has_stroke_check)
            appearance_actions_layout.addWidget(self.need_inpaint_check)
            appearance_actions_layout.addStretch(1)
            appearance_actions_layout.addWidget(self.save_button)

            box_actions_row = QWidget()
            box_actions_layout = QHBoxLayout(box_actions_row)
            box_actions_layout.setContentsMargins(0, 0, 0, 0)
            box_actions_layout.setSpacing(8)
            box_actions_layout.addWidget(self.copy_bt_button, 1)
            box_actions_layout.addWidget(self.add_empty_bt_button, 1)

            clipboard_actions_row = QWidget()
            clipboard_actions_layout = QHBoxLayout(clipboard_actions_row)
            clipboard_actions_layout.setContentsMargins(0, 0, 0, 0)
            clipboard_actions_layout.setSpacing(8)
            clipboard_actions_layout.addWidget(self.copy_to_clipboard_button, 1)
            clipboard_actions_layout.addWidget(self.delete_clipboard_button, 1)

            layout.addWidget(QLabel('文字'))
            layout.addWidget(self.bt_text_edit)
            layout.addWidget(typography_row)
            layout.addWidget(rotation_row)
            layout.addWidget(box_actions_row)
            layout.addWidget(self.measure_preview_button)
            self.save_button.clicked.connect(self.save_pending_changes)
            layout.addWidget(appearance_row)
            layout.addWidget(appearance_actions_row)
            layout.addWidget(title)
            layout.addWidget(QLabel('文字框剪貼簿（跨專案保留）'))
            self.bt_clipboard_list.setMinimumHeight(130)
            self.bt_clipboard_list.setToolTip('單擊一項即可暫存複製它；可用 F2 貼上。')
            layout.addWidget(self.bt_clipboard_list)
            layout.addWidget(clipboard_actions_row)
            layout.addWidget(self.regularize_font_button)
            layout.addWidget(self.font_size_table, 1)
            layout.addWidget(self.even_font_button)
            self.set_box_editor_enabled(False)
            self.update_bt_clipboard_list()

            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(panel)
            scroll.setMinimumWidth(280)
            return scroll

        def _build_layer_panel(self) -> None:
            panel = QWidget()
            panel.setMinimumWidth(220)
            layout = QVBoxLayout(panel)
            layout.setContentsMargins(10, 10, 10, 10)
            layout.setSpacing(8)

            layout.addWidget(QLabel('圖層'))
            for widget in (
                self.show_mask,
                self.show_npz_smoothed,
                self.show_npz_outer,
                self.show_block_boxes,
                self.show_align_boxes,
                self.show_line_polygons,
                self.show_char_boxes,
                self.show_font_labels,
            ):
                layout.addWidget(widget)

            opacity_row = QHBoxLayout()
            opacity_row.addWidget(QLabel('疊圖透明度'))
            opacity_row.addWidget(self.opacity_slider)
            layout.addLayout(opacity_row)

            layout.addWidget(QLabel('流程'))
            reload_button = QPushButton('重新載入目前頁')
            reload_button.clicked.connect(self.reload_current_page)
            layout.addWidget(reload_button)
            layout.addWidget(QLabel('字級計算方法'))
            self.ocr_aligned_font_radio.setToolTip(
                '使用 mit48 整行 OCR 對齊可靠文字，再依字型墨跡緩存逐字計算字級。'
            )
            self.char_box_font_radio.setToolTip(
                '使用原有遮罩切分單字框的字級計算方法，不執行 mit48 OCR。'
            )
            self.ocr_aligned_font_radio.toggled.connect(self.auto_calibrate_device_combo.setEnabled)
            layout.addWidget(self.ocr_aligned_font_radio)
            layout.addWidget(self.char_box_font_radio)
            layout.addWidget(self.auto_calibrate_device_combo)
            self.generate_button.clicked.connect(self.generate_ctd)
            layout.addWidget(self.generate_button)
            layout.addWidget(self.font_calibration_progress_label)
            self.edit_measure_button.clicked.connect(self.open_measure_editor)
            layout.addWidget(self.edit_measure_button)
            self.import_labelplus_button.clicked.connect(self.import_labelplus_txt)
            layout.addWidget(self.import_labelplus_button)
            layout.addStretch(1)

            dock = QDockWidget('圖層 / 流程', self)
            dock.setWidget(panel)
            dock.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
            self.right_layer_dock = dock
            self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

        def _connect_signals(self) -> None:
            self.page_list.currentRowChanged.connect(self.handle_page_row_changed)
            self.bt_item_list.currentRowChanged.connect(self.handle_bt_item_row_changed)
            self.bt_item_list.itemSelectionChanged.connect(self.handle_bt_item_selection_changed)
            for widget in (
                self.show_mask,
                self.show_npz_smoothed,
                self.show_npz_outer,
                self.show_block_boxes,
                self.show_align_boxes,
                self.show_line_polygons,
                self.show_char_boxes,
                self.show_font_labels,
            ):
                widget.stateChanged.connect(self.handle_layer_checkbox_changed)
            self.opacity_slider.valueChanged.connect(self.render_current_page)
            self.view.imageMouseMoved.connect(self.update_hover_char_box)
            self.view.imageMouseLeft.connect(self.clear_hover_char_box)
            self.view.imageMousePressed.connect(self.handle_image_mouse_press)
            self.bt_view.imageMousePressed.connect(self.handle_bt_mouse_press)
            self.bt_view.imageMouseMoved.connect(self.update_bt_view_cursor)
            self.bt_view.imageMouseLeft.connect(self.clear_bt_view_cursor)
            self.bt_view.imageMouseDragged.connect(self.handle_bt_mouse_drag)
            self.bt_view.imageMouseReleased.connect(self.handle_bt_mouse_release)
            self.bt_view.fontSizeWheelRequested.connect(self.nudge_selected_font_size)
            self.view.fontSizeWheelRequested.connect(self.nudge_selected_font_size)
            self.bt_view.viewportChanged.connect(self.sync_viewports_from)
            self.view.viewportChanged.connect(self.sync_viewports_from)
            self.navigator.navigateRequested.connect(self.center_views_on)
            self.show_popover_check.stateChanged.connect(self.handle_match_popover_setting_changed)
            self.show_bt_font_labels_check.stateChanged.connect(self.handle_bt_font_label_setting_changed)
            self.bt_text_edit.textChanged.connect(self.apply_editor_changes_to_selected_box)
            self.bt_text_popover.text_edit.textChanged.connect(self.apply_bt_text_popover_changes)
            self.font_size_spin.valueChanged.connect(self.handle_typography_control_changed)
            self.orientation_vertical_button.clicked.connect(self.apply_editor_changes_to_selected_box)
            self.orientation_horizontal_button.clicked.connect(self.apply_editor_changes_to_selected_box)
            self.rotation_spin.valueChanged.connect(self.handle_typography_control_changed)
            self.font_size_spin.editingFinished.connect(self._commit_bt_editor_preview)
            self.rotation_spin.editingFinished.connect(self._commit_bt_editor_preview)
            self.copy_bt_button.clicked.connect(self.copy_selected_box)
            self.copy_to_clipboard_button.clicked.connect(self.copy_selected_box_to_clipboard)
            self.delete_clipboard_button.clicked.connect(self.delete_selected_clipboard_item)
            self.bt_clipboard_list.itemClicked.connect(self.copy_bt_clipboard_list_item)
            self.measure_preview_button.clicked.connect(self.open_bt_measurement_dialog)
            self.add_empty_bt_button.clicked.connect(self.add_empty_bt_box)
            self.color_black_button.clicked.connect(self.apply_editor_changes_to_selected_box)
            self.color_white_button.clicked.connect(self.apply_editor_changes_to_selected_box)
            self.stroke_weight_spin.valueChanged.connect(self.apply_editor_changes_to_selected_box)
            self.text_has_stroke_check.stateChanged.connect(self.apply_editor_changes_to_selected_box)
            self.need_inpaint_check.stateChanged.connect(self.apply_editor_changes_to_selected_box)
            self.font_size_table.cellClicked.connect(self.apply_font_size_from_table)

        def choose_folder(self) -> None:
            start_dir = self.current_image_dir or self._last_existing_image_dir() or str(Path.home())
            folder = QFileDialog.getExistingDirectory(self, '選擇包含原圖和 ctd 的圖片資料夾', start_dir)
            if folder:
                self.load_folder(folder)

        def _layer_checkboxes(self) -> tuple[tuple[str, QCheckBox], ...]:
            return (
                ('show_mask', self.show_mask),
                ('show_npz_smoothed', self.show_npz_smoothed),
                ('show_npz_outer', self.show_npz_outer),
                ('show_block_boxes', self.show_block_boxes),
                ('show_align_boxes', self.show_align_boxes),
                ('show_line_polygons', self.show_line_polygons),
                ('show_char_boxes', self.show_char_boxes),
                ('show_font_labels', self.show_font_labels),
            )

        def _settings_bool(self, key: str, default: bool) -> bool:
            value = self.settings.value(key, default)
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)):
                return bool(value)
            if isinstance(value, str):
                lowered = value.strip().lower()
                if lowered in ('1', 'true', 'yes', 'on'):
                    return True
                if lowered in ('0', 'false', 'no', 'off'):
                    return False
            return default

        def _restore_layer_checkbox_states(self) -> None:
            for name, checkbox in self._layer_checkboxes():
                default = name in self.DEFAULT_LAYER_CHECKED
                checkbox.setChecked(self._settings_bool(f'layers/{name}', default))

        def _save_layer_checkbox_states(self) -> None:
            for name, checkbox in self._layer_checkboxes():
                self.settings.setValue(f'layers/{name}', checkbox.isChecked())

        def handle_layer_checkbox_changed(self, *_: object) -> None:
            self._save_layer_checkbox_states()
            self.render_current_page()

        def handle_bt_font_label_setting_changed(self, *_: object) -> None:
            self.settings.setValue('ui/show_bt_font_labels', self.show_bt_font_labels_check.isChecked())
            self.render_bt_page(refit=False)

        def match_popover_enabled(self) -> bool:
            return self.show_popover_check.isChecked() and not self.has_multiple_bt_selection()

        def show_bt_text_popover(self, item: dict[str, Any]) -> None:
            if self.has_multiple_bt_selection() or self.selected_bt_item() is None:
                self.bt_text_popover.hide()
                return
            self.bt_text_popover.set_text(str(item.get('text') or ''))
            self.position_bt_text_popover(item)
            self.bt_text_popover.show()
            self.bt_text_popover.raise_()

        def position_bt_text_popover(self, item: dict[str, Any]) -> None:
            xyxy = self.bt_xyxy_from_item(item)
            if xyxy is None:
                self.bt_text_popover.hide()
                return
            x1, y1, x2, y2 = xyxy
            p1 = self.bt_view.mapFromScene(QPointF(x1, y1))
            p2 = self.bt_view.mapFromScene(QPointF(x2, y2))
            selected_rect = QRectF(p1, p2).normalized().adjusted(-8, -8, 8, 8)
            popover_size = self.bt_text_popover.sizeHint()
            gap = 12
            left_local = QPointF(
                selected_rect.left() - gap - popover_size.width(),
                selected_rect.center().y() - popover_size.height() / 2,
            )
            global_pos = self.bt_view.viewport().mapToGlobal(left_local.toPoint())
            self.bt_text_popover.move(global_pos)

        def apply_bt_text_popover_changes(self) -> None:
            if self._updating_editor or self.has_multiple_bt_selection():
                return
            item = self.selected_bt_item()
            if item is None:
                return
            text = self.bt_text_popover.text_edit.toPlainText()
            changed = self.apply_selected_box_updates(
                {'text': text},
                status='已修改當前 _bt 文字，尚未保存。',
                refresh_editor=False,
            )
            if not changed:
                return
            self._updating_editor = True
            self.bt_text_edit.setPlainText(text)
            self._updating_editor = False
            self.position_bt_text_popover(item)

        def sync_bt_text_popover_from_item(self) -> None:
            if not self.bt_text_popover.isVisible() or self.has_multiple_bt_selection():
                return
            item = self.selected_bt_item()
            if item is not None:
                self.bt_text_popover.set_text(str(item.get('text') or ''))
                self.position_bt_text_popover(item)

        def handle_match_popover_setting_changed(self, *_: object) -> None:
            enabled = self.show_popover_check.isChecked()
            self.settings.setValue('ui/show_match_popover', enabled)
            if not self.match_popover_enabled():
                self.bt_match_popover.hide()
                return
            item = self.selected_bt_item()
            if item is not None:
                self.show_bt_match_popover(item)

        def toggle_match_popover_enabled(self, *_: object) -> None:
            self.show_popover_check.setChecked(not self.show_popover_check.isChecked())

        def restore_right_panel_state(self) -> None:
            collapsed = self._settings_bool('right_panel/collapsed', False)
            width = self.settings.value('right_panel/width', self._last_right_splitter_width)
            try:
                self._last_right_splitter_width = max(160, int(width))
            except (TypeError, ValueError):
                self._last_right_splitter_width = 520
            self.set_right_panel_collapsed(collapsed, remember=False)

        def toggle_right_panel(self, checked: bool = False) -> None:
            self.set_right_panel_collapsed(checked, remember=True)

        def set_right_panel_collapsed(self, collapsed: bool, *, remember: bool) -> None:
            if not hasattr(self, 'main_splitter'):
                return
            sizes = self.main_splitter.sizes()
            if len(sizes) < 3:
                return

            if collapsed:
                if sizes[2] > 0:
                    self._last_right_splitter_width = sizes[2]
                sizes[0] = max(160, sizes[0] + sizes[2])
                sizes[2] = 0
                self.main_splitter.setSizes(sizes)
                self.view.hide()
                if self.right_layer_dock is not None:
                    self.right_layer_dock.hide()
                label = '展開右側'
            else:
                self.view.show()
                if self.right_layer_dock is not None:
                    self.right_layer_dock.show()
                restored_width = max(160, self._last_right_splitter_width)
                if sizes[2] <= 0:
                    sizes[2] = restored_width
                    sizes[0] = max(160, sizes[0] - restored_width)
                self.main_splitter.setSizes(sizes)
                label = '收起右側'

            if self.toggle_right_panel_action is not None:
                self.toggle_right_panel_action.blockSignals(True)
                self.toggle_right_panel_action.setChecked(collapsed)
                self.toggle_right_panel_action.setText(label)
                self.toggle_right_panel_action.blockSignals(False)
            if remember:
                self.settings.setValue('right_panel/collapsed', collapsed)
                if self._last_right_splitter_width > 0:
                    self.settings.setValue('right_panel/width', self._last_right_splitter_width)
            self.bt_match_popover.hide()
            self.fit_both_views()

        def is_right_panel_collapsed(self) -> bool:
            return hasattr(self, 'view') and not self.view.isVisible()

        def load_folder(self, image_dir: str) -> None:
            self._commit_bt_editor_preview()
            if self.bt_dirty and not self.save_pending_changes(auto=True):
                return
            image_dir = str(Path(image_dir).expanduser().resolve())
            self.current_image_dir = image_dir
            self._clear_bt_image_cache()
            self.clear_hover_char_box(render=False)
            self._bt_cursor_image_pos = None
            self.page = None
            self.selected_box_index = None
            self.selected_bt_index = None
            self.bt_data = None
            self.bt_path = None
            self.bt_dirty = False
            self.bt_undo_stack.clear()
            self.bt_path_label.setText('尚未載入 _bt.json')
            self.update_bt_item_list()
            self.undo_stack.clear()
            self.measure_dirty = False
            self.current_page_row = -1
            self.update_action_state()
            self.set_box_editor_enabled(False)
            try:
                self.processor = CtdOverlayProcessor(image_dir)
                self.page_names = self.processor.page_names()
            except Exception as exc:
                show_exception_details(self, '載入失敗', '無法載入資料夾。下方是完整可複製的出錯信息。', exc)
                return

            self._save_last_image_dir(image_dir)
            self.page_list.clear()
            self.page_list.addItems(self.page_names)
            self.update_font_size_list()
            summary = self.processor.summary()
            self.status_label.setText(self._summary_text(summary))
            if self.page_names:
                self.page_list.setCurrentRow(0)
                self._autoload_mapped_bt_json()
            else:
                self.page = None
                self.current_page_row = -1
                self.view.set_pixmap(QPixmap(), fit=False)
                self.bt_view.set_pixmap(QPixmap(), fit=False)
                self.update_font_size_list()
                self.status_label.setText(f'{self.status_label.text()}\n此資料夾沒有可顯示的圖片。')

        def handle_page_row_changed(self, row: int) -> None:
            if row == self.current_page_row:
                return
            previous_row = self.current_page_row
            if self.bt_dirty and not self.save_pending_changes(auto=True):
                self.page_list.blockSignals(True)
                self.page_list.setCurrentRow(previous_row)
                self.page_list.blockSignals(False)
                return
            self.undo_stack.clear()
            self.update_action_state()
            self.load_page_at_row(row)

        def go_to_relative_page(self, delta: int) -> None:
            if not self.page_names:
                return
            current = self.page_list.currentRow()
            if current < 0:
                current = self.current_page_row
            target = max(0, min(current + delta, len(self.page_names) - 1))
            if target != current:
                self.page_list.setCurrentRow(target)

        def fit_both_views(self) -> None:
            if not hasattr(self, 'bt_view') or not hasattr(self, 'view'):
                return
            if self._fitting_views:
                return
            self._fitting_views = True
            views = (self.bt_view,) if self.is_right_panel_collapsed() else (self.bt_view, self.view)
            for view in views:
                if view.sceneRect().isValid() and not view.sceneRect().isEmpty():
                    view.fitInView(view.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
            self._fitting_views = False
            if self.is_right_panel_collapsed():
                self.sync_viewports_from(self.bt_view)
            else:
                self.sync_viewports_from(self.view)
            self.update_navigator()
            self.schedule_bt_html_overlay_update()

        def sync_viewports_from(self, source: ImageView) -> None:
            if self._syncing_views or self._fitting_views:
                return
            self._pending_viewport_source = source
            if not self._viewport_sync_timer.isActive():
                self._viewport_sync_timer.start()

        def _flush_viewport_sync(self) -> None:
            source = self._pending_viewport_source
            self._pending_viewport_source = None
            if source is None or self._syncing_views or self._fitting_views:
                return
            target = self.view if source is self.bt_view else self.bt_view
            if source.sceneRect().isEmpty() or target.sceneRect().isEmpty():
                return
            self._syncing_views = True
            target.setTransform(source.transform())
            center = source.mapToScene(source.viewport().rect().center())
            target.centerOn(center)
            self._syncing_views = False
            self.update_navigator()
            self.schedule_bt_html_overlay_update()
            if self._popover_bt_item is not None and self.bt_match_popover.isVisible():
                self.position_bt_match_popover(self._popover_bt_item)
            item = self.selected_bt_item()
            if item is not None and self.bt_text_popover.isVisible():
                self.position_bt_text_popover(item)

        def center_views_on(self, x: float, y: float) -> None:
            self._syncing_views = True
            center = QPointF(x, y)
            self.bt_view.centerOn(center)
            self.view.centerOn(center)
            self._syncing_views = False
            self.update_navigator()
            self.schedule_bt_html_overlay_update()

        def update_navigator(self) -> None:
            if not hasattr(self, 'navigator'):
                return
            source_view = self.bt_view if self.is_right_panel_collapsed() else self.view
            pixmap = source_view.current_pixmap()
            if pixmap.isNull():
                self.navigator.set_navigator_state(QPixmap(), (0, 0), QRectF())
                return
            visible_polygon = source_view.mapToScene(source_view.viewport().rect())
            visible_rect = visible_polygon.boundingRect()
            self.navigator.set_navigator_state(
                pixmap,
                (pixmap.width(), pixmap.height()),
                visible_rect,
            )

        def resizeEvent(self, event) -> None:
            super().resizeEvent(event)
            if not hasattr(self, '_fitting_views'):
                return
            self.fit_both_views()
            self.schedule_bt_html_overlay_update()

        def load_bt_clipboard_items(self) -> list[dict[str, Any]]:
            """Load the app-wide text-box clipboard without trusting stored data."""
            raw_value = self.settings.value('bt_clipboard/items', '[]', str)
            try:
                decoded = json.loads(raw_value) if isinstance(raw_value, str) else raw_value
            except (TypeError, ValueError, json.JSONDecodeError):
                return []
            if not isinstance(decoded, list):
                return []
            return [copy.deepcopy(item) for item in decoded if isinstance(item, dict)][:50]

        def save_bt_clipboard_items(self) -> None:
            self.settings.setValue(
                'bt_clipboard/items',
                json.dumps(self.bt_clipboard_items[:50], ensure_ascii=False, separators=(',', ':')),
            )

        def bt_clipboard_item_label(self, item: dict[str, Any]) -> str:
            text = ' '.join(str(item.get('text') or '').split())
            if len(text) > 32:
                text = f'{text[:32]}...'
            if not text:
                text = '(空文字)'
            font_size = item.get('font-size', item.get('font_size', '?'))
            orientation = '直' if str(item.get('orientation') or 'vertical') == 'vertical' else '橫'
            return f'[{orientation} {font_size}px] {text}'

        def update_bt_clipboard_list(self) -> None:
            if not hasattr(self, 'bt_clipboard_list'):
                return
            self.bt_clipboard_list.blockSignals(True)
            self.bt_clipboard_list.clear()
            if not self.bt_clipboard_items:
                self.bt_clipboard_list.addItem('剪貼簿目前是空的')
                self.bt_clipboard_list.item(0).setFlags(Qt.ItemFlag.NoItemFlags)
            else:
                for index, item in enumerate(self.bt_clipboard_items):
                    active_prefix = '● ' if index == self.active_bt_clipboard_index else '○ '
                    self.bt_clipboard_list.addItem(f'{active_prefix}{self.bt_clipboard_item_label(item)}')
                    list_item = self.bt_clipboard_list.item(index)
                    list_item.setData(Qt.ItemDataRole.UserRole, index)
                    list_item.setToolTip('單擊即可複製為目前剪貼簿內容。')
                if self.active_bt_clipboard_index is not None and 0 <= self.active_bt_clipboard_index < len(self.bt_clipboard_items):
                    self.bt_clipboard_list.setCurrentRow(self.active_bt_clipboard_index)
                else:
                    self.bt_clipboard_list.setCurrentRow(-1)
            self.bt_clipboard_list.blockSignals(False)
            self.update_action_state()

        def active_bt_clipboard_item(self) -> dict[str, Any] | None:
            if self.active_bt_clipboard_index is None or not 0 <= self.active_bt_clipboard_index < len(self.bt_clipboard_items):
                return None
            return self.bt_clipboard_items[self.active_bt_clipboard_index]

        def activate_bt_clipboard_item(self, index: int, *, announce: bool = True) -> None:
            if not 0 <= index < len(self.bt_clipboard_items):
                return
            self.active_bt_clipboard_index = index
            self.memory_bt_clipboard_item = copy.deepcopy(self.bt_clipboard_items[index])
            self.update_bt_clipboard_list()
            if announce:
                self.status_label.setText('已暫存複製剪貼簿項目；可用 F2 在游標位置貼上。')

        def copy_bt_clipboard_list_item(self, list_item) -> None:
            index = list_item.data(Qt.ItemDataRole.UserRole) if list_item is not None else None
            if isinstance(index, int):
                self.activate_bt_clipboard_item(index)

        def copy_selected_box_to_clipboard(self) -> None:
            item = self.selected_bt_item()
            if item is None:
                self.status_label.setText('請先選擇一條 _bt 文字，再加入持久剪貼簿列表。')
                return
            self.bt_clipboard_items.insert(0, copy.deepcopy(item))
            self.bt_clipboard_items = self.bt_clipboard_items[:50]
            self.save_bt_clipboard_items()
            self.update_bt_clipboard_list()
            self.status_label.setText('已將目前文字框加入持久剪貼簿列表。')

        def copy_selected_box_to_memory(self) -> None:
            item = self.selected_bt_item()
            if item is None:
                return
            self.memory_bt_clipboard_item = copy.deepcopy(item)
            self.active_bt_clipboard_index = None
            self.update_bt_clipboard_list()
            self.status_label.setText('已暫存複製目前文字框；可用 F2 在游標位置貼上。')

        def paste_box_from_memory(self) -> None:
            if self.bt_data is None:
                return
            page_name = self.current_page_name()
            source_item = self.memory_bt_clipboard_item
            if page_name is None or source_item is None:
                return
            cursor = self._bt_cursor_image_pos
            if cursor is None:
                return
            items = self.bt_items_for_page(page_name)
            before_items = copy.deepcopy(items)
            before_selected = self.selected_bt_index
            before_selected_indices = set(self.selected_bt_indices)
            new_item = copy.deepcopy(source_item)
            new_item['index'] = self.next_bt_index_for_page(items)
            new_item['match_status'] = 'manual'
            xyxy = self.bt_xyxy_from_item(new_item)
            if xyxy is not None:
                x1, y1, x2, y2 = xyxy
                width, height = x2 - x1, y2 - y1
                center_x, center_y = (int(round(cursor[0])), int(round(cursor[1])))
                xyxy = (center_x - width // 2, center_y - height // 2, center_x - width // 2 + width, center_y - height // 2 + height)
                self.set_bt_xyxy(new_item, self.clamp_xyxy(xyxy))
            else:
                center_x, center_y = (int(round(cursor[0])), int(round(cursor[1])))
                self.set_bt_xyxy(new_item, self.clamp_xyxy((center_x - 25, center_y - 25, center_x + 25, center_y + 25)))
            items.append(new_item)
            new_index = len(items) - 1
            self.selected_bt_index = new_index
            self.selected_bt_indices = {new_index}
            self.push_bt_items_undo_snapshot(
                '從文字框剪貼簿建立 _bt 條目',
                before_items,
                before_selected,
                before_selected_indices,
            )
            self.mark_bt_dirty()
            self.populate_box_editor_for_selection()
            self.update_bt_item_list()
            self.render_bt_page(refit=False)
            self.status_label.setText('已在游標位置貼上暫存文字框，尚未保存。')

        def delete_selected_clipboard_item(self) -> None:
            list_item = self.bt_clipboard_list.currentItem() if hasattr(self, 'bt_clipboard_list') else None
            index = list_item.data(Qt.ItemDataRole.UserRole) if list_item is not None else None
            if not isinstance(index, int) or not 0 <= index < len(self.bt_clipboard_items):
                self.status_label.setText('請先選擇一個剪貼簿文字框。')
                return
            del self.bt_clipboard_items[index]
            if self.active_bt_clipboard_index == index:
                self.active_bt_clipboard_index = None
            elif self.active_bt_clipboard_index is not None and self.active_bt_clipboard_index > index:
                self.active_bt_clipboard_index -= 1
            self.save_bt_clipboard_items()
            self.update_bt_clipboard_list()
            self.status_label.setText('已從全域剪貼簿刪除文字框。')

        def current_page_name(self) -> str | None:
            if self.page is not None:
                return self.page.page_name
            if 0 <= self.current_page_row < len(self.page_names):
                return self.page_names[self.current_page_row]
            return None

        def bt_items_for_page(self, page_name: str | None = None) -> list[dict[str, Any]]:
            if self.bt_data is None:
                return []
            page_name = page_name or self.current_page_name()
            if not page_name:
                return []
            items = self.bt_data.get('transMap', {}).get(page_name, [])
            return items if isinstance(items, list) else []

        def _bt_total_count(self) -> int:
            if self.bt_data is None:
                return 0
            trans_map = self.bt_data.get('transMap', {})
            if not isinstance(trans_map, dict):
                return 0
            return sum(len(items) for items in trans_map.values() if isinstance(items, list))

        def _bt_remaining_count_from_current_page(self) -> int:
            if self.bt_data is None or not self.page_names:
                return 0
            row = self.current_page_row
            if row < 0:
                page_name = self.current_page_name()
                row = self.page_names.index(page_name) if page_name in self.page_names else 0
            return sum(len(self.bt_items_for_page(page_name)) for page_name in self.page_names[row:])

        def update_bt_stats_label(self) -> None:
            if not hasattr(self, 'bt_stats_label'):
                return
            if self.bt_data is None:
                self.bt_stats_label.setText('_bt 統計：未載入')
                return
            current_count = len(self.bt_items_for_page())
            remaining_count = self._bt_remaining_count_from_current_page()
            self.bt_stats_label.setText(
                f'_bt 總條數：{self._bt_total_count()}，'
                f'當前頁：{current_count}，'
                f'當前頁起剩餘：{remaining_count}'
            )

        def selected_bt_item(self) -> dict[str, Any] | None:
            items = self.bt_items_for_page()
            if self.selected_bt_index is None or self.selected_bt_index < 0 or self.selected_bt_index >= len(items):
                return None
            item = items[self.selected_bt_index]
            return item if isinstance(item, dict) else None

        def selected_bt_indices_list(self) -> list[int]:
            items = self.bt_items_for_page()
            valid = [
                index for index in sorted(self.selected_bt_indices)
                if 0 <= index < len(items) and isinstance(items[index], dict)
            ]
            if self.selected_bt_index is not None and self.selected_bt_index not in valid:
                if 0 <= self.selected_bt_index < len(items) and isinstance(items[self.selected_bt_index], dict):
                    valid.append(self.selected_bt_index)
            return sorted(valid)

        def selected_bt_items(self) -> list[tuple[int, dict[str, Any]]]:
            items = self.bt_items_for_page()
            return [
                (index, items[index])
                for index in self.selected_bt_indices_list()
                if isinstance(items[index], dict)
            ]

        def has_multiple_bt_selection(self) -> bool:
            return len(self.selected_bt_indices_list()) > 1

        def active_bt_index_from_selection(self, indices: set[int] | list[int], fallback: int | None = None) -> int | None:
            items = self.bt_items_for_page()
            valid = sorted(index for index in indices if 0 <= index < len(items) and isinstance(items[index], dict))
            if fallback is not None and fallback in valid:
                return fallback
            if self.selected_bt_index in valid:
                return self.selected_bt_index
            return valid[0] if valid else None

        def bt_item_status(self, item: dict[str, Any]) -> str:
            raw_status = str(item.get('match_status') or '').lower()
            if raw_status == 'manual':
                return '手動'
            if raw_status == 'auto':
                return '自動'
            if raw_status == 'unmatched':
                return '未匹配'
            if raw_status in {'duplicate', 'fallback'}:
                return '待確認'
            if self.bt_item_needs_review(item):
                return '待確認'
            return '自動'

        def bt_item_needs_review(self, item: dict[str, Any]) -> bool:
            raw_status = str(item.get('match_status') or '').lower()
            if raw_status in {'manual', 'auto'}:
                return False
            if raw_status in {'unmatched', 'duplicate', 'fallback'}:
                return True
            xyxy = self.bt_xyxy_from_item(item)
            if xyxy is None:
                return True
            x1, y1, x2, y2 = xyxy
            return abs((x2 - x1) - 50) <= 1 and abs((y2 - y1) - 50) <= 1

        def bt_item_list_label(self, index: int, item: dict[str, Any]) -> str:
            status = self.bt_item_status(item)
            text = ' '.join(str(item.get('text') or '').split())
            if len(text) > 28:
                text = f'{text[:28]}...'
            if not text:
                text = '(空文字)'
            entry_index = item.get('index', index + 1)
            group_id = item.get('groupId', '-')
            return f'{index + 1:03d} [{status}] index={entry_index} g={group_id}  {text}'

        def update_bt_item_list(self) -> None:
            if not hasattr(self, 'bt_item_list'):
                return
            self.bt_item_list.blockSignals(True)
            self.bt_item_list.clear()
            items = self.bt_items_for_page()
            if self.bt_data is None:
                self.bt_item_list.addItem('未載入 _bt.json')
                self.bt_item_list.item(0).setFlags(Qt.ItemFlag.NoItemFlags)
            elif not items:
                self.bt_item_list.addItem('本頁沒有 _bt 條目')
                self.bt_item_list.item(0).setFlags(Qt.ItemFlag.NoItemFlags)
            else:
                for index, item in enumerate(items):
                    label = self.bt_item_list_label(index, item if isinstance(item, dict) else {})
                    self.bt_item_list.addItem(label)
                    list_item = self.bt_item_list.item(index)
                    list_item.setData(Qt.ItemDataRole.UserRole, index)
                    status = self.bt_item_status(item if isinstance(item, dict) else {})
                    if status in {'未匹配', '待確認'}:
                        list_item.setBackground(QBrush(QColor(255, 236, 194)))
                        list_item.setForeground(QBrush(QColor(92, 50, 0)))
                        list_item.setToolTip('需要人工確認：可先選左側此條，再點右側 measure 框套用。')
                    elif status == '手動':
                        list_item.setBackground(QBrush(QColor(214, 245, 223)))
                        list_item.setForeground(QBrush(QColor(18, 92, 50)))
                        list_item.setToolTip('已手動套用 measure 框。')
            selected_indices = set(self.selected_bt_indices_list())
            if self.selected_bt_index is not None:
                selected_indices.add(self.selected_bt_index)
            active_index = self.active_bt_index_from_selection(selected_indices, self.selected_bt_index)
            for index in selected_indices:
                if 0 <= index < self.bt_item_list.count():
                    self.bt_item_list.item(index).setSelected(True)
            if active_index is not None and 0 <= active_index < len(items):
                self.bt_item_list.setCurrentRow(active_index)
            elif not selected_indices:
                self.bt_item_list.clearSelection()
                self.bt_item_list.setCurrentRow(-1)
            self.bt_item_list.blockSignals(False)
            self.update_bt_stats_label()

        def handle_bt_item_row_changed(self, row: int) -> None:
            item = self.bt_item_list.item(row)
            if item is None:
                return
            index = item.data(Qt.ItemDataRole.UserRole)
            if not isinstance(index, int):
                self.select_bt_item(None)
                return
            selected = {
                selected_item.data(Qt.ItemDataRole.UserRole)
                for selected_item in self.bt_item_list.selectedItems()
            }
            selected_indices = {value for value in selected if isinstance(value, int)}
            if not selected_indices:
                selected_indices = {index}
            self.set_bt_selection(selected_indices, active_index=index, center=True, sync_list=False)

        def handle_bt_item_selection_changed(self) -> None:
            selected = {
                selected_item.data(Qt.ItemDataRole.UserRole)
                for selected_item in self.bt_item_list.selectedItems()
            }
            selected_indices = {value for value in selected if isinstance(value, int)}
            current_item = self.bt_item_list.currentItem()
            current_index = current_item.data(Qt.ItemDataRole.UserRole) if current_item is not None else None
            active_index = current_index if isinstance(current_index, int) else None
            self.set_bt_selection(selected_indices, active_index=active_index, sync_list=False)

        def load_bt_json_path(self, path: Path, *, remember: bool = True) -> None:
            self._commit_bt_editor_preview()
            path = path.expanduser().resolve()
            data = json.loads(path.read_text(encoding='utf-8'))
            if not isinstance(data, dict) or not isinstance(data.get('transMap'), dict):
                raise ValueError('不是有效的 _bt/MEO JSON：缺少 transMap。')
            self.bt_data = data
            self.bt_path = path
            self.bt_dirty = False
            self.selected_bt_index = None
            self.selected_bt_indices.clear()
            self.bt_undo_stack.clear()
            self.bt_path_label.setText(str(path))
            self.set_box_editor_enabled(False)
            self.update_bt_item_list()
            self.render_bt_page(refit=True)
            self.update_action_state()
            if remember:
                self._save_bt_mapping(path)

        def open_bt_json(self) -> None:
            start_dir = self.current_image_dir or self._last_existing_image_dir() or str(Path.home())
            path_text, _ = QFileDialog.getOpenFileName(self, '打開 _bt.json', start_dir, 'JSON (*.json)')
            if not path_text:
                return
            try:
                self.load_bt_json_path(Path(path_text).expanduser().resolve())
            except Exception as exc:
                show_exception_details(self, '打開失敗', '無法打開 _bt.json。下方是完整可複製的出錯信息。', exc)

        def save_bt_json(self) -> None:
            self._commit_bt_editor_preview()
            if self.bt_data is None or self.bt_path is None:
                return
            self.bt_path.write_text(
                json.dumps(self.bt_data, ensure_ascii=False, indent=2) + '\n',
                encoding='utf-8',
            )
            self.bt_dirty = False
            self.update_action_state()
            self.schedule_bt_html_overlay_update()

        def mark_bt_dirty(self) -> None:
            self.bt_dirty = True
            self.update_action_state()

        def bt_xyxy_from_item(self, item: dict[str, Any]) -> tuple[int, int, int, int] | None:
            xyxy = item.get('xyxy_pixel')
            if isinstance(xyxy, list) and len(xyxy) == 4:
                return tuple(int(round(float(v))) for v in xyxy)
            page_name = self.current_page_name()
            if not page_name or self.processor is None:
                return None
            image_size = qimage_size(self.processor.image_dir / page_name)
            if image_size is None:
                return None
            width, height = image_size
            try:
                cx = float(item.get('x')) * width
                cy = float(item.get('y')) * height
            except (TypeError, ValueError):
                return None
            half = 25
            return int(round(cx - half)), int(round(cy - half)), int(round(cx + half)), int(round(cy + half))

        def set_bt_xyxy(self, item: dict[str, Any], xyxy: tuple[int, int, int, int]) -> None:
            item['xyxy_pixel'] = list(xyxy)
            if self.page is not None:
                image_size = self._bt_cached_image_size or qimage_size(self.page.image_path)
                center = normalized_center_from_xyxy(xyxy, image_size)
                if center is not None:
                    item['x'] = center[0]
                    item['y'] = center[1]

        def bt_text_color(self, item: dict[str, Any]) -> str:
            color = str(item.get('color') or '#000000').lower()
            return 'white' if color in {'#ffffff', 'ffffff', 'white'} else 'black'

        def normalized_rotation(self, value: object) -> float:
            try:
                angle = float(value)
            except (TypeError, ValueError):
                return 0.0
            while angle > 180.0:
                angle -= 360.0
            while angle <= -180.0:
                angle += 360.0
            return round(angle, 2)

        def bt_font_label(self, item: dict[str, Any]) -> str:
            orientation = str(item.get('orientation') or 'vertical')
            direction = 'H' if orientation == 'horizontal' else 'V'
            font_size = positive_int(item.get('font-size'), 0)
            parts = [f'{font_size}{direction}' if font_size > 0 else direction]
            color_name = self.bt_text_color(item)
            parts.append('白' if color_name == 'white' else '黑')
            if positive_int(item.get('stroke-weight'), 0) > 0:
                parts.append('描邊')
            return ','.join(parts)

        def set_bt_text_color(self, item: dict[str, Any], value: str) -> None:
            if value == 'white':
                item['color'] = '#FFFFFF'
                item['stroke-color'] = '#000000'
            else:
                item['color'] = '#000000'
                item['stroke-color'] = '#FFFFFF'

        def push_bt_undo(self, description: str) -> None:
            item = self.selected_bt_item()
            page_name = self.current_page_name()
            if item is None or page_name is None or self.selected_bt_index is None:
                return
            self.bt_undo_stack.append({
                'page_name': page_name,
                'item_index': self.selected_bt_index,
                'item': copy.deepcopy(item),
                'description': description,
            })
            if len(self.bt_undo_stack) > 200:
                self.bt_undo_stack = self.bt_undo_stack[-200:]

        def push_bt_items_undo_snapshot(
            self,
            description: str,
            items: list[dict[str, Any]],
            selected_index: int | None,
            selected_indices: set[int] | list[int] | None = None,
        ) -> None:
            page_name = self.current_page_name()
            if page_name is None:
                return
            self.bt_undo_stack.append({
                'page_name': page_name,
                'items': copy.deepcopy(items),
                'selected_index': selected_index,
                'selected_indices': sorted(selected_indices or []),
                'description': description,
            })
            if len(self.bt_undo_stack) > 200:
                self.bt_undo_stack = self.bt_undo_stack[-200:]

        def push_bt_changes_undo_snapshot(
            self,
            description: str,
            before_items: dict[int, dict[str, Any]],
            selected_index: int | None,
            selected_indices: set[int] | list[int] | None = None,
        ) -> None:
            page_name = self.current_page_name()
            if page_name is None or not before_items:
                return
            self.bt_undo_stack.append({
                'page_name': page_name,
                'changes': [
                    {'index': index, 'item': item}
                    for index, item in sorted(before_items.items())
                ],
                'selected_index': selected_index,
                'selected_indices': sorted(selected_indices or []),
                'description': description,
            })
            if len(self.bt_undo_stack) > 200:
                self.bt_undo_stack = self.bt_undo_stack[-200:]

        def next_bt_index_for_page(self, items: list[dict[str, Any]]) -> int:
            indexes = []
            for item in items:
                try:
                    indexes.append(int(item.get('index')))
                except (AttributeError, TypeError, ValueError):
                    continue
            return max(indexes, default=0) + 1

        def font_size_counts(self) -> dict[int, int]:
            if self.processor is None:
                return {}
            counts = {}
            for items in (self.processor.measure.get('pages') or {}).values():
                if not isinstance(items, list):
                    continue
                for item in items:
                    if not isinstance(item, dict) or item.get('font_size') is None:
                        continue
                    try:
                        size = int(round(float(item['font_size'])))
                    except (TypeError, ValueError):
                        continue
                    if size > 0:
                        counts[size] = counts.get(size, 0) + 1
            return counts

        def bt_font_size_counts(self) -> dict[int, int]:
            if self.bt_data is None:
                return {}
            counts: dict[int, int] = {}
            pages = self.bt_data.get('transMap') or {}
            if not isinstance(pages, dict):
                return counts
            for items in pages.values():
                if not isinstance(items, list):
                    continue
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    size = positive_int(item.get('font-size'), 0)
                    if size > 0:
                        counts[size] = counts.get(size, 0) + 1
            return counts

        def open_regularize_font_size_dialog(self) -> None:
            if self.bt_data is None:
                self.status_label.setText('請先打開 _bt.json，再規整字體大小。')
                return
            counts = self.bt_font_size_counts()
            if not counts:
                self.status_label.setText('_bt.json 中沒有可規整的字體大小。')
                return
            dialog = FontSizeRegularizeDialog(counts, self)
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return
            lower, upper = dialog.selected_range()
            target = dialog.target_size()
            self.regularize_bt_font_sizes(lower, upper, target)

        def regularize_bt_font_sizes(self, lower: int, upper: int, target: int) -> None:
            if self.bt_data is None:
                return
            lower, upper = sorted((int(lower), int(upper)))
            target = max(1, int(target))
            pages = self.bt_data.get('transMap') or {}
            if not isinstance(pages, dict):
                return
            changed = 0
            affected = 0
            for items in pages.values():
                if not isinstance(items, list):
                    continue
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    size = positive_int(item.get('font-size'), 0)
                    if lower <= size <= upper:
                        affected += 1
                        if size != target:
                            item['font-size'] = target
                            changed += 1
            if changed <= 0:
                self.status_label.setText(f'{lower}-{upper} px 範圍內沒有需要改為 {target} px 的 _bt 條目。')
                return
            self.mark_bt_dirty()
            if self.selected_bt_items():
                self.populate_box_editor_for_selection()
            else:
                self.set_box_editor_enabled(False)
            self.update_bt_item_list()
            self.update_font_size_list()
            self.render_bt_page(refit=False)
            self.update_action_state()
            self.status_label.setText(
                f'已將 {lower}-{upper} px 範圍內 {affected} 條 _bt 中的 {changed} 條改為 {target} px，尚未保存。'
            )

        def even_font_size_preview(self) -> tuple[dict[int, int], int]:
            if self.processor is None:
                return {}, 0
            counts = {}
            changed = 0
            for items in (self.processor.measure.get('pages') or {}).values():
                if not isinstance(items, list):
                    continue
                for item in items:
                    if not isinstance(item, dict) or item.get('font_size') is None:
                        continue
                    new_size = even_font_size(item.get('font_size'))
                    if new_size is None:
                        continue
                    try:
                        old_size = int(round(float(item['font_size'])))
                    except (TypeError, ValueError):
                        old_size = new_size
                    if new_size != old_size:
                        changed += 1
                    counts[new_size] = counts.get(new_size, 0) + 1
            return counts, changed

        def format_font_size_counts(self, counts: dict[int, int]) -> str:
            if not counts:
                return '沒有可預覽的字級資料。'
            lines = ['字級  數目', '----------']
            for size in sorted(counts):
                lines.append(f'{size:>4}  {counts[size]:>4}')
            return '\n'.join(lines)

        def preview_even_font_sizes(self) -> None:
            QMessageBox.information(self, '功能已移除', 'ctd/measure.json 生成後不再支持批量修改字體大小。')

        def apply_even_font_sizes(self) -> None:
            return

        def current_page_font_sizes(self) -> set[int]:
            if self.page is None:
                return set()
            sizes = set()
            for box in self.page.boxes:
                if box.font_size is None:
                    continue
                size = int(round(float(box.font_size)))
                if size > 0:
                    sizes.add(size)
            return sizes

        def update_font_size_list(self) -> None:
            current_sizes = self.current_page_font_sizes()
            counts = self.font_size_counts()
            self.font_size_table.setRowCount(0)
            for row, size in enumerate(sorted(counts)):
                self.font_size_table.insertRow(row)
                size_item = QTableWidgetItem(str(size))
                count_item = QTableWidgetItem(str(counts[size]))
                for item in (size_item, count_item):
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                if size in current_sizes:
                    for item in (size_item, count_item):
                        item.setBackground(QBrush(QColor(255, 220, 92)))
                        item.setForeground(QBrush(QColor(0, 0, 0)))
                        item.setToolTip('目前頁面使用')
                else:
                    for item in (size_item, count_item):
                        item.setForeground(QBrush(QColor(225, 229, 233)))
                self.font_size_table.setItem(row, 0, size_item)
                self.font_size_table.setItem(row, 1, count_item)
            self.font_size_table.resizeRowsToContents()

        def selected_box(self) -> BoxOverlay | None:
            if self.page is None or self.selected_box_index is None:
                return None
            if self.selected_box_index < 0 or self.selected_box_index >= len(self.page.boxes):
                return None
            return self.page.boxes[self.selected_box_index]

        def set_box_editor_enabled(self, enabled: bool) -> None:
            for widget in (
                self.bt_text_edit,
                self.font_size_spin,
                self.orientation_vertical_button,
                self.orientation_horizontal_button,
                self.rotation_spin,
                self.copy_bt_button,
                self.measure_preview_button,
                self.color_black_button,
                self.color_white_button,
                self.stroke_weight_spin,
                self.text_has_stroke_check,
                self.need_inpaint_check,
            ):
                widget.setEnabled(enabled)
            if not enabled:
                self.box_editor_title.setText('未選擇 _bt 條目')
                self.copy_bt_button.setText('複製文字框')
                self.bt_text_popover.hide()
                previous_updating = self._updating_editor
                self._updating_editor = True
                self.bt_text_edit.clear()
                self.rotation_spin.setValue(0.0)
                self._updating_editor = previous_updating
            self.update_bt_item_list()
            self.update_action_state()

        def selection_mixed_bt_fields(self, selected_items: list[tuple[int, dict[str, Any]]]) -> list[str]:
            if len(selected_items) <= 1:
                return []
            getters = (
                ('字體', lambda item: positive_int(item.get('font-size'), 40)),
                ('旋轉', lambda item: self.normalized_rotation(item.get('rotation'))),
                ('方向', lambda item: item.get('orientation') or 'vertical'),
                ('顏色', lambda item: self.bt_text_color(item)),
                ('描邊', lambda item: positive_int(item.get('stroke-weight'), 0)),
                ('修復', lambda item: item.get('need_inpaint') is True),
            )
            mixed: list[str] = []
            for label, getter in getters:
                values = [getter(item) for _, item in selected_items]
                if any(value != values[0] for value in values[1:]):
                    mixed.append(label)
            return mixed

        def populate_box_editor_for_selection(self) -> None:
            selected_items = self.selected_bt_items()
            if not selected_items:
                self.set_box_editor_enabled(False)
                return
            active_item = self.selected_bt_item() or selected_items[0][1]
            self.populate_box_editor_from_bt(active_item)
            if len(selected_items) <= 1:
                self.bt_text_edit.setEnabled(True)
                self.measure_preview_button.setEnabled(True)
                self.copy_bt_button.setText('複製文字框')
                self.show_bt_text_popover(active_item)
                return
            self._updating_editor = True
            self.bt_text_edit.setPlainText('多選時不批量修改文字')
            self._updating_editor = False
            mixed = self.selection_mixed_bt_fields(selected_items)
            suffix = f'；混合：{", ".join(mixed)}' if mixed else ''
            active_index = self.selected_bt_index if self.selected_bt_index is not None else selected_items[0][0]
            self.box_editor_title.setText(f'已選擇 {len(selected_items)} 條；active={active_index + 1}{suffix}')
            self.bt_text_edit.setEnabled(False)
            self.measure_preview_button.setEnabled(True)
            self.copy_bt_button.setText('複製選中文字框（⌘/Ctrl+D）')
            self.bt_match_popover.hide()
            self._popover_bt_item = None
            self.bt_text_popover.hide()

        def set_bt_selection(
            self,
            indices: set[int] | list[int] | None,
            *,
            active_index: int | None = None,
            center: bool = False,
            sync_list: bool = True,
        ) -> None:
            self._commit_bt_editor_preview()
            items = self.bt_items_for_page()
            valid_indices = {
                index for index in (indices or set())
                if 0 <= index < len(items) and isinstance(items[index], dict)
            }
            active_index = self.active_bt_index_from_selection(valid_indices, active_index)
            if active_index is None:
                self.selected_bt_index = None
                self.selected_bt_indices.clear()
                self._popover_bt_item = None
                self.bt_match_popover.hide()
                self.set_box_editor_enabled(False)
                if sync_list:
                    self.update_bt_item_list()
                self.render_bt_page(refit=False)
                return
            valid_indices.add(active_index)
            self.selected_bt_index = active_index
            self.selected_bt_indices = valid_indices
            self.populate_box_editor_for_selection()
            if sync_list:
                self.update_bt_item_list()
            self.render_bt_page(refit=False)
            if len(valid_indices) > 1:
                self._popover_bt_item = None
                self.bt_match_popover.hide()
            if center:
                self.center_views_on_bt_item(items[active_index])

        def select_bt_item(self, index: int | None, *, center: bool = False) -> None:
            self.set_bt_selection({index} if index is not None else set(), active_index=index, center=center)

        def clear_current_bt_focus(self) -> None:
            """Clear the current _bt selection without changing its contents."""
            if not self.selected_bt_indices_list():
                return
            self.select_bt_item(None)
            self.bt_view.setFocus()
            self.status_label.setText('已取消目前文字框的選取。')

        def center_views_on_bt_item(self, item: dict[str, Any]) -> None:
            xyxy = self.bt_xyxy_from_item(item)
            if xyxy is not None:
                x1, y1, x2, y2 = xyxy
                self.center_views_on((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                return
            pixmap = self.bt_view.pixmap_item.pixmap()
            if pixmap.isNull():
                return
            center = self.bt_center_pixel_from_item(item)
            if center is not None:
                self.center_views_on(center[0], center[1])

        def measure_box_for_bt_item(self, item: dict[str, Any]) -> BoxOverlay | None:
            if self.page is None:
                return None
            measure_index = item.get('match_measure_item_index')
            if isinstance(measure_index, int):
                for box in self.page.boxes:
                    if box.measure_item_index == measure_index:
                        return box
            source_index = item.get('match_source_block_index')
            if isinstance(source_index, int):
                for box in self.page.boxes:
                    if box.source_block_index == source_index:
                        return box
            return None

        def box_preview_content(
            self,
            box: BoxOverlay | None,
            *,
            char_box: dict | None = None,
        ) -> tuple[QPixmap, list[tuple[QRectF, str]]] | None:
            if box is None or self.page is None:
                return None
            image = QImage(str(self.page.image_path)).convertToFormat(QImage.Format.Format_RGBA8888)
            if image.isNull():
                return None
            x1, y1, x2, y2 = box.xyxy_pixel
            pad = max(2, int(round(max(x2 - x1, y2 - y1) * 0.04)))
            crop_x1 = max(0, x1 - pad)
            crop_y1 = max(0, y1 - pad)
            crop_x2 = min(image.width(), x2 + pad)
            crop_y2 = min(image.height(), y2 + pad)
            crop_rect = QRectF(crop_x1, crop_y1, crop_x2 - crop_x1, crop_y2 - crop_y1).toRect()
            if crop_rect.isEmpty():
                return None
            crop = image.copy(crop_rect)
            char_regions: list[tuple[QRectF, str]] = []
            for char_item in self.page.char_boxes:
                bbox = char_item.get('bbox')
                if not isinstance(bbox, list) or len(bbox) != 4:
                    continue
                cx1, cy1, cx2, cy2 = [float(value) for value in bbox]
                if cx2 < x1 or cx1 > x2 or cy2 < y1 or cy1 > y2:
                    continue
                rect = QRectF(cx1 - crop_x1, cy1 - crop_y1, cx2 - cx1, cy2 - cy1)
                label = char_box_label(char_item) or 'W-H-'
                char_regions.append((rect, label))

            highlight_rect: QRectF | None = None
            if char_box is not None:
                bbox = char_box.get('bbox')
                if isinstance(bbox, list) and len(bbox) == 4:
                    cx1, cy1, cx2, cy2 = [float(value) for value in bbox]
                    highlight_rect = QRectF(cx1 - crop_x1, cy1 - crop_y1, cx2 - cx1, cy2 - cy1)
            pixmap = QPixmap.fromImage(crop)
            target_w = 460
            target_h = 360
            scaled = pixmap.scaled(
                target_w,
                target_h,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            scale_x = scaled.width() / max(1, crop.width())
            scale_y = scaled.height() / max(1, crop.height())
            scaled_regions = [
                (
                    QRectF(
                        rect.x() * scale_x,
                        rect.y() * scale_y,
                        rect.width() * scale_x,
                        rect.height() * scale_y,
                    ),
                    tooltip,
                )
                for rect, tooltip in char_regions
            ]
            scaled_highlight = None
            if highlight_rect is not None:
                scaled_highlight = QRectF(
                    highlight_rect.x() * scale_x,
                    highlight_rect.y() * scale_y,
                    highlight_rect.width() * scale_x,
                    highlight_rect.height() * scale_y,
                )

            painter = QPainter(scaled)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
            painter.setBrush(QColor(245, 170, 35, 32))
            painter.setPen(QPen(QColor(245, 170, 35), 2))
            for rect, _tooltip in scaled_regions:
                painter.drawRect(rect)
            if scaled_highlight is not None:
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.setPen(QPen(QColor(255, 40, 120), 3))
                painter.drawRect(scaled_highlight)

            label_font = QFont('Helvetica Neue', 13)
            label_font.setWeight(QFont.Weight.DemiBold)
            label_font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
            painter.setFont(label_font)
            metrics = painter.fontMetrics()
            painter.setPen(QPen(QColor(74, 42, 12), 1))
            for rect, label in scaled_regions:
                text_w = metrics.horizontalAdvance(label)
                text_h = metrics.ascent() + metrics.descent()
                label_x = int(round(rect.center().x() - text_w / 2))
                label_y = int(round(rect.top() - 3))
                if label_y - text_h < 2:
                    label_y = int(round(rect.bottom() + text_h + 3))
                label_x = max(2, min(label_x, max(2, scaled.width() - text_w - 2)))
                label_y = max(text_h + 2, min(label_y, scaled.height() - 2))
                painter.drawText(QPointF(label_x, label_y), label)
            painter.end()
            return scaled, scaled_regions

        def show_bt_match_popover(self, item: dict[str, Any]) -> None:
            if not self.match_popover_enabled():
                self._popover_bt_item = None
                self.bt_match_popover.hide()
                return
            self._popover_bt_item = item
            box = self.measure_box_for_bt_item(item)
            if box is None:
                self.bt_match_popover.hide()
                return
            content = self.box_preview_content(box)
            if content is None:
                self.bt_match_popover.hide()
                return
            preview, regions = content
            if preview.isNull():
                self.bt_match_popover.hide()
                return
            self.bt_match_popover.set_content(preview, regions)
            self.position_bt_match_popover(item)
            self.bt_match_popover.show()
            self.bt_match_popover.raise_()

        def position_bt_match_popover(self, item: dict[str, Any]) -> None:
            xyxy = self.bt_xyxy_from_item(item)
            if xyxy is None:
                return
            x1, y1, x2, y2 = xyxy
            view_rect = self.bt_view.viewport().rect()
            p1 = self.bt_view.mapFromScene(QPointF(x1, y1))
            p2 = self.bt_view.mapFromScene(QPointF(x2, y2))
            selected_rect = QRectF(p1, p2).normalized().adjusted(-8, -8, 8, 8)
            popover_size = self.bt_match_popover.sizeHint()
            gap = 12
            margin = 12
            min_x = margin
            max_x = max(margin, view_rect.width() - popover_size.width() - margin)
            min_y = margin
            max_y = max(margin, view_rect.height() - popover_size.height() - margin)
            x = int(round(selected_rect.center().x() - popover_size.width() / 2))
            x = max(min_x, min(x, max_x))

            top_y = int(round(selected_rect.top() - popover_size.height() - gap))
            bottom_y = int(round(selected_rect.bottom() + gap))
            if top_y >= min_y:
                y = top_y
            elif bottom_y + popover_size.height() <= view_rect.height() - margin:
                y = bottom_y
            else:
                space_above = selected_rect.top() - margin
                space_below = view_rect.height() - selected_rect.bottom() - margin
                y = top_y if space_above >= space_below else bottom_y
                y = max(min_y, min(y, max_y))
            global_pos = self.bt_view.viewport().mapToGlobal(QPointF(x, y).toPoint())
            self.bt_match_popover.move(global_pos)

        def update_popover_with_char_box(self, item: dict) -> None:
            if not self.match_popover_enabled():
                self.bt_match_popover.hide()
                return
            source_index = item.get('source_block_index', '-')
            box = None
            try:
                source_number = int(source_index)
            except (TypeError, ValueError):
                source_number = None
            if self.page is not None and source_number is not None:
                for candidate in self.page.boxes:
                    if candidate.source_block_index == source_number:
                        box = candidate
                        break
            if box is None and self._popover_bt_item is not None:
                box = self.measure_box_for_bt_item(self._popover_bt_item)
            content = self.box_preview_content(box, char_box=item)
            if content is None:
                self.bt_match_popover.hide()
                return
            preview, regions = content
            if preview.isNull():
                self.bt_match_popover.hide()
                return
            self.bt_match_popover.set_content(preview, regions)
            if self._popover_bt_item is not None:
                self.position_bt_match_popover(self._popover_bt_item)
            else:
                self.position_char_popover(item)
            self.bt_match_popover.show()
            self.bt_match_popover.raise_()

        def position_char_popover(self, item: dict) -> None:
            bbox = item.get('bbox')
            if not isinstance(bbox, list) or len(bbox) != 4:
                return
            x1, y1, x2, y2 = [float(value) for value in bbox]
            view_rect = self.view.viewport().rect()
            p1 = self.view.mapFromScene(QPointF(x1, y1))
            p2 = self.view.mapFromScene(QPointF(x2, y2))
            hover_rect = QRectF(p1, p2).normalized().adjusted(-8, -8, 8, 8)
            popover_size = self.bt_match_popover.sizeHint()
            gap = 12
            x = hover_rect.right() + gap
            y = hover_rect.top()
            if x + popover_size.width() > view_rect.width() - gap:
                x = hover_rect.left() - popover_size.width() - gap
            if y + popover_size.height() > view_rect.height() - gap:
                y = view_rect.height() - popover_size.height() - gap
            x = max(gap, min(x, max(gap, view_rect.width() - popover_size.width() - gap)))
            y = max(gap, min(y, max(gap, view_rect.height() - popover_size.height() - gap)))
            global_pos = self.view.viewport().mapToGlobal(QPointF(x, y).toPoint())
            self.bt_match_popover.move(global_pos)

        def select_box(self, index: int | None) -> None:
            if self.page is None or index is None or index < 0 or index >= len(self.page.boxes):
                self.selected_box_index = None
                self.render_current_page(refit=False)
                return

            self.selected_box_index = index
            self.render_current_page(refit=False)

        def editor_orientation(self) -> str:
            return 'horizontal' if self.orientation_horizontal_button.isChecked() else 'vertical'

        def set_editor_orientation(self, orientation: object) -> None:
            self.orientation_horizontal_button.setChecked(orientation == 'horizontal')
            self.orientation_vertical_button.setChecked(orientation != 'horizontal')

        def editor_text_color(self) -> str:
            return 'white' if self.color_white_button.isChecked() else 'black'

        def set_editor_text_color(self, color: object) -> None:
            self.color_white_button.setChecked(color == 'white')
            self.color_black_button.setChecked(color != 'white')

        def populate_box_editor(self, box: BoxOverlay) -> None:
            self._updating_editor = True
            self.set_box_editor_enabled(True)
            x1, y1, x2, y2 = box.xyxy_pixel
            self.box_editor_title.setText(
                f'區塊 {box.source_block_index}  框：{x1},{y1},{x2},{y2}'
            )
            self.font_size_spin.setValue(max(1, int(round(float(box.font_size or 1)))))
            self.set_editor_text_color(box.text_color or 'black')
            self.text_has_stroke_check.setChecked(box.text_has_stroke is True)
            self.need_inpaint_check.setChecked(box.need_inpaint is True)
            self._updating_editor = False

        def populate_box_editor_from_bt(self, item: dict[str, Any]) -> None:
            self._updating_editor = True
            self.set_box_editor_enabled(True)
            xyxy = self.bt_xyxy_from_item(item)
            box_text = '無框' if xyxy is None else ','.join(str(v) for v in xyxy)
            self.box_editor_title.setText(
                f'index={item.get("index", "-")}  groupId={item.get("groupId", "-")}  框：{box_text}'
            )
            self.bt_text_edit.setPlainText(str(item.get('text') or ''))
            self.font_size_spin.setValue(positive_int(item.get('font-size'), 40))
            self.set_editor_orientation(item.get('orientation') or 'vertical')
            self.rotation_spin.setValue(self.normalized_rotation(item.get('rotation')))
            self.set_editor_text_color(self.bt_text_color(item))
            stroke_weight = int(round(float(item.get('stroke-weight') or 0)))
            self.stroke_weight_spin.setValue(max(0, min(99, stroke_weight)))
            self.text_has_stroke_check.setChecked(stroke_weight > 0)
            self.need_inpaint_check.setChecked(item.get('need_inpaint') is True)
            self._updating_editor = False

        def selected_box_updates_from_editor(self) -> dict[str, object] | None:
            if self._updating_editor or self.selected_bt_item() is None:
                return None
            color = self.editor_text_color()
            stroke_weight = int(self.stroke_weight_spin.value())
            if self.text_has_stroke_check.isChecked() and stroke_weight <= 0:
                stroke_weight = max(1, int(np.ceil(float(self.font_size_spin.value()) / 8.0)))
            if self.has_multiple_bt_selection():
                sender = self.sender()
                if sender is self.font_size_spin:
                    return {'font-size': int(self.font_size_spin.value())}
                if sender in (self.orientation_vertical_button, self.orientation_horizontal_button):
                    return {'orientation': self.editor_orientation()}
                if sender is self.rotation_spin:
                    return {'rotation': self.normalized_rotation(self.rotation_spin.value())}
                if sender in (self.color_black_button, self.color_white_button):
                    return {
                        'color': '#FFFFFF' if color == 'white' else '#000000',
                        'stroke-color': '#000000' if color == 'white' else '#FFFFFF',
                    }
                if sender is self.stroke_weight_spin:
                    return {'stroke-weight': stroke_weight}
                if sender is self.text_has_stroke_check:
                    return {'stroke-weight': stroke_weight if self.text_has_stroke_check.isChecked() else 0}
                if sender is self.need_inpaint_check:
                    return {'need_inpaint': self.need_inpaint_check.isChecked()}
                return None
            return {
                'text': self.bt_text_edit.toPlainText(),
                'font-size': int(self.font_size_spin.value()),
                'orientation': self.editor_orientation(),
                'rotation': self.normalized_rotation(self.rotation_spin.value()),
                'color': '#FFFFFF' if color == 'white' else '#000000',
                'stroke-color': '#000000' if color == 'white' else '#FFFFFF',
                'stroke-weight': stroke_weight,
                'need_inpaint': self.need_inpaint_check.isChecked(),
            }

        def open_bt_measurement_dialog(self) -> None:
            selected_indices = self.selected_bt_indices_list()
            page_name = self.current_page_name()
            if not selected_indices or page_name is None:
                self.status_label.setText('請先選擇一條 _bt 文字，再打開測量角度。')
                return
            image = self.load_bt_source_image(page_name)
            if image is None or image.isNull():
                self.status_label.setText('無法讀取目前頁面的圖片。')
                return
            view_crop_rect, view_display_scale = self.measurement_view_crop_rect(image)
            items = self.bt_items_for_page(page_name)
            targets: list[tuple[int, dict[str, Any], tuple[float, float], tuple[int, int, int, int] | None]] = []
            target_rects: list[QRectF] = []
            centers: list[QPointF] = []
            for index in selected_indices:
                if index < 0 or index >= len(items) or not isinstance(items[index], dict):
                    continue
                item = items[index]
                center = self.bt_center_pixel_from_item(item)
                if center is None:
                    continue
                xyxy = self.bt_xyxy_from_item(item)
                centers.append(QPointF(center[0], center[1]))
                if xyxy is not None:
                    x1, y1, x2, y2 = xyxy
                    target_rects.append(QRectF(x1, y1, max(1, x2 - x1), max(1, y2 - y1)))
                else:
                    target_rects.append(QRectF(center[0] - 25, center[1] - 25, 50, 50))
                targets.append((index, item, center, xyxy))
            if not targets:
                self.status_label.setText('無法建立測量角度裁切圖。')
                return
            crop_rect, display_scale = self.measurement_multi_crop_rect(
                image,
                target_rects,
                centers,
                view_crop_rect,
                view_display_scale,
            )
            crop = image.copy(crop_rect.toRect())
            if crop.isNull():
                self.status_label.setText('無法建立測量角度裁切圖。')
                return
            mask_crop = None
            if self.processor is not None:
                mask_path = self.processor.mask_dir / f'{Path(page_name).stem}.png'
                mask_image = QImage(str(mask_path))
                if not mask_image.isNull():
                    mask_crop = mask_image.copy(crop_rect.toRect())
            entries: list[dict[str, object]] = []
            for index, item, center, _xyxy in targets:
                local_center = QPointF(center[0] - crop_rect.left(), center[1] - crop_rect.top())
                font_size = positive_int(item.get('font-size'), 40)
                entries.append({
                    'item_index': index,
                    'item': item,
                    'crop_origin': QPointF(crop_rect.left(), crop_rect.top()),
                    'center': local_center,
                    'font_size': font_size,
                    'rotation': self.normalized_rotation(item.get('rotation')),
                    'display_scale': display_scale,
                    'orientation': str(item.get('orientation') or 'vertical'),
                    'color': self.qcolor_from_bt_value(item.get('color'), QColor(0, 0, 0)),
                    'stroke_color': self.qcolor_from_bt_value(item.get('stroke-color'), QColor(255, 255, 255)),
                    'stroke_weight': max(0.0, float(item.get('stroke-weight') or 0)),
                    'font_family': str(item.get('font') or '').strip() or 'Helvetica Neue',
                })
            if not entries:
                self.status_label.setText('無法建立測量角度裁切圖。')
                return
            dialog = BtMeasurementDialog(
                crop,
                entries=entries,
                mask=mask_crop,
                display_scale=display_scale,
                parent=self,
            )
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return
            single_char_measurement = dialog.canvas.interaction_mode == 'detect'
            results = dialog.result_updates()
            if not results:
                return
            if single_char_measurement:
                apply_status = '已套用單字測量的字體大小與角度，尚未保存。'
            else:
                apply_status = (
                    f'已套用 {len(results)} 條測量角度，尚未保存。'
                    if len(results) > 1
                    else '已套用測量角度，尚未保存。'
                )
            changed = self.apply_bt_updates_to_indices(
                list(results.keys()),
                lambda index, item: self.measurement_updates_for_item(
                    item,
                    results[index],
                    image.width(),
                    image.height(),
                ) if index in results else {},
                status=apply_status,
            )
            active_result = results.get(self.selected_bt_index if self.selected_bt_index is not None else selected_indices[0])
            if changed and active_result is not None:
                center_pixel = active_result.get('center_pixel')
                if isinstance(center_pixel, list) and len(center_pixel) == 2:
                    self.center_views_on(float(center_pixel[0]), float(center_pixel[1]))

        def qcolor_from_bt_value(self, value: object, fallback: QColor) -> QColor:
            text = str(value or '').strip()
            if not text:
                return QColor(fallback)
            if text.lower() == 'black':
                return QColor(0, 0, 0)
            if text.lower() == 'white':
                return QColor(255, 255, 255)
            color = QColor(text if text.startswith('#') else f'#{text}')
            return color if color.isValid() else QColor(fallback)

        def measurement_multi_crop_rect(
            self,
            image: QImage,
            target_rects: list[QRectF],
            centers: list[QPointF],
            view_crop_rect: QRectF | None,
            view_display_scale: float | None,
        ) -> tuple[QRectF, float | None]:
            image_rect = QRectF(0, 0, image.width(), image.height())
            if view_crop_rect is not None and centers and all(view_crop_rect.contains(center) for center in centers):
                return view_crop_rect.intersected(image_rect), view_display_scale
            if not target_rects:
                return image_rect, None
            union = QRectF(target_rects[0])
            for rect in target_rects[1:]:
                union = union.united(rect)
            pad = max(96, int(round(max(union.width(), union.height()) * 0.45)))
            desired = union.adjusted(-pad, -pad, pad, pad)
            crop_x1 = max(0, int(math.floor(desired.left())))
            crop_y1 = max(0, int(math.floor(desired.top())))
            crop_x2 = min(image.width(), int(math.ceil(desired.right())))
            crop_y2 = min(image.height(), int(math.ceil(desired.bottom())))
            if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
                return image_rect, None
            return QRectF(crop_x1, crop_y1, crop_x2 - crop_x1, crop_y2 - crop_y1), None

        def measurement_view_crop_rect(self, image: QImage) -> tuple[QRectF | None, float | None]:
            if self.bt_view.sceneRect().isEmpty():
                return None, None
            viewport_rect = self.bt_view.viewport().rect()
            if viewport_rect.isEmpty():
                return None, None
            top_left = self.bt_view.mapToScene(viewport_rect.topLeft())
            bottom_right = self.bt_view.mapToScene(viewport_rect.bottomRight())
            visible = QRectF(top_left, bottom_right).normalized()
            image_rect = QRectF(0, 0, image.width(), image.height())
            crop = visible.intersected(image_rect)
            if crop.isEmpty():
                return None, None
            crop_x1 = max(0, int(math.floor(crop.left())))
            crop_y1 = max(0, int(math.floor(crop.top())))
            crop_x2 = min(image.width(), int(math.ceil(crop.right())))
            crop_y2 = min(image.height(), int(math.ceil(crop.bottom())))
            if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
                return None, None
            transform = self.bt_view.transform()
            display_scale = max(0.01, min(abs(transform.m11()), abs(transform.m22())))
            return QRectF(crop_x1, crop_y1, crop_x2 - crop_x1, crop_y2 - crop_y1), display_scale

        def measurement_crop_rect(
            self,
            image: QImage,
            center: tuple[float, float],
            xyxy: tuple[int, int, int, int] | None,
        ) -> QRectF:
            if xyxy is not None:
                x1, y1, x2, y2 = xyxy
                width = max(1, x2 - x1)
                height = max(1, y2 - y1)
                pad = max(96, int(round(max(width, height) * 1.6)))
                crop_x1 = x1 - pad
                crop_y1 = y1 - pad
                crop_x2 = x2 + pad
                crop_y2 = y2 + pad
            else:
                crop_w = min(image.width(), 640)
                crop_h = min(image.height(), 640)
                crop_x1 = int(round(center[0] - crop_w / 2.0))
                crop_y1 = int(round(center[1] - crop_h / 2.0))
                crop_x2 = crop_x1 + crop_w
                crop_y2 = crop_y1 + crop_h
            crop_x1 = max(0, crop_x1)
            crop_y1 = max(0, crop_y1)
            crop_x2 = min(image.width(), max(crop_x1 + 1, crop_x2))
            crop_y2 = min(image.height(), max(crop_y1 + 1, crop_y2))
            return QRectF(crop_x1, crop_y1, crop_x2 - crop_x1, crop_y2 - crop_y1)

        def measurement_updates_for_item(
            self,
            item: dict[str, Any],
            result: dict[str, object],
            image_width: int,
            image_height: int,
        ) -> dict[str, object]:
            center_values = result.get('center_pixel')
            if not isinstance(center_values, list) or len(center_values) != 2:
                return {
                    'font-size': result['font-size'],
                    'rotation': result['rotation'],
                }
            cx = float(center_values[0])
            cy = float(center_values[1])
            updates: dict[str, object] = {
                'font-size': result['font-size'],
                'rotation': result['rotation'],
            }
            xyxy = self.bt_xyxy_from_item(item)
            if xyxy is not None:
                x1, y1, x2, y2 = xyxy
                width = max(1, x2 - x1)
                height = max(1, y2 - y1)
                half_w = width / 2.0
                half_h = height / 2.0
                cx = max(half_w, min(cx, max(half_w, image_width - half_w)))
                cy = max(half_h, min(cy, max(half_h, image_height - half_h)))
                updates['xyxy_pixel'] = [
                    int(round(cx - width / 2.0)),
                    int(round(cy - height / 2.0)),
                    int(round(cx + width / 2.0)),
                    int(round(cy + height / 2.0)),
                ]
            else:
                updates['x'] = max(0.0, min(1.0, cx / max(1, image_width)))
                updates['y'] = max(0.0, min(1.0, cy / max(1, image_height)))
            return updates

        def measure_box_updates_for_bt(self, box: BoxOverlay) -> dict[str, object]:
            item = self.selected_bt_item() or {}
            updates: dict[str, object] = {
                'xyxy_pixel': list(box.xyxy_pixel),
                'orientation': box.orientation or item.get('orientation') or 'vertical',
                'match_status': 'manual',
                'match_source_block_index': box.source_block_index,
            }
            if box.measure_item_index is not None:
                updates['match_measure_item_index'] = box.measure_item_index
            if box.center_normalized is not None:
                updates['x'] = box.center_normalized[0]
                updates['y'] = box.center_normalized[1]

            if box.font_size is not None:
                font_size = max(1, int(round(float(box.font_size))))
                updates['font-size'] = font_size
            else:
                font_size = positive_int(
                    item.get('font-size'),
                    positive_int(self.font_size_spin.value(), 40),
                )

            color = str(box.text_color or self.bt_text_color(item)).lower()
            if color not in {'black', 'white'}:
                color = 'black'
            updates['color'] = '#FFFFFF' if color == 'white' else '#000000'
            updates['stroke-color'] = '#000000' if color == 'white' else '#FFFFFF'
            needs_stroke = box.text_has_stroke is True or box.need_inpaint is True
            updates['stroke-weight'] = int(np.ceil(font_size / 8.0)) if needs_stroke else 0
            updates['need_inpaint'] = box.need_inpaint is True
            return updates

        def apply_measure_box_to_selected_bt(self, box: BoxOverlay) -> bool:
            if self.has_multiple_bt_selection():
                self.status_label.setText('右側 measure 框套用只支持單個 _bt 條目；請先取消多選。')
                return False
            item = self.selected_bt_item()
            if item is None:
                self.status_label.setText('請先在左側選擇一條 _bt，再點右側 measure 框套用。')
                return False
            entry_index = item.get('index', self.selected_bt_index)
            source_index = box.source_block_index
            status = f'已將右側 measure 區塊 {source_index} 套用到左側 _bt index={entry_index}，尚未保存。'
            changed = self.apply_selected_box_updates(
                self.measure_box_updates_for_bt(box),
                status=status,
            )
            if not changed:
                self.status_label.setText(f'右側 measure 區塊 {source_index} 與左側 _bt 目前內容相同。')
            return changed

        def apply_editor_changes_to_selected_box(self) -> None:
            updates = self.selected_box_updates_from_editor()
            if updates is None:
                return
            refresh_editor = QApplication.focusWidget() is not self.bt_text_edit
            if self.has_multiple_bt_selection():
                count = len(self.selected_bt_indices_list())
                self.apply_bt_updates_to_indices(
                    self.selected_bt_indices_list(),
                    lambda _index, _item: dict(updates),
                    status=f'已批量修改 {count} 條 _bt 條目，尚未保存。',
                    refresh_editor=refresh_editor,
                )
            else:
                self.apply_selected_box_updates(
                    updates,
                    status='已修改當前 _bt 條目，尚未保存。',
                    refresh_editor=refresh_editor,
                )
            if QApplication.focusWidget() is self.bt_text_edit:
                self.sync_bt_text_popover_from_item()

        def _begin_bt_editor_preview(self) -> None:
            if self._bt_editor_preview_snapshot is not None:
                return
            page_name = self.current_page_name()
            if page_name is None:
                return
            items = self.bt_items_for_page(page_name)
            indices = self.selected_bt_indices_list()
            self._bt_editor_preview_snapshot = {
                'page_name': page_name,
                'items': {
                    index: copy.deepcopy(items[index])
                    for index in indices
                    if 0 <= index < len(items) and isinstance(items[index], dict)
                },
                'selected_index': self.selected_bt_index,
                'selected_indices': set(self.selected_bt_indices),
            }

        def _schedule_bt_editor_preview_render(self) -> None:
            if not self._bt_editor_preview_render_timer.isActive():
                self._bt_editor_preview_render_timer.start()

        def _render_bt_editor_preview(self) -> None:
            if self._bt_editor_preview_snapshot is None:
                return
            self.render_bt_page(refit=False)

        def _commit_bt_editor_preview(self, *_args: object) -> None:
            render_pending = self._bt_editor_preview_render_timer.isActive()
            self._bt_editor_preview_render_timer.stop()
            self._bt_editor_preview_commit_timer.stop()
            snapshot = self._bt_editor_preview_snapshot
            self._bt_editor_preview_snapshot = None
            if snapshot is None:
                return

            page_name = snapshot.get('page_name')
            if not isinstance(page_name, str) or page_name != self.current_page_name():
                self._bt_editor_preview_status = None
                return
            items = self.bt_items_for_page(page_name)
            before_items = snapshot.get('items')
            if not isinstance(before_items, dict):
                self._bt_editor_preview_status = None
                return
            changed_before_items = {
                index: item
                for index, item in before_items.items()
                if isinstance(index, int)
                and 0 <= index < len(items)
                and isinstance(item, dict)
                and items[index] != item
            }
            if not changed_before_items:
                self._bt_editor_preview_status = None
                return

            selected_index = snapshot.get('selected_index')
            selected_indices = snapshot.get('selected_indices')
            self.push_bt_changes_undo_snapshot(
                self._bt_editor_preview_status or '修改 _bt 字體/旋轉',
                changed_before_items,
                selected_index if isinstance(selected_index, int) else None,
                selected_indices if isinstance(selected_indices, set) else None,
            )
            self.mark_bt_dirty()
            self.update_bt_item_list()
            self.update_action_state()
            self.status_label.setText(self._bt_editor_preview_status or '已修改 _bt，尚未保存。')
            self._bt_editor_preview_status = None
            if render_pending:
                self.render_bt_page(refit=False)

        def _preview_bt_updates_to_indices(
            self,
            indices: list[int] | set[int],
            update_builder,
            *,
            status: str,
        ) -> bool:
            page_name = self.current_page_name()
            if page_name is None:
                return False
            items = self.bt_items_for_page(page_name)
            target_indices = [
                index for index in sorted(set(indices))
                if 0 <= index < len(items) and isinstance(items[index], dict)
            ]
            if not target_indices:
                return False

            changes: list[tuple[int, dict[str, object]]] = []
            for index in target_indices:
                updates = update_builder(index, items[index])
                if not updates:
                    continue
                normalized_updates = self.normalized_bt_updates(updates)
                if self.updates_change_item(items[index], normalized_updates):
                    changes.append((index, normalized_updates))
            if not changes:
                return False

            self._begin_bt_editor_preview()
            for index, normalized_updates in changes:
                items[index].update(normalized_updates)
            self._bt_editor_preview_status = status
            self._schedule_bt_editor_preview_render()
            self._bt_editor_preview_commit_timer.start()
            return True

        def _sync_typography_controls_from_preview(self) -> None:
            item = self.selected_bt_item()
            if item is None:
                return
            self._updating_editor = True
            self.font_size_spin.setValue(positive_int(item.get('font-size'), 40))
            self.rotation_spin.setValue(self.normalized_rotation(item.get('rotation')))
            self._updating_editor = False

        def handle_typography_control_changed(self, value: object) -> None:
            if self._updating_editor:
                return
            selected_indices = self.selected_bt_indices_list()
            if not selected_indices:
                return
            sender = self.sender()
            if sender is self.font_size_spin:
                size = max(1, min(999, int(value)))
                self._preview_bt_updates_to_indices(
                    selected_indices,
                    lambda _index, _item: {'font-size': size},
                    status=f'已把 {len(selected_indices)} 條 _bt 字體大小預覽為 {size}px，尚未保存。',
                )
            elif sender is self.rotation_spin:
                rotation = self.normalized_rotation(value)
                self._preview_bt_updates_to_indices(
                    selected_indices,
                    lambda _index, _item: {'rotation': rotation},
                    status=f'已把 {len(selected_indices)} 條 _bt 旋轉預覽為 {rotation:g} 度，尚未保存。',
                )

        def apply_font_size_from_table(self, row: int, column: int) -> None:
            selected_indices = self.selected_bt_indices_list()
            if not selected_indices:
                self.status_label.setText('請先在左側選擇一條 _bt 文字，再點右側字級。')
                return
            item = self.font_size_table.item(row, 0)
            if item is None:
                return
            try:
                size = int(item.text())
            except ValueError:
                return
            if len(selected_indices) > 1:
                self.apply_bt_updates_to_indices(
                    selected_indices,
                    lambda _index, _item: {'font-size': size},
                    status=f'已把 {len(selected_indices)} 條 _bt 條目字體大小改為 {size}，尚未保存。',
                )
            else:
                self.apply_selected_box_updates({'font-size': size}, status=f'已把當前 _bt 條目字體大小改為 {size}，尚未保存。')

        def nudge_selected_font_size(self, delta: int) -> None:
            selected_indices = self.selected_bt_indices_list()
            if not selected_indices:
                self.status_label.setText('請先選擇一條 _bt 文字，再使用字體大小快捷鍵。')
                return
            sign = '+' if delta > 0 else ''
            changed = self._preview_bt_updates_to_indices(
                selected_indices,
                lambda _index, item: {
                    'font-size': max(1, min(999, positive_int(item.get('font-size'), 40) + delta))
                },
                status=f'已將 {len(selected_indices)} 條 _bt 條目字體大小各自 {sign}{delta}，尚未保存。',
            )
            if changed:
                self._sync_typography_controls_from_preview()

        def nudge_selected_rotation(self, delta: float) -> None:
            selected_indices = self.selected_bt_indices_list()
            if not selected_indices:
                self.status_label.setText('請先選擇一條 _bt 文字，再使用旋轉快捷鍵。')
                return
            sign = '+' if delta > 0 else ''
            changed = self._preview_bt_updates_to_indices(
                selected_indices,
                lambda _index, item: {
                    'rotation': self.normalized_rotation(self.normalized_rotation(item.get('rotation')) + delta)
                },
                status=f'已將 {len(selected_indices)} 條 _bt 條目各自旋轉 {sign}{delta:g} 度，尚未保存。',
            )
            if changed:
                self._sync_typography_controls_from_preview()

        def nudge_selected_box_position(self, dx: int, dy: int) -> None:
            selected_indices = self.selected_bt_indices_list()
            if not selected_indices:
                self.status_label.setText('請先選擇一條 _bt 文字，再使用方向鍵移動。')
                return
            move_text = f'{dx:+d},{dy:+d}'
            if len(selected_indices) > 1:
                self.apply_bt_updates_to_indices(
                    selected_indices,
                    lambda _index, item: self.move_bt_item_updates(item, dx, dy),
                    status=f'已用方向鍵移動 {len(selected_indices)} 條 _bt 條目 {move_text}，尚未保存。',
                )
                return
            item = self.selected_bt_item()
            if item is None:
                return
            xyxy = self.bt_xyxy_from_item(item)
            if xyxy is None:
                return
            x1, y1, x2, y2 = xyxy
            new_xyxy = self.clamp_xyxy((x1 + dx, y1 + dy, x2 + dx, y2 + dy))
            if new_xyxy == xyxy:
                return
            self.apply_selected_box_updates(
                {'xyxy_pixel': list(new_xyxy), 'match_status': 'manual'},
                status=f'已用方向鍵移動當前 _bt 條目 {move_text}，尚未保存。',
            )

        def move_bt_item_updates(self, item: dict[str, Any], dx: int, dy: int) -> dict[str, object]:
            xyxy = self.bt_xyxy_from_item(item)
            if xyxy is None:
                return {}
            x1, y1, x2, y2 = xyxy
            new_xyxy = self.clamp_xyxy((x1 + dx, y1 + dy, x2 + dx, y2 + dy))
            if new_xyxy == xyxy:
                return {}
            return {'xyxy_pixel': list(new_xyxy), 'match_status': 'manual'}

        def copy_selected_box(self) -> None:
            if self.bt_data is None:
                return
            page_name = self.current_page_name()
            selected_indices = self.selected_bt_indices_list()
            if not selected_indices or page_name is None:
                self.status_label.setText('請先選擇一條 _bt 文字，再複製。')
                return
            items = self.bt_items_for_page(page_name)
            before_items = copy.deepcopy(items)
            before_selected = self.selected_bt_index
            before_selected_indices = set(self.selected_bt_indices)
            next_index = self.next_bt_index_for_page(items)
            new_items: list[dict[str, Any]] = []
            for source_index in selected_indices:
                if source_index < 0 or source_index >= len(items) or not isinstance(items[source_index], dict):
                    continue
                new_item = copy.deepcopy(items[source_index])
                new_item['index'] = next_index
                next_index += 1
                new_item['match_status'] = 'manual'
                xyxy = self.bt_xyxy_from_item(new_item)
                if xyxy is not None:
                    x1, y1, x2, y2 = xyxy
                    self.set_bt_xyxy(new_item, self.clamp_xyxy((x1 + 16, y1 + 16, x2 + 16, y2 + 16)))
                else:
                    try:
                        new_item['x'] = min(1.0, max(0.0, float(new_item.get('x')) + 0.01))
                        new_item['y'] = min(1.0, max(0.0, float(new_item.get('y')) + 0.01))
                    except (TypeError, ValueError):
                        pass
                new_items.append(new_item)
            if not new_items:
                return
            insert_at = max(selected_indices) + 1
            for offset, new_item in enumerate(new_items):
                items.insert(insert_at + offset, new_item)
            new_selection = set(range(insert_at, insert_at + len(new_items)))
            self.selected_bt_indices = new_selection
            self.selected_bt_index = insert_at
            self.push_bt_items_undo_snapshot('複製 _bt 條目', before_items, before_selected, before_selected_indices)
            self.mark_bt_dirty()
            self.populate_box_editor_for_selection()
            self.update_bt_item_list()
            self.render_bt_page(refit=False)
            if len(new_items) > 1:
                self.status_label.setText(f'已複製 {len(new_items)} 條 _bt 條目，尚未保存。')
            else:
                self.status_label.setText('已複製 _bt 條目，尚未保存。')

        def add_empty_bt_box(self) -> None:
            if self.bt_data is None:
                self.status_label.setText('請先打開 _bt.json，再新增空文案。')
                return
            page_name = self.current_page_name()
            cursor = self._bt_cursor_image_pos
            if page_name is None or cursor is None:
                self.status_label.setText('請先將鼠標移到左側圖片上，再新增空文案。')
                return
            items = self.bt_items_for_page(page_name)
            image_size = qimage_size(self.page.image_path) if self.page is not None else None
            if image_size is None:
                self.status_label.setText('目前頁面沒有可用的圖片尺寸。')
                return

            before_items = copy.deepcopy(items)
            before_selected = self.selected_bt_index
            before_selected_indices = set(self.selected_bt_indices)
            center_x = int(round(cursor[0]))
            center_y = int(round(cursor[1]))
            half_size = 15
            xyxy = self.clamp_xyxy(
                (
                    center_x - half_size,
                    center_y - half_size,
                    center_x + half_size,
                    center_y + half_size,
                )
            )
            clipboard_text = QApplication.clipboard().text()
            initial_text = clipboard_text if clipboard_text.strip() else ''
            new_item: dict[str, Any] = {
                'index': self.next_bt_index_for_page(items),
                'text': initial_text,
                'font-size': 30,
                'orientation': 'vertical',
                'rotation': 0,
                'color': '#000000',
                'stroke-color': '#FFFFFF',
                'stroke-weight': 4,
                'need_inpaint': False,
                'match_status': 'manual',
                'xyxy_pixel': list(xyxy),
            }
            self.set_bt_xyxy(new_item, xyxy)
            items.append(new_item)
            new_index = len(items) - 1
            self.selected_bt_index = new_index
            self.selected_bt_indices = {new_index}
            self.push_bt_items_undo_snapshot(
                '新增空文案',
                before_items,
                before_selected,
                before_selected_indices,
            )
            self.mark_bt_dirty()
            self.populate_box_editor_for_selection()
            self.update_bt_item_list()
            self.render_bt_page(refit=False)
            if initial_text:
                self.status_label.setText('已在鼠標位置新增文案並帶入系統剪貼簿文字，尚未保存。')
            else:
                self.status_label.setText('已在鼠標位置新增空文案，尚未保存。')

        def delete_selected_box(self) -> None:
            if self.bt_data is None:
                return
            selected_indices = self.selected_bt_indices_list()
            if not selected_indices:
                self.status_label.setText('請先選擇一條 _bt 文字，再刪除。')
                return
            page_name = self.current_page_name()
            items = self.bt_items_for_page(page_name)
            if page_name is None:
                return
            before_items = copy.deepcopy(items)
            before_selected = self.selected_bt_index
            before_selected_indices = set(self.selected_bt_indices)
            for index in sorted(selected_indices, reverse=True):
                if 0 <= index < len(items):
                    del items[index]
            self.selected_bt_index = None
            self.selected_bt_indices.clear()
            self.push_bt_items_undo_snapshot('刪除 _bt 條目', before_items, before_selected, before_selected_indices)
            self.mark_bt_dirty()
            self.set_box_editor_enabled(False)
            self.update_bt_item_list()
            self.render_bt_page(refit=False)
            if len(selected_indices) > 1:
                self.status_label.setText(f'已刪除 {len(selected_indices)} 條 _bt 條目，尚未保存。')
            else:
                self.status_label.setText('已刪除 _bt 條目，尚未保存。')

        def update_action_state(self) -> None:
            can_save = self.bt_data is not None and self.bt_dirty
            self.save_button.setEnabled(can_save)
            if self.save_action is not None:
                self.save_action.setEnabled(can_save)
            if self.undo_action is not None:
                self.undo_action.setEnabled(bool(self.bt_undo_stack))
            has_pages = bool(self.page_names)
            if self.prev_page_action is not None:
                self.prev_page_action.setEnabled(has_pages and self.page_list.currentRow() > 0)
            if self.next_page_action is not None:
                self.next_page_action.setEnabled(has_pages and self.page_list.currentRow() < len(self.page_names) - 1)
            has_selected_bt = bool(self.selected_bt_indices_list())
            if self.increase_font_action is not None:
                self.increase_font_action.setEnabled(has_selected_bt)
            if self.decrease_font_action is not None:
                self.decrease_font_action.setEnabled(has_selected_bt)
            if self.increase_font_10_action is not None:
                self.increase_font_10_action.setEnabled(has_selected_bt)
            if self.decrease_font_10_action is not None:
                self.decrease_font_10_action.setEnabled(has_selected_bt)
            if self.rotate_counterclockwise_action is not None:
                self.rotate_counterclockwise_action.setEnabled(has_selected_bt)
            if self.rotate_clockwise_action is not None:
                self.rotate_clockwise_action.setEnabled(has_selected_bt)
            if self.rotate_counterclockwise_5_action is not None:
                self.rotate_counterclockwise_5_action.setEnabled(has_selected_bt)
            if self.rotate_clockwise_5_action is not None:
                self.rotate_clockwise_5_action.setEnabled(has_selected_bt)
            if self.copy_box_action is not None:
                self.copy_box_action.setEnabled(has_selected_bt)
            if hasattr(self, 'copy_to_clipboard_button'):
                self.copy_to_clipboard_button.setEnabled(has_selected_bt)
            if self.copy_to_memory_action is not None:
                self.copy_to_memory_action.setEnabled(has_selected_bt)
            has_memory_clipboard_item = self.memory_bt_clipboard_item is not None
            can_paste_from_memory = (
                self.bt_data is not None
                and self.current_page_name() is not None
                and self._bt_cursor_image_pos is not None
                and has_memory_clipboard_item
            )
            if self.paste_from_memory_action is not None:
                self.paste_from_memory_action.setEnabled(can_paste_from_memory)
            if hasattr(self, 'delete_clipboard_button'):
                list_item = self.bt_clipboard_list.currentItem()
                list_index = list_item.data(Qt.ItemDataRole.UserRole) if list_item is not None else None
                self.delete_clipboard_button.setEnabled(isinstance(list_index, int))
            if self.measure_angle_action is not None:
                self.measure_angle_action.setEnabled(has_selected_bt)
            can_add_empty_box = (
                self.bt_data is not None
                and self.current_page_name() is not None
                and self._bt_cursor_image_pos is not None
            )
            if self.add_empty_box_action is not None:
                self.add_empty_box_action.setEnabled(can_add_empty_box)
            if hasattr(self, 'add_empty_bt_button'):
                self.add_empty_bt_button.setEnabled(can_add_empty_box)
            if self.delete_box_action is not None:
                self.delete_box_action.setEnabled(has_selected_bt)
            for action in self.move_box_actions:
                action.setEnabled(has_selected_bt)
            suffix = ' *' if self.bt_dirty else ''
            self.setWindowTitle(f'CTD / MEO BT 編輯器{suffix}')

        def mark_measure_dirty(self) -> None:
            self.measure_dirty = False
            self.update_action_state()

        def save_pending_changes(self, *_, auto: bool = False) -> bool:
            self._commit_bt_editor_preview()
            if self.bt_data is None:
                if not auto:
                    self.status_label.setText('目前沒有載入 _bt.json。')
                return True
            if not self.bt_dirty:
                if not auto:
                    self.status_label.setText('目前沒有需要保存的 _bt 修改。')
                return True
            try:
                self.save_bt_json()
            except Exception as exc:
                show_exception_details(self, '保存失敗', '無法寫入 _bt.json。下方是完整可複製的出錯信息。', exc)
                return False
            self.status_label.setText(f'已保存：{self.bt_path}')
            return True

        def build_box_updates(self, box: BoxOverlay, updates: dict[str, object]) -> dict[str, object]:
            result = dict(updates)
            if 'xyxy_pixel' in result:
                xyxy = tuple(int(v) for v in result['xyxy_pixel'])
                result['xyxy_pixel'] = list(self.clamp_xyxy(xyxy))
                image_size = qimage_size(self.page.image_path) if self.page is not None else None
                center = normalized_center_from_xyxy(tuple(result['xyxy_pixel']), image_size)
                if center is not None:
                    result['center_normalized'] = list(center)
            return result

        def values_equal(self, left, right) -> bool:
            if isinstance(left, (int, float)) and isinstance(right, (int, float)):
                return abs(float(left) - float(right)) < 1e-6
            return left == right

        def updates_change_item(self, item: dict, updates: dict[str, object]) -> bool:
            return any(not self.values_equal(item.get(key), value) for key, value in updates.items())

        def normalized_bt_updates(self, updates: dict[str, object]) -> dict[str, object]:
            normalized_updates = dict(updates)
            if 'xyxy_pixel' in normalized_updates:
                xyxy = tuple(int(v) for v in normalized_updates['xyxy_pixel'])
                normalized_updates['xyxy_pixel'] = list(self.clamp_xyxy(xyxy))
            if 'rotation' in normalized_updates:
                normalized_updates['rotation'] = self.normalized_rotation(normalized_updates['rotation'])
            return normalized_updates

        def apply_bt_updates_to_indices(
            self,
            indices: list[int] | set[int],
            update_builder,
            *,
            status: str = '已修改，尚未保存。',
            refresh_editor: bool = True,
        ) -> bool:
            self._commit_bt_editor_preview()
            page_name = self.current_page_name()
            if page_name is None:
                return False
            items = self.bt_items_for_page(page_name)
            target_indices = [
                index for index in sorted(set(indices))
                if 0 <= index < len(items) and isinstance(items[index], dict)
            ]
            if not target_indices:
                return False

            changes: list[tuple[int, dict[str, object]]] = []
            for index in target_indices:
                updates = update_builder(index, items[index])
                if not updates:
                    continue
                normalized_updates = self.normalized_bt_updates(updates)
                if self.updates_change_item(items[index], normalized_updates):
                    changes.append((index, normalized_updates))
            if not changes:
                return False

            before_items = {
                index: copy.deepcopy(items[index])
                for index, _updates in changes
                if 0 <= index < len(items) and isinstance(items[index], dict)
            }
            before_selected = self.selected_bt_index
            before_selected_indices = set(self.selected_bt_indices)
            self.push_bt_changes_undo_snapshot(status, before_items, before_selected, before_selected_indices)
            for index, normalized_updates in changes:
                item = items[index]
                if 'xyxy_pixel' in normalized_updates:
                    self.set_bt_xyxy(item, tuple(int(v) for v in normalized_updates.pop('xyxy_pixel')))
                item.update(normalized_updates)
            self.mark_bt_dirty()
            if refresh_editor:
                self.populate_box_editor_for_selection()
            self.update_bt_item_list()
            self.render_bt_page(refit=False)
            self.status_label.setText(status)
            return True

        def sync_box_from_measure_item(self, box: BoxOverlay, item: dict) -> None:
            xyxy = xyxy_from_item(item)
            if xyxy is not None:
                box.xyxy_pixel = xyxy
            image_size = qimage_size(self.page.image_path) if self.page is not None else None
            box.center_normalized = tuple_center(item.get('center_normalized')) or normalized_center_from_xyxy(box.xyxy_pixel, image_size)
            box.font_size = float(item['font_size']) if item.get('font_size') is not None else None
            box.text_color = item.get('text_color')
            box.text_has_stroke = (
                bool(item.get('text_has_stroke'))
                if item.get('text_has_stroke') is not None
                else None
            )
            box.need_inpaint = (
                bool(item.get('need_inpaint'))
                if item.get('need_inpaint') is not None
                else None
            )
            box.raw_measure = dict(item)

        def apply_selected_box_updates(
            self,
            updates: dict[str, object],
            *,
            status: str = '已修改，尚未保存。',
            refresh_editor: bool = True,
        ) -> bool:
            if self.selected_bt_item() is None or self.selected_bt_index is None:
                return False
            return self.apply_bt_updates_to_indices(
                [self.selected_bt_index],
                lambda _index, _item: dict(updates),
                status=status,
                refresh_editor=refresh_editor,
            )

        def undo_last_edit(self) -> None:
            if self.bt_data is None or not self.bt_undo_stack:
                self.status_label.setText('沒有可撤銷的 _bt 修改。')
                self.update_action_state()
                return
            entry = self.bt_undo_stack.pop()
            page_name = str(entry.get('page_name') or '')
            if self.page is None or page_name != self.page.page_name:
                self.status_label.setText('撤銷只支持當前頁；已忽略其它頁面的撤銷記錄。')
                self.update_action_state()
                return
            if isinstance(entry.get('changes'), list):
                items = self.bt_items_for_page(page_name)
                for change in entry['changes']:
                    if not isinstance(change, dict):
                        continue
                    item_index = change.get('index')
                    before_item = change.get('item')
                    if (
                        isinstance(item_index, int)
                        and 0 <= item_index < len(items)
                        and isinstance(before_item, dict)
                    ):
                        items[item_index] = copy.deepcopy(before_item)
                selected_index = entry.get('selected_index')
                selected_indices = entry.get('selected_indices')
                restored_indices = {
                    index for index in selected_indices
                    if isinstance(index, int) and 0 <= index < len(items)
                } if isinstance(selected_indices, list) else set()
                if isinstance(selected_index, int) and 0 <= selected_index < len(items):
                    restored_indices.add(selected_index)
                    self.selected_bt_index = selected_index
                else:
                    self.selected_bt_index = self.active_bt_index_from_selection(restored_indices)
                self.selected_bt_indices = restored_indices
                self.mark_bt_dirty()
                if self.selected_bt_items():
                    self.populate_box_editor_for_selection()
                else:
                    self.set_box_editor_enabled(False)
                self.update_bt_item_list()
                self.render_bt_page(refit=False)
                self.status_label.setText('已撤銷上一個 _bt 修改，尚未保存。')
                return
            if isinstance(entry.get('items'), list):
                items = self.bt_items_for_page(page_name)
                items[:] = copy.deepcopy(entry.get('items') or [])
                selected_index = entry.get('selected_index')
                selected_indices = entry.get('selected_indices')
                restored_indices = {
                    index for index in selected_indices
                    if isinstance(index, int) and 0 <= index < len(items)
                } if isinstance(selected_indices, list) else set()
                if isinstance(selected_index, int) and 0 <= selected_index < len(items):
                    restored_indices.add(selected_index)
                    self.selected_bt_index = selected_index
                else:
                    self.selected_bt_index = self.active_bt_index_from_selection(restored_indices)
                self.selected_bt_indices = restored_indices
                self.mark_bt_dirty()
                if self.selected_bt_items():
                    self.populate_box_editor_for_selection()
                else:
                    self.set_box_editor_enabled(False)
                self.update_bt_item_list()
                self.render_bt_page(refit=False)
                self.status_label.setText('已撤銷上一個 _bt 修改，尚未保存。')
                return
            item = copy.deepcopy(entry.get('item') or {})
            item_index = entry.get('item_index')
            items = self.bt_items_for_page(page_name)
            if isinstance(item_index, int) and 0 <= item_index < len(items):
                items[item_index] = item
            else:
                insert_index = item_index if isinstance(item_index, int) else len(items)
                items.insert(max(0, min(insert_index, len(items))), item)
            self.selected_bt_index = item_index if isinstance(item_index, int) else None
            self.selected_bt_indices = {self.selected_bt_index} if self.selected_bt_index is not None else set()
            self.mark_bt_dirty()
            if self.selected_bt_item() is not None:
                self.populate_box_editor_from_bt(self.selected_bt_item())
            self.update_bt_item_list()
            self.render_bt_page(refit=False)
            self.status_label.setText('已撤銷上一個 _bt 修改，尚未保存。')

        def refresh_current_page_from_measure(
            self,
            *,
            selected_source_index: int | None = None,
            status: str | None = None,
            refit: bool = False,
        ) -> None:
            if self.processor is None or self.page is None:
                self.update_action_state()
                return
            page_name = self.page.page_name
            if selected_source_index is None:
                box = self.selected_box()
                selected_source_index = box.source_block_index if box is not None else None
            image_size = qimage_size(self.processor.image_dir / page_name)
            self.page = self.processor.load_page(page_name, image_size=image_size)
            self.selected_box_index = None
            if selected_source_index is not None:
                for index, box in enumerate(self.page.boxes):
                    if box.source_block_index == selected_source_index:
                        self.selected_box_index = index
                        break
            if self.selected_box_index is not None:
                pass
            else:
                self.selected_box_index = None
            self.update_font_size_list()
            self.render_current_page(refit=refit)
            if status:
                self.status_label.setText(status)
            self.update_action_state()

        def hit_test_box(self, x: float, y: float) -> tuple[int | None, str | None]:
            if self.page is None:
                return None, None
            handles = (
                ('tl', -1, -1), ('t', 0, -1), ('tr', 1, -1),
                ('l', -1, 0), ('r', 1, 0),
                ('bl', -1, 1), ('b', 0, 1), ('br', 1, 1),
            )
            tolerance = 8.0
            matches = []
            for index, box in enumerate(self.page.boxes):
                x1, y1, x2, y2 = box.xyxy_pixel
                for mode, sx, sy in handles:
                    hx = (x1 + x2) / 2 if sx == 0 else (x1 if sx < 0 else x2)
                    hy = (y1 + y2) / 2 if sy == 0 else (y1 if sy < 0 else y2)
                    if abs(x - hx) <= tolerance and abs(y - hy) <= tolerance:
                        matches.append((0, index, mode))
                if x1 <= x <= x2 and y1 <= y <= y2:
                    area = max(0, x2 - x1) * max(0, y2 - y1)
                    matches.append((area, index, 'move'))
            if not matches:
                return None, None
            _, index, mode = min(matches, key=lambda item: item[0])
            return index, mode

        def handle_image_mouse_press(self, x: float, y: float) -> None:
            index, _mode = self.hit_test_box(x, y)
            self.select_box(index)
            self._box_drag_mode = None
            self._box_drag_start = None
            self._box_drag_original = None
            self._box_drag_temporary = False
            if index is not None:
                box = self.page.boxes[index] if self.page is not None else None
                if box is not None and self.selected_bt_item() is not None:
                    self.apply_measure_box_to_selected_bt(box)
                else:
                    self.status_label.setText('已選中 CTD measure 框；請先在左側選擇 _bt 條目後再點右側套用。')

        def handle_image_mouse_drag(self, x: float, y: float) -> None:
            return

        def handle_image_mouse_release(self, x: float, y: float) -> None:
            self._box_drag_mode = None
            self._box_drag_start = None
            self._box_drag_original = None
            self._box_drag_temporary = False

        def clamp_xyxy(self, xyxy: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
            if self.page is None:
                return xyxy
            image_size = self._bt_cached_image_size or qimage_size(self.page.image_path)
            width, height = image_size or (10_000, 10_000)
            x1, y1, x2, y2 = xyxy
            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(1, min(x2, width))
            y2 = max(1, min(y2, height))
            if x2 - x1 < 4:
                if self._box_drag_mode and 'l' in self._box_drag_mode:
                    x1 = max(0, x2 - 4)
                else:
                    x2 = min(width, x1 + 4)
            if y2 - y1 < 4:
                if self._box_drag_mode and 't' in self._box_drag_mode:
                    y1 = max(0, y2 - 4)
                else:
                    y2 = min(height, y1 + 4)
            return x1, y1, x2, y2

        def _summary_text(self, summary: dict) -> str:
            if summary.get('has_overlay_data'):
                return (
                    f'已載入 {summary["pages"]} 頁，'
                    f'{summary["boxes"]} 個區塊，{summary["lines"]} 條文字行。'
                )
            issues = summary.get('overlay_data_issues') or []
            missing = summary.get('missing_required_data') or []
            issue_text = '、'.join(issues) if issues else '、'.join(Path(path).name for path in missing)
            return (
                f'已載入 {summary["source_images"]} 張原圖，但尚未找到完整 CTD 疊圖資料。'
                f'\n問題：{issue_text or "ctd 資料不完整"}'
                '\n請按「生成/更新 CTD」建立資料，或選擇已產生 ctd/ 的資料夾。'
            )

        def import_labelplus_txt(self) -> None:
            if self.processor is None:
                QMessageBox.information(self, '尚未選擇資料夾', '請先選擇包含原圖的圖片資料夾。')
                return
            if not self.processor.ctd_measure_path.is_file():
                QMessageBox.information(self, '缺少 measure.json', '請先生成 CTD，確保 ctd/measure.json 已存在。')
                return

            start_dir = self.current_image_dir or self._last_existing_image_dir() or str(Path.home())
            txt_path, _ = QFileDialog.getOpenFileName(
                self,
                '導入 LabelPlus txt',
                start_dir,
                'LabelPlus/Text (*.txt);;All Files (*)',
            )
            if not txt_path:
                return

            try:
                result = build_bt_from_labelplus_txt(
                    txt_path,
                    self.processor.ctd_measure_path,
                    self.processor.image_dir,
                )
            except Exception as exc:
                show_exception_details(
                    self,
                    '導入失敗',
                    '無法由 LabelPlus txt 生成 _meo_bt.json。下方是完整可複製的出錯信息。',
                    exc,
                )
                return

            filtered_path = result.get('filtered_path')
            filtered_text = f'\n{filtered_path}' if filtered_path is not None else ''
            message = (
                '已生成：\n'
                f'{result["meo_path"]}'
                f'{filtered_text}\n'
                f'{result["bt_path"]}\n\n'
                f'頁數：{result["pages"]}，條目：{result["labels"]}\n'
                f'未匹配：{result["unmatched_pages"]} 頁，{result["unmatched_labels"]} 條'
            )
            self.status_label.setText(f'已生成 {Path(result["bt_path"]).name}')
            self.load_bt_json_path(Path(result['bt_path']))
            QMessageBox.information(self, '生成完成', message)

        def generate_ctd(self) -> None:
            if not self.current_image_dir:
                QMessageBox.information(self, '尚未選擇資料夾', '請先選擇包含原圖的圖片資料夾。')
                return
            if self.detect_process is not None or self.font_calibration_process is not None:
                QMessageBox.information(self, '正在處理', 'CTD 資料正在生成中，請稍候。')
                return

            script = Path(__file__).resolve().parent / 'run_detection.py'
            if not script.is_file():
                QMessageBox.critical(self, '找不到生成器', f'找不到：\n{script}')
                return

            calibration_settings = self._ask_generated_font_calibration_settings()
            if calibration_settings is None:
                return
            self.generated_default_font_size, self.generated_font_size_step = calibration_settings
            self.generated_font_size_method = (
                'ocr_aligned' if self.ocr_aligned_font_radio.isChecked() else 'char_box'
            )
            self._capture_generated_calibration_backup()

            self.generate_button.setEnabled(False)
            method_label = (
                'OCR 對齊逐字計算'
                if self.generated_font_size_method == 'ocr_aligned'
                else '單字框計算'
            )
            self.status_label.setText(
                f'正在生成 CTD 資料（{method_label}），'
                f'字級候選基準 {self.generated_default_font_size:.1f}、'
                f'Step {self.generated_font_size_step:.1f}...'
            )
            process = QProcess(self)
            args = [
                str(script),
                self.current_image_dir,
                '--font-size-calculation-method',
                self.generated_font_size_method,
                '--default-font-size',
                f'{self.generated_default_font_size:.1f}',
                '--font-size-step',
                f'{self.generated_font_size_step:.1f}',
            ]
            self.detect_command = [sys.executable, *args]
            self.detect_output_chunks = []
            process.setProgram(sys.executable)
            process.setArguments(args)
            process.setWorkingDirectory(str(Path(__file__).resolve().parents[1]))
            process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
            process.readyReadStandardOutput.connect(self._read_detection_output)
            process.errorOccurred.connect(self._detection_process_error)
            process.finished.connect(self._detection_finished)
            self.detect_process = process
            process.start()

        def _ask_generated_font_calibration_settings(self) -> tuple[float, float] | None:
            dialog = QDialog(self)
            dialog.setWindowTitle('生成／更新 CTD 字級設定')
            layout = QVBoxLayout(dialog)
            if self.ocr_aligned_font_radio.isChecked():
                description_text = (
                    'OCR 對齊逐字計算會先排除不可靠字元，再從「預設字級＋Step」形成的候選中選擇總誤差最低者。'
                )
            else:
                description_text = (
                    '單字框計算沿用原有遮罩切框與段落字級估算，最後依「預設字級＋Step」選擇最接近的候選字級。'
                )
            description = QLabel(description_text)
            description.setWordWrap(True)
            layout.addWidget(description)

            default_row = QHBoxLayout()
            default_row.addWidget(QLabel('預設字級'))
            default_spin = QDoubleSpinBox()
            default_spin.setRange(0.1, 999.0)
            default_spin.setDecimals(1)
            default_spin.setSingleStep(0.1)
            try:
                saved_default = float(self.settings.value('font_calibration/default_font_size', DEFAULT_FONT_SIZE_BASE))
            except (TypeError, ValueError):
                saved_default = DEFAULT_FONT_SIZE_BASE
            default_spin.setValue(max(0.1, min(999.0, saved_default)))
            default_row.addWidget(default_spin, 1)
            layout.addLayout(default_row)

            step_row = QHBoxLayout()
            step_row.addWidget(QLabel('Step'))
            step_spin = QDoubleSpinBox()
            step_spin.setRange(0.1, 999.0)
            step_spin.setDecimals(1)
            step_spin.setSingleStep(0.1)
            try:
                saved_step = float(self.settings.value('font_calibration/font_size_step', DEFAULT_FONT_SIZE_STEP))
            except (TypeError, ValueError):
                saved_step = DEFAULT_FONT_SIZE_STEP
            step_spin.setValue(max(0.1, min(999.0, saved_step)))
            step_row.addWidget(step_spin, 1)
            layout.addLayout(step_row)

            preview = QLabel()
            preview.setWordWrap(True)

            def update_preview() -> None:
                base = float(default_spin.value())
                step = float(step_spin.value())
                preview.setText(
                    '候選示例：'
                    + '、'.join(f'{base + offset * step:.1f}' for offset in range(-2, 3) if base + offset * step > 0)
                )

            default_spin.valueChanged.connect(update_preview)
            step_spin.valueChanged.connect(update_preview)
            update_preview()
            layout.addWidget(preview)

            buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
            buttons.button(QDialogButtonBox.StandardButton.Ok).setText('開始生成')
            buttons.button(QDialogButtonBox.StandardButton.Cancel).setText('取消')
            buttons.accepted.connect(dialog.accept)
            buttons.rejected.connect(dialog.reject)
            layout.addWidget(buttons)
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return None
            default_font_size = round(float(default_spin.value()), 1)
            font_size_step = round(float(step_spin.value()), 1)
            self.settings.setValue('font_calibration/default_font_size', default_font_size)
            self.settings.setValue('font_calibration/font_size_step', font_size_step)
            return default_font_size, font_size_step

        def _generated_calibration_paths(self) -> tuple[Path, ...]:
            if not self.current_image_dir:
                return ()
            ctd_dir = Path(self.current_image_dir) / 'ctd'
            return (
                ctd_dir / 'measure.json',
                ctd_dir / 'measure.debug.json',
                ctd_dir / 'measure_ocr.json',
            )

        def _capture_generated_calibration_backup(self) -> None:
            self.generated_calibration_backup = {
                path: path.read_bytes() if path.is_file() else None
                for path in self._generated_calibration_paths()
            }

        def _restore_generated_calibration_backup(self) -> None:
            backup = self.generated_calibration_backup
            self.generated_calibration_backup = None
            if backup is None:
                return
            for path, content in backup.items():
                if content is None:
                    path.unlink(missing_ok=True)
                    continue
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)

        def _clear_generated_calibration_backup(self) -> None:
            self.generated_calibration_backup = None

        def _read_detection_output(self) -> None:
            if self.detect_process is None:
                return
            text = bytes(self.detect_process.readAllStandardOutput()).decode('utf-8', errors='replace')
            if text:
                self.detect_output_chunks.append(text)
                lines = text.strip().splitlines()
                if lines:
                    self.status_label.setText(lines[-1])

        def _detection_process_error(self, error) -> None:
            process = self.detect_process
            command_text = ' '.join(self.detect_command)
            details = (
                '【命令】\n'
                f'{command_text}\n\n'
                '【工作目錄】\n'
                f'{Path(__file__).resolve().parents[1]}\n\n'
                '【Python】\n'
                f'{sys.executable}\n\n'
                '【QProcess 錯誤】\n'
                f'{error}\n\n'
                '【完整輸出】\n'
                f'{"".join(self.detect_output_chunks).strip() or "(沒有輸出)"}'
            )
            show_error_details(
                self,
                '生成 CTD 失敗',
                '生成程序無法啟動或執行中斷。下方是完整可複製的出錯信息。',
                details,
            )
            self.generate_button.setEnabled(True)
            self.status_label.setText('生成 CTD 失敗。')
            self._restore_generated_calibration_backup()
            self.detect_process = None
            if process is not None:
                process.deleteLater()

        def _detection_finished(self, exit_code: int, exit_status) -> None:
            process = self.detect_process
            self.detect_process = None
            if process is None:
                return
            trailing_output = bytes(process.readAllStandardOutput()).decode('utf-8', errors='replace')
            if trailing_output:
                self.detect_output_chunks.append(trailing_output)
            output = ''.join(self.detect_output_chunks).strip()
            if exit_code != 0:
                command_text = ' '.join(self.detect_command)
                details = (
                    '【命令】\n'
                    f'{command_text}\n\n'
                    '【工作目錄】\n'
                    f'{Path(__file__).resolve().parents[1]}\n\n'
                    '【Python】\n'
                    f'{sys.executable}\n\n'
                    '【退出碼】\n'
                    f'{exit_code}\n\n'
                    '【完整輸出】\n'
                    f'{output or "(沒有輸出)"}'
                )
                show_error_details(
                    self,
                    '生成 CTD 失敗',
                    f'生成失敗，退出碼：{exit_code}。\n下方是完整可複製的出錯信息。',
                    details,
                )
                self.status_label.setText('生成 CTD 失敗。')
                self.generate_button.setEnabled(True)
                self._restore_generated_calibration_backup()
                return
            if self.generated_font_size_method == 'ocr_aligned':
                self.status_label.setText('CTD 資料生成完成，正在啟動 mit48 字級校準...')
                self._start_generated_font_calibration()
                return
            self._clear_generated_calibration_backup()
            self.generate_button.setEnabled(True)
            self.status_label.setText('CTD 與單字框字級計算完成，正在重新載入...')
            if self.current_image_dir:
                self.load_folder(self.current_image_dir)

        def _start_generated_font_calibration(self) -> None:
            if not self.current_image_dir:
                self.generate_button.setEnabled(True)
                self._restore_generated_calibration_backup()
                return
            project_root = Path(__file__).resolve().parents[1]
            script = project_root / 'measure_ocr.py'
            if not script.is_file():
                self.generate_button.setEnabled(True)
                self._restore_generated_calibration_backup()
                QMessageBox.critical(self, '找不到 OCR 腳本', f'找不到：\n{script}')
                return
            device = str(self.auto_calibrate_device_combo.currentData() or 'cpu')
            args = [
                '-u',
                str(script),
                self.current_image_dir,
                '--device',
                device,
                '--apply-font-sizes',
                '--default-font-size',
                f'{self.generated_default_font_size:.1f}',
                '--font-size-step',
                f'{self.generated_font_size_step:.1f}',
            ]
            process = QProcess(self)
            process.setProgram(sys.executable)
            process.setArguments(args)
            process.setWorkingDirectory(str(project_root))
            process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
            process.readyReadStandardOutput.connect(self._read_font_calibration_output)
            process.errorOccurred.connect(self._font_calibration_process_error)
            process.finished.connect(self._font_calibration_finished)
            self.font_calibration_process = process
            self.font_calibration_command = [sys.executable, *args]
            self.font_calibration_output_chunks = []
            progress_text = f'字級 OCR：正在以 {device.upper()} 載入 mit48 模型...'
            self.status_label.setText(progress_text)
            self.font_calibration_progress_label.setText(progress_text)
            self.font_calibration_progress_label.show()
            process.start()

        def _read_font_calibration_output(self) -> None:
            if self.font_calibration_process is None:
                return
            text = bytes(self.font_calibration_process.readAllStandardOutput()).decode('utf-8', errors='replace')
            if not text:
                return
            self.font_calibration_output_chunks.append(text)
            lines = text.strip().splitlines()
            if lines:
                progress_text = lines[-1]
                self.status_label.setText(progress_text)
                self.font_calibration_progress_label.setText(progress_text)
                self.font_calibration_progress_label.show()

        def _font_calibration_process_error(self, error) -> None:
            process = self.font_calibration_process
            command_text = ' '.join(self.font_calibration_command)
            details = (
                '【命令】\n'
                f'{command_text}\n\n'
                '【QProcess 錯誤】\n'
                f'{error}\n\n'
                '【完整輸出】\n'
                f'{"".join(self.font_calibration_output_chunks).strip() or "(沒有輸出)"}'
            )
            show_error_details(
                self,
                '自動字級校準失敗',
                'CTD 已生成，但 mit48 字級校準程序中斷。可稍後在 measure 編輯器中重新執行。',
                details,
            )
            self.font_calibration_process = None
            self.generate_button.setEnabled(True)
            self.font_calibration_progress_label.hide()
            self._restore_generated_calibration_backup()
            self.status_label.setText('OCR 字級校準失敗，已恢復執行前的字級結果。')
            if process is not None:
                process.deleteLater()
            if self.current_image_dir:
                self.load_folder(self.current_image_dir)

        def _font_calibration_finished(self, exit_code: int, exit_status) -> None:
            process = self.font_calibration_process
            self.font_calibration_process = None
            if process is None:
                return
            trailing_output = bytes(process.readAllStandardOutput()).decode('utf-8', errors='replace')
            if trailing_output:
                self.font_calibration_output_chunks.append(trailing_output)
            process.deleteLater()
            if exit_code != 0:
                details = (
                    '【命令】\n'
                    f'{" ".join(self.font_calibration_command)}\n\n'
                    '【退出碼】\n'
                    f'{exit_code}\n\n'
                    '【完整輸出】\n'
                    f'{"".join(self.font_calibration_output_chunks).strip() or "(沒有輸出)"}'
                )
                show_error_details(
                    self,
                    '自動字級校準失敗',
                    'CTD 已生成，但 mit48 字級校準失敗。可稍後在 measure 編輯器中重新執行。',
                    details,
                )
                self._restore_generated_calibration_backup()
                self.status_label.setText('OCR 字級校準失敗，已恢復執行前的字級結果。')
            else:
                self._clear_generated_calibration_backup()
                self.status_label.setText('CTD 與字型緩存字級校準完成，正在重新載入...')
            self.generate_button.setEnabled(True)
            self.font_calibration_progress_label.hide()
            if self.current_image_dir:
                self.load_folder(self.current_image_dir)

        def load_page_at_row(self, row: int) -> None:
            if row < 0 or row >= len(self.page_names) or self.processor is None:
                return
            self._commit_bt_editor_preview()
            page_name = self.page_names[row]
            try:
                self._clear_bt_image_cache()
                self.clear_hover_char_box(render=False)
                self.selected_box_index = None
                self.selected_bt_index = None
                self.selected_bt_indices.clear()
                self.show_bt_inpainted = True
                self.set_box_editor_enabled(False)
                image_size = qimage_size(self.processor.image_dir / page_name)
                self._bt_cached_image_size = image_size
                self.page = self.processor.load_page(page_name, image_size=image_size)
                self.current_page_row = row
                self.update_font_size_list()
                self.render_current_page(refit=False)
                self.select_bt_item(None)
                self.update_bt_item_list()
                self.render_bt_page(refit=False)
                self.fit_both_views()
                self.schedule_bt_html_overlay_update()
            except Exception as exc:
                show_exception_details(self, '頁面載入失敗', '無法載入此頁。下方是完整可複製的出錯信息。', exc)

        def update_hover_char_box(self, x: float, y: float) -> None:
            if self.page is None or not self.show_char_boxes.isChecked():
                self.clear_hover_char_box()
                return

            matches = []
            for item in self.page.char_boxes:
                bbox = item.get('bbox')
                if not isinstance(bbox, list) or len(bbox) != 4:
                    continue
                x1, y1, x2, y2 = [float(value) for value in bbox]
                if x1 <= x <= x2 and y1 <= y <= y2:
                    matches.append((max(0.0, x2 - x1) * max(0.0, y2 - y1), item))

            if not matches:
                self.clear_hover_char_box()
                return

            item = min(matches, key=lambda entry: entry[0])[1]
            if item is self.hover_char_box:
                return
            self.hover_char_box = item
            self.char_info_label.setText(self._char_info_text(item))
            self.update_popover_with_char_box(item)
            self.render_current_page(refit=False)

        def clear_hover_char_box(self, render: bool = True) -> None:
            had_hover = self.hover_char_box is not None
            self.hover_char_box = None
            self.char_info_label.setText('游標單字框：未選中')
            if self._popover_bt_item is not None and self.bt_match_popover.isVisible():
                self.show_bt_match_popover(self._popover_bt_item)
            elif self.bt_match_popover.isVisible():
                self.bt_match_popover.hide()
            if render and had_hover:
                self.render_current_page(refit=False)

        def _char_info_text(self, item: dict) -> str:
            bbox = item.get('bbox')
            width_text = compact_px(item.get('width')) or '-'
            height_text = compact_px(item.get('height')) or '-'
            source_index = item.get('source_block_index', '-')
            line_index = item.get('line_index', '-')
            bbox_text = ', '.join(str(int(round(float(value)))) for value in bbox) if isinstance(bbox, list) else '-'
            try:
                estimated_font_size = float(item.get('estimated_font_size', item.get('calculated_font_size')))
                font_size_text = f'{estimated_font_size:.1f}' if estimated_font_size > 0 else '-'
            except (TypeError, ValueError):
                font_size_text = '-'
            return (
                '游標單字框：\n'
                f'寬：{width_text}px  高：{height_text}px  單字字級：{font_size_text}px\n'
                f'區塊：{source_index}  行：{line_index}\n'
                f'bbox：{bbox_text}'
            )

        def reload_current_page(self) -> None:
            row = self.page_list.currentRow()
            if row >= 0:
                self.load_page_at_row(row)

        def open_measure_editor(self) -> None:
            if self.processor is None:
                QMessageBox.information(self, '尚未載入資料夾', '請先選擇包含 ctd/measure.json 的圖片資料夾。')
                return
            if not self.processor.measure_path.is_file():
                QMessageBox.information(self, '缺少 measure.json', '請先生成 CTD，確保 ctd/measure.json 已存在。')
                return
            try:
                from .measure_editor import MeasureEditorWindow
            except ImportError:
                from measure_editor import MeasureEditorWindow
            editor = MeasureEditorWindow(self.processor, self.current_page_name(), self)
            editor.saved.connect(self.handle_measure_editor_saved)
            editor.destroyed.connect(lambda _=None, editor=editor: self.forget_measure_editor(editor))
            self.measure_editor_windows.append(editor)
            editor.show()

        def forget_measure_editor(self, editor: QMainWindow) -> None:
            if editor in self.measure_editor_windows:
                self.measure_editor_windows.remove(editor)

        def handle_measure_editor_saved(self) -> None:
            if self.current_image_dir:
                try:
                    current_name = self.current_page_name()
                    self.processor = CtdOverlayProcessor(self.current_image_dir)
                    self.page_names = self.processor.page_names()
                    if current_name in self.page_names:
                        self.load_page_at_row(self.page_names.index(current_name))
                    elif self.page_names:
                        self.load_page_at_row(max(0, min(self.current_page_row, len(self.page_names) - 1)))
                except Exception as exc:
                    show_exception_details(self, '重新載入失敗', 'measure.json 已保存，但主視窗重新載入失敗。', exc)

        def bt_inpainted_overlay_path(self, page_name: str) -> Path | None:
            if self.processor is None:
                return None
            stem = Path(page_name).stem
            candidates = [
                self.processor.ctd_dir / 'inpainted' / f'{stem}.png',
                self.processor.image_dir / 'inpainted' / f'{stem}.png',
            ]
            for path in candidates:
                if path.is_file():
                    return path
            return None

        def load_bt_source_image(self, page_name: str) -> QImage | None:
            if self.processor is None:
                return None
            image_path = self.processor.image_dir / page_name
            if not image_path.is_file() and self.page is not None:
                image_path = self.page.image_path
            if self._bt_cached_page_name == page_name and self._bt_cached_source_image is not None:
                return self._bt_cached_source_image.copy()
            image = QImage(str(image_path))
            if image.isNull():
                return None
            image = image.convertToFormat(QImage.Format.Format_RGBA8888)
            self._bt_cached_page_name = page_name
            self._bt_cached_source_image = image.copy()
            self._bt_cached_base_image = None
            self._bt_cached_base_show_inpainted = None
            return image

        def load_bt_base_image(self, page_name: str) -> QImage | None:
            if (
                self._bt_cached_page_name == page_name
                and self._bt_cached_base_image is not None
                and self._bt_cached_base_show_inpainted == self.show_bt_inpainted
            ):
                return self._bt_cached_base_image.copy()
            image = self.load_bt_source_image(page_name)
            if image is None:
                return None
            if not self.show_bt_inpainted:
                self._bt_cached_base_image = image.copy()
                self._bt_cached_base_show_inpainted = False
                return image
            overlay_path = self.bt_inpainted_overlay_path(page_name)
            if overlay_path is None:
                self._bt_cached_base_image = image.copy()
                self._bt_cached_base_show_inpainted = True
                return image
            overlay = QImage(str(overlay_path))
            if overlay.isNull():
                self._bt_cached_base_image = image.copy()
                self._bt_cached_base_show_inpainted = True
                return image
            overlay = overlay.convertToFormat(QImage.Format.Format_RGBA8888)
            if overlay.size() != image.size():
                overlay = overlay.scaled(
                    image.size(),
                    Qt.AspectRatioMode.IgnoreAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            painter = QPainter(image)
            painter.drawImage(0, 0, overlay)
            painter.end()
            self._bt_cached_base_image = image.copy()
            self._bt_cached_base_show_inpainted = True
            return image

        def render_bt_page(self, *_, refit: bool = True) -> None:
            if self.processor is None:
                return
            page_name = self.current_page_name()
            if not page_name:
                return
            base_key = (page_name, bool(self.show_bt_inpainted))
            base_changed = (
                self._bt_displayed_base_key != base_key
                or self.bt_view.current_pixmap().isNull()
            )
            if base_changed:
                image = self.load_bt_base_image(page_name)
                if image is None:
                    return
                self.bt_annotation_item.set_image_size(image.width(), image.height())
                self.bt_view.set_pixmap(QPixmap.fromImage(image), fit=refit)
                self._bt_displayed_base_key = base_key
            else:
                self.bt_annotation_item.set_image_size(
                    self.bt_view.current_pixmap().width(),
                    self.bt_view.current_pixmap().height(),
                )
                if refit and not self.bt_view.sceneRect().isEmpty():
                    self.bt_view._zoom = 1.0
                    self.bt_view.resetTransform()
                    self.bt_view.fitInView(
                        self.bt_view.sceneRect(),
                        Qt.AspectRatioMode.KeepAspectRatio,
                    )
                    self.bt_view.viewportChanged.emit(self.bt_view)
            self.bt_annotation_item.update()
            self.update_navigator()
            self.schedule_bt_html_overlay_update()

        def update_bt_html_overlay(self) -> None:
            overlay = getattr(self, 'bt_html_overlay', None)
            if overlay is None:
                return
            if self._bt_drag_active:
                overlay.suspend()
                return
            overlay.resume()
            overlay.update_geometry()
            items = self.build_bt_html_items()
            if items:
                overlay.set_items(items)
            else:
                overlay.hide()

        def schedule_bt_html_overlay_update(self) -> None:
            if getattr(self, 'bt_html_overlay', None) is None:
                return
            if getattr(self, '_bt_html_overlay_update_scheduled', False):
                return
            self._bt_html_overlay_update_scheduled = True
            QTimer.singleShot(16, self._flush_bt_html_overlay_update)

        def _flush_bt_html_overlay_update(self) -> None:
            self._bt_html_overlay_update_scheduled = False
            self.update_bt_html_overlay()

        def build_bt_html_items(self) -> list[dict[str, object]]:
            if self.bt_data is None:
                return []
            viewport_size = self.bt_view.viewport().size()
            width = max(1, viewport_size.width())
            height = max(1, viewport_size.height())
            scale = max(0.01, min(abs(self.bt_view.transform().m11()), abs(self.bt_view.transform().m22())))
            items = []
            for index, item in enumerate(self.bt_items_for_page()):
                text = str(item.get('text') or '').strip()
                if not text:
                    continue
                center = self.bt_center_pixel_from_item(item)
                if center is None:
                    continue
                point = self.bt_view.mapFromScene(QPointF(center[0], center[1]))
                if point.x() < -width or point.x() > width * 2 or point.y() < -height or point.y() > height * 2:
                    continue
                font_size = max(1, min(999, positive_int(item.get('font-size'), 40)))
                display_font_size = max(1, font_size * scale)
                color_name = self.bt_css_text_color(item)
                stroke_weight, stroke_color_name = self.bt_css_stroke(item, font_size, scale, color_name)
                font_family = self.bt_html_font_family(item)
                orientation = str(item.get('orientation') or 'vertical')
                items.append({
                    'id': str(item.get('index', index)),
                    'x': round(float(point.x()), 3),
                    'y': round(float(point.y()), 3),
                    'rotation': self.normalized_rotation(item.get('rotation')),
                    'text': self.prepare_html_bt_text(text, orientation),
                    'fontSize': round(float(display_font_size), 3),
                    'fontFamily': font_family,
                    'color': color_name,
                    'textShadow': self.bt_text_shadow_css(stroke_weight, stroke_color_name),
                    'vertical': orientation == 'vertical',
                })
            return items

        def bt_css_text_color(self, item: dict[str, Any]) -> str:
            color = str(item.get('color') or '#000000').strip()
            if not color:
                return '#000000'
            if color.lower() == 'black':
                return '#000000'
            if color.lower() == 'white':
                return '#FFFFFF'
            return color if color.startswith('#') else f'#{color}'

        def bt_css_stroke(self, item: dict[str, Any], font_size: int, scale: float, color_name: str) -> tuple[float, str]:
            stroke_weight = max(0.0, float(item.get('stroke-weight') or 0) * scale)
            stroke_color = str(item.get('stroke-color') or '').strip()
            if not stroke_color:
                stroke_color = '#000000' if color_name.lower() in {'#ffffff', 'white'} else '#FFFFFF'
            if stroke_color.lower() == 'black':
                stroke_color = '#000000'
            elif stroke_color.lower() == 'white':
                stroke_color = '#FFFFFF'
            elif stroke_color and not stroke_color.startswith('#'):
                stroke_color = f'#{stroke_color}'
            return min(stroke_weight, max(1.0, float(font_size) * scale / 2.0)), stroke_color

        def prepare_html_bt_text(self, text: str, orientation: str) -> str:
            prepared = text.replace('\\n', '\n').replace('\r\n', '\n').replace('\r', '\n')
            prepared = prepared.translate(str.maketrans({
                '「': '｢',
                '」': '｣',
                '“': '‶',
                '”': '〟',
            }))
            prepared = prepared.translate(str.maketrans({
                '!': '！',
                '"': '＂',
                '#': '＃',
                '$': '＄',
                '%': '％',
                '&': '＆',
                "'": '＇',
                '(': '（',
                ')': '）',
                '*': '＊',
                '+': '＋',
                ',': '，',
                '-': '－',
                '.': '．',
                '/': '／',
                ':': '：',
                ';': '；',
                '<': '＜',
                '=': '＝',
                '>': '＞',
                '?': '？',
                '@': '＠',
                '[': '［',
                '\\': '＼',
                ']': '］',
                '^': '＾',
                '_': '＿',
                '`': '｀',
                '{': '｛',
                '|': '｜',
                '}': '｝',
                '~': '～',
            }))
            if orientation == 'vertical':
                prepared = ''.join(
                    chr(ord(char) + 0xFEE0) if '0' <= char <= '9' else char
                    for char in prepared
                )
            return prepared

        def bt_html_font_family(self, item: dict[str, Any]) -> str:
            font = str(item.get('font') or '').strip()
            fallback = '"Noto Sans TC", "Hiragino Sans", "PingFang TC", "PingFang SC", sans-serif'
            if not font:
                return fallback
            escaped = font.replace('"', '\\"')
            return f'"{escaped}", {fallback}'

        def bt_center_pixel_from_item(
            self,
            item: dict[str, Any],
        ) -> tuple[float, float] | None:
            xyxy = self.bt_xyxy_from_item(item)
            if xyxy is not None:
                x1, y1, x2, y2 = xyxy
                return (x1 + x2) / 2.0, (y1 + y2) / 2.0
            try:
                x = float(item.get('x'))
                y = float(item.get('y'))
            except (TypeError, ValueError):
                x = y = None
            if x is not None and y is not None:
                pixmap = self.bt_view.pixmap_item.pixmap()
                image_width = pixmap.width()
                image_height = pixmap.height()
                return x * image_width, y * image_height
            return None

        def bt_text_shadow_css(self, stroke_weight: float, stroke_color_name: str) -> str:
            if stroke_weight <= 0:
                return ''
            shadows = []
            for angle in range(0, 360, 45):
                rad = np.deg2rad(angle)
                x = round(float(np.cos(rad) * stroke_weight), 2)
                y = round(float(np.sin(rad) * stroke_weight), 2)
                shadows.append(f'{x}px {y}px 0 {stroke_color_name}')
            for angle in range(22, 360, 45):
                rad = np.deg2rad(angle)
                x = round(float(np.cos(rad) * stroke_weight), 2)
                y = round(float(np.sin(rad) * stroke_weight), 2)
                shadows.append(f'{x}px {y}px 0 {stroke_color_name}')
            return 'text-shadow:' + ','.join(shadows) + ';'

        def _draw_bt_items(self, painter: QPainter, image_width: int, image_height: int) -> None:
            items = self.bt_items_for_page()
            if not items:
                font = QFont('Helvetica', 24)
                font.setBold(True)
                painter.setFont(font)
                painter.setPen(QPen(QColor(245, 245, 245), 1))
                painter.drawText(
                    QRectF(0, 0, image_width, image_height),
                    Qt.AlignmentFlag.AlignCenter,
                    '未載入 _bt.json',
                )
                return

            selected_indices = set(self.selected_bt_indices_list())
            multiple_selected = len(selected_indices) > 1
            dragging = self._bt_drag_active
            for index, item in enumerate(items):
                xyxy = self.bt_xyxy_from_item(item)
                if xyxy is None:
                    continue
                x1, y1, x2, y2 = xyxy
                selected = index in selected_indices
                if dragging and not selected:
                    continue
                frame_color = QColor(255, 236, 150, 210)
                if selected:
                    painter.setBrush(Qt.BrushStyle.NoBrush)
                    painter.setPen(QPen(frame_color, 2))
                    painter.drawRect(QRectF(x1, y1, max(1, x2 - x1), max(1, y2 - y1)))
                if selected and index == self.selected_bt_index and not multiple_selected:
                    painter.setBrush(frame_color)
                    for hx, hy in (
                        (x1, y1), ((x1 + x2) / 2, y1), (x2, y1),
                        (x1, (y1 + y2) / 2), (x2, (y1 + y2) / 2),
                        (x1, y2), ((x1 + x2) / 2, y2), (x2, y2),
                    ):
                        painter.drawRect(QRectF(hx - 3, hy - 3, 6, 6))
                if self._bt_drag_active and selected:
                    center_x = (x1 + x2) / 2.0
                    center_y = (y1 + y2) / 2.0
                    center_color = QColor(255, 92, 92, 235) if index == self.selected_bt_index else QColor(80, 210, 255, 230)
                    painter.setBrush(center_color)
                    painter.setPen(QPen(center_color, 2))
                    painter.drawEllipse(QPointF(center_x, center_y), 4, 4)
                    painter.drawLine(QPointF(center_x - 11, center_y), QPointF(center_x + 11, center_y))
                    painter.drawLine(QPointF(center_x, center_y - 11), QPointF(center_x, center_y + 11))
                if not dragging and self.show_bt_font_labels_check.isChecked():
                    self.draw_bt_font_label(
                        painter,
                        QRectF(x1, y1, x2 - x1, y2 - y1),
                        self.bt_font_label(item),
                        image_width,
                        image_height,
                    )

        def draw_bt_font_label(
            self,
            painter: QPainter,
            item_rect: QRectF,
            label: str,
            image_width: int,
            image_height: int,
        ) -> None:
            if not label:
                return
            font = QFont('Helvetica', max(16, min(30, (image_width // 100) * 2)))
            font.setBold(True)
            painter.setFont(font)
            metrics = painter.fontMetrics()
            text_rect = metrics.boundingRect(label)
            pad = 8
            label_w = text_rect.width() + pad * 2
            label_h = text_rect.height() + pad * 2
            x = min(max(int(round(item_rect.right() + 10)), 2), max(2, image_width - label_w - 2))
            y = min(max(int(round(item_rect.bottom() + 10)), label_h + 2), max(label_h + 2, image_height - 2))
            painter.fillRect(QRectF(x, y - label_h, label_w, label_h), QColor(255, 255, 255, 230))
            painter.setPen(QPen(QColor(20, 20, 20), 1))
            painter.drawText(QPointF(x + pad, y - pad), label)

        def hit_test_bt_item(self, x: float, y: float) -> tuple[int | None, str | None]:
            handles = (
                ('tl', -1, -1), ('t', 0, -1), ('tr', 1, -1),
                ('l', -1, 0), ('r', 1, 0),
                ('bl', -1, 1), ('b', 0, 1), ('br', 1, 1),
            )
            tolerance = 8.0
            matches = []
            for index, item in enumerate(self.bt_items_for_page()):
                xyxy = self.bt_xyxy_from_item(item)
                if xyxy is None:
                    continue
                x1, y1, x2, y2 = xyxy
                for mode, sx, sy in handles:
                    hx = (x1 + x2) / 2 if sx == 0 else (x1 if sx < 0 else x2)
                    hy = (y1 + y2) / 2 if sy == 0 else (y1 if sy < 0 else y2)
                    if abs(x - hx) <= tolerance and abs(y - hy) <= tolerance:
                        matches.append((0, index, mode))
                if x1 <= x <= x2 and y1 <= y <= y2:
                    area = max(0, x2 - x1) * max(0, y2 - y1)
                    matches.append((area, index, 'move'))
            if not matches:
                return None, None
            _, index, mode = min(matches, key=lambda item: item[0])
            return index, mode

        def _schedule_bt_drag_render(self) -> None:
            if not self._bt_drag_render_timer.isActive():
                self._bt_drag_render_timer.start()

        def _render_bt_drag_preview(self) -> None:
            if self._bt_drag_active:
                self.bt_annotation_item.update()

        def handle_bt_mouse_press(self, x: float, y: float) -> None:
            self._bt_drag_active = False
            self._bt_drag_render_timer.stop()
            index, mode = self.hit_test_bt_item(x, y)
            self.bt_view.set_background_pan_enabled(index is None)
            modifiers = QApplication.keyboardModifiers()
            toggle_selection = bool(
                modifiers & (Qt.KeyboardModifier.MetaModifier | Qt.KeyboardModifier.ControlModifier)
            )
            if toggle_selection and index is not None:
                selected = set(self.selected_bt_indices_list())
                if index in selected and len(selected) > 1:
                    selected.remove(index)
                    active_index = self.active_bt_index_from_selection(selected)
                else:
                    selected.add(index)
                    active_index = index
                self.set_bt_selection(selected, active_index=active_index)
                self._bt_drag_mode = None
                self._bt_drag_start = None
                self._bt_drag_original = None
                self._bt_drag_original_item = None
                self._bt_drag_original_items = None
                self._bt_drag_original_xyxys = {}
                self._bt_drag_indices = []
                self._bt_drag_temporary = False
                return
            if index is None:
                self.select_bt_item(None)
            elif index in self.selected_bt_indices_list() and self.has_multiple_bt_selection():
                self.set_bt_selection(set(self.selected_bt_indices_list()), active_index=index)
            else:
                self.select_bt_item(index)
            item = self.selected_bt_item()
            xyxy = self.bt_xyxy_from_item(item) if item is not None else None
            if item is None or xyxy is None or mode is None:
                self._bt_drag_mode = None
                self._bt_drag_start = None
                self._bt_drag_original = None
                self._bt_drag_original_item = None
                self._bt_drag_original_items = None
                self._bt_drag_original_xyxys = {}
                self._bt_drag_indices = []
                self._bt_drag_temporary = False
                self._popover_bt_item = None
                self.bt_match_popover.hide()
                return
            temporary = bool(QApplication.keyboardModifiers() & Qt.KeyboardModifier.AltModifier)
            drag_indices = self.selected_bt_indices_list()
            if self.has_multiple_bt_selection():
                mode = 'move'
                drag_indices = [
                    selected_index for selected_index in drag_indices
                    if self.bt_xyxy_from_item(self.bt_items_for_page()[selected_index]) is not None
                ]
            self._bt_drag_mode = 'move' if temporary else mode
            self._bt_drag_start = (x, y)
            self._bt_drag_original = xyxy
            self._bt_drag_original_item = copy.deepcopy(item)
            self._bt_drag_indices = drag_indices
            self._bt_drag_original_xyxys = {}
            items = self.bt_items_for_page()
            self._bt_drag_original_items = {
                selected_index: copy.deepcopy(items[selected_index])
                for selected_index in drag_indices
                if 0 <= selected_index < len(items) and isinstance(items[selected_index], dict)
            }
            for selected_index in drag_indices:
                if 0 <= selected_index < len(items):
                    selected_xyxy = self.bt_xyxy_from_item(items[selected_index])
                    if selected_xyxy is not None:
                        self._bt_drag_original_xyxys[selected_index] = selected_xyxy
            self._bt_drag_temporary = temporary
            self._bt_drag_active = True
            if self.bt_html_overlay is not None:
                self.bt_html_overlay.suspend()
            self.show_bt_match_popover(item)

        def update_bt_view_cursor(self, x: float, y: float) -> None:
            self._bt_cursor_image_pos = (float(x), float(y))
            index, _ = self.hit_test_bt_item(x, y)
            self.bt_view.set_background_pan_enabled(index is None)
            self.update_action_state()

        def clear_bt_view_cursor(self) -> None:
            self._bt_cursor_image_pos = None
            self.bt_view.set_background_pan_enabled(False)
            self.update_action_state()

        def handle_bt_mouse_drag(self, x: float, y: float) -> None:
            item = self.selected_bt_item()
            if item is None or self._bt_drag_mode is None or self._bt_drag_start is None or self._bt_drag_original is None:
                return
            dx = int(round(x - self._bt_drag_start[0]))
            dy = int(round(y - self._bt_drag_start[1]))
            items = self.bt_items_for_page()
            if self.has_multiple_bt_selection() and self._bt_drag_mode == 'move':
                for index, xyxy in self._bt_drag_original_xyxys.items():
                    if 0 <= index < len(items):
                        x1, y1, x2, y2 = xyxy
                        self.set_bt_xyxy(items[index], self.clamp_xyxy((x1 + dx, y1 + dy, x2 + dx, y2 + dy)))
                self._schedule_bt_drag_render()
                return
            x1, y1, x2, y2 = self._bt_drag_original
            if self._bt_drag_mode == 'move':
                new_xyxy = (x1 + dx, y1 + dy, x2 + dx, y2 + dy)
            else:
                nx1, ny1, nx2, ny2 = x1, y1, x2, y2
                if 'l' in self._bt_drag_mode:
                    nx1 += dx
                if 'r' in self._bt_drag_mode:
                    nx2 += dx
                if 't' in self._bt_drag_mode:
                    ny1 += dy
                if 'b' in self._bt_drag_mode:
                    ny2 += dy
                new_xyxy = (nx1, ny1, nx2, ny2)
            self.set_bt_xyxy(item, self.clamp_xyxy(new_xyxy))
            self._schedule_bt_drag_render()

        def handle_bt_mouse_release(self, x: float, y: float) -> None:
            self._bt_drag_render_timer.stop()
            self._bt_drag_active = False
            item = self.selected_bt_item()
            if (
                self._bt_drag_temporary
                and (self._bt_drag_original_item is not None or self._bt_drag_original_items is not None)
                and self.selected_bt_index is not None
            ):
                page_name = self.current_page_name()
                items = self.bt_items_for_page(page_name)
                if self._bt_drag_original_items is not None:
                    for index, original_item in self._bt_drag_original_items.items():
                        if 0 <= index < len(items):
                            items[index] = copy.deepcopy(original_item)
                    self.populate_box_editor_for_selection()
                    self.update_bt_item_list()
                    self.render_bt_page(refit=False)
                    self.status_label.setText('臨時移動結束，已回到原位。')
                self._bt_drag_mode = None
                self._bt_drag_start = None
                self._bt_drag_original = None
                self._bt_drag_original_item = None
                self._bt_drag_original_items = None
                self._bt_drag_original_xyxys = {}
                self._bt_drag_indices = []
                self._bt_drag_temporary = False
                return
            if (
                self.has_multiple_bt_selection()
                and self._bt_drag_original_items is not None
                and self._bt_drag_original_xyxys
            ):
                page_name = self.current_page_name()
                items = self.bt_items_for_page(page_name)
                changed = False
                for index, original_xyxy in self._bt_drag_original_xyxys.items():
                    if 0 <= index < len(items):
                        xyxy = self.bt_xyxy_from_item(items[index])
                        if xyxy is not None and xyxy != original_xyxy:
                            items[index]['match_status'] = 'manual'
                            changed = True
                if changed:
                    self.push_bt_changes_undo_snapshot(
                        '移動 _bt 多選框',
                        self._bt_drag_original_items,
                        self.selected_bt_index,
                        self.selected_bt_indices,
                    )
                    self.mark_bt_dirty()
                    self.populate_box_editor_for_selection()
                    self.update_bt_item_list()
                    self.status_label.setText(f'已移動 {len(self._bt_drag_original_xyxys)} 條 _bt 條目，尚未保存。')
                self.render_bt_page(refit=False)
                self._bt_drag_mode = None
                self._bt_drag_start = None
                self._bt_drag_original = None
                self._bt_drag_original_item = None
                self._bt_drag_original_items = None
                self._bt_drag_original_xyxys = {}
                self._bt_drag_indices = []
                self._bt_drag_temporary = False
                return
            if item is not None and self._bt_drag_original is not None and self._bt_drag_original_item is not None:
                xyxy = self.bt_xyxy_from_item(item)
                if xyxy is not None and xyxy != self._bt_drag_original:
                    page_name = self.current_page_name()
                    if page_name is not None and self.selected_bt_index is not None:
                        self.bt_undo_stack.append({
                            'page_name': page_name,
                            'item_index': self.selected_bt_index,
                            'item': self._bt_drag_original_item,
                            'description': '移動/調整 _bt 框',
                        })
                    self.mark_bt_dirty()
                    item['match_status'] = 'manual'
                    self.populate_box_editor_from_bt(item)
                    self.update_bt_item_list()
                    self.status_label.setText('已修改 _bt 框，尚未保存。')
            self.render_bt_page(refit=False)
            self._bt_drag_mode = None
            self._bt_drag_start = None
            self._bt_drag_original = None
            self._bt_drag_original_item = None
            self._bt_drag_original_items = None
            self._bt_drag_original_xyxys = {}
            self._bt_drag_indices = []
            self._bt_drag_temporary = False

        def render_current_page(self, *_, refit: bool = True) -> None:
            if self.page is None:
                return
            image = QImage(str(self.page.image_path))
            if image.isNull():
                show_error_details(
                    self,
                    '圖片讀取失敗',
                    '無法讀取圖片。下方是完整可複製的出錯信息。',
                    f'【圖片路徑】\n{self.page.image_path}',
                )
                return
            image = image.convertToFormat(QImage.Format.Format_RGBA8888)

            painter = QPainter(image)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

            opacity = self.opacity_slider.value() / 100.0
            self._draw_mask_layer(painter, opacity)
            if self.show_npz_smoothed.isChecked():
                self._draw_npz_layer(painter, 'smoothed', QColor(72, 155, 255), opacity)
            if self.show_npz_outer.isChecked():
                self._draw_npz_layer(painter, 'outer', QColor(255, 170, 58), opacity)
            if self.show_block_boxes.isChecked():
                self._draw_block_boxes(painter)
            if self.show_line_polygons.isChecked():
                self._draw_line_polygons(painter)
            if self.show_char_boxes.isChecked():
                self._draw_char_boxes(painter)
            if self.show_align_boxes.isChecked():
                self._draw_aligned_boxes(painter)
            if self.show_font_labels.isChecked():
                self._draw_font_labels(painter, image.width(), image.height())

            painter.end()
            self.view.set_pixmap(QPixmap.fromImage(image), fit=refit)
            self.update_navigator()
            if self.processor is not None and not self.processor.has_overlay_data():
                self.status_label.setText(
                    f'{self.page.page_name}\n'
                    '目前只顯示原圖。此資料夾尚未找到完整 CTD 疊圖資料。'
                )
            else:
                self.status_label.setText(
                    f'{self.page.page_name}：{len(self.page.boxes)} 個區塊，'
                    f'{len(self.page.lines)} 條文字行，{len(self.page.char_boxes)} 個單字框。'
                )

        def _draw_mask_layer(self, painter: QPainter, opacity: float) -> None:
            if self.page is None or not self.show_mask.isChecked() or self.page.mask_path is None:
                return
            mask_image = QImage(str(self.page.mask_path)).convertToFormat(QImage.Format.Format_Grayscale8)
            if mask_image.isNull():
                return
            ptr = mask_image.constBits()
            arr = np.frombuffer(ptr, dtype=np.uint8).reshape(mask_image.height(), mask_image.bytesPerLine())
            mask = arr[:, :mask_image.width()].copy()
            pixmap = mask_to_pixmap(mask, QColor(255, 65, 140), opacity)
            painter.drawPixmap(0, 0, pixmap)

        def _draw_npz_layer(self, painter: QPainter, layer: str, color: QColor, opacity: float) -> None:
            if self.page is None or self.page.align_masks is None:
                return
            masks = (
                self.page.align_masks.smoothed_masks
                if layer == 'smoothed'
                else self.page.align_masks.outer_body_masks
            )
            mask = union_mask(masks)
            if mask is None:
                return
            painter.drawPixmap(0, 0, mask_to_pixmap(mask, color, opacity))

        def _draw_block_boxes(self, painter: QPainter) -> None:
            if self.page is None:
                return
            pen = QPen(QColor(55, 110, 220), 2, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            for box in self.page.boxes:
                if box.block_xyxy_pixel is None:
                    continue
                x1, y1, x2, y2 = box.block_xyxy_pixel
                painter.drawRect(QRectF(x1, y1, x2 - x1, y2 - y1))

        def _draw_aligned_boxes(self, painter: QPainter) -> None:
            if self.page is None:
                return
            height = int(painter.device().height())
            width = int(painter.device().width())
            painter.setBrush(Qt.BrushStyle.NoBrush)
            for index, box in enumerate(self.page.boxes):
                color = QColor(20, 175, 95)
                if box.accepted is False:
                    color = QColor(32, 32, 32)
                if box.error_route:
                    color = QColor(215, 50, 50)
                selected = index == self.selected_box_index
                painter.setPen(QPen(QColor(255, 210, 40) if selected else color, 5 if selected else 3))
                x1, y1, x2, y2 = box.xyxy_pixel
                painter.drawRect(QRectF(x1, y1, x2 - x1, y2 - y1))
                if selected:
                    painter.setBrush(QColor(255, 210, 40))
                    for hx, hy in (
                        (x1, y1), ((x1 + x2) / 2, y1), (x2, y1),
                        (x1, (y1 + y2) / 2), (x2, (y1 + y2) / 2),
                        (x1, y2), ((x1 + x2) / 2, y2), (x2, y2),
                    ):
                        painter.drawRect(QRectF(hx - 4, hy - 4, 8, 8))
                    painter.setBrush(Qt.BrushStyle.NoBrush)
                cx, cy = box.center_pixel
                side = max(4, int(round(float(box.font_size or 0))))
                half = side / 2.0
                marker_x1 = max(0, int(round(cx - half)))
                marker_y1 = max(0, int(round(cy - half)))
                marker_x2 = min(width, int(round(cx + half)))
                marker_y2 = min(height, int(round(cy + half)))
                if marker_x2 > marker_x1 and marker_y2 > marker_y1:
                    painter.fillRect(
                        QRectF(
                            marker_x1,
                            marker_y1,
                            marker_x2 - marker_x1,
                            marker_y2 - marker_y1,
                        ),
                        color,
                    )

        def _draw_line_polygons(self, painter: QPainter) -> None:
            if self.page is None:
                return
            painter.setPen(QPen(QColor(0, 145, 175), 2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            for line in self.page.lines:
                points = [QPointF(x, y) for x, y in line.polygon]
                if len(points) >= 2:
                    painter.drawPolygon(points)

        def _draw_char_boxes(self, painter: QPainter) -> None:
            if self.page is None:
                return
            image_height = int(painter.device().height())
            image_width = int(painter.device().width())
            painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
            normal_font = QFont('Helvetica Neue', max(6, min(8, image_width // 240)))
            normal_font.setWeight(getattr(QFont.Weight, 'Thin', QFont.Weight.Light))
            normal_font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
            painter.setFont(normal_font)
            metrics = painter.fontMetrics()
            painter.setPen(QPen(QColor(245, 170, 35), 1))
            painter.setBrush(QColor(245, 170, 35, 32))
            for item in self.page.char_boxes:
                bbox = item.get('bbox')
                if not isinstance(bbox, list) or len(bbox) != 4:
                    continue
                x1, y1, x2, y2 = [int(round(float(value))) for value in bbox]
                painter.drawRect(QRectF(x1, y1, x2 - x1, y2 - y1))
                if item is self.hover_char_box:
                    continue
                label = char_box_label(item)
                if label is None:
                    continue
                text_w = metrics.horizontalAdvance(label)
                text_h = metrics.ascent() + metrics.descent()
                label_x = int(round((x1 + x2) / 2 - text_w / 2))
                label_y = y1 - 2
                if label_y - text_h < 1:
                    label_y = y2 + text_h + 1
                label_x = max(1, min(label_x, max(1, image_width - text_w - 1)))
                label_y = max(text_h + 1, min(label_y, image_height - 1))

                painter.setPen(QPen(QColor(85, 48, 12), 1))
                painter.drawText(QPointF(label_x, label_y), label)
                painter.setPen(QPen(QColor(245, 170, 35), 1))
            if self.hover_char_box is not None:
                bbox = self.hover_char_box.get('bbox')
                if isinstance(bbox, list) and len(bbox) == 4:
                    x1, y1, x2, y2 = [int(round(float(value))) for value in bbox]
                    painter.setPen(QPen(QColor(255, 40, 120), 3))
                    painter.setBrush(Qt.BrushStyle.NoBrush)
                    painter.drawRect(QRectF(x1, y1, x2 - x1, y2 - y1))
                    label = char_box_label(self.hover_char_box)
                    if label is not None:
                        view_scale = 1.0
                        if hasattr(self, 'view'):
                            transform = self.view.transform()
                            view_scale = max(0.05, min(abs(transform.m11()), abs(transform.m22())))
                        hover_font = QFont('Helvetica Neue')
                        hover_font.setPixelSize(max(14, int(round(24 / view_scale))))
                        hover_font.setWeight(QFont.Weight.Bold)
                        hover_font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
                        painter.setFont(hover_font)
                        hover_metrics = painter.fontMetrics()
                        pad_x = max(3, int(round(7 / view_scale)))
                        pad_y = max(2, int(round(4 / view_scale)))
                        offset_y = max(2, int(round(4 / view_scale)))
                        text_w = hover_metrics.horizontalAdvance(label)
                        text_h = hover_metrics.ascent() + hover_metrics.descent()
                        label_w = text_w + pad_x * 2
                        label_h = text_h + pad_y * 2
                        box_x = int(round((x1 + x2) / 2 - label_w / 2))
                        box_y = y1 - label_h - offset_y
                        box_x = max(2, min(box_x, max(2, image_width - label_w - 2)))
                        box_y = max(2, min(box_y, max(2, image_height - label_h - 2)))

                        painter.setPen(QPen(QColor(20, 20, 20), 2))
                        painter.setBrush(QColor(255, 255, 245, 245))
                        painter.drawRect(QRectF(box_x, box_y, label_w, label_h))
                        painter.setPen(QPen(QColor(0, 0, 0), 1))
                        painter.drawText(QRectF(box_x, box_y, label_w, label_h), Qt.AlignmentFlag.AlignCenter, label)

        def _draw_font_labels(self, painter: QPainter, image_width: int, image_height: int) -> None:
            if self.page is None:
                return
            font = QFont('Helvetica', max(20, min(36, (image_width // 85) * 2)))
            font.setBold(True)
            painter.setFont(font)
            metrics = painter.fontMetrics()
            for box in self.page.boxes:
                label = box.font_label
                x1, y1, x2, y2 = box.xyxy_pixel
                text_rect = metrics.boundingRect(label)
                pad = 10
                label_w = text_rect.width() + pad * 2
                label_h = text_rect.height() + pad * 2
                x = min(max(x2 + 12, 2), max(2, image_width - label_w - 2))
                y = min(max(y2 + 12, label_h + 2), max(label_h + 2, image_height - 2))
                painter.fillRect(QRectF(x, y - label_h, label_w, label_h), QColor(255, 255, 255, 230))
                painter.setPen(QPen(QColor(20, 20, 20), 1))
                painter.drawText(QPointF(x + pad, y - pad), label)


else:
    CtdOverlayViewer = None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='開啟 PySide6 CTD 即時疊圖檢視器。')
    parser.add_argument('image_dir', nargs='?', default=None, help='包含原圖和 ctd/ 的圖片資料夾。')
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if QT_IMPORT_ERROR is not None:
        print(
            '尚未安裝 PySide6。請安裝完整 PySide6 後再啟動。',
            file=sys.stderr,
        )
        raise SystemExit(1)
    app = QApplication(sys.argv)
    viewer = CtdOverlayViewer(args.image_dir)
    viewer.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
