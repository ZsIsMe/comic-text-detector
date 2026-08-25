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

    try:
        from .viewer_ui_mixin import ViewerUIMixin
    except ImportError:
        from viewer_ui_mixin import ViewerUIMixin

    try:
        from .viewer_session_mixin import ViewerSessionMixin
    except ImportError:
        from viewer_session_mixin import ViewerSessionMixin
    try:
        from .viewer_bt_data_mixin import ViewerBTDataMixin
    except ImportError:
        from viewer_bt_data_mixin import ViewerBTDataMixin
    try:
        from .viewer_bt_editor_mixin import ViewerBTEditorMixin
    except ImportError:
        from viewer_bt_editor_mixin import ViewerBTEditorMixin
    try:
        from .viewer_bt_measurement_mixin import ViewerBTMeasurementMixin
    except ImportError:
        from viewer_bt_measurement_mixin import ViewerBTMeasurementMixin
    try:
        from .viewer_bt_update_mixin import ViewerBTUpdateMixin
    except ImportError:
        from viewer_bt_update_mixin import ViewerBTUpdateMixin
    try:
        from .viewer_workflow_mixin import ViewerWorkflowMixin
    except ImportError:
        from viewer_workflow_mixin import ViewerWorkflowMixin

    class CtdOverlayViewer(
        ViewerUIMixin,
        ViewerSessionMixin,
        ViewerBTDataMixin,
        ViewerBTEditorMixin,
        ViewerBTMeasurementMixin,
        ViewerBTUpdateMixin,
        ViewerWorkflowMixin,
        QMainWindow,
    ):
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

            super().resizeEvent(event)
            if not hasattr(self, '_fitting_views'):
                return
            self.fit_both_views()
            self.schedule_bt_html_overlay_update()

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
