#!/usr/bin/env python3
"""ctd JSON/NPZ 即時疊圖檢視器。

檢視器會在記憶體中根據原圖、ctd/measure.json、ctd/progressing/*.json
和 align/masks/*.npz 重建只讀疊圖，不會輸出預覽 PNG。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

QT_IMPORT_ERROR: ModuleNotFoundError | None = None
QT_WEBENGINE_AVAILABLE = False
QWebEngineView = None
BtMeasurementDialog = None
try:
    from PySide6.QtCore import QProcess, QSettings, Qt, QTimer
    from PySide6.QtGui import QAction, QFont, QImage, QKeyEvent
    from PySide6.QtWidgets import (
        QApplication,
        QButtonGroup,
        QCheckBox,
        QComboBox,
        QDialog,
        QDockWidget,
        QLabel,
        QListWidget,
        QMainWindow,
        QPlainTextEdit,
        QPushButton,
        QRadioButton,
        QSplitter,
        QDoubleSpinBox,
        QSpinBox,
        QSlider,
        QTableWidget,
        QVBoxLayout,
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
        CtdOverlayProcessor,
        PageOverlay,
    )
except ImportError:
    from font_size_calibration import DEFAULT_FONT_SIZE_BASE, DEFAULT_FONT_SIZE_STEP
    from processor import (
        CtdOverlayProcessor,
        PageOverlay,
    )


if QT_IMPORT_ERROR is None:
    try:
        from .viewer_widgets import BtAnnotationItem, ImageView, NavigatorWidget
    except ImportError:
        from viewer_widgets import BtAnnotationItem, ImageView, NavigatorWidget
    try:
        from .viewer_popovers import BtMatchPopover, BtTextEditPopover, HtmlTextOverlay
    except ImportError:
        from viewer_popovers import BtMatchPopover, BtTextEditPopover, HtmlTextOverlay
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
    try:
        from .viewer_render_mixin import ViewerRenderMixin
    except ImportError:
        from viewer_render_mixin import ViewerRenderMixin

    class CtdOverlayViewer(
        ViewerUIMixin,
        ViewerSessionMixin,
        ViewerBTDataMixin,
        ViewerBTEditorMixin,
        ViewerBTMeasurementMixin,
        ViewerBTUpdateMixin,
        ViewerWorkflowMixin,
        ViewerRenderMixin,
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
