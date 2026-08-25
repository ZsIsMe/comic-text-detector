#!/usr/bin/env python3
"""獨立編輯 ctd/measure.json 的 PySide6 視窗。"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from PySide6.QtCore import QPointF, QSettings, QTimer, Qt, Signal
from PySide6.QtGui import QAction, QImage, QKeySequence, QPainter, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

try:
    from .measure_view import (
        MeasureImageView,
        char_info_text,
        compact_int_px,
        draw_char_boxes,
        draw_font_labels,
        draw_measure_boxes,
    )
    from .processor import (
        CtdOverlayProcessor,
        PageOverlay,
        normalize_measure_map,
        normalized_center_from_xyxy,
        resolve_image_path,
        xyxy_from_item,
    )
    from .measure_ocr_dialog import MeasureOcrDialog
except ImportError:
    from measure_view import (
        MeasureImageView,
        char_info_text,
        compact_int_px,
        draw_char_boxes,
        draw_font_labels,
        draw_measure_boxes,
    )
    from processor import (
        CtdOverlayProcessor,
        PageOverlay,
        normalize_measure_map,
        normalized_center_from_xyxy,
        resolve_image_path,
        xyxy_from_item,
    )
    from measure_ocr_dialog import MeasureOcrDialog


def qimage_size(path: Path) -> tuple[int, int] | None:
    image = QImage(str(path))
    if image.isNull():
        return None
    return image.width(), image.height()


def positive_int(value, default: int, *, minimum: int = 1) -> int:
    try:
        number = int(round(float(value)))
    except (TypeError, ValueError):
        return default
    return number if number >= minimum else default


class MeasureEditorWindow(QMainWindow):
    saved = Signal()

    def __init__(
        self,
        processor: CtdOverlayProcessor,
        current_page_name: str | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle('編輯 measure.json')
        self.resize(1180, 820)
        self.processor = CtdOverlayProcessor(processor.image_dir)
        self.measure = copy.deepcopy(processor.measure)
        self.processor.measure = self.measure
        self.measure_path = processor.measure_path
        self.settings = QSettings('comic-text-detector', 'measure-editor')
        folder_key = hashlib.sha1(str(self.processor.image_dir).encode('utf-8')).hexdigest()
        self.settings_prefix = f'folders/{folder_key}'
        self.page_names = processor.page_names()
        self.page: PageOverlay | None = None
        self.current_page_row = -1
        self.selected_index: int | None = None
        self.hover_char_box: dict[str, Any] | None = None
        self.dirty = False
        self.undo_stack: list[dict[str, Any]] = []
        self._updating_editor = False
        self._drag_mode: str | None = None
        self._drag_start: tuple[float, float] | None = None
        self._drag_original: tuple[int, int, int, int] | None = None
        self._drag_original_items: list[dict[str, Any]] | None = None
        self._drag_changed = False

        self.view = MeasureImageView()
        self.page_list = QListWidget()
        self.item_list = QListWidget()
        self.title_label = QLabel('未選擇文字框')
        self.title_label.setWordWrap(True)
        self.char_info_label = QLabel('游標單字框：未選中')
        self.char_info_label.setWordWrap(True)
        self.x1_spin = QSpinBox()
        self.y1_spin = QSpinBox()
        self.x2_spin = QSpinBox()
        self.y2_spin = QSpinBox()
        self.font_size_spin = QDoubleSpinBox()
        self.orientation_combo = QComboBox()
        self.color_combo = QComboBox()
        self.text_has_stroke_check = QCheckBox('原字描邊')
        self.need_inpaint_check = QCheckBox('需要修復/描邊')
        self.show_center_marker = QCheckBox('中心點')
        self.show_char_boxes = QCheckBox('單字框')
        self.show_font_labels = QCheckBox('字級標籤')
        self.font_size_list = QListWidget()
        self.save_button = QPushButton('保存 measure.json')
        self.undo_button = QPushButton('撤銷')
        self.copy_button = QPushButton('複製當前文字框')
        self.delete_button = QPushButton('刪除當前文字框')
        self.uniform_font_button = QPushButton('統一調整字體大小')
        self.ocr_button = QPushButton('逐字 OCR 校準字級')
        self.status_label = QLabel('尚未修改')
        self.status_label.setWordWrap(True)
        self.save_action = QAction('保存 measure.json', self)
        self.undo_action = QAction('撤銷', self)
        self.delete_action = QAction('刪除文字框', self)
        self.increase_font_action = QAction('字體+2', self)
        self.decrease_font_action = QAction('字體-2', self)
        self.increase_font_10_action = QAction('字體+10', self)
        self.decrease_font_10_action = QAction('字體-10', self)

        self._build_ui()
        self._connect_signals()
        self._build_shortcuts()
        self.page_list.addItems(self.page_names)
        initial_page_name = self.remembered_page_name() or current_page_name
        if initial_page_name in self.page_names:
            self.page_list.setCurrentRow(self.page_names.index(initial_page_name))
        elif self.page_names:
            self.page_list.setCurrentRow(0)
        self.update_action_state()

    def _build_ui(self) -> None:
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.setCentralWidget(splitter)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(8, 8, 8, 8)
        left_layout.setSpacing(8)
        left_layout.addWidget(QLabel('頁面'))
        left_layout.addWidget(self.page_list, 1)
        left_layout.addWidget(QLabel('文字框'))
        left_layout.addWidget(self.item_list, 2)
        left_layout.addWidget(QLabel('顯示'))
        self.show_center_marker.setChecked(True)
        self.show_char_boxes.setChecked(True)
        self.show_font_labels.setChecked(True)
        left_layout.addWidget(self.show_center_marker)
        left_layout.addWidget(self.show_char_boxes)
        left_layout.addWidget(self.show_font_labels)
        left_layout.addWidget(self.char_info_label)
        splitter.addWidget(left_panel)

        splitter.addWidget(self.view)

        font_panel = QWidget()
        font_layout = QVBoxLayout(font_panel)
        font_layout.setContentsMargins(8, 8, 8, 8)
        font_layout.setSpacing(8)
        font_layout.addWidget(QLabel('字體大小'))
        self.font_size_list.setMinimumWidth(88)
        for size in range(6, 1000):
            self.font_size_list.addItem(QListWidgetItem(f'{float(size):.1f}'))
        font_layout.addWidget(self.font_size_list, 1)
        splitter.addWidget(font_panel)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(10, 10, 10, 10)
        right_layout.setSpacing(8)
        right_layout.addWidget(QLabel('當前 measure 文字框'))
        right_layout.addWidget(self.title_label)
        xyxy_layout = QGridLayout()
        for spin in (self.x1_spin, self.y1_spin, self.x2_spin, self.y2_spin):
            spin.setRange(0, 99999)
        xyxy_layout.addWidget(QLabel('x1'), 0, 0)
        xyxy_layout.addWidget(self.x1_spin, 0, 1)
        xyxy_layout.addWidget(QLabel('y1'), 0, 2)
        xyxy_layout.addWidget(self.y1_spin, 0, 3)
        xyxy_layout.addWidget(QLabel('x2'), 1, 0)
        xyxy_layout.addWidget(self.x2_spin, 1, 1)
        xyxy_layout.addWidget(QLabel('y2'), 1, 2)
        xyxy_layout.addWidget(self.y2_spin, 1, 3)
        right_layout.addWidget(QLabel('xyxy_pixel'))
        right_layout.addLayout(xyxy_layout)
        self.font_size_spin.setRange(0.1, 999.0)
        self.font_size_spin.setDecimals(1)
        self.font_size_spin.setSingleStep(0.1)
        self.font_size_spin.setSuffix(' px')
        self.orientation_combo.addItem('直排', 'vertical')
        self.orientation_combo.addItem('橫排', 'horizontal')
        self.color_combo.addItem('黑', 'black')
        self.color_combo.addItem('白', 'white')
        right_layout.addWidget(QLabel('字體大小'))
        right_layout.addWidget(self.font_size_spin)
        right_layout.addWidget(QLabel('方向'))
        right_layout.addWidget(self.orientation_combo)
        right_layout.addWidget(QLabel('文字顏色'))
        right_layout.addWidget(self.color_combo)
        right_layout.addWidget(self.text_has_stroke_check)
        right_layout.addWidget(self.need_inpaint_check)
        right_layout.addWidget(self.copy_button)
        right_layout.addWidget(self.delete_button)
        right_layout.addWidget(self.uniform_font_button)
        right_layout.addWidget(self.ocr_button)
        right_layout.addWidget(self.undo_button)
        right_layout.addWidget(self.save_button)
        right_layout.addWidget(self.status_label)
        right_layout.addStretch(1)
        splitter.addWidget(right_panel)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setStretchFactor(3, 0)
        splitter.setSizes([240, 700, 110, 260])

    def _build_shortcuts(self) -> None:
        self.save_action.setShortcuts([QKeySequence('Meta+S'), QKeySequence('Ctrl+S')])
        self.save_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        self.save_action.triggered.connect(self.save_measure)
        self.addAction(self.save_action)

        self.undo_action.setShortcuts([QKeySequence('Meta+Z'), QKeySequence('Ctrl+Z')])
        self.undo_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        self.addAction(self.undo_action)

        self.delete_action.setShortcuts([QKeySequence(Qt.Key.Key_Delete), QKeySequence(Qt.Key.Key_Backspace)])
        self.delete_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        self.delete_action.triggered.connect(self.delete_selected_item)
        self.addAction(self.delete_action)

        self.increase_font_action.setShortcuts([
            QKeySequence('Meta++'),
            QKeySequence('Meta+='),
            QKeySequence('Ctrl++'),
            QKeySequence('Ctrl+='),
        ])
        self.increase_font_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        self.increase_font_action.triggered.connect(lambda: self.nudge_selected_font_size(2))
        self.addAction(self.increase_font_action)

        self.decrease_font_action.setShortcuts([QKeySequence('Meta+-'), QKeySequence('Ctrl+-')])
        self.decrease_font_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        self.decrease_font_action.triggered.connect(lambda: self.nudge_selected_font_size(-2))
        self.addAction(self.decrease_font_action)

        self.increase_font_10_action.setShortcuts([
            QKeySequence('Meta+Alt++'),
            QKeySequence('Meta+Alt+='),
            QKeySequence('Ctrl+Alt++'),
            QKeySequence('Ctrl+Alt+='),
        ])
        self.increase_font_10_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        self.increase_font_10_action.triggered.connect(lambda: self.nudge_selected_font_size(10))
        self.addAction(self.increase_font_10_action)

        self.decrease_font_10_action.setShortcuts([
            QKeySequence('Meta+Alt+-'),
            QKeySequence('Ctrl+Alt+-'),
        ])
        self.decrease_font_10_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        self.decrease_font_10_action.triggered.connect(lambda: self.nudge_selected_font_size(-10))
        self.addAction(self.decrease_font_10_action)

    def _connect_signals(self) -> None:
        self.page_list.currentRowChanged.connect(self.handle_page_changed)
        self.item_list.currentRowChanged.connect(self.handle_item_row_changed)
        self.view.imageMouseMoved.connect(self.update_hover_char_box)
        self.view.imageMouseLeft.connect(self.clear_hover_char_box)
        self.view.imageMousePressed.connect(self.handle_mouse_press)
        self.view.imageMouseDragged.connect(self.handle_mouse_drag)
        self.view.imageMouseReleased.connect(self.handle_mouse_release)
        self.view.fontSizeWheelRequested.connect(self.nudge_selected_font_size)
        self.x1_spin.valueChanged.connect(self.apply_editor_changes)
        self.y1_spin.valueChanged.connect(self.apply_editor_changes)
        self.x2_spin.valueChanged.connect(self.apply_editor_changes)
        self.y2_spin.valueChanged.connect(self.apply_editor_changes)
        self.font_size_spin.valueChanged.connect(self.apply_editor_changes)
        self.orientation_combo.currentIndexChanged.connect(self.apply_editor_changes)
        self.color_combo.currentIndexChanged.connect(self.apply_editor_changes)
        self.text_has_stroke_check.stateChanged.connect(self.apply_editor_changes)
        self.need_inpaint_check.stateChanged.connect(self.apply_editor_changes)
        self.show_center_marker.stateChanged.connect(lambda *_: self.render_page(refit=False))
        self.show_char_boxes.stateChanged.connect(lambda *_: self.render_page(refit=False))
        self.show_font_labels.stateChanged.connect(lambda *_: self.render_page(refit=False))
        self.font_size_list.itemClicked.connect(self.apply_font_size_from_list)
        self.save_button.clicked.connect(self.save_measure)
        self.undo_button.clicked.connect(self.undo_last_change)
        self.undo_action.triggered.connect(self.undo_last_change)
        self.copy_button.clicked.connect(self.copy_selected_item)
        self.delete_button.clicked.connect(self.delete_selected_item)
        self.uniform_font_button.clicked.connect(self.show_uniform_font_size_dialog)
        self.ocr_button.clicked.connect(self.open_ocr_dialog)

    def current_page_name(self) -> str | None:
        if 0 <= self.current_page_row < len(self.page_names):
            return self.page_names[self.current_page_row]
        return None

    def settings_key(self, *parts: str) -> str:
        return '/'.join((self.settings_prefix, *parts))

    def page_settings_key(self, page_name: str, field: str) -> str:
        page_key = hashlib.sha1(page_name.encode('utf-8')).hexdigest()
        return self.settings_key('pages', page_key, field)

    def remembered_page_name(self) -> str | None:
        value = self.settings.value(self.settings_key('last_page_name'), '')
        return str(value) if value else None

    def remembered_selected_index(self, page_name: str) -> int | None:
        value = self.settings.value(self.page_settings_key(page_name, 'selected_index'), -1)
        try:
            index = int(value)
        except (TypeError, ValueError):
            return None
        return index if index >= 0 else None

    def save_editor_position(self) -> None:
        page_name = self.current_page_name()
        if not page_name:
            return
        self.settings.setValue(self.settings_key('last_page_name'), page_name)
        self.settings.setValue(
            self.page_settings_key(page_name, 'selected_index'),
            self.selected_index if self.selected_index is not None else -1,
        )
        self.settings.setValue(self.page_settings_key(page_name, 'scroll_x'), self.view.horizontalScrollBar().value())
        self.settings.setValue(self.page_settings_key(page_name, 'scroll_y'), self.view.verticalScrollBar().value())

    def restore_current_view_state(self) -> None:
        page_name = self.current_page_name()
        if not page_name:
            return
        try:
            scroll_x = int(self.settings.value(self.page_settings_key(page_name, 'scroll_x'), 0))
            scroll_y = int(self.settings.value(self.page_settings_key(page_name, 'scroll_y'), 0))
        except (TypeError, ValueError):
            return
        self.view.horizontalScrollBar().setValue(scroll_x)
        self.view.verticalScrollBar().setValue(scroll_y)

    def current_items(self) -> list[dict[str, Any]]:
        page_name = self.current_page_name()
        if not page_name:
            return []
        pages = self.measure.setdefault('pages', {})
        items = pages.setdefault(page_name, [])
        return items if isinstance(items, list) else []

    def selected_item(self) -> dict[str, Any] | None:
        items = self.current_items()
        if self.selected_index is None or self.selected_index < 0 or self.selected_index >= len(items):
            return None
        item = items[self.selected_index]
        return item if isinstance(item, dict) else None

    def refresh_page_overlay(self) -> None:
        page_name = self.current_page_name()
        if not page_name:
            self.page = None
            return
        self.processor.measure = self.measure
        self.page = self.processor.load_page(page_name, image_size=self.image_size())

    def refresh_after_measure_change(self, *, refit: bool = False) -> None:
        self.refresh_page_overlay()
        self.update_item_list()
        self.populate_editor(self.selected_item())
        self.update_font_size_selection()
        self.render_page(refit=refit)
        self.update_action_state()

    def image_size(self) -> tuple[int, int] | None:
        page_name = self.current_page_name()
        if not page_name:
            return None
        try:
            return qimage_size(resolve_image_path(self.processor.image_dir, page_name))
        except Exception:
            return None

    def handle_page_changed(self, row: int) -> None:
        if row < 0 or row >= len(self.page_names):
            return
        if self.current_page_row >= 0 and row != self.current_page_row:
            self.save_editor_position()
        self.current_page_row = row
        page_name = self.current_page_name()
        remembered_index = self.remembered_selected_index(page_name) if page_name else None
        self.selected_index = (
            remembered_index
            if remembered_index is not None and 0 <= remembered_index < len(self.current_items())
            else None
        )
        self.clear_hover_char_box(render=False)
        self.refresh_page_overlay()
        self.update_item_list()
        self.render_page(refit=True)
        self.populate_editor(self.selected_item())
        self.update_font_size_selection()
        QTimer.singleShot(0, self.restore_current_view_state)
        self.update_action_state()

    def update_item_list(self) -> None:
        self.item_list.blockSignals(True)
        self.item_list.clear()
        for index, item in enumerate(self.current_items()):
            xyxy = xyxy_from_item(item) if isinstance(item, dict) else None
            try:
                font_size = f'{float(item.get("font_size")):.1f}' if isinstance(item, dict) else None
            except (TypeError, ValueError):
                font_size = None
            orientation = str(item.get('orientation') or 'vertical') if isinstance(item, dict) else 'vertical'
            direction = 'H' if orientation == 'horizontal' else 'V'
            source_index = item.get('source_block_index', index) if isinstance(item, dict) else index
            box_text = '-' if xyxy is None else ','.join(str(v) for v in xyxy)
            self.item_list.addItem(f'{index + 1:03d}  src={source_index}  {font_size or "-"}{direction}  {box_text}')
        if self.selected_index is not None and 0 <= self.selected_index < self.item_list.count():
            self.item_list.setCurrentRow(self.selected_index)
        else:
            self.item_list.setCurrentRow(-1)
        self.item_list.blockSignals(False)

    def update_font_size_selection(self) -> None:
        item = self.selected_item()
        try:
            target = round(float(item.get('font_size')), 1) if item is not None else 0.0
        except (TypeError, ValueError):
            target = 0.0
        self.font_size_list.blockSignals(True)
        self.font_size_list.setCurrentRow(-1)
        if target > 0:
            for row in range(self.font_size_list.count()):
                list_item = self.font_size_list.item(row)
                if list_item is not None and abs(float(list_item.text()) - target) < 0.05:
                    self.font_size_list.setCurrentRow(row)
                    break
        self.font_size_list.blockSignals(False)

    def handle_item_row_changed(self, row: int) -> None:
        if row < 0 or row >= len(self.current_items()):
            self.select_item(None)
        else:
            self.select_item(row, center=True)

    def select_item(self, index: int | None, *, center: bool = False) -> None:
        self.selected_index = index
        item = self.selected_item()
        self.populate_editor(item)
        self.update_item_list()
        self.update_font_size_selection()
        self.render_page(refit=False)
        if center and item is not None:
            xyxy = xyxy_from_item(item)
            if xyxy is not None:
                x1, y1, x2, y2 = xyxy
                self.view.centerOn(QPointF((x1 + x2) / 2.0, (y1 + y2) / 2.0))
        self.save_editor_position()
        self.update_action_state()

    def select_box_item(self, index: int | None) -> None:
        if index is None:
            self.select_item(None)
            return
        self.selected_index = index
        self.populate_editor(self.selected_item())
        self.update_item_list()
        self.update_font_size_selection()
        self.render_page(refit=False)
        self.save_editor_position()
        self.update_action_state()

    def populate_editor(self, item: dict[str, Any] | None) -> None:
        self._updating_editor = True
        enabled = item is not None
        for widget in (
            self.x1_spin,
            self.y1_spin,
            self.x2_spin,
            self.y2_spin,
            self.font_size_spin,
            self.orientation_combo,
            self.color_combo,
            self.text_has_stroke_check,
            self.need_inpaint_check,
            self.copy_button,
            self.delete_button,
        ):
            widget.setEnabled(enabled)
        if item is None:
            self.title_label.setText('未選擇文字框')
            self._updating_editor = False
            return
        xyxy = xyxy_from_item(item)
        box_text = '無框' if xyxy is None else ','.join(str(v) for v in xyxy)
        self.title_label.setText(f'index={self.selected_index}  source={item.get("source_block_index", "-")}  框：{box_text}')
        if xyxy is not None:
            x1, y1, x2, y2 = xyxy
            self.x1_spin.setValue(x1)
            self.y1_spin.setValue(y1)
            self.x2_spin.setValue(x2)
            self.y2_spin.setValue(y2)
        try:
            font_size = float(item.get('font_size'))
        except (TypeError, ValueError):
            font_size = 40.0
        self.font_size_spin.setValue(max(0.1, min(999.0, font_size)))
        orientation_index = self.orientation_combo.findData(item.get('orientation') or 'vertical')
        self.orientation_combo.setCurrentIndex(max(0, orientation_index))
        color_index = self.color_combo.findData(item.get('text_color') or 'black')
        self.color_combo.setCurrentIndex(max(0, color_index))
        self.text_has_stroke_check.setChecked(item.get('text_has_stroke') is True)
        self.need_inpaint_check.setChecked(item.get('need_inpaint') is True)
        self._updating_editor = False
        self.update_font_size_selection()

    def apply_editor_changes(self) -> None:
        if self._updating_editor:
            return
        item = self.selected_item()
        if item is None:
            return
        before_items = copy.deepcopy(self.current_items())
        before_selected = self.selected_index
        updated = {
            'font_size': round(float(self.font_size_spin.value()), 1),
            'orientation': self.orientation_combo.currentData() or 'vertical',
            'text_color': self.color_combo.currentData() or 'black',
            'text_has_stroke': self.text_has_stroke_check.isChecked(),
            'need_inpaint': self.need_inpaint_check.isChecked(),
        }
        updated_xyxy = (
            int(self.x1_spin.value()),
            int(self.y1_spin.value()),
            int(self.x2_spin.value()),
            int(self.y2_spin.value()),
        )
        current_xyxy = self.measure_xyxy_from_item(item)
        if current_xyxy == self.clamp_xyxy(updated_xyxy) and all(item.get(key) == value for key, value in updated.items()):
            return
        self.push_undo_snapshot('修改屬性', before_items, before_selected)
        self.set_measure_xyxy(item, updated_xyxy)
        item.update(updated)
        self.mark_dirty('已修改 measure 屬性，尚未保存。')
        self.refresh_after_measure_change(refit=False)

    def apply_font_size_from_list(self, item: QListWidgetItem) -> None:
        try:
            size = float(item.text())
        except ValueError:
            return
        self.set_selected_font_size(size, f'已把當前 measure 字體大小改為 {size:.1f}，尚未保存。')

    def nudge_selected_font_size(self, delta: int) -> None:
        item = self.selected_item()
        if item is None:
            self.status_label.setText('請先選擇一個 measure 文字框，再使用字體大小快捷鍵。')
            return
        try:
            current = float(item.get('font_size'))
        except (TypeError, ValueError):
            current = float(self.font_size_spin.value())
        size = round(max(0.1, min(999.0, current + delta)), 1)
        if abs(size - current) < 0.05:
            return
        sign = '+' if delta > 0 else ''
        self.set_selected_font_size(size, f'已將當前 measure 字體大小 {sign}{delta} 到 {size}，尚未保存。')

    def set_selected_font_size(self, size: float, status: str) -> None:
        item = self.selected_item()
        if item is None:
            self.status_label.setText('請先選擇一個 measure 文字框。')
            return
        size = round(max(0.1, min(999.0, float(size))), 1)
        try:
            current = float(item.get('font_size'))
        except (TypeError, ValueError):
            current = float(self.font_size_spin.value())
        if abs(current - size) < 0.05:
            return
        before_items = copy.deepcopy(self.current_items())
        before_selected = self.selected_index
        self.push_undo_snapshot('修改字體大小', before_items, before_selected)
        item['font_size'] = size
        self.mark_dirty(status)
        self.refresh_after_measure_change(refit=False)

    def show_uniform_font_size_dialog(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle('統一調整字體大小')
        layout = QVBoxLayout(dialog)

        scope_row = QHBoxLayout()
        scope_row.addWidget(QLabel('範圍'))
        scope_combo = QComboBox()
        scope_combo.addItem('當頁', 'current')
        scope_combo.addItem('全部', 'all')
        scope_row.addWidget(scope_combo, 1)
        layout.addLayout(scope_row)

        size_row = QHBoxLayout()
        size_row.addWidget(QLabel('字體大小'))
        size_spin = QDoubleSpinBox()
        size_spin.setRange(0.1, 999.0)
        size_spin.setDecimals(1)
        size_spin.setSingleStep(0.1)
        try:
            uniform_size = float(self.settings.value(self.settings_key('uniform_font_size'), 24.0))
        except (TypeError, ValueError):
            uniform_size = 24.0
        size_spin.setValue(max(0.1, min(999.0, uniform_size)))
        size_spin.selectAll()
        size_row.addWidget(size_spin, 1)
        layout.addLayout(size_row)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText('確認')
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText('取消')
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        size = round(float(size_spin.value()), 1)
        self.settings.setValue(self.settings_key('uniform_font_size'), size)
        self.apply_uniform_font_size(scope_combo.currentData() or 'current', size)

    def apply_uniform_font_size(self, scope: str, size: float) -> None:
        size = round(max(0.1, min(999.0, float(size))), 1)
        pages = self.measure.setdefault('pages', {})
        if scope == 'all':
            target_page_names = [
                page_name
                for page_name, items in pages.items()
                if isinstance(items, list)
            ]
        else:
            current_name = self.current_page_name()
            target_page_names = [current_name] if current_name else []
        if not target_page_names:
            self.status_label.setText('沒有可調整的 measure 頁面。')
            return

        before_pages: dict[str, list[dict[str, Any]]] = {}
        changed = 0
        for page_name in target_page_names:
            items = pages.get(page_name)
            if not isinstance(items, list):
                continue
            before_pages[page_name] = copy.deepcopy(items)
            for item in items:
                if not isinstance(item, dict):
                    continue
                if item.get('font_size') == size:
                    continue
                item['font_size'] = size
                changed += 1
        if changed == 0:
            self.status_label.setText(f'字體大小已經都是 {size}，沒有修改。')
            return

        self.push_undo_pages_snapshot(
            '統一調整字體大小',
            before_pages,
            self.current_page_name(),
            self.selected_index,
        )
        scope_text = '全部頁面' if scope == 'all' else '當前頁'
        self.mark_dirty(f'已把{scope_text} {changed} 個文字框字體大小改為 {size}，尚未保存。')
        self.refresh_after_measure_change(refit=False)

    def measure_xyxy_from_item(self, item: dict[str, Any]) -> tuple[int, int, int, int] | None:
        return xyxy_from_item(item)

    def set_measure_xyxy(self, item: dict[str, Any], xyxy: tuple[int, int, int, int]) -> None:
        clamped = self.clamp_xyxy(xyxy)
        item['xyxy_pixel'] = list(clamped)
        center = normalized_center_from_xyxy(clamped, self.image_size())
        if center is not None:
            item['center_normalized'] = list(center)

    def clamp_xyxy(self, xyxy: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        image_size = self.image_size()
        if image_size is None:
            return xyxy
        width, height = image_size
        x1, y1, x2, y2 = [int(round(float(v))) for v in xyxy]
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(1, min(x2, width))
        y2 = max(1, min(y2, height))
        if x2 - x1 < 4:
            x2 = min(width, x1 + 4)
        if y2 - y1 < 4:
            y2 = min(height, y1 + 4)
        return x1, y1, x2, y2

    def hit_test_item(self, x: float, y: float) -> tuple[int | None, str | None]:
        handles = (
            ('tl', -1, -1), ('t', 0, -1), ('tr', 1, -1),
            ('l', -1, 0), ('r', 1, 0),
            ('bl', -1, 1), ('b', 0, 1), ('br', 1, 1),
        )
        tolerance = 8.0
        matches = []
        boxes: list[tuple[int, tuple[int, int, int, int]]] = []
        if self.page is not None:
            for fallback_index, box in enumerate(self.page.boxes):
                index = box.measure_item_index if box.measure_item_index is not None else fallback_index
                boxes.append((index, box.xyxy_pixel))
        else:
            for index, item in enumerate(self.current_items()):
                if not isinstance(item, dict):
                    continue
                xyxy = self.measure_xyxy_from_item(item)
                if xyxy is not None:
                    boxes.append((index, xyxy))
        for index, xyxy in boxes:
            x1, y1, x2, y2 = xyxy
            for mode, sx, sy in handles:
                hx = (x1 + x2) / 2 if sx == 0 else (x1 if sx < 0 else x2)
                hy = (y1 + y2) / 2 if sy == 0 else (y1 if sy < 0 else y2)
                if abs(x - hx) <= tolerance and abs(y - hy) <= tolerance:
                    matches.append((0, index, mode))
            if x1 <= x <= x2 and y1 <= y <= y2:
                matches.append(((x2 - x1) * (y2 - y1), index, 'move'))
        if not matches:
            return None, None
        _, index, mode = min(matches, key=lambda entry: entry[0])
        return index, mode

    def handle_mouse_press(self, x: float, y: float) -> None:
        index, mode = self.hit_test_item(x, y)
        self.select_box_item(index)
        item = self.selected_item()
        xyxy = self.measure_xyxy_from_item(item) if item is not None else None
        if item is None or xyxy is None or mode is None:
            self._drag_mode = None
            self._drag_start = None
            self._drag_original = None
            return
        self._drag_mode = mode
        self._drag_start = (x, y)
        self._drag_original = xyxy
        self._drag_original_items = copy.deepcopy(self.current_items())
        self._drag_changed = False

    def handle_mouse_drag(self, x: float, y: float) -> None:
        item = self.selected_item()
        if item is None or self._drag_mode is None or self._drag_start is None or self._drag_original is None:
            return
        dx = int(round(x - self._drag_start[0]))
        dy = int(round(y - self._drag_start[1]))
        x1, y1, x2, y2 = self._drag_original
        if self._drag_mode == 'move':
            new_xyxy = (x1 + dx, y1 + dy, x2 + dx, y2 + dy)
        else:
            nx1, ny1, nx2, ny2 = x1, y1, x2, y2
            if 'l' in self._drag_mode:
                nx1 += dx
            if 'r' in self._drag_mode:
                nx2 += dx
            if 't' in self._drag_mode:
                ny1 += dy
            if 'b' in self._drag_mode:
                ny2 += dy
            new_xyxy = (nx1, ny1, nx2, ny2)
        self.set_measure_xyxy(item, new_xyxy)
        self._drag_changed = True
        self.refresh_after_measure_change(refit=False)

    def handle_mouse_release(self, x: float, y: float) -> None:
        if self._drag_changed and self._drag_original_items is not None:
            self.push_undo_snapshot('移動/縮放文字框', self._drag_original_items, self.selected_index)
            self.mark_dirty('已修改 measure 框，尚未保存。')
            self.update_action_state()
        self._drag_mode = None
        self._drag_start = None
        self._drag_original = None
        self._drag_original_items = None
        self._drag_changed = False

    def update_hover_char_box(self, x: float, y: float) -> None:
        if self.page is None or not self.show_char_boxes.isChecked() or self._drag_mode is not None:
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
        self.char_info_label.setText(char_info_text(item))
        self.render_page(refit=False)

    def clear_hover_char_box(self, render: bool = True) -> None:
        had_hover = self.hover_char_box is not None
        self.hover_char_box = None
        self.char_info_label.setText('游標單字框：未選中')
        if render and had_hover:
            self.render_page(refit=False)

    def copy_selected_item(self) -> None:
        item = self.selected_item()
        if item is None:
            return
        before_items = copy.deepcopy(self.current_items())
        before_selected = self.selected_index
        new_item = copy.deepcopy(item)
        xyxy = self.measure_xyxy_from_item(new_item)
        if xyxy is not None:
            x1, y1, x2, y2 = xyxy
            self.set_measure_xyxy(new_item, (x1 + 16, y1 + 16, x2 + 16, y2 + 16))
        items = self.current_items()
        insert_at = len(items) if self.selected_index is None else self.selected_index + 1
        items.insert(insert_at, new_item)
        self.selected_index = insert_at
        self.push_undo_snapshot('複製文字框', before_items, before_selected)
        self.mark_dirty('已複製 measure 文字框，尚未保存。')
        self.refresh_after_measure_change(refit=False)

    def delete_selected_item(self) -> None:
        if self.selected_index is None:
            return
        items = self.current_items()
        if not (0 <= self.selected_index < len(items)):
            return
        before_items = copy.deepcopy(items)
        before_selected = self.selected_index
        del items[self.selected_index]
        self.selected_index = None
        self.push_undo_snapshot('刪除文字框', before_items, before_selected)
        self.mark_dirty('已刪除 measure 文字框，尚未保存。')
        self.refresh_after_measure_change(refit=False)

    def push_undo_snapshot(
        self,
        label: str,
        before_items: list[dict[str, Any]],
        before_selected: int | None,
    ) -> None:
        page_name = self.current_page_name()
        if not page_name:
            return
        self.undo_stack.append(
            {
                'label': label,
                'page_name': page_name,
                'items': copy.deepcopy(before_items),
                'selected_index': before_selected,
            }
        )

    def push_undo_pages_snapshot(
        self,
        label: str,
        before_pages: dict[str, list[dict[str, Any]]],
        selected_page_name: str | None,
        before_selected: int | None,
    ) -> None:
        if not before_pages:
            return
        self.undo_stack.append(
            {
                'label': label,
                'pages': copy.deepcopy(before_pages),
                'selected_page_name': selected_page_name,
                'selected_index': before_selected,
            }
        )

    def undo_last_change(self) -> None:
        if not self.undo_stack:
            self.status_label.setText('沒有可撤銷的 measure 修改。')
            self.update_action_state()
            return
        entry = self.undo_stack.pop()
        if isinstance(entry.get('pages'), dict):
            pages = self.measure.setdefault('pages', {})
            for page_name, items in entry['pages'].items():
                pages[str(page_name)] = copy.deepcopy(items)
            selected_page_name = str(entry.get('selected_page_name') or '')
            if selected_page_name in self.page_names and selected_page_name != self.current_page_name():
                self.page_list.setCurrentRow(self.page_names.index(selected_page_name))
            selected = entry.get('selected_index')
            self.selected_index = selected if isinstance(selected, int) else None
            if self.selected_index is not None and not (0 <= self.selected_index < len(self.current_items())):
                self.selected_index = None
            self.mark_dirty(f'已撤銷：{entry.get("label", "measure 修改")}，尚未保存。')
            self.refresh_after_measure_change(refit=False)
            return
        page_name = str(entry.get('page_name') or '')
        if page_name not in self.page_names:
            self.status_label.setText('撤銷失敗：找不到原頁面。')
            self.update_action_state()
            return
        if page_name != self.current_page_name():
            self.page_list.setCurrentRow(self.page_names.index(page_name))
        pages = self.measure.setdefault('pages', {})
        pages[page_name] = copy.deepcopy(entry.get('items') or [])
        selected = entry.get('selected_index')
        self.selected_index = selected if isinstance(selected, int) else None
        if self.selected_index is not None and not (0 <= self.selected_index < len(self.current_items())):
            self.selected_index = None
        self.mark_dirty(f'已撤銷：{entry.get("label", "measure 修改")}，尚未保存。')
        self.refresh_after_measure_change(refit=False)

    def mark_dirty(self, message: str) -> None:
        self.dirty = True
        self.status_label.setText(message)
        self.update_action_state()

    def update_action_state(self) -> None:
        self.save_button.setEnabled(self.dirty)
        self.save_action.setEnabled(self.dirty)
        has_item = self.selected_item() is not None
        self.copy_button.setEnabled(has_item)
        self.delete_button.setEnabled(has_item)
        self.delete_action.setEnabled(has_item)
        self.font_size_list.setEnabled(has_item)
        self.undo_button.setEnabled(bool(self.undo_stack))
        self.undo_action.setEnabled(bool(self.undo_stack))
        for action in (
            self.increase_font_action,
            self.decrease_font_action,
            self.increase_font_10_action,
            self.decrease_font_10_action,
        ):
            action.setEnabled(has_item)
        suffix = ' *' if self.dirty else ''
        self.setWindowTitle(f'編輯 measure.json{suffix}')

    def save_measure(self) -> None:
        normalize_measure_map(self.measure)
        self.measure_path.write_text(
            json.dumps(self.measure, ensure_ascii=False, indent=2) + '\n',
            encoding='utf-8',
        )
        self.processor.measure = copy.deepcopy(self.measure)
        self.dirty = False
        self.undo_stack.clear()
        self.save_editor_position()
        self.status_label.setText(f'已保存：{self.measure_path}')
        self.update_action_state()
        self.saved.emit()

    def open_ocr_dialog(self) -> None:
        if not self.measure_path.is_file() and not self.dirty:
            QMessageBox.information(self, '缺少 measure.json', '請先生成 CTD，確保 ctd/measure.json 已存在。')
            return
        if self.dirty:
            result = QMessageBox.question(
                self,
                '尚未保存',
                'OCR 會讀取磁碟上的 measure.json。要先保存目前修改再開始嗎？',
                QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Save,
            )
            if result != QMessageBox.StandardButton.Save:
                return
            self.save_measure()

        selected = self.selected_item()
        source_block_index = None
        if selected is not None:
            try:
                source_block_index = int(selected.get('source_block_index', self.selected_index))
            except (TypeError, ValueError):
                source_block_index = self.selected_index
        dialog = MeasureOcrDialog(
            self.processor,
            self.current_page_name(),
            source_block_index,
            self,
        )
        dialog.completed.connect(self.handle_ocr_completed)
        dialog.applyRequested.connect(self.apply_ocr_font_size_updates)
        dialog.exec()

    def handle_ocr_completed(self, output_path: str) -> None:
        self.processor.measure_ocr_path = Path(output_path)
        self.processor.measure_ocr = json.loads(Path(output_path).read_text(encoding='utf-8'))
        self.refresh_after_measure_change(refit=False)
        self.status_label.setText(f'OCR 已完成：{output_path}')

    def apply_ocr_font_size_updates(self, updates: object) -> None:
        if not isinstance(updates, dict):
            return
        previous_method = self.measure.get('font_size_calculation_method')
        pages = self.measure.setdefault('pages', {})
        before_pages: dict[str, list[dict[str, Any]]] = {}
        changed = 0
        for page_name, page_updates in updates.items():
            items = pages.get(page_name)
            if not isinstance(items, list) or not isinstance(page_updates, dict):
                continue
            original_items = copy.deepcopy(items)
            page_changed = 0
            for raw_index, raw_size in page_updates.items():
                try:
                    item_index = int(raw_index)
                    font_size = round(max(0.1, min(999.0, float(raw_size))), 1)
                except (TypeError, ValueError):
                    continue
                if not (0 <= item_index < len(items)) or not isinstance(items[item_index], dict):
                    continue
                item = items[item_index]
                try:
                    current_size = float(item.get('font_size'))
                except (TypeError, ValueError):
                    current_size = 0.0
                if abs(current_size - font_size) < 0.05:
                    continue
                item.setdefault('font_size_detected', item.get('font_size'))
                item['font_size'] = font_size
                item['font_size_method'] = 'mit48_cached_font_ink_candidate_grid'
                page_changed += 1
            if page_changed:
                before_pages[str(page_name)] = original_items
                changed += page_changed

        calibration = self.processor.measure_ocr.get('font_calibration') or {}
        self.measure['font_size_calculation_method'] = 'ocr_aligned'
        self.measure['font_size_calculation_settings'] = {
            'default_font_size': round(float(calibration.get('default_font_size', 24.0)), 1),
            'font_size_step': round(float(calibration.get('font_size_step', 2.0)), 1),
        }
        self.processor.font_size_calculation_method = 'ocr_aligned'
        method_changed = previous_method != 'ocr_aligned'

        if changed == 0 and not method_changed:
            self.status_label.setText('OCR 字級建議與目前值相同，沒有修改。')
            return
        if changed:
            self.push_undo_pages_snapshot(
                '套用 OCR 字級校準',
                before_pages,
                self.current_page_name(),
                self.selected_index,
            )
        if changed:
            status = f'已套用 {changed} 個 OCR 字級建議，尚未保存。'
        else:
            status = '字級數值相同，已切換為 OCR 對齊逐字計算，尚未保存。'
        self.mark_dirty(status)
        self.refresh_after_measure_change(refit=False)

    def render_page(self, *_, refit: bool = True) -> None:
        if self.page is None:
            self.refresh_page_overlay()
        if self.page is None:
            return
        image = QImage(str(self.page.image_path))
        if image.isNull():
            return
        image = image.convertToFormat(QImage.Format.Format_RGBA8888)
        painter = QPainter(image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        if self.show_char_boxes.isChecked():
            draw_char_boxes(painter, self.page, self.hover_char_box, self.view)
        draw_measure_boxes(
            painter,
            self.page,
            self.selected_index,
            show_center_marker=self.show_center_marker.isChecked(),
        )
        if self.show_font_labels.isChecked():
            draw_font_labels(painter, self.page, image.width(), image.height())
        painter.end()
        self.view.set_pixmap(QPixmap.fromImage(image), fit=refit)

    def closeEvent(self, event) -> None:
        self.save_editor_position()
        if not self.dirty:
            event.accept()
            return
        result = QMessageBox.question(
            self,
            '尚未保存',
            'measure.json 有未保存修改，要保存後關閉嗎？',
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save,
        )
        if result == QMessageBox.StandardButton.Cancel:
            event.ignore()
            return
        if result == QMessageBox.StandardButton.Save:
            self.save_measure()
        event.accept()
