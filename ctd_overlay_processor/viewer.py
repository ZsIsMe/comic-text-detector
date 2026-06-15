#!/usr/bin/env python3
"""ctd JSON/NPZ 即時疊圖檢視器。

檢視器會在記憶體中根據原圖、measure.custom.json、ctd/progressing/*.json
和 align/masks/*.npz 重建疊圖，不會輸出預覽 PNG。
"""

from __future__ import annotations

import argparse
import copy
import sys
import traceback
from pathlib import Path

import numpy as np

QT_IMPORT_ERROR: ModuleNotFoundError | None = None
try:
    from PySide6.QtCore import QPointF, QProcess, QRectF, QSettings, Qt, Signal
    from PySide6.QtGui import QAction, QBrush, QColor, QFont, QImage, QKeySequence, QPainter, QPen, QPixmap
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QDialog,
        QDialogButtonBox,
        QDockWidget,
        QFileDialog,
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

try:
    from .processor import (
        BoxOverlay,
        CtdOverlayProcessor,
        PageOverlay,
        normalized_center_from_xyxy,
        tuple_center,
        xyxy_from_item,
    )
except ImportError:
    from processor import (
        BoxOverlay,
        CtdOverlayProcessor,
        PageOverlay,
        normalized_center_from_xyxy,
        tuple_center,
        xyxy_from_item,
    )


if QT_IMPORT_ERROR is None:


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


    class ImageView(QGraphicsView):
        imageMouseMoved = Signal(float, float)
        imageMouseLeft = Signal()
        imageMousePressed = Signal(float, float)
        imageMouseDragged = Signal(float, float)
        imageMouseReleased = Signal(float, float)

        def __init__(self) -> None:
            super().__init__()
            self.setScene(QGraphicsScene(self))
            self.pixmap_item = QGraphicsPixmapItem()
            self.scene().addItem(self.pixmap_item)
            self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
            self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
            self.setBackgroundBrush(QColor(32, 34, 36))
            self.setMouseTracking(True)
            self.viewport().setMouseTracking(True)
            self._zoom = 1.0
            self._mouse_down_on_image = False

        def set_pixmap(self, pixmap: QPixmap, fit: bool = True) -> None:
            if self.pixmap_item.scene() is None:
                self.scene().addItem(self.pixmap_item)
            self.pixmap_item.setPixmap(pixmap)
            self.scene().setSceneRect(QRectF(pixmap.rect()))
            if fit:
                self._zoom = 1.0
                self.resetTransform()
                self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

        def wheelEvent(self, event) -> None:
            factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
            self._zoom *= factor
            self.scale(factor, factor)

        def mouseMoveEvent(self, event) -> None:
            super().mouseMoveEvent(event)
            scene_point = self.mapToScene(event.position().toPoint())
            pixmap_rect = QRectF(self.pixmap_item.pixmap().rect())
            if pixmap_rect.contains(scene_point):
                if self._mouse_down_on_image and event.buttons() & Qt.MouseButton.LeftButton:
                    self.imageMouseDragged.emit(scene_point.x(), scene_point.y())
                    event.accept()
                    return
                self.imageMouseMoved.emit(scene_point.x(), scene_point.y())
            else:
                self.imageMouseLeft.emit()

        def mousePressEvent(self, event) -> None:
            scene_point = self.mapToScene(event.position().toPoint())
            pixmap_rect = QRectF(self.pixmap_item.pixmap().rect())
            if event.button() == Qt.MouseButton.LeftButton and pixmap_rect.contains(scene_point):
                self._mouse_down_on_image = True
                self.imageMousePressed.emit(scene_point.x(), scene_point.y())
                event.accept()
                return
            super().mousePressEvent(event)

        def mouseReleaseEvent(self, event) -> None:
            scene_point = self.mapToScene(event.position().toPoint())
            if self._mouse_down_on_image and event.button() == Qt.MouseButton.LeftButton:
                self.imageMouseReleased.emit(scene_point.x(), scene_point.y())
                self._mouse_down_on_image = False
                event.accept()
                return
            self._mouse_down_on_image = False
            super().mouseReleaseEvent(event)

        def leaveEvent(self, event) -> None:
            self.imageMouseLeft.emit()
            super().leaveEvent(event)


    class ErrorDetailsDialog(QDialog):
        def __init__(self, title: str, summary: str, details: str, parent=None) -> None:
            super().__init__(parent)
            self.setWindowTitle(title)
            self.resize(820, 560)
            self.details = details

            layout = QVBoxLayout(self)
            label = QLabel(summary)
            label.setWordWrap(True)
            layout.addWidget(label)

            text = QPlainTextEdit()
            text.setPlainText(details)
            text.setReadOnly(True)
            text.selectAll()
            layout.addWidget(text, 1)

            copy_button = QPushButton('複製完整信息')
            copy_button.clicked.connect(self.copy_details)
            close_button = QPushButton('關閉')
            close_button.clicked.connect(self.accept)
            buttons = QDialogButtonBox()
            buttons.addButton(copy_button, QDialogButtonBox.ButtonRole.ActionRole)
            buttons.addButton(close_button, QDialogButtonBox.ButtonRole.AcceptRole)
            layout.addWidget(buttons)

        def copy_details(self) -> None:
            QApplication.clipboard().setText(self.details)


    class ConfirmPreviewDialog(QDialog):
        def __init__(self, title: str, summary: str, preview: str, parent=None) -> None:
            super().__init__(parent)
            self.setWindowTitle(title)
            self.resize(520, 640)

            layout = QVBoxLayout(self)
            label = QLabel(summary)
            label.setWordWrap(True)
            layout.addWidget(label)

            text = QPlainTextEdit()
            text.setPlainText(preview)
            text.setReadOnly(True)
            layout.addWidget(text, 1)

            buttons = QDialogButtonBox(
                QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
            )
            buttons.button(QDialogButtonBox.StandardButton.Ok).setText('確定修改')
            buttons.button(QDialogButtonBox.StandardButton.Cancel).setText('取消')
            buttons.accepted.connect(self.accept)
            buttons.rejected.connect(self.reject)
            layout.addWidget(buttons)


    def show_error_details(parent, title: str, summary: str, details: str) -> None:
        dialog = ErrorDetailsDialog(title, summary, details, parent)
        dialog.exec()


    def show_exception_details(parent, title: str, summary: str, exc: BaseException) -> None:
        details = (
            '【錯誤類型】\n'
            f'{type(exc).__name__}\n\n'
            '【錯誤內容】\n'
            f'{exc}\n\n'
            '【完整 traceback】\n'
            f'{traceback.format_exc()}'
        )
        show_error_details(parent, title, summary, details)


    def compact_px(value) -> str | None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if number <= 0:
            return None
        if abs(number - round(number)) < 0.05:
            return str(int(round(number)))
        return f'{number:.1f}'


    def compact_int_px(value) -> str | None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if number <= 0:
            return None
        return str(int(round(number)))


    def even_font_size(value) -> int | None:
        try:
            size = int(round(float(value)))
        except (TypeError, ValueError):
            return None
        if size <= 0:
            return None
        if size % 2:
            size += 1
        return size


    class CtdOverlayViewer(QMainWindow):
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
            self.detect_process: QProcess | None = None
            self.detect_output_chunks: list[str] = []
            self.detect_command: list[str] = []
            self.hover_char_box: dict | None = None
            self.selected_box_index: int | None = None
            self.undo_stack: list[dict[str, object]] = []
            self.save_action: QAction | None = None
            self.undo_action: QAction | None = None
            self.prev_page_action: QAction | None = None
            self.next_page_action: QAction | None = None
            self.increase_font_action: QAction | None = None
            self.decrease_font_action: QAction | None = None
            self.measure_dirty = False
            self.current_page_row = -1
            self._updating_editor = False
            self._box_drag_mode: str | None = None
            self._box_drag_start: tuple[float, float] | None = None
            self._box_drag_original: tuple[int, int, int, int] | None = None

            self.view = ImageView()
            self.setCentralWidget(self.view)

            self.page_list = QListWidget()
            self.font_size_table = QTableWidget()
            self.status_label = QLabel('尚未選擇資料夾')
            self.status_label.setWordWrap(True)
            self.show_mask = QCheckBox('文字遮罩')
            self.show_block_boxes = QCheckBox('原始區塊框')
            self.show_align_boxes = QCheckBox('重定位框')
            self.show_line_polygons = QCheckBox('文字行多邊形')
            self.show_char_boxes = QCheckBox('單字框')
            self.show_npz_smoothed = QCheckBox('NPZ 平滑遮罩')
            self.show_npz_outer = QCheckBox('NPZ 外輪廓遮罩')
            self.show_font_labels = QCheckBox('字級標籤')
            self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
            self.generate_button = QPushButton('生成/更新 CTD')
            self.save_button = QPushButton('保存修改')
            self.even_font_button = QPushButton('字體取偶數')
            self.char_info_label = QLabel('游標單字框：未選中')
            self.char_info_label.setWordWrap(True)
            self.box_editor_title = QLabel('未選擇文字框')
            self.box_editor_title.setWordWrap(True)
            self.font_size_spin = QSpinBox()
            self.color_combo = QComboBox()
            self.text_has_stroke_check = QCheckBox('原字描邊')
            self.need_inpaint_check = QCheckBox('需要修復/描邊')

            for checkbox in (
                self.show_align_boxes,
                self.show_font_labels,
                self.show_mask,
            ):
                checkbox.setChecked(True)
            self.opacity_slider.setRange(5, 90)
            self.opacity_slider.setValue(35)

            self._build_toolbar()
            self._build_side_panel()
            self._build_font_size_panel()
            self._connect_signals()

            if startup_image_dir:
                self.load_folder(startup_image_dir)

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

            fit_action = QAction('適合視窗', self)
            fit_action.triggered.connect(lambda: self.view.fitInView(self.view.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio))
            toolbar.addAction(fit_action)

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

            self.save_action = QAction('保存', self)
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
            self.addAction(self.increase_font_action)

            self.decrease_font_action = QAction('字體-2', self)
            self.decrease_font_action.setShortcuts([QKeySequence('Meta+-'), QKeySequence('Ctrl+-')])
            self.decrease_font_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
            self.decrease_font_action.triggered.connect(lambda: self.nudge_selected_font_size(-2))
            self.addAction(self.decrease_font_action)

        def _build_side_panel(self) -> None:
            panel = QWidget()
            layout = QVBoxLayout(panel)
            layout.setContentsMargins(10, 10, 10, 10)
            layout.setSpacing(8)

            layout.addWidget(QLabel('頁面'))
            layout.addWidget(self.page_list, 1)
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
            layout.addWidget(self.status_label)
            layout.addWidget(self.char_info_label)
            layout.addWidget(QLabel('當前文字框'))
            layout.addWidget(self.box_editor_title)
            self.font_size_spin.setRange(1, 999)
            self.font_size_spin.setSuffix(' px')
            self.color_combo.addItem('黑', 'black')
            self.color_combo.addItem('白', 'white')
            layout.addWidget(QLabel('字體大小'))
            layout.addWidget(self.font_size_spin)
            layout.addWidget(QLabel('文字顏色'))
            layout.addWidget(self.color_combo)
            layout.addWidget(self.text_has_stroke_check)
            layout.addWidget(self.need_inpaint_check)
            self.save_button.clicked.connect(self.save_pending_changes)
            layout.addWidget(self.save_button)
            hint = QLabel('修改後先暫存；Command+S、保存、或切換頁面時寫入 measure.custom.json')
            hint.setWordWrap(True)
            layout.addWidget(hint)
            self.set_box_editor_enabled(False)

            reload_button = QPushButton('重新載入目前頁')
            reload_button.clicked.connect(self.reload_current_page)
            layout.addWidget(reload_button)
            self.generate_button.clicked.connect(self.generate_ctd)
            layout.addWidget(self.generate_button)

            dock = QDockWidget('CTD 資料', self)
            dock.setWidget(panel)
            dock.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
            self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)

        def _build_font_size_panel(self) -> None:
            panel = QWidget()
            layout = QVBoxLayout(panel)
            layout.setContentsMargins(10, 10, 10, 10)
            layout.setSpacing(8)

            title = QLabel('全部字級')
            title.setWordWrap(True)
            layout.addWidget(title)
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
            layout.addWidget(self.font_size_table, 1)
            self.even_font_button.clicked.connect(self.preview_even_font_sizes)
            layout.addWidget(self.even_font_button)

            dock = QDockWidget('字級列表', self)
            dock.setWidget(panel)
            dock.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
            self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

        def _connect_signals(self) -> None:
            self.page_list.currentRowChanged.connect(self.handle_page_row_changed)
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
                widget.stateChanged.connect(self.render_current_page)
            self.opacity_slider.valueChanged.connect(self.render_current_page)
            self.view.imageMouseMoved.connect(self.update_hover_char_box)
            self.view.imageMouseLeft.connect(self.clear_hover_char_box)
            self.view.imageMousePressed.connect(self.handle_image_mouse_press)
            self.view.imageMouseDragged.connect(self.handle_image_mouse_drag)
            self.view.imageMouseReleased.connect(self.handle_image_mouse_release)
            self.font_size_spin.valueChanged.connect(self.apply_editor_changes_to_selected_box)
            self.color_combo.currentIndexChanged.connect(self.apply_editor_changes_to_selected_box)
            self.text_has_stroke_check.stateChanged.connect(self.apply_editor_changes_to_selected_box)
            self.need_inpaint_check.stateChanged.connect(self.apply_editor_changes_to_selected_box)
            self.font_size_table.cellClicked.connect(self.apply_font_size_from_table)

        def choose_folder(self) -> None:
            start_dir = self.current_image_dir or self._last_existing_image_dir() or str(Path.home())
            folder = QFileDialog.getExistingDirectory(self, '選擇包含原圖和 ctd 的圖片資料夾', start_dir)
            if folder:
                self.load_folder(folder)

        def load_folder(self, image_dir: str) -> None:
            image_dir = str(Path(image_dir).expanduser().resolve())
            self.current_image_dir = image_dir
            self.clear_hover_char_box(render=False)
            self.page = None
            self.selected_box_index = None
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
            else:
                self.page = None
                self.current_page_row = -1
                self.view.scene().clear()
                self.update_font_size_list()
                self.status_label.setText(f'{self.status_label.text()}\n此資料夾沒有可顯示的圖片。')

        def handle_page_row_changed(self, row: int) -> None:
            if row == self.current_page_row:
                return
            previous_row = self.current_page_row
            if self.measure_dirty and not self.save_pending_changes(auto=True):
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
            if self.processor is None:
                QMessageBox.information(self, '尚未載入資料', '請先選擇圖片資料夾。')
                return
            if not self.processor.measure_path.is_file():
                QMessageBox.information(self, '找不到 measure.custom.json', '目前資料夾沒有可修改的 measure.custom.json。')
                return

            counts, changed = self.even_font_size_preview()
            if not counts:
                QMessageBox.information(self, '沒有字級資料', 'measure.custom.json 裡沒有可修改的 font_size。')
                return
            if changed == 0:
                QMessageBox.information(self, '不需要修改', '全部 font_size 取整後已經是偶數。')
                return

            dialog = ConfirmPreviewDialog(
                '字體取偶數',
                f'將修改全部頁面 {changed} 個區塊的 font_size。下方是保存後全部字級的數目；確定後會立即寫入 measure.custom.json，且不可撤銷。',
                self.format_font_size_counts(counts),
                self,
            )
            if dialog.exec() == QDialog.DialogCode.Accepted:
                self.apply_even_font_sizes()

        def apply_even_font_sizes(self) -> None:
            if self.processor is None:
                return
            pages = self.processor.measure.get('pages') or {}
            changed = 0
            for items in pages.values():
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
                        item['font_size'] = new_size
                        changed += 1

            if changed == 0:
                return

            self.undo_stack.clear()
            try:
                self.processor.save_measure()
            except Exception as exc:
                show_exception_details(self, '保存失敗', '無法寫入 measure.custom.json。下方是完整可複製的出錯信息。', exc)
                return
            self.measure_dirty = False
            self.refresh_current_page_from_measure(status=f'已對全部頁面套用字體取偶數並保存：{changed} 個區塊。')
            QMessageBox.information(self, '修改完成', f'已對全部頁面套用並保存 {changed} 個區塊。此操作不可撤銷。')

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
                self.font_size_spin,
                self.color_combo,
                self.text_has_stroke_check,
                self.need_inpaint_check,
            ):
                widget.setEnabled(enabled)
            if not enabled:
                self.box_editor_title.setText('未選擇文字框')
            self.update_action_state()

        def select_box(self, index: int | None) -> None:
            if self.page is None or index is None or index < 0 or index >= len(self.page.boxes):
                self.selected_box_index = None
                self.set_box_editor_enabled(False)
                self.render_current_page(refit=False)
                return

            self.selected_box_index = index
            self.populate_box_editor(self.page.boxes[index])
            self.render_current_page(refit=False)

        def populate_box_editor(self, box: BoxOverlay) -> None:
            self._updating_editor = True
            self.set_box_editor_enabled(True)
            x1, y1, x2, y2 = box.xyxy_pixel
            self.box_editor_title.setText(
                f'區塊 {box.source_block_index}  框：{x1},{y1},{x2},{y2}'
            )
            self.font_size_spin.setValue(max(1, int(round(float(box.font_size or 1)))))
            color_index = self.color_combo.findData(box.text_color or 'black')
            self.color_combo.setCurrentIndex(max(0, color_index))
            self.text_has_stroke_check.setChecked(box.text_has_stroke is True)
            self.need_inpaint_check.setChecked(box.need_inpaint is True)
            self._updating_editor = False

        def selected_box_updates_from_editor(self) -> dict[str, object] | None:
            if self._updating_editor or self.selected_box() is None:
                return None
            return {
                'font_size': int(self.font_size_spin.value()),
                'text_color': self.color_combo.currentData() or 'black',
                'fg_color_rgb': [255, 255, 255] if self.color_combo.currentData() == 'white' else [0, 0, 0],
                'text_has_stroke': self.text_has_stroke_check.isChecked(),
                'need_inpaint': self.need_inpaint_check.isChecked(),
            }

        def apply_editor_changes_to_selected_box(self) -> None:
            updates = self.selected_box_updates_from_editor()
            if updates is None:
                return
            self.apply_selected_box_updates(updates, status='已修改當前文字框，尚未保存。')

        def apply_font_size_from_table(self, row: int, column: int) -> None:
            if self.selected_box() is None:
                self.status_label.setText('請先在圖片中選擇一個文字框，再點右側字級。')
                return
            item = self.font_size_table.item(row, 0)
            if item is None:
                return
            try:
                size = int(item.text())
            except ValueError:
                return
            self.apply_selected_box_updates({'font_size': size}, status=f'已把當前文字框字體大小改為 {size}，尚未保存。')

        def nudge_selected_font_size(self, delta: int) -> None:
            box = self.selected_box()
            if box is None:
                self.status_label.setText('請先選擇一個文字框，再使用字體大小快捷鍵。')
                return
            current = int(round(float(box.font_size or self.font_size_spin.value() or 1)))
            size = max(1, min(999, current + delta))
            if size == current:
                return
            sign = '+' if delta > 0 else ''
            self.apply_selected_box_updates({'font_size': size}, status=f'已將當前文字框字體大小 {sign}{delta} 到 {size}，尚未保存。')

        def update_action_state(self) -> None:
            can_save = self.processor is not None and self.measure_dirty
            self.save_button.setEnabled(can_save)
            if self.save_action is not None:
                self.save_action.setEnabled(can_save)
            if self.undo_action is not None:
                self.undo_action.setEnabled(bool(self.undo_stack))
            has_pages = bool(self.page_names)
            if self.prev_page_action is not None:
                self.prev_page_action.setEnabled(has_pages and self.page_list.currentRow() > 0)
            if self.next_page_action is not None:
                self.next_page_action.setEnabled(has_pages and self.page_list.currentRow() < len(self.page_names) - 1)
            has_selected_box = self.selected_box() is not None
            if self.increase_font_action is not None:
                self.increase_font_action.setEnabled(has_selected_box)
            if self.decrease_font_action is not None:
                self.decrease_font_action.setEnabled(has_selected_box)
            suffix = ' *' if self.measure_dirty else ''
            self.setWindowTitle(f'CTD 疊圖檢視器{suffix}')

        def mark_measure_dirty(self) -> None:
            self.measure_dirty = True
            self.update_action_state()

        def save_pending_changes(self, *_, auto: bool = False) -> bool:
            if self.processor is None:
                return True
            if not self.measure_dirty:
                if not auto:
                    self.status_label.setText('目前沒有需要保存的修改。')
                return True
            try:
                self.processor.save_measure()
            except Exception as exc:
                show_exception_details(self, '保存失敗', '無法寫入 measure.custom.json。下方是完整可複製的出錯信息。', exc)
                return False

            self.measure_dirty = False
            self.update_action_state()
            page_text = self.page.page_name if self.page is not None else '目前資料'
            self.status_label.setText(f'{page_text}：已保存整頁修改到 measure.custom.json。')
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

        def apply_selected_box_updates(self, updates: dict[str, object], *, status: str = '已修改，尚未保存。') -> bool:
            if self.processor is None or self.page is None:
                return False
            box = self.selected_box()
            if box is None:
                return False

            normalized_updates = self.build_box_updates(box, updates)
            found = self.processor.find_measure_item(
                self.page.page_name,
                box.source_block_index,
                fallback_index=box.measure_item_index,
            )
            if found is None:
                QMessageBox.warning(self, '修改失敗', '找不到對應的 measure.custom.json 區塊。')
                return False
            item_index, old_item = found
            if not self.updates_change_item(old_item, normalized_updates):
                return False

            self.undo_stack.append({
                'kind': 'box',
                'page_name': self.page.page_name,
                'item_index': item_index,
                'source_block_index': box.source_block_index,
                'selected_box_index': self.selected_box_index,
                'item': copy.deepcopy(old_item),
                'description': status,
            })
            if len(self.undo_stack) > 200:
                self.undo_stack = self.undo_stack[-200:]

            item = self.processor.update_measure_item(
                self.page.page_name,
                box.source_block_index,
                normalized_updates,
                fallback_index=box.measure_item_index,
            )
            if item is None:
                QMessageBox.warning(self, '修改失敗', '找不到對應的 measure.custom.json 區塊。')
                return False
            box.measure_item_index = item_index
            self.sync_box_from_measure_item(box, item)
            self.mark_measure_dirty()
            self.populate_box_editor(box)
            self.update_font_size_list()
            self.render_current_page(refit=False)
            self.status_label.setText(status)
            return True

        def undo_last_edit(self) -> None:
            if self.processor is None or not self.undo_stack:
                return
            entry = self.undo_stack.pop()
            kind = entry.get('kind')
            if kind != 'box':
                self.update_action_state()
                return

            page_name = str(entry.get('page_name') or '')
            if self.page is None or page_name != self.page.page_name:
                self.status_label.setText('撤銷只支持當前頁；已忽略其它頁面的撤銷記錄。')
                self.update_action_state()
                return
            item = copy.deepcopy(entry.get('item') or {})
            item_index = entry.get('item_index')
            source_block_index = entry.get('source_block_index')
            items = self.processor.measure_items_for_page(page_name)
            restored = False
            if isinstance(item_index, int) and 0 <= item_index < len(items):
                items[item_index] = item
                restored = True
            elif isinstance(source_block_index, int):
                found = self.processor.find_measure_item(page_name, source_block_index)
                if found is not None:
                    index, _ = found
                    items[index] = item
                    restored = True
            if not restored:
                QMessageBox.warning(self, '撤銷失敗', '找不到可恢復的 measure.custom.json 區塊。')
                self.update_action_state()
                return

            self.mark_measure_dirty()
            if self.page is not None and self.page.page_name == page_name:
                selected_source = source_block_index if isinstance(source_block_index, int) else None
                self.refresh_current_page_from_measure(
                    selected_source_index=selected_source,
                    status='已撤銷上一個修改，尚未保存。',
                    refit=False,
                )
            else:
                self.update_font_size_list()
                self.status_label.setText(f'{page_name}：已撤銷上一個修改，尚未保存。')

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
                self.populate_box_editor(self.page.boxes[self.selected_box_index])
            else:
                self.set_box_editor_enabled(False)
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
            index, mode = self.hit_test_box(x, y)
            self.select_box(index)
            box = self.selected_box()
            if box is None or mode is None:
                self._box_drag_mode = None
                self._box_drag_start = None
                self._box_drag_original = None
                return
            self._box_drag_mode = mode
            self._box_drag_start = (x, y)
            self._box_drag_original = box.xyxy_pixel

        def handle_image_mouse_drag(self, x: float, y: float) -> None:
            if self.page is None or self._box_drag_mode is None or self._box_drag_start is None or self._box_drag_original is None:
                return
            box = self.selected_box()
            if box is None:
                return
            dx = int(round(x - self._box_drag_start[0]))
            dy = int(round(y - self._box_drag_start[1]))
            x1, y1, x2, y2 = self._box_drag_original
            if self._box_drag_mode == 'move':
                new_xyxy = (x1 + dx, y1 + dy, x2 + dx, y2 + dy)
            else:
                nx1, ny1, nx2, ny2 = x1, y1, x2, y2
                if 'l' in self._box_drag_mode:
                    nx1 += dx
                if 'r' in self._box_drag_mode:
                    nx2 += dx
                if 't' in self._box_drag_mode:
                    ny1 += dy
                if 'b' in self._box_drag_mode:
                    ny2 += dy
                new_xyxy = (nx1, ny1, nx2, ny2)
            box.xyxy_pixel = self.clamp_xyxy(new_xyxy)
            self.populate_box_editor(box)
            self.render_current_page(refit=False)

        def handle_image_mouse_release(self, x: float, y: float) -> None:
            box = self.selected_box()
            if box is not None and self._box_drag_mode is not None:
                if self._box_drag_original is None or box.xyxy_pixel != self._box_drag_original:
                    self.apply_selected_box_updates({'xyxy_pixel': list(box.xyxy_pixel)})
            self._box_drag_mode = None
            self._box_drag_start = None
            self._box_drag_original = None

        def clamp_xyxy(self, xyxy: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
            if self.page is None:
                return xyxy
            image_size = qimage_size(self.page.image_path)
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

        def generate_ctd(self) -> None:
            if not self.current_image_dir:
                QMessageBox.information(self, '尚未選擇資料夾', '請先選擇包含原圖的圖片資料夾。')
                return
            if self.detect_process is not None:
                QMessageBox.information(self, '正在處理', 'CTD 資料正在生成中，請稍候。')
                return

            script = Path(__file__).resolve().parent / 'run_detection.py'
            if not script.is_file():
                QMessageBox.critical(self, '找不到生成器', f'找不到：\n{script}')
                return

            self.generate_button.setEnabled(False)
            self.status_label.setText('正在生成 CTD 資料，模型推理可能需要一段時間...')
            process = QProcess(self)
            args = [str(script), self.current_image_dir]
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
            self.detect_process = None
            if process is not None:
                process.deleteLater()

        def _detection_finished(self, exit_code: int, exit_status) -> None:
            self.generate_button.setEnabled(True)
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
                return
            self.status_label.setText('CTD 資料生成完成，正在重新載入...')
            if self.current_image_dir:
                self.load_folder(self.current_image_dir)

        def load_page_at_row(self, row: int) -> None:
            if row < 0 or row >= len(self.page_names) or self.processor is None:
                return
            page_name = self.page_names[row]
            try:
                self.clear_hover_char_box(render=False)
                self.selected_box_index = None
                self.set_box_editor_enabled(False)
                image_size = qimage_size(self.processor.image_dir / page_name)
                self.page = self.processor.load_page(page_name, image_size=image_size)
                self.current_page_row = row
                self.update_font_size_list()
                self.render_current_page()
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
            self.render_current_page(refit=False)

        def clear_hover_char_box(self, render: bool = True) -> None:
            had_hover = self.hover_char_box is not None
            self.hover_char_box = None
            self.char_info_label.setText('游標單字框：未選中')
            if render and had_hover:
                self.render_current_page(refit=False)

        def _char_info_text(self, item: dict) -> str:
            bbox = item.get('bbox')
            width_text = compact_px(item.get('width')) or '-'
            height_text = compact_px(item.get('height')) or '-'
            source_index = item.get('source_block_index', '-')
            line_index = item.get('line_index', '-')
            bbox_text = ', '.join(str(int(round(float(value)))) for value in bbox) if isinstance(bbox, list) else '-'
            return (
                '游標單字框：\n'
                f'寬：{width_text}px  高：{height_text}px\n'
                f'區塊：{source_index}  行：{line_index}\n'
                f'bbox：{bbox_text}'
            )

        def reload_current_page(self) -> None:
            row = self.page_list.currentRow()
            if row >= 0:
                self.load_page_at_row(row)

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
                width_text = compact_int_px(item.get('width'))
                height_text = compact_int_px(item.get('height'))
                if width_text is None or height_text is None:
                    continue

                label = f'W{width_text}H{height_text}'
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
                    width_text = compact_int_px(self.hover_char_box.get('width'))
                    height_text = compact_int_px(self.hover_char_box.get('height'))
                    if width_text is not None and height_text is not None:
                        label = f'W{width_text}H{height_text}'
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
            '尚未安裝 PySide6。請安裝 PySide6 或 PySide6-Essentials 後再啟動。',
            file=sys.stderr,
        )
        raise SystemExit(1)
    app = QApplication(sys.argv)
    viewer = CtdOverlayViewer(args.image_dir)
    viewer.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
