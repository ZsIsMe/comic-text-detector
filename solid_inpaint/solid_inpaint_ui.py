#!/usr/bin/env python3
"""PySide6 UI for Solid Inpaint."""

from __future__ import annotations

import os
import os.path as osp
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QObject, QPoint, QSettings, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QAction, QColor, QImage, QKeySequence, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QGraphicsEllipseItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QCheckBox,
    QProgressBar,
    QSlider,
    QSizePolicy,
    QSplitter,
    QStatusBar,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from detect_solid_inpaint_folder import (
    _compose_overlay_preview,
    _ensure_dirs,
    _mask_path,
    _other_mask_path,
    _output_path,
    _write_preview_pdf,
    build_report,
    create_detector,
    image_files_in_folder,
    load_report,
    process_image_with_detector,
    regenerate_image_from_mask,
    write_report,
)
from utils.io_utils import imread, imwrite


STATUS_OK = ''
STATUS_OTHER = '有 OTHER'
STATUS_FAILED = '失敗'
STATUS_TODO = '未處理'
MAX_RECENT_FOLDERS = 12
MAX_UNDO_STEPS = 30
DEFAULT_BRUSH_RADIUS = 24
MIN_BRUSH_RADIUS = 2
MAX_BRUSH_RADIUS = 160
APP_VERSION = '0.2.0'


def _qimage_from_bgr(img: np.ndarray) -> QImage:
    if len(img.shape) == 2:
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, rgb.strides[0], QImage.Format.Format_RGB888).copy()


def _qimage_from_rgba(img: np.ndarray) -> QImage:
    rgba = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    h, w = rgba.shape[:2]
    return QImage(rgba.data, w, h, rgba.strides[0], QImage.Format.Format_RGBA8888).copy()


def _optional_imread(path: str, flags: int) -> np.ndarray | None:
    if not osp.isfile(path):
        return None
    try:
        return imread(path, flags)
    except Exception:
        return None


def _mask_overlay_image(
    base: np.ndarray,
    mask: np.ndarray | None,
    alpha: float,
    color_bgr: tuple[int, int, int],
) -> np.ndarray:
    base_bgr = base[:, :, :3].copy() if len(base.shape) == 3 else cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    if mask is None:
        return base_bgr
    mask_active = mask > 0
    dimmed = (base_bgr.astype(np.float32) * max(0.0, 1.0 - alpha)).astype(np.uint8)
    color = np.zeros_like(base_bgr)
    color[:, :, 0] = color_bgr[0]
    color[:, :, 1] = color_bgr[1]
    color[:, :, 2] = color_bgr[2]
    blended = dimmed.copy()
    blended[mask_active] = (
        base_bgr[mask_active].astype(np.float32) * (1.0 - alpha)
        + color[mask_active].astype(np.float32) * alpha
    ).astype(np.uint8)
    return blended


def _overlay_mask_on_bgr(
    base_bgr: np.ndarray,
    mask: np.ndarray | None,
    alpha: float,
    color_bgr: tuple[int, int, int],
) -> np.ndarray:
    if mask is None:
        return base_bgr
    active = mask > 0
    if not np.any(active):
        return base_bgr
    color = np.zeros_like(base_bgr)
    color[:, :, 0] = color_bgr[0]
    color[:, :, 1] = color_bgr[1]
    color[:, :, 2] = color_bgr[2]
    output = base_bgr.copy()
    output[active] = (
        output[active].astype(np.float32) * (1.0 - alpha)
        + color[active].astype(np.float32) * alpha
    ).astype(np.uint8)
    return output


class ImageView(QGraphicsView):
    def __init__(self) -> None:
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene().addItem(self.pixmap_item)
        self.setBackgroundBrush(QColor('#0b0d10'))
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self._zoom = 1.0

    def set_qimage(self, image: QImage | None, keep_view: bool = False) -> None:
        if image is None:
            self.pixmap_item.setPixmap(QPixmap())
            self.scene().setSceneRect(0, 0, 1, 1)
            return
        pixmap = QPixmap.fromImage(image)
        self.pixmap_item.setPixmap(pixmap)
        self.scene().setSceneRect(pixmap.rect())
        if not keep_view:
            self.fit()

    def fit(self) -> None:
        pixmap = self.pixmap_item.pixmap()
        if pixmap.isNull():
            return
        self.resetTransform()
        self.fitInView(self.pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
        self._zoom = 1.0

    def actual_size(self) -> None:
        self.resetTransform()
        self._zoom = 1.0

    def wheelEvent(self, event) -> None:
        if self.pixmap_item.pixmap().isNull():
            return
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self._zoom *= factor
        self.scale(factor, factor)


class MaskEditorView(ImageView):
    editStarted = Signal()
    maskEdited = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self.tool = 'brush'
        self.brush_radius = DEFAULT_BRUSH_RADIUS
        self.mask: np.ndarray | None = None
        self.image_shape: tuple[int, int] | None = None
        self._active_button: Qt.MouseButton | None = None
        self._drag_start: tuple[int, int] | None = None
        self._last_brush_point: tuple[int, int] | None = None
        self._panning = False
        self._pan_last_pos: QPoint | None = None
        self._rubber_band: QGraphicsRectItem | None = None
        self._brush_cursor: QGraphicsEllipseItem | None = None
        self._edit_started = False
        self._rect_pen_add = QPen(QColor('#e9fffb'), 2, Qt.PenStyle.DashLine)
        self._rect_pen_remove = QPen(QColor('#ff8f8f'), 2, Qt.PenStyle.DashLine)
        self._brush_pen = QPen(QColor('#e9fffb'), 2)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)

    def set_mask(self, mask: np.ndarray | None, shape: tuple[int, int]) -> None:
        self.image_shape = shape
        if mask is None:
            self.mask = np.zeros(shape, dtype=np.uint8)
        else:
            self.mask = np.where(mask > 0, 255, 0).astype(np.uint8)

    def set_tool(self, tool: str) -> None:
        self.tool = tool
        self._clear_rubber_band()
        self._update_brush_cursor_visibility()

    def set_brush_radius(self, radius: int) -> None:
        self.brush_radius = max(MIN_BRUSH_RADIUS, min(MAX_BRUSH_RADIUS, int(radius)))
        self._move_brush_cursor_to_last_position()

    def image_point_from_view(self, pos: QPoint, clamp: bool = False) -> tuple[int, int] | None:
        if self.image_shape is None:
            return None
        pixmap = self.pixmap_item.pixmap()
        if pixmap.isNull():
            return None
        scene_pos = self.mapToScene(pos)
        item_pos = self.pixmap_item.mapFromScene(scene_pos)
        x = int(round(item_pos.x()))
        y = int(round(item_pos.y()))
        height, width = self.image_shape
        if clamp:
            return max(0, min(width - 1, x)), max(0, min(height - 1, y))
        if x < 0 or y < 0 or x >= width or y >= height:
            return None
        return x, y

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._is_pan_modifier(event.modifiers()):
            self._panning = True
            self._pan_last_pos = event.position().toPoint()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        if event.button() not in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton):
            super().mousePressEvent(event)
            return
        point = self.image_point_from_view(event.position().toPoint())
        if point is None or self.mask is None:
            super().mousePressEvent(event)
            return
        self._active_button = event.button()
        self._drag_start = point
        self._last_brush_point = point
        self._edit_started = False
        if self.tool == 'brush':
            self._begin_edit_once()
            self._paint_brush(point, event.button())
            self.maskEdited.emit(self.mask.copy())
        else:
            self._update_rubber_band(point, point, event.button())
        event.accept()

    def mouseMoveEvent(self, event) -> None:
        if self._panning:
            current = event.position().toPoint()
            if self._pan_last_pos is not None:
                delta = current - self._pan_last_pos
                self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
                self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            self._pan_last_pos = current
            event.accept()
            return
        hover_point = self.image_point_from_view(event.position().toPoint())
        self._update_brush_cursor(hover_point)
        if self._active_button is None:
            event.accept()
            return
        point = self.image_point_from_view(event.position().toPoint(), clamp=True)
        if point is None or self.mask is None:
            event.accept()
            return
        if self.tool == 'brush':
            self._begin_edit_once()
            self._paint_line(self._last_brush_point or point, point, self._active_button)
            self._last_brush_point = point
            self.maskEdited.emit(self.mask.copy())
        else:
            self._update_rubber_band(self._drag_start or point, point, self._active_button)
        event.accept()

    def mouseReleaseEvent(self, event) -> None:
        if self._panning and event.button() == Qt.MouseButton.LeftButton:
            self._panning = False
            self._pan_last_pos = None
            self.unsetCursor()
            event.accept()
            return
        if event.button() != self._active_button:
            super().mouseReleaseEvent(event)
            return
        point = self.image_point_from_view(event.position().toPoint(), clamp=True)
        if self.tool == 'rect' and self.mask is not None and self._drag_start is not None and point is not None:
            self._begin_edit_once()
            self._apply_rect(self._drag_start, point, self._active_button)
            self.maskEdited.emit(self.mask.copy())
        self._active_button = None
        self._drag_start = None
        self._last_brush_point = None
        self._edit_started = False
        self._clear_rubber_band()
        event.accept()

    def leaveEvent(self, event) -> None:
        self._update_brush_cursor(None)
        super().leaveEvent(event)

    def _is_pan_modifier(self, modifiers: Qt.KeyboardModifier) -> bool:
        return bool(
            modifiers & Qt.KeyboardModifier.MetaModifier
            or modifiers & Qt.KeyboardModifier.ControlModifier
        )

    def _begin_edit_once(self) -> None:
        if self._edit_started:
            return
        self._edit_started = True
        self.editStarted.emit()

    def _paint_brush(self, point: tuple[int, int], button: Qt.MouseButton) -> None:
        if self.mask is None:
            return
        color = 255 if button == Qt.MouseButton.LeftButton else 0
        cv2.circle(self.mask, point, self.brush_radius, color, thickness=-1, lineType=cv2.LINE_8)

    def _paint_line(
        self,
        start: tuple[int, int],
        end: tuple[int, int],
        button: Qt.MouseButton,
    ) -> None:
        if self.mask is None:
            return
        color = 255 if button == Qt.MouseButton.LeftButton else 0
        cv2.line(self.mask, start, end, color, thickness=self.brush_radius * 2, lineType=cv2.LINE_8)
        cv2.circle(self.mask, end, self.brush_radius, color, thickness=-1, lineType=cv2.LINE_8)

    def _apply_rect(
        self,
        start: tuple[int, int],
        end: tuple[int, int],
        button: Qt.MouseButton,
    ) -> None:
        if self.mask is None:
            return
        x1, x2 = sorted((start[0], end[0]))
        y1, y2 = sorted((start[1], end[1]))
        if x2 < x1 or y2 < y1:
            return
        value = 255 if button == Qt.MouseButton.LeftButton else 0
        self.mask[y1:y2 + 1, x1:x2 + 1] = value

    def _update_rubber_band(
        self,
        start: tuple[int, int],
        end: tuple[int, int],
        button: Qt.MouseButton,
    ) -> None:
        x1, x2 = sorted((start[0], end[0]))
        y1, y2 = sorted((start[1], end[1]))
        if self._rubber_band is None:
            self._rubber_band = QGraphicsRectItem()
            self._rubber_band.setBrush(QColor(255, 255, 255, 30))
            self.scene().addItem(self._rubber_band)
        self._rubber_band.setPen(
            self._rect_pen_add if button == Qt.MouseButton.LeftButton else self._rect_pen_remove
        )
        self._rubber_band.setRect(x1, y1, max(1, x2 - x1), max(1, y2 - y1))

    def _clear_rubber_band(self) -> None:
        if self._rubber_band is not None:
            self.scene().removeItem(self._rubber_band)
            self._rubber_band = None

    def _update_brush_cursor(self, point: tuple[int, int] | None) -> None:
        if self.tool != 'brush' or point is None:
            if self._brush_cursor is not None:
                self._brush_cursor.setVisible(False)
            return
        if self._brush_cursor is None:
            self._brush_cursor = QGraphicsEllipseItem()
            self._brush_cursor.setBrush(QColor(233, 255, 251, 28))
            self._brush_cursor.setPen(self._brush_pen)
            self._brush_cursor.setZValue(10)
            self.scene().addItem(self._brush_cursor)
        radius = self.brush_radius
        self._brush_cursor.setRect(point[0] - radius, point[1] - radius, radius * 2, radius * 2)
        self._brush_cursor.setVisible(True)

    def _update_brush_cursor_visibility(self) -> None:
        if self.tool != 'brush' and self._brush_cursor is not None:
            self._brush_cursor.setVisible(False)

    def _move_brush_cursor_to_last_position(self) -> None:
        if self._brush_cursor is None or not self._brush_cursor.isVisible():
            return
        rect = self._brush_cursor.rect()
        center = rect.center()
        self._update_brush_cursor((int(center.x()), int(center.y())))


class FolderWorker(QObject):
    progress = Signal(int, int, str)
    finished = Signal(dict)
    failed = Signal(str)

    def __init__(self, folder: str, mode: str, image_paths: list[str] | None = None) -> None:
        super().__init__()
        self.folder = folder
        self.mode = mode
        self.image_paths = image_paths

    def run(self) -> None:
        try:
            paths = _ensure_dirs(self.folder)
            imglist = self.image_paths or image_files_in_folder(self.folder)
            detector = create_detector() if self.mode == 'detect' else None
            existing = load_report(paths)
            pages = dict(existing.get('pages', {}))
            total = len(imglist)
            for index, img_path in enumerate(imglist, start=1):
                name = osp.basename(img_path)
                self.progress.emit(index, total, name)
                try:
                    if self.mode == 'detect':
                        pages[name] = process_image_with_detector(img_path, paths, detector)
                    else:
                        pages[name] = regenerate_image_from_mask(img_path, paths)
                except Exception as exc:
                    pages[name] = {'error': str(exc)}
            report = build_report(self.folder, paths, image_files_in_folder(self.folder), pages)
            write_report(paths, report)
            self.finished.emit(report)
        except Exception as exc:
            self.failed.emit(str(exc))


class PageRegenerateWorker(QObject):
    finished = Signal(str, dict)
    failed = Signal(str, str)

    def __init__(self, folder: str, paths: dict[str, str], imglist: list[str], img_path: str, mask: np.ndarray) -> None:
        super().__init__()
        self.folder = folder
        self.paths = paths
        self.imglist = imglist
        self.img_path = img_path
        self.mask = np.where(mask > 0, 255, 0).astype(np.uint8)

    def run(self) -> None:
        try:
            existing = load_report(self.paths)
            pages = dict(existing.get('pages', {}))
            summary = regenerate_image_from_mask(self.img_path, self.paths, self.mask)
            pages[osp.basename(self.img_path)] = summary
            report = build_report(self.folder, self.paths, self.imglist, pages)
            write_report(self.paths, report)
            self.finished.emit(self.img_path, report)
        except Exception as exc:
            self.failed.emit(self.img_path, str(exc))


class PdfWorker(QObject):
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, imglist: list[str], paths: dict[str, str], report: dict) -> None:
        super().__init__()
        self.imglist = list(imglist)
        self.paths = dict(paths)
        self.report = dict(report)

    def run(self) -> None:
        try:
            pdf_path = _write_preview_pdf(self.imglist, self.paths, self.report)
            self.finished.emit(pdf_path)
        except Exception as exc:
            self.failed.emit(str(exc))


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle('Solid Inpaint')
        self.resize(1500, 900)
        self.folder = ''
        self.paths: dict[str, str] = {}
        self.imglist: list[str] = []
        self.report: dict = {}
        self.current_img_path = ''
        self.current_base: np.ndarray | None = None
        self.current_mask: np.ndarray | None = None
        self.alpha = 1.0
        self.show_other_mask = True
        self.undo_stack: list[np.ndarray] = []
        self.redo_stack: list[np.ndarray] = []
        self.worker_thread: QThread | None = None
        self.worker: FolderWorker | None = None
        self.page_worker_thread: QThread | None = None
        self.page_worker: PageRegenerateWorker | None = None
        self.pdf_worker_thread: QThread | None = None
        self.pdf_worker: PdfWorker | None = None
        self.pending_render_img_path = ''
        self.pending_render_mask: np.ndarray | None = None
        self.suppress_list_selection = False
        self.render_timer = QTimer(self)
        self.render_timer.setSingleShot(True)
        self.render_timer.timeout.connect(self.start_pending_render)
        self.settings = QSettings('ComicTextDetector', 'SolidInpaintUI')
        self.recent_folders = self._load_recent_folders()

        self._build_ui()
        self._apply_style()

    def _build_ui(self) -> None:
        toolbar = QToolBar('工具')
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        choose_action = QAction('選擇文件夾', self)
        choose_action.triggered.connect(self.choose_folder)
        toolbar.addAction(choose_action)

        self.recent_menu = QMenu(self)
        self.recent_button = QToolButton()
        self.recent_button.setText('打開最近列表')
        self.recent_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.recent_button.setMenu(self.recent_menu)
        toolbar.addWidget(self.recent_button)
        self.update_recent_menu()

        prev_action = QAction('上一頁', self)
        prev_action.setShortcuts([QKeySequence('Left'), QKeySequence('PageUp')])
        prev_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        prev_action.triggered.connect(self.previous_image)
        toolbar.addAction(prev_action)

        next_action = QAction('下一頁', self)
        next_action.setShortcuts([QKeySequence('Right'), QKeySequence('PageDown')])
        next_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        next_action.triggered.connect(self.next_image)
        toolbar.addAction(next_action)

        open_output_action = QAction('打開輸出', self)
        open_output_action.triggered.connect(self.open_output)
        toolbar.addAction(open_output_action)

        generate_pdf_action = QAction('生成 PDF', self)
        generate_pdf_action.triggered.connect(self.generate_pdf)
        toolbar.addAction(generate_pdf_action)

        open_pdf_action = QAction('打開 PDF', self)
        open_pdf_action.triggered.connect(self.open_pdf)
        toolbar.addAction(open_pdf_action)

        help_action = QAction('說明', self)
        help_action.triggered.connect(self.show_help)
        toolbar.addAction(help_action)

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        toolbar.addWidget(spacer)

        self.detect_button = QToolButton()
        self.detect_button.setText('偵測並生成')
        self.detect_button.setProperty('danger', True)
        self.detect_button.clicked.connect(self.run_or_load)
        toolbar.addWidget(self.detect_button)

        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(10, 8, 10, 10)
        root_layout.setSpacing(8)

        top_row = QHBoxLayout()
        self.folder_label = QLabel('未選擇文件夾')
        self.progress_label = QLabel('')
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        top_row.addWidget(self.folder_label, 4)
        top_row.addWidget(self.progress_label, 2)
        top_row.addWidget(self.progress, 3)
        root_layout.addLayout(top_row)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root_layout.addWidget(splitter, 1)

        left_panel = QFrame()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(10, 10, 10, 10)
        left_layout.addWidget(QLabel('圖片列表'))
        self.summary_label = QLabel('共 0 張')
        left_layout.addWidget(self.summary_label)
        self.list_widget = QListWidget()
        self.list_widget.currentRowChanged.connect(self.on_image_selected)
        left_layout.addWidget(self.list_widget, 1)
        splitter.addWidget(left_panel)

        center_panel = QFrame()
        center_layout = QVBoxLayout(center_panel)
        center_layout.setContentsMargins(10, 10, 10, 10)
        center_layout.addWidget(QLabel('Mask / 原圖'))
        edit_toolbar = QHBoxLayout()
        self.rect_btn = QPushButton('矩形')
        self.rect_btn.setCheckable(True)
        self.rect_btn.clicked.connect(lambda: self.set_edit_tool('rect'))
        self.brush_btn = QPushButton('筆刷')
        self.brush_btn.setCheckable(True)
        self.brush_btn.setChecked(True)
        self.brush_btn.clicked.connect(lambda: self.set_edit_tool('brush'))
        self.undo_btn = QPushButton('撤銷')
        self.undo_btn.clicked.connect(self.undo_mask)
        self.redo_btn = QPushButton('重做')
        self.redo_btn.clicked.connect(self.redo_mask)
        self.brush_down_btn = QPushButton('-')
        self.brush_down_btn.clicked.connect(lambda: self.change_brush_radius(-4))
        self.brush_up_btn = QPushButton('+')
        self.brush_up_btn.clicked.connect(lambda: self.change_brush_radius(4))
        self.brush_label = QLabel(f'筆刷 {DEFAULT_BRUSH_RADIUS}px')
        edit_toolbar.addWidget(self.brush_btn)
        edit_toolbar.addWidget(self.rect_btn)
        edit_toolbar.addSpacing(10)
        edit_toolbar.addWidget(self.brush_down_btn)
        edit_toolbar.addWidget(self.brush_label)
        edit_toolbar.addWidget(self.brush_up_btn)
        edit_toolbar.addSpacing(10)
        edit_toolbar.addWidget(self.undo_btn)
        edit_toolbar.addWidget(self.redo_btn)
        edit_toolbar.addStretch()
        center_layout.addLayout(edit_toolbar)
        self.mask_view = MaskEditorView()
        self.mask_view.editStarted.connect(self.push_undo_snapshot)
        self.mask_view.maskEdited.connect(self.on_mask_edited)
        center_layout.addWidget(self.mask_view, 1)
        slider_row = QHBoxLayout()
        slider_row.addWidget(QLabel('Mask 顯示'))
        self.alpha_slider = QSlider(Qt.Orientation.Horizontal)
        self.alpha_slider.setRange(0, 100)
        self.alpha_slider.setInvertedAppearance(True)
        self.alpha_slider.setValue(100)
        self.alpha_slider.valueChanged.connect(self.on_alpha_changed)
        self.alpha_label = QLabel('目前 100%')
        slider_row.addWidget(QLabel('100%'))
        slider_row.addWidget(self.alpha_slider, 1)
        slider_row.addWidget(QLabel('0%'))
        slider_row.addWidget(self.alpha_label)
        center_layout.addLayout(slider_row)
        view_buttons = QHBoxLayout()
        fit_btn = QPushButton('適應')
        fit_btn.clicked.connect(self.mask_view.fit)
        actual_btn = QPushButton('100%')
        actual_btn.clicked.connect(self.mask_view.actual_size)
        view_buttons.addWidget(fit_btn)
        view_buttons.addWidget(actual_btn)
        view_buttons.addStretch()
        center_layout.addLayout(view_buttons)
        splitter.addWidget(center_panel)

        right_panel = QFrame()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(10, 10, 10, 10)
        right_layout.addWidget(QLabel('Inpainted 預覽'))
        self.other_mask_checkbox = QCheckBox('顯示 other_mask')
        self.other_mask_checkbox.setChecked(True)
        self.other_mask_checkbox.stateChanged.connect(self.on_show_other_mask_changed)
        right_layout.addWidget(self.other_mask_checkbox)
        self.preview_view = ImageView()
        right_layout.addWidget(self.preview_view, 1)
        self.stats_label = QLabel('未載入')
        self.stats_label.setWordWrap(True)
        right_layout.addWidget(self.stats_label)
        preview_buttons = QHBoxLayout()
        fit_btn2 = QPushButton('適應')
        fit_btn2.clicked.connect(self.preview_view.fit)
        actual_btn2 = QPushButton('100%')
        actual_btn2.clicked.connect(self.preview_view.actual_size)
        preview_buttons.addWidget(fit_btn2)
        preview_buttons.addWidget(actual_btn2)
        preview_buttons.addStretch()
        right_layout.addLayout(preview_buttons)
        splitter.addWidget(right_panel)

        splitter.setSizes([250, 650, 650])
        self.setCentralWidget(root)

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage('第二版：可編輯 mask，編輯後自動生成預覽。')

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background: #111418; color: #e6ebef; }
            QToolBar { background: #171b20; border: 0; spacing: 8px; padding: 8px; }
            QToolButton, QPushButton {
                background: #242b33; color: #e6ebef; border: 1px solid #3a4551;
                border-radius: 5px; padding: 6px 12px;
            }
            QToolButton:hover, QPushButton:hover { background: #2b3440; }
            QPushButton:checked {
                background: #e9fffb; color: #10161a; border-color: #e9fffb;
                font-weight: 600;
            }
            QToolButton[danger="true"] { color: #ff6b6b; border-color: #6d3438; }
            QToolButton[danger="true"]:hover { background: #352329; border-color: #a8454d; }
            QToolButton::menu-indicator { image: none; width: 0; }
            QFrame { background: #1c2128; border: 1px solid #2e3640; border-radius: 6px; }
            QLabel { color: #dfe5ea; border: 0; }
            QMenu { background: #1c2128; color: #e6ebef; border: 1px solid #3a4551; padding: 4px; }
            QMenu::item { padding: 7px 22px 7px 10px; border-radius: 4px; }
            QMenu::item:selected { background: #2b3440; }
            QMenu::item:disabled { color: #7c8792; }
            QListWidget { background: #1c2128; border: 0; color: #dfe5ea; outline: none; }
            QListWidget::item { padding: 8px 6px; border-bottom: 1px solid #27303a; }
            QListWidget::item:selected { background: #243039; color: #ffffff; }
            QProgressBar { background: #27303a; border: 0; border-radius: 4px; height: 8px; text-align: center; }
            QProgressBar::chunk { background: #38a996; border-radius: 4px; }
            QSlider::groove:horizontal { background: #303944; height: 6px; border-radius: 3px; }
            QSlider::handle:horizontal { background: #e9fffb; border: 2px solid #38a996; width: 14px; margin: -5px 0; border-radius: 7px; }
            """
        )

    def choose_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, '選擇圖片文件夾', self.folder or str(Path.home()))
        if not folder:
            return
        self.load_folder(folder)

    def load_folder(self, folder: str) -> None:
        if not osp.isdir(folder):
            QMessageBox.warning(self, '文件夾不存在', folder)
            self.remove_recent_folder(folder)
            return
        self.folder = folder
        self.paths = _ensure_dirs(folder)
        self.imglist = image_files_in_folder(folder)
        self.report = load_report(self.paths)
        self.folder_label.setText(folder)
        self.add_recent_folder(folder)
        self.refresh_list()
        if self.imglist:
            self.list_widget.setCurrentRow(0)

    def _load_recent_folders(self) -> list[str]:
        value = self.settings.value('recent_folders', [])
        if isinstance(value, str):
            folders = [value]
        elif isinstance(value, (list, tuple)):
            folders = [str(item) for item in value]
        else:
            folders = []
        seen = set()
        result = []
        for folder in folders:
            normalized = osp.abspath(osp.expanduser(folder))
            if normalized in seen or not osp.isdir(normalized):
                continue
            seen.add(normalized)
            result.append(normalized)
        return result[:MAX_RECENT_FOLDERS]

    def save_recent_folders(self) -> None:
        self.settings.setValue('recent_folders', self.recent_folders)

    def add_recent_folder(self, folder: str) -> None:
        normalized = osp.abspath(osp.expanduser(folder))
        self.recent_folders = [item for item in self.recent_folders if item != normalized]
        self.recent_folders.insert(0, normalized)
        self.recent_folders = self.recent_folders[:MAX_RECENT_FOLDERS]
        self.save_recent_folders()
        self.update_recent_menu()

    def remove_recent_folder(self, folder: str) -> None:
        normalized = osp.abspath(osp.expanduser(folder))
        self.recent_folders = [item for item in self.recent_folders if item != normalized]
        self.save_recent_folders()
        self.update_recent_menu()

    def clear_recent_folders(self) -> None:
        self.recent_folders = []
        self.save_recent_folders()
        self.update_recent_menu()

    def update_recent_menu(self) -> None:
        self.recent_menu.clear()
        if self.recent_folders:
            for folder in self.recent_folders:
                action = QAction(folder, self)
                action.triggered.connect(lambda checked=False, path=folder: self.load_folder(path))
                self.recent_menu.addAction(action)
        else:
            empty_action = QAction('沒有最近文件夾', self)
            empty_action.setEnabled(False)
            self.recent_menu.addAction(empty_action)

        self.recent_menu.addSeparator()
        clear_action = QAction('清除', self)
        clear_action.setEnabled(bool(self.recent_folders))
        clear_action.triggered.connect(self.clear_recent_folders)
        self.recent_menu.addAction(clear_action)

    def run_or_load(self) -> None:
        if not self.folder:
            self.choose_folder()
            if not self.folder:
                return
        if self.page_worker_thread is not None or self.render_timer.isActive():
            QMessageBox.information(self, '正在生成預覽', '請等待當前頁預覽生成完成後再重新偵測。')
            return
        if self.has_existing_masks():
            reply = QMessageBox.warning(
                self,
                '確認重新偵測',
                '當前文件夾已存在 mask。\n\n'
                '「偵測並生成」會重新跑 detector，並覆蓋已有的 mask、other_mask 和 inpainted 輸出。\n'
                '如果你已經手動修改過 mask，這些修改會丟失。\n\n'
                '確定要繼續嗎？',
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
        self.start_worker('detect', self.imglist)

    def has_existing_masks(self) -> bool:
        if not self.paths:
            return False
        mask_dir = self.paths.get('mask')
        if not mask_dir or not osp.isdir(mask_dir):
            return False
        return any(
            osp.isfile(_mask_path(self.paths, img_path))
            for img_path in self.imglist
        )

    def start_worker(self, mode: str, image_paths: list[str]) -> None:
        if self.worker_thread is not None:
            QMessageBox.information(self, '正在執行', '已有任務在執行中。')
            return
        if self.page_worker_thread is not None:
            QMessageBox.information(self, '正在生成預覽', '請等待當前頁預覽生成完成。')
            return
        self.progress.setValue(0)
        self.worker_thread = QThread()
        self.worker = FolderWorker(self.folder, mode, image_paths)
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.on_worker_progress)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.failed.connect(self.on_worker_failed)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.failed.connect(self.worker_thread.quit)
        self.worker_thread.finished.connect(self.cleanup_worker)
        self.worker_thread.start()

    def on_worker_progress(self, current: int, total: int, name: str) -> None:
        percent = int(current * 100 / max(1, total))
        self.progress.setValue(percent)
        self.progress_label.setText(f'{current} / {total}  {name}')

    def on_worker_finished(self, report: dict) -> None:
        self.report = report
        self.progress.setValue(100)
        self.refresh_list()
        self.reload_current()
        self.status.showMessage('任務完成。')

    def on_worker_failed(self, message: str) -> None:
        QMessageBox.critical(self, '執行失敗', message)

    def cleanup_worker(self) -> None:
        self.worker = None
        self.worker_thread = None

    def refresh_list(self) -> None:
        previous_path = self.current_img_path
        self.suppress_list_selection = True
        self.list_widget.clear()
        pages = self.report.get('pages', {})
        other_count = 0
        failed_count = 0
        selected_row = -1
        for img_path in self.imglist:
            name = osp.basename(img_path)
            info = pages.get(name, {})
            status = STATUS_TODO
            if 'error' in info:
                status = STATUS_FAILED
                failed_count += 1
            elif info:
                if int(info.get('other_pixels', 0)) > 0:
                    status = STATUS_OTHER
                    other_count += 1
                else:
                    status = STATUS_OK
            item = QListWidgetItem(f'{name}    {status}')
            item.setData(Qt.ItemDataRole.UserRole, img_path)
            if status == STATUS_OTHER:
                item.setForeground(QColor('#d59a45'))
            elif status == STATUS_FAILED:
                item.setForeground(QColor('#e06767'))
            elif status == STATUS_OK:
                item.setForeground(QColor('#57b66f'))
            self.list_widget.addItem(item)
            if img_path == previous_path:
                selected_row = self.list_widget.count() - 1
        self.summary_label.setText(f'共 {len(self.imglist)} 張    OTHER {other_count}    失敗 {failed_count}')
        if selected_row >= 0:
            self.list_widget.setCurrentRow(selected_row)
        self.suppress_list_selection = False

    def on_image_selected(self, row: int) -> None:
        if self.suppress_list_selection:
            return
        if row < 0:
            return
        item = self.list_widget.item(row)
        if item is None:
            return
        self.current_img_path = item.data(Qt.ItemDataRole.UserRole)
        self.undo_stack = []
        self.redo_stack = []
        self.reload_current()

    def previous_image(self) -> None:
        if not self.imglist:
            return
        row = self.list_widget.currentRow()
        if row <= 0:
            return
        self.list_widget.setCurrentRow(row - 1)

    def next_image(self) -> None:
        if not self.imglist:
            return
        row = self.list_widget.currentRow()
        if row < 0:
            self.list_widget.setCurrentRow(0)
            return
        if row < self.list_widget.count() - 1:
            self.list_widget.setCurrentRow(row + 1)

    def reload_current(self) -> None:
        if not self.current_img_path:
            return
        base = _optional_imread(self.current_img_path, cv2.IMREAD_UNCHANGED)
        mask = _optional_imread(_mask_path(self.paths, self.current_img_path), cv2.IMREAD_GRAYSCALE)
        other_mask = _optional_imread(_other_mask_path(self.paths, self.current_img_path), cv2.IMREAD_GRAYSCALE)
        overlay = _optional_imread(_output_path(self.paths, self.current_img_path), cv2.IMREAD_UNCHANGED)
        if base is None:
            return
        self.current_base = base
        shape = base.shape[:2]
        self.current_mask = np.where(mask > 0, 255, 0).astype(np.uint8) if mask is not None else np.zeros(shape, dtype=np.uint8)
        self.mask_view.set_mask(self.current_mask, shape)
        self.refresh_mask_preview(keep_view=False)

        if overlay is not None:
            preview = _compose_overlay_preview(base, overlay)
            if self.show_other_mask:
                preview = _overlay_mask_on_bgr(preview, other_mask, 0.38, (165, 110, 255))
            self.preview_view.set_qimage(_qimage_from_bgr(preview))
        else:
            preview = base[:, :, :3].copy() if len(base.shape) == 3 else cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
            if self.show_other_mask:
                preview = _overlay_mask_on_bgr(preview, other_mask, 0.38, (165, 110, 255))
            self.preview_view.set_qimage(_qimage_from_bgr(preview))

        info = self.report.get('pages', {}).get(osp.basename(self.current_img_path), {})
        if 'error' in info:
            self.stats_label.setText(
                f'{osp.basename(self.current_img_path)}\n失敗：{info["error"]}'
            )
        elif info:
            blocks = info.get('blocks', 0)
            auto_blocks = info.get('auto_blocks', 0)
            other_blocks = info.get('other_blocks', 0)
            other_pixels = info.get('other_pixels', 0)
            self.stats_label.setText(
                f'{osp.basename(self.current_img_path)}\n'
                f'blocks {blocks}    auto {auto_blocks}    '
                f'other {other_blocks}    other_pixels {other_pixels}'
            )
        else:
            self.stats_label.setText(f'{osp.basename(self.current_img_path)}\n未處理')

        self.update_edit_buttons()

    def refresh_mask_preview(self, keep_view: bool = True) -> None:
        if self.current_base is None:
            self.mask_view.set_qimage(None)
            return
        mask_preview = _mask_overlay_image(self.current_base, self.current_mask, self.alpha, (255, 255, 255))
        self.mask_view.set_qimage(_qimage_from_bgr(mask_preview), keep_view=keep_view)
        if self.current_mask is not None:
            self.mask_view.set_mask(self.current_mask, self.current_base.shape[:2])

    def on_alpha_changed(self, value: int) -> None:
        self.alpha = value / 100.0
        self.alpha_label.setText(f'目前 {value}%')
        self.refresh_mask_preview(keep_view=True)

    def on_show_other_mask_changed(self, state: int) -> None:
        self.show_other_mask = state == Qt.CheckState.Checked.value
        self.reload_current()

    def set_edit_tool(self, tool: str) -> None:
        self.mask_view.set_tool(tool)
        self.brush_btn.setChecked(tool == 'brush')
        self.rect_btn.setChecked(tool == 'rect')

    def change_brush_radius(self, delta: int) -> None:
        self.mask_view.set_brush_radius(self.mask_view.brush_radius + delta)
        self.brush_label.setText(f'筆刷 {self.mask_view.brush_radius}px')

    def keyPressEvent(self, event) -> None:
        if event.key() in (Qt.Key.Key_Left, Qt.Key.Key_PageUp):
            self.previous_image()
            return
        if event.key() in (Qt.Key.Key_Right, Qt.Key.Key_PageDown):
            self.next_image()
            return
        if event.key() == Qt.Key.Key_B:
            self.set_edit_tool('brush')
            return
        if event.key() == Qt.Key.Key_R:
            self.set_edit_tool('rect')
            return
        if event.key() == Qt.Key.Key_BracketLeft:
            self.change_brush_radius(-4)
            return
        if event.key() == Qt.Key.Key_BracketRight:
            self.change_brush_radius(4)
            return
        if event.matches(QKeySequence.StandardKey.Undo):
            self.undo_mask()
            return
        if event.matches(QKeySequence.StandardKey.Redo):
            self.redo_mask()
            return
        super().keyPressEvent(event)

    def push_undo_snapshot(self) -> None:
        if self.current_mask is None:
            return
        self.undo_stack.append(self.current_mask.copy())
        self.undo_stack = self.undo_stack[-MAX_UNDO_STEPS:]
        self.redo_stack = []
        self.update_edit_buttons()

    def on_mask_edited(self, mask: object) -> None:
        self.current_mask = np.where(np.asarray(mask) > 0, 255, 0).astype(np.uint8)
        self.save_current_mask()
        self.refresh_mask_preview(keep_view=True)
        self.queue_auto_render()
        self.update_edit_buttons()

    def undo_mask(self) -> None:
        if self.current_mask is None or not self.undo_stack:
            return
        self.redo_stack.append(self.current_mask.copy())
        self.current_mask = self.undo_stack.pop()
        self.save_current_mask()
        self.refresh_mask_preview(keep_view=True)
        self.queue_auto_render()
        self.update_edit_buttons()

    def redo_mask(self) -> None:
        if self.current_mask is None or not self.redo_stack:
            return
        self.undo_stack.append(self.current_mask.copy())
        self.current_mask = self.redo_stack.pop()
        self.save_current_mask()
        self.refresh_mask_preview(keep_view=True)
        self.queue_auto_render()
        self.update_edit_buttons()

    def update_edit_buttons(self) -> None:
        self.undo_btn.setEnabled(bool(self.undo_stack))
        self.redo_btn.setEnabled(bool(self.redo_stack))

    def save_current_mask(self) -> None:
        if not self.current_img_path or self.current_mask is None:
            return
        imwrite(_mask_path(self.paths, self.current_img_path), self.current_mask)

    def queue_auto_render(self) -> None:
        if not self.current_img_path or self.current_mask is None:
            return
        self.pending_render_img_path = self.current_img_path
        self.pending_render_mask = self.current_mask.copy()
        self.stats_label.setText(f'{osp.basename(self.current_img_path)}\n正在生成預覽...')
        self.render_timer.start(450)

    def start_pending_render(self) -> None:
        if not self.pending_render_img_path or self.pending_render_mask is None:
            return
        if self.worker_thread is not None:
            self.render_timer.start(450)
            return
        if self.page_worker_thread is not None:
            return
        img_path = self.pending_render_img_path
        mask = self.pending_render_mask.copy()
        self.pending_render_img_path = ''
        self.pending_render_mask = None
        self.page_worker_thread = QThread()
        self.page_worker = PageRegenerateWorker(self.folder, self.paths, self.imglist, img_path, mask)
        self.page_worker.moveToThread(self.page_worker_thread)
        self.page_worker_thread.started.connect(self.page_worker.run)
        self.page_worker.finished.connect(self.on_page_render_finished)
        self.page_worker.failed.connect(self.on_page_render_failed)
        self.page_worker.finished.connect(self.page_worker_thread.quit)
        self.page_worker.failed.connect(self.page_worker_thread.quit)
        self.page_worker_thread.finished.connect(self.cleanup_page_worker)
        self.page_worker_thread.start()

    def on_page_render_finished(self, img_path: str, report: dict) -> None:
        self.report = report
        if self.pending_render_img_path == img_path and self.pending_render_mask is not None:
            imwrite(_mask_path(self.paths, img_path), self.pending_render_mask)
        self.refresh_list()
        if img_path == self.current_img_path:
            if self.pending_render_img_path:
                self.status.showMessage('預覽已更新，等待最新編輯。')
            else:
                self.reload_current()
                self.status.showMessage('預覽已更新。')
        if self.pending_render_img_path:
            self.render_timer.start(50)

    def on_page_render_failed(self, img_path: str, message: str) -> None:
        pages = dict(self.report.get('pages', {}))
        pages[osp.basename(img_path)] = {'error': message}
        self.report = build_report(self.folder, self.paths, self.imglist, pages)
        write_report(self.paths, self.report)
        self.refresh_list()
        if img_path == self.current_img_path:
            self.stats_label.setText(f'{osp.basename(img_path)}\n失敗：{message}')
        self.status.showMessage('預覽生成失敗。')
        if self.pending_render_img_path:
            self.render_timer.start(50)

    def cleanup_page_worker(self) -> None:
        self.page_worker = None
        self.page_worker_thread = None
        if self.pending_render_img_path:
            self.render_timer.start(50)

    def generate_pdf(self) -> None:
        if not self.paths or not self.imglist:
            return
        if self.pdf_worker_thread is not None:
            QMessageBox.information(self, '正在生成 PDF', 'PDF 已在生成中。')
            return
        if self.page_worker_thread is not None or self.render_timer.isActive():
            QMessageBox.information(self, '正在生成預覽', '請等待當前頁預覽生成完成後再生成 PDF。')
            return
        report = self.report or load_report(self.paths)
        if not report:
            QMessageBox.information(self, '沒有報告', '請先偵測並生成，或載入已有輸出。')
            return
        self.status.showMessage('正在生成 PDF...')
        self.pdf_worker_thread = QThread()
        self.pdf_worker = PdfWorker(self.imglist, self.paths, report)
        self.pdf_worker.moveToThread(self.pdf_worker_thread)
        self.pdf_worker_thread.started.connect(self.pdf_worker.run)
        self.pdf_worker.finished.connect(self.on_pdf_finished)
        self.pdf_worker.failed.connect(self.on_pdf_failed)
        self.pdf_worker.finished.connect(self.pdf_worker_thread.quit)
        self.pdf_worker.failed.connect(self.pdf_worker_thread.quit)
        self.pdf_worker_thread.finished.connect(self.cleanup_pdf_worker)
        self.pdf_worker_thread.start()

    def on_pdf_finished(self, pdf_path: object) -> None:
        if pdf_path:
            self.status.showMessage(f'PDF 已生成：{pdf_path}')
        else:
            self.status.showMessage('沒有可生成 PDF 的頁面。')

    def on_pdf_failed(self, message: str) -> None:
        QMessageBox.critical(self, 'PDF 生成失敗', message)
        self.status.showMessage('PDF 生成失敗。')

    def cleanup_pdf_worker(self) -> None:
        self.pdf_worker = None
        self.pdf_worker_thread = None

    def show_help(self) -> None:
        QMessageBox.information(
            self,
            'Solid Inpaint 說明',
            f'Solid Inpaint UI\n'
            f'版本：{APP_VERSION}\n\n'
            '滑鼠操作：\n'
            '左鍵：添加 mask\n'
            '右鍵：去掉 mask\n'
            'Command/Ctrl + 左鍵拖拽：平移畫布\n'
            '滾輪：縮放畫布\n\n'
            '快捷鍵：\n'
            '← / PageUp：上一頁\n'
            '→ / PageDown：下一頁\n'
            'B：筆刷工具\n'
            'R：矩形工具\n'
            '[：縮小筆刷\n'
            ']：放大筆刷\n'
            'Ctrl+Z：撤銷\n'
            'Ctrl+Shift+Z：重做\n\n'
            '注意：紅色「偵測並生成」會重新跑 detector，可能覆蓋已有 mask。',
        )

    def open_output(self) -> None:
        if not self.paths:
            return
        self._open_path(self.paths['output'])

    def open_pdf(self) -> None:
        if not self.paths:
            return
        pdf_path = osp.join(self.paths['output'], 'preview_report.pdf')
        if osp.isfile(pdf_path):
            self._open_path(pdf_path)

    def _open_path(self, path: str) -> None:
        if sys.platform == 'darwin':
            subprocess.Popen(['open', path])
        elif os.name == 'nt':
            os.startfile(path)  # type: ignore[attr-defined]
        else:
            subprocess.Popen(['xdg-open', path])


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
