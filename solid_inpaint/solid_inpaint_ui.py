#!/usr/bin/env python3
"""PySide6 UI for Solid Inpaint V1."""

from __future__ import annotations

import os
import os.path as osp
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QObject, QSettings, Qt, QThread, Signal
from PySide6.QtGui import QAction, QColor, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QGraphicsPixmapItem,
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
from utils.io_utils import imread


STATUS_OK = '完成'
STATUS_OTHER = '有 OTHER'
STATUS_FAILED = '失敗'
STATUS_TODO = '未處理'
MAX_RECENT_FOLDERS = 12


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

    def set_qimage(self, image: QImage | None) -> None:
        if image is None:
            self.pixmap_item.setPixmap(QPixmap())
            self.scene().setSceneRect(0, 0, 1, 1)
            return
        pixmap = QPixmap.fromImage(image)
        self.pixmap_item.setPixmap(pixmap)
        self.scene().setSceneRect(pixmap.rect())
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
            _write_preview_pdf(image_files_in_folder(self.folder), paths, report)
            self.finished.emit(report)
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
        self.alpha = 1.0
        self.show_other_mask = False
        self.worker_thread: QThread | None = None
        self.worker: FolderWorker | None = None
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

        run_action = QAction('偵測並生成', self)
        run_action.triggered.connect(self.run_or_load)
        toolbar.addAction(run_action)

        regen_page_action = QAction('用現有 Mask 重算當前頁', self)
        regen_page_action.triggered.connect(self.regenerate_current)
        toolbar.addAction(regen_page_action)

        regen_all_action = QAction('用現有 Mask 重算全部', self)
        regen_all_action.triggered.connect(self.regenerate_all)
        toolbar.addAction(regen_all_action)

        open_output_action = QAction('打開輸出', self)
        open_output_action.triggered.connect(self.open_output)
        toolbar.addAction(open_output_action)

        open_pdf_action = QAction('打開 PDF', self)
        open_pdf_action.triggered.connect(self.open_pdf)
        toolbar.addAction(open_pdf_action)

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
        self.mask_view = ImageView()
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
        self.status.showMessage('V1：不包含 mask 編輯。')

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
        self.start_worker('detect', self.imglist)

    def regenerate_current(self) -> None:
        if not self.current_img_path:
            return
        self.start_worker('regenerate', [self.current_img_path])

    def regenerate_all(self) -> None:
        if not self.folder:
            return
        self.start_worker('regenerate', self.imglist)

    def start_worker(self, mode: str, image_paths: list[str]) -> None:
        if self.worker_thread is not None:
            QMessageBox.information(self, '正在執行', '已有任務在執行中。')
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
        self.list_widget.clear()
        pages = self.report.get('pages', {})
        other_count = 0
        failed_count = 0
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
        self.summary_label.setText(f'共 {len(self.imglist)} 張    OTHER {other_count}    失敗 {failed_count}')

    def on_image_selected(self, row: int) -> None:
        if row < 0:
            return
        item = self.list_widget.item(row)
        if item is None:
            return
        self.current_img_path = item.data(Qt.ItemDataRole.UserRole)
        self.reload_current()

    def reload_current(self) -> None:
        if not self.current_img_path:
            return
        base = imread(self.current_img_path, cv2.IMREAD_UNCHANGED)
        mask = imread(_mask_path(self.paths, self.current_img_path), cv2.IMREAD_GRAYSCALE)
        other_mask = imread(_other_mask_path(self.paths, self.current_img_path), cv2.IMREAD_GRAYSCALE)
        overlay = imread(_output_path(self.paths, self.current_img_path), cv2.IMREAD_UNCHANGED)
        if base is None:
            return
        if mask is not None:
            mask_preview = _mask_overlay_image(base, mask, self.alpha, (255, 255, 255))
            self.mask_view.set_qimage(_qimage_from_bgr(mask_preview))
        else:
            self.mask_view.set_qimage(_qimage_from_bgr(base))

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

    def on_alpha_changed(self, value: int) -> None:
        self.alpha = value / 100.0
        self.alpha_label.setText(f'目前 {value}%')
        self.reload_current()

    def on_show_other_mask_changed(self, state: int) -> None:
        self.show_other_mask = state == Qt.CheckState.Checked.value
        self.reload_current()

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
