#!/usr/bin/env python3
"""ctd JSON/NPZ 即時疊圖檢視器。

檢視器會在記憶體中根據原圖、ctd/progressing/*.json、ctd/measure.json
和 align/masks/*.npz 重建疊圖，不會輸出預覽 PNG。
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np

QT_IMPORT_ERROR: ModuleNotFoundError | None = None
try:
    from PySide6.QtCore import QPointF, QProcess, QRectF, Qt
    from PySide6.QtGui import QAction, QColor, QFont, QImage, QPainter, QPen, QPixmap
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
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
        QSlider,
        QToolBar,
        QVBoxLayout,
        QWidget,
    )
except ModuleNotFoundError as exc:
    QT_IMPORT_ERROR = exc

try:
    from .processor import BoxOverlay, CtdOverlayProcessor, PageOverlay
except ImportError:
    from processor import BoxOverlay, CtdOverlayProcessor, PageOverlay


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
        def __init__(self) -> None:
            super().__init__()
            self.setScene(QGraphicsScene(self))
            self.pixmap_item = QGraphicsPixmapItem()
            self.scene().addItem(self.pixmap_item)
            self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
            self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
            self.setBackgroundBrush(QColor(32, 34, 36))
            self._zoom = 1.0

        def set_pixmap(self, pixmap: QPixmap) -> None:
            self.scene().clear()
            self.pixmap_item = QGraphicsPixmapItem(pixmap)
            self.scene().addItem(self.pixmap_item)
            self.scene().setSceneRect(QRectF(pixmap.rect()))
            self._zoom = 1.0
            self.resetTransform()
            self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

        def wheelEvent(self, event) -> None:
            factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
            self._zoom *= factor
            self.scale(factor, factor)


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


    class CtdOverlayViewer(QMainWindow):
        def __init__(self, image_dir: str | None = None) -> None:
            super().__init__()
            self.setWindowTitle('CTD 疊圖檢視器')
            self.resize(1280, 860)
            self.processor: CtdOverlayProcessor | None = None
            self.page: PageOverlay | None = None
            self.page_names: list[str] = []
            self.current_image_dir: str | None = image_dir
            self.detect_process: QProcess | None = None
            self.detect_output_chunks: list[str] = []
            self.detect_command: list[str] = []

            self.view = ImageView()
            self.setCentralWidget(self.view)

            self.page_list = QListWidget()
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
            self._connect_signals()

            if image_dir:
                self.load_folder(image_dir)

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

            reload_button = QPushButton('重新載入目前頁')
            reload_button.clicked.connect(self.reload_current_page)
            layout.addWidget(reload_button)
            self.generate_button.clicked.connect(self.generate_ctd)
            layout.addWidget(self.generate_button)

            dock = QDockWidget('CTD 資料', self)
            dock.setWidget(panel)
            dock.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
            self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)

        def _connect_signals(self) -> None:
            self.page_list.currentRowChanged.connect(self.load_page_at_row)
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

        def choose_folder(self) -> None:
            folder = QFileDialog.getExistingDirectory(self, '選擇包含原圖和 ctd 的圖片資料夾')
            if folder:
                self.load_folder(folder)

        def load_folder(self, image_dir: str) -> None:
            self.current_image_dir = image_dir
            try:
                self.processor = CtdOverlayProcessor(image_dir)
                self.page_names = self.processor.page_names()
            except Exception as exc:
                show_exception_details(self, '載入失敗', '無法載入資料夾。下方是完整可複製的出錯信息。', exc)
                return

            self.page_list.clear()
            self.page_list.addItems(self.page_names)
            summary = self.processor.summary()
            self.status_label.setText(self._summary_text(summary))
            if self.page_names:
                self.page_list.setCurrentRow(0)
            else:
                self.page = None
                self.view.scene().clear()
                self.status_label.setText(f'{self.status_label.text()}\n此資料夾沒有可顯示的圖片。')

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
                image_size = qimage_size(self.processor.image_dir / page_name)
                self.page = self.processor.load_page(page_name, image_size=image_size)
                self.render_current_page()
            except Exception as exc:
                show_exception_details(self, '頁面載入失敗', '無法載入此頁。下方是完整可複製的出錯信息。', exc)

        def reload_current_page(self) -> None:
            row = self.page_list.currentRow()
            if row >= 0:
                self.load_page_at_row(row)

        def render_current_page(self) -> None:
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
            self.view.set_pixmap(QPixmap.fromImage(image))
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
            for box in self.page.boxes:
                color = QColor(20, 175, 95)
                if box.accepted is False:
                    color = QColor(32, 32, 32)
                if box.error_route:
                    color = QColor(215, 50, 50)
                painter.setPen(QPen(color, 3))
                x1, y1, x2, y2 = box.xyxy_pixel
                painter.drawRect(QRectF(x1, y1, x2 - x1, y2 - y1))
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
            height = int(painter.device().height())
            width = int(painter.device().width())
            font = QFont('Helvetica', max(8, min(13, width // 120)))
            font.setBold(True)
            painter.setFont(font)
            metrics = painter.fontMetrics()
            painter.setPen(QPen(QColor(245, 170, 35), 1))
            painter.setBrush(QColor(245, 170, 35, 32))
            for item in self.page.char_boxes:
                bbox = item.get('bbox')
                if not isinstance(bbox, list) or len(bbox) != 4:
                    continue
                x1, y1, x2, y2 = [int(round(float(value))) for value in bbox]
                painter.drawRect(QRectF(x1, y1, x2 - x1, y2 - y1))
                width_text = compact_px(item.get('width'))
                height_text = compact_px(item.get('height'))
                if width_text is None or height_text is None:
                    continue

                label = f'寬{width_text} 高{height_text}'
                text_rect = metrics.boundingRect(label)
                pad_x = 3
                pad_y = 2
                label_w = text_rect.width() + pad_x * 2
                label_h = text_rect.height() + pad_y * 2
                label_x = int(round((x1 + x2) / 2 - label_w / 2))
                label_y = y1 - 3
                if label_y - label_h < 1:
                    label_y = y2 + label_h + 3
                label_y = max(label_h + 1, min(label_y, height - 1))
                label_x = max(1, min(label_x, max(1, width - label_w - 1)))

                painter.fillRect(
                    QRectF(label_x, label_y - label_h, label_w, label_h),
                    QColor(255, 255, 255, 225),
                )
                painter.setPen(QPen(QColor(70, 45, 0), 1))
                painter.drawText(QPointF(label_x + pad_x, label_y - pad_y), label)
                painter.setPen(QPen(QColor(245, 170, 35), 1))

        def _draw_font_labels(self, painter: QPainter, image_width: int, image_height: int) -> None:
            if self.page is None:
                return
            font = QFont('Helvetica', max(10, min(18, image_width // 85)))
            font.setBold(True)
            painter.setFont(font)
            metrics = painter.fontMetrics()
            for box in self.page.boxes:
                label = box.font_label
                x1, y1, x2, y2 = box.xyxy_pixel
                text_rect = metrics.boundingRect(label)
                pad = 5
                label_w = text_rect.width() + pad * 2
                label_h = text_rect.height() + pad * 2
                x = min(max(x2 + 6, 2), max(2, image_width - label_w - 2))
                y = min(max(y2 + 6, label_h + 2), max(label_h + 2, image_height - 2))
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
