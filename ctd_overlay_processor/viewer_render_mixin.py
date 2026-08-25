"""Image, BT overlay and annotation rendering for the CTC viewer."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap

try:
    from .measure_view import char_box_label
    from .viewer_dialogs import show_error_details
    from .viewer_render_utils import mask_to_pixmap, union_mask
except ImportError:
    from measure_view import char_box_label
    from viewer_dialogs import show_error_details
    from viewer_render_utils import mask_to_pixmap, union_mask


class ViewerRenderMixin:

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
