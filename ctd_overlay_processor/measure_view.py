#!/usr/bin/env python3
"""Shared measure overlay view and drawing helpers."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QGraphicsPixmapItem, QGraphicsScene, QGraphicsView, QSizePolicy

try:
    from .processor import PageOverlay
except ImportError:
    from processor import PageOverlay


def compact_int_px(value) -> str | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number <= 0:
        return None
    return str(int(round(number)))


def char_info_text(item: dict[str, Any]) -> str:
    bbox = item.get('bbox')
    width_text = compact_int_px(item.get('width')) or '-'
    height_text = compact_int_px(item.get('height')) or '-'
    source_index = item.get('source_block_index', '-')
    line_index = item.get('line_index', '-')
    bbox_text = ', '.join(str(int(round(float(value)))) for value in bbox) if isinstance(bbox, list) else '-'
    calculated_text = compact_int_px(item.get('calculated_font_size')) or '-'
    ocr_text = str(item.get('ocr_text') or '-')
    probability = item.get('ocr_probability')
    probability_text = f'{float(probability):.1%}' if isinstance(probability, (int, float)) else '-'
    return (
        '游標單字框：\n'
        f'寬：{width_text}px  高：{height_text}px  計算字級：{calculated_text}px\n'
        f'OCR：{ocr_text}  信心：{probability_text}  狀態：{item.get("status") or "-"}\n'
        f'區塊：{source_index}  行：{line_index}\n'
        f'bbox：{bbox_text}'
    )


def char_box_label(item: dict[str, Any]) -> str | None:
    width_text = compact_int_px(item.get('width'))
    height_text = compact_int_px(item.get('height'))
    if width_text is None or height_text is None:
        return None
    label = f'W{width_text}H{height_text}'
    font_size_text = compact_int_px(item.get('calculated_font_size'))
    if font_size_text is not None:
        label += f'FS{font_size_text}'
    return label


class MeasureImageView(QGraphicsView):
    imageMouseMoved = Signal(float, float)
    imageMouseLeft = Signal()
    imageMousePressed = Signal(float, float)
    imageMouseDragged = Signal(float, float)
    imageMouseReleased = Signal(float, float)
    fontSizeWheelRequested = Signal(int)

    def __init__(self) -> None:
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene().addItem(self.pixmap_item)
        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setBackgroundBrush(QColor(32, 34, 36))
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)
        self._mouse_down_on_image = False
        self._fit_on_resize = True

    def set_pixmap(self, pixmap: QPixmap, fit: bool = True) -> None:
        if self.pixmap_item.scene() is None:
            self.scene().addItem(self.pixmap_item)
        self.pixmap_item.setPixmap(pixmap)
        self.scene().setSceneRect(QRectF(pixmap.rect()))
        if fit and not pixmap.isNull():
            self._fit_on_resize = True
            self.fit_to_view()

    def fit_to_view(self) -> None:
        if self.pixmap_item.pixmap().isNull():
            return
        self.resetTransform()
        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._fit_on_resize:
            self.fit_to_view()

    def wheelEvent(self, event) -> None:
        modifiers = event.modifiers()
        angle = event.angleDelta()
        pixel = event.pixelDelta()
        delta_y = angle.y() or pixel.y()
        horizontal_modifier = Qt.KeyboardModifier.MetaModifier | Qt.KeyboardModifier.ControlModifier
        if modifiers & Qt.KeyboardModifier.AltModifier:
            if delta_y:
                step = 1 if delta_y > 0 else -1
                self.fontSizeWheelRequested.emit(step * 2)
            event.accept()
            return
        if modifiers & horizontal_modifier:
            delta = angle.y() or angle.x() or pixel.y() or pixel.x()
            if delta:
                bar = self.horizontalScrollBar()
                bar.setValue(bar.value() - delta)
            event.accept()
            return
        if delta_y:
            self._fit_on_resize = False
            factor = 1.15 if delta_y > 0 else 1 / 1.15
            self.scale(factor, factor)
            event.accept()
            return
        super().wheelEvent(event)

    def mousePressEvent(self, event) -> None:
        scene_point = self.mapToScene(event.position().toPoint())
        pixmap_rect = QRectF(self.pixmap_item.pixmap().rect())
        if event.button() == Qt.MouseButton.LeftButton and pixmap_rect.contains(scene_point):
            self._mouse_down_on_image = True
            self.imageMousePressed.emit(scene_point.x(), scene_point.y())
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        scene_point = self.mapToScene(event.position().toPoint())
        pixmap_rect = QRectF(self.pixmap_item.pixmap().rect())
        if self._mouse_down_on_image and event.buttons() & Qt.MouseButton.LeftButton:
            self.imageMouseDragged.emit(scene_point.x(), scene_point.y())
            event.accept()
            return
        super().mouseMoveEvent(event)
        if pixmap_rect.contains(scene_point):
            self.imageMouseMoved.emit(scene_point.x(), scene_point.y())
        else:
            self.imageMouseLeft.emit()

    def mouseReleaseEvent(self, event) -> None:
        if self._mouse_down_on_image and event.button() == Qt.MouseButton.LeftButton:
            scene_point = self.mapToScene(event.position().toPoint())
            self.imageMouseReleased.emit(scene_point.x(), scene_point.y())
            self._mouse_down_on_image = False
            event.accept()
            return
        self._mouse_down_on_image = False
        super().mouseReleaseEvent(event)


def draw_measure_boxes(
    painter: QPainter,
    page: PageOverlay,
    selected_index: int | None,
    *,
    show_center_marker: bool = True,
) -> None:
    height = int(painter.device().height())
    width = int(painter.device().width())
    painter.setBrush(Qt.BrushStyle.NoBrush)
    for fallback_index, box in enumerate(page.boxes):
        item_index = box.measure_item_index if box.measure_item_index is not None else fallback_index
        color = QColor(20, 175, 95)
        if box.accepted is False:
            color = QColor(32, 32, 32)
        if box.error_route:
            color = QColor(215, 50, 50)
        selected = item_index == selected_index
        painter.setPen(QPen(QColor(255, 224, 118) if selected else color, 3))
        x1, y1, x2, y2 = box.xyxy_pixel
        painter.drawRect(QRectF(x1, y1, x2 - x1, y2 - y1))
        if selected:
            painter.setBrush(QColor(255, 224, 118, 180))
            for hx, hy in (
                (x1, y1), ((x1 + x2) / 2, y1), (x2, y1),
                (x1, (y1 + y2) / 2), (x2, (y1 + y2) / 2),
                (x1, y2), ((x1 + x2) / 2, y2), (x2, y2),
            ):
                painter.drawRect(QRectF(hx - 4, hy - 4, 8, 8))
            painter.setBrush(Qt.BrushStyle.NoBrush)
        if show_center_marker:
            cx, cy = box.center_pixel
            side = max(4, int(round(float(box.font_size or 0))))
            half = side / 2.0
            marker_x1 = max(0, int(round(cx - half)))
            marker_y1 = max(0, int(round(cy - half)))
            marker_x2 = min(width, int(round(cx + half)))
            marker_y2 = min(height, int(round(cy + half)))
            if marker_x2 > marker_x1 and marker_y2 > marker_y1:
                painter.fillRect(
                    QRectF(marker_x1, marker_y1, marker_x2 - marker_x1, marker_y2 - marker_y1),
                    color,
                )


def draw_char_boxes(
    painter: QPainter,
    page: PageOverlay,
    hover_char_box: dict[str, Any] | None,
    view: QGraphicsView,
) -> None:
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
    for item in page.char_boxes:
        bbox = item.get('bbox')
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [int(round(float(value))) for value in bbox]
        painter.drawRect(QRectF(x1, y1, x2 - x1, y2 - y1))
        if item is hover_char_box:
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
    if hover_char_box is not None:
        _draw_hover_char_box(painter, hover_char_box, image_width, image_height, view)


def _draw_hover_char_box(
    painter: QPainter,
    hover_char_box: dict[str, Any],
    image_width: int,
    image_height: int,
    view: QGraphicsView,
) -> None:
    bbox = hover_char_box.get('bbox')
    if not isinstance(bbox, list) or len(bbox) != 4:
        return
    x1, y1, x2, y2 = [int(round(float(value))) for value in bbox]
    painter.setPen(QPen(QColor(255, 40, 120), 3))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawRect(QRectF(x1, y1, x2 - x1, y2 - y1))
    label = char_box_label(hover_char_box)
    if label is None:
        return
    transform = view.transform()
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


def draw_font_labels(painter: QPainter, page: PageOverlay, image_width: int, image_height: int) -> None:
    font = QFont('Helvetica', max(20, min(36, (image_width // 85) * 2)))
    font.setBold(True)
    painter.setFont(font)
    metrics = painter.fontMetrics()
    for box in page.boxes:
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
