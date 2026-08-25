"""Reusable graphics widgets for the CTC overlay viewer."""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QGraphicsItem, QGraphicsPixmapItem, QGraphicsScene, QGraphicsView, QWidget


class ImageView(QGraphicsView):
    imageMouseMoved = Signal(float, float)
    imageMouseLeft = Signal()
    imageMousePressed = Signal(float, float)
    imageMouseDragged = Signal(float, float)
    imageMouseReleased = Signal(float, float)
    viewportChanged = Signal(object)
    fontSizeWheelRequested = Signal(int)

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
        self._background_pan_enabled = False
        self._background_panning = False
        self._background_pan_start = None
        self._background_pan_h = 0
        self._background_pan_v = 0
        self._background_pan_button = Qt.MouseButton.NoButton

    def set_background_pan_enabled(self, enabled: bool) -> None:
        self._background_pan_enabled = enabled
        self.setCursor(Qt.CursorShape.OpenHandCursor if enabled else Qt.CursorShape.ArrowCursor)

    def _ensure_pixmap_item(self) -> QGraphicsPixmapItem:
        try:
            item_scene = self.pixmap_item.scene()
        except RuntimeError:
            self.pixmap_item = QGraphicsPixmapItem()
            item_scene = None
        if item_scene is None:
            self.scene().addItem(self.pixmap_item)
        return self.pixmap_item

    def current_pixmap(self) -> QPixmap:
        return self._ensure_pixmap_item().pixmap()

    def set_pixmap(self, pixmap: QPixmap, fit: bool = True) -> None:
        pixmap_item = self._ensure_pixmap_item()
        pixmap_item.setPixmap(pixmap)
        self.scene().setSceneRect(QRectF(pixmap.rect()))
        if fit:
            self._zoom = 1.0
            self.resetTransform()
            self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self.viewportChanged.emit(self)

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
            factor = 1.15 if delta_y > 0 else 1 / 1.15
            self._zoom *= factor
            self.scale(factor, factor)
            self.viewportChanged.emit(self)
            event.accept()
            return
        super().wheelEvent(event)

    def scrollContentsBy(self, dx: int, dy: int) -> None:
        super().scrollContentsBy(dx, dy)
        self.viewportChanged.emit(self)

    def mouseMoveEvent(self, event) -> None:
        if self._background_panning:
            delta = event.position().toPoint() - self._background_pan_start
            self.horizontalScrollBar().setValue(self._background_pan_h - delta.x())
            self.verticalScrollBar().setValue(self._background_pan_v - delta.y())
            self.viewportChanged.emit(self)
            event.accept()
            return
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

    def _start_background_pan(self, event) -> None:
        self._background_panning = True
        self._background_pan_button = event.button()
        self._background_pan_start = event.position().toPoint()
        self._background_pan_h = self.horizontalScrollBar().value()
        self._background_pan_v = self.verticalScrollBar().value()
        self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def _finish_background_pan(self) -> None:
        self._background_panning = False
        self._background_pan_button = Qt.MouseButton.NoButton
        self._background_pan_start = None
        self.setCursor(Qt.CursorShape.OpenHandCursor if self._background_pan_enabled else Qt.CursorShape.ArrowCursor)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.RightButton:
            self._start_background_pan(event)
            event.accept()
            return
        scene_point = self.mapToScene(event.position().toPoint())
        pixmap_rect = QRectF(self.pixmap_item.pixmap().rect())
        if event.button() == Qt.MouseButton.LeftButton and pixmap_rect.contains(scene_point):
            self._mouse_down_on_image = True
            self.imageMousePressed.emit(scene_point.x(), scene_point.y())
            if self._background_pan_enabled:
                self._start_background_pan(event)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._background_panning and event.button() == self._background_pan_button:
            if self._mouse_down_on_image and event.button() == Qt.MouseButton.LeftButton:
                scene_point = self.mapToScene(event.position().toPoint())
                self.imageMouseReleased.emit(scene_point.x(), scene_point.y())
                self._mouse_down_on_image = False
            self._finish_background_pan()
            event.accept()
            return
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


class BtAnnotationItem(QGraphicsItem):
    """獨立繪製 _bt 框，避免拖動時複製整張背景圖片。"""

    def __init__(self, owner) -> None:
        super().__init__()
        self.owner = owner
        self._image_size = (1, 1)
        self.setZValue(1.0)
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)

    def set_image_size(self, width: int, height: int) -> None:
        width = max(1, int(width))
        height = max(1, int(height))
        if self._image_size == (width, height):
            return
        self.prepareGeometryChange()
        self._image_size = (width, height)

    def boundingRect(self) -> QRectF:
        width, height = self._image_size
        return QRectF(0, 0, width, height)

    def paint(self, painter, _option, _widget=None) -> None:
        width, height = self._image_size
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        self.owner._draw_bt_items(painter, width, height)


class NavigatorWidget(QWidget):
    navigateRequested = Signal(float, float)

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumSize(180, 130)
        self.setMaximumHeight(180)
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        self._pixmap = QPixmap()
        self._image_size = (0, 0)
        self._visible_rect = QRectF()
        self._thumb_rect = QRectF()
        self._dragging = False

    def set_navigator_state(
        self,
        pixmap: QPixmap,
        image_size: tuple[int, int],
        visible_rect: QRectF,
    ) -> None:
        self._pixmap = pixmap
        self._image_size = image_size
        self._visible_rect = visible_rect
        self.update()

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QColor(24, 26, 28))
        if self._pixmap.isNull() or self._image_size[0] <= 0 or self._image_size[1] <= 0:
            painter.setPen(QPen(QColor(180, 184, 190), 1))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, '導航器')
            painter.end()
            return

        margin = 8
        available_w = max(1, self.width() - margin * 2)
        available_h = max(1, self.height() - margin * 2)
        image_w, image_h = self._image_size
        scale = min(available_w / image_w, available_h / image_h)
        thumb_w = image_w * scale
        thumb_h = image_h * scale
        thumb_x = (self.width() - thumb_w) / 2.0
        thumb_y = (self.height() - thumb_h) / 2.0
        self._thumb_rect = QRectF(thumb_x, thumb_y, thumb_w, thumb_h)

        painter.drawPixmap(self._thumb_rect, self._pixmap, QRectF(self._pixmap.rect()))
        painter.setPen(QPen(QColor(70, 74, 80), 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(self._thumb_rect)

        rect = self._visible_rect.intersected(QRectF(0, 0, image_w, image_h))
        if rect.isValid() and not rect.isEmpty():
            view_rect = QRectF(
                thumb_x + rect.x() * scale,
                thumb_y + rect.y() * scale,
                rect.width() * scale,
                rect.height() * scale,
            )
            painter.setBrush(QColor(255, 245, 180, 46))
            painter.setPen(QPen(QColor(255, 235, 120), 2))
            painter.drawRect(view_rect)
        painter.end()

    def _emit_navigation(self, point: QPointF) -> None:
        if self._thumb_rect.isEmpty() or self._image_size[0] <= 0 or self._image_size[1] <= 0:
            return
        x = min(max(point.x(), self._thumb_rect.left()), self._thumb_rect.right())
        y = min(max(point.y(), self._thumb_rect.top()), self._thumb_rect.bottom())
        ratio_x = (x - self._thumb_rect.left()) / max(1.0, self._thumb_rect.width())
        ratio_y = (y - self._thumb_rect.top()) / max(1.0, self._thumb_rect.height())
        self.navigateRequested.emit(ratio_x * self._image_size[0], ratio_y * self._image_size[1])

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            self._emit_navigation(event.position())
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._dragging and event.buttons() & Qt.MouseButton.LeftButton:
            self._emit_navigation(event.position())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._dragging and event.button() == Qt.MouseButton.LeftButton:
            self._dragging = False
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            self._emit_navigation(event.position())
            event.accept()
            return
        super().mouseReleaseEvent(event)
