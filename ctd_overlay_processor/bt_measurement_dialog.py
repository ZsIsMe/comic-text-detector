from __future__ import annotations

import math

from PySide6.QtCore import QEvent, QPointF, QRectF, QSize, QSettings, Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QKeyEvent, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QButtonGroup,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


def normalized_rotation(value: float) -> float:
    angle = float(value)
    while angle > 180.0:
        angle -= 360.0
    while angle <= -180.0:
        angle += 360.0
    return round(angle, 2)


def line_from_measurement(center: QPointF, length: float, rotation: float) -> tuple[QPointF, QPointF]:
    length = max(1.0, float(length))
    radians = math.radians(float(rotation))
    dx = math.cos(radians) * length / 2.0
    dy = -math.sin(radians) * length / 2.0
    return QPointF(center.x() - dx, center.y() - dy), QPointF(center.x() + dx, center.y() + dy)


class BtMeasurementMultiCanvas(QWidget):
    changed = Signal()

    def __init__(
        self,
        image: QImage,
        *,
        entries: list[dict[str, object]],
        display_scale: float | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setMouseTracking(True)
        self._image = image.convertToFormat(QImage.Format.Format_RGBA8888)
        self._pixmap = QPixmap.fromImage(self._image)
        self._display_scale = display_scale
        self.entries = entries
        self.active_index = 0
        self.interaction_mode = 'line'
        self._drag_entry_index: int | None = None
        self._drag_mode: str | None = None
        self._press_widget_point: QPointF | None = None
        self._pending_drag_entry_index: int | None = None
        self._pending_drag_mode: str | None = None
        self._drag_started = False
        self._drag_threshold = 3.0
        self._undo_stack: list[dict[str, object]] = []
        self._redo_stack: list[dict[str, object]] = []
        self._view_zoom = 1.0
        self._view_pan = QPointF(0, 0)
        self._panning = False
        self._pan_start_point: QPointF | None = None
        self._pan_start_offset = QPointF(0, 0)
        self.text_preview_opacity = 1.0
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        for entry in self.entries:
            self.entry_state(entry)
        self.setMinimumSize(self.sizeHint())

    def sizeHint(self) -> QSize:
        margin = 24
        if self._display_scale is None:
            return QSize(720, 520)
        return QSize(
            max(320, int(round(self._image.width() * self._display_scale)) + margin),
            max(240, int(round(self._image.height() * self._display_scale)) + margin),
        )

    def entry_state(self, entry: dict[str, object]) -> dict[str, object]:
        state = entry.get('state')
        if isinstance(state, dict):
            return state
        center = entry.get('center')
        if not isinstance(center, QPointF):
            center = QPointF(0, 0)
        font_size = max(1.0, float(entry.get('font_size') or 40))
        rotation = normalized_rotation(float(entry.get('rotation') or 0.0))
        line_start, line_end = line_from_measurement(center, font_size, rotation)
        state = {
            'text_center': QPointF(center),
            'line_start': line_start,
            'line_end': line_end,
            'changed': False,
        }
        entry['state'] = state
        return state

    def active_entry(self) -> dict[str, object] | None:
        if not self.entries:
            return None
        self.active_index = max(0, min(self.active_index, len(self.entries) - 1))
        return self.entries[self.active_index]

    def active_font_size(self) -> float:
        entry = self.active_entry()
        return self.font_size_for_entry(entry) if entry is not None else 1.0

    def active_rotation(self) -> float:
        entry = self.active_entry()
        return self.rotation_for_entry(entry) if entry is not None else 0.0

    def font_size_for_entry(self, entry: dict[str, object]) -> float:
        state = self.entry_state(entry)
        line_start = state.get('line_start')
        line_end = state.get('line_end')
        if isinstance(line_start, QPointF) and isinstance(line_end, QPointF):
            return max(1.0, math.hypot(line_end.x() - line_start.x(), line_end.y() - line_start.y()))
        return max(1.0, float(entry.get('font_size') or 40))

    def rotation_for_entry(self, entry: dict[str, object]) -> float:
        state = self.entry_state(entry)
        line_start = state.get('line_start')
        line_end = state.get('line_end')
        if isinstance(line_start, QPointF) and isinstance(line_end, QPointF):
            dx = line_end.x() - line_start.x()
            dy = line_end.y() - line_start.y()
            if abs(dx) >= 1e-6 or abs(dy) >= 1e-6:
                return normalized_rotation(math.degrees(math.atan2(-dy, dx)))
        return normalized_rotation(float(entry.get('rotation') or 0.0))

    def set_interaction_mode(self, mode: str) -> None:
        if mode not in {'line', 'text'}:
            return
        self.interaction_mode = mode
        self._drag_entry_index = None
        self._drag_mode = None
        self.update()

    def toggle_interaction_mode(self) -> None:
        self.set_interaction_mode('text' if self.interaction_mode == 'line' else 'line')

    def set_text_preview_opacity(self, opacity: float) -> None:
        self.text_preview_opacity = max(0.05, min(1.0, float(opacity)))
        self.update()

    def _base_scale(self) -> float:
        image_w = max(1, self._image.width())
        image_h = max(1, self._image.height())
        if self._display_scale is None:
            margin = 12
            available_w = max(1, self.width() - margin * 2)
            available_h = max(1, self.height() - margin * 2)
            return max(0.01, min(available_w / image_w, available_h / image_h))
        return max(0.01, self._display_scale)

    def _display_rect(self) -> QRectF:
        image_w = max(1, self._image.width())
        image_h = max(1, self._image.height())
        scale = self._scale()
        display_w = image_w * scale
        display_h = image_h * scale
        return QRectF(
            (self.width() - display_w) / 2.0 + self._view_pan.x(),
            (self.height() - display_h) / 2.0 + self._view_pan.y(),
            display_w,
            display_h,
        )

    def _scale(self) -> float:
        return self._base_scale() * self._view_zoom

    def _to_widget(self, point: QPointF) -> QPointF:
        rect = self._display_rect()
        scale = self._scale()
        return QPointF(rect.left() + point.x() * scale, rect.top() + point.y() * scale)

    def _to_image(self, point: QPointF) -> QPointF:
        rect = self._display_rect()
        scale = max(1e-6, self._scale())
        x = (point.x() - rect.left()) / scale
        y = (point.y() - rect.top()) / scale
        return self._clamp_image_point(QPointF(x, y))

    def _clamp_image_point(self, point: QPointF) -> QPointF:
        return QPointF(
            max(0.0, min(point.x(), max(0.0, self._image.width() - 1.0))),
            max(0.0, min(point.y(), max(0.0, self._image.height() - 1.0))),
        )

    def state_snapshot(self) -> dict[str, object]:
        states = []
        for entry in self.entries:
            state = self.entry_state(entry)
            text_center = state.get('text_center')
            line_start = state.get('line_start')
            line_end = state.get('line_end')
            states.append({
                'text_center': QPointF(text_center) if isinstance(text_center, QPointF) else QPointF(0, 0),
                'line_start': QPointF(line_start) if isinstance(line_start, QPointF) else QPointF(0, 0),
                'line_end': QPointF(line_end) if isinstance(line_end, QPointF) else QPointF(0, 0),
                'changed': bool(state.get('changed')),
            })
        return {'active_index': self.active_index, 'states': states}

    def restore_snapshot(self, snapshot: dict[str, object]) -> None:
        states = snapshot.get('states')
        if not isinstance(states, list):
            return
        for entry, saved in zip(self.entries, states):
            if not isinstance(saved, dict):
                continue
            state = self.entry_state(entry)
            for key in ('text_center', 'line_start', 'line_end'):
                point = saved.get(key)
                if isinstance(point, QPointF):
                    state[key] = QPointF(point)
            state['changed'] = bool(saved.get('changed'))
        active_index = snapshot.get('active_index')
        if isinstance(active_index, int):
            self.active_index = max(0, min(active_index, max(0, len(self.entries) - 1)))
        self._drag_entry_index = None
        self._drag_mode = None
        self._pending_drag_entry_index = None
        self._pending_drag_mode = None
        self._drag_started = False
        self.changed.emit()
        self.update()

    def push_undo_snapshot(self) -> None:
        self._undo_stack.append(self.state_snapshot())
        if len(self._undo_stack) > 100:
            self._undo_stack = self._undo_stack[-100:]
        self._redo_stack.clear()

    def can_undo(self) -> bool:
        return bool(self._undo_stack)

    def can_redo(self) -> bool:
        return bool(self._redo_stack)

    def undo(self) -> bool:
        if not self._undo_stack:
            return False
        self._redo_stack.append(self.state_snapshot())
        self.restore_snapshot(self._undo_stack.pop())
        return True

    def redo(self) -> bool:
        if not self._redo_stack:
            return False
        self._undo_stack.append(self.state_snapshot())
        self.restore_snapshot(self._redo_stack.pop())
        return True

    def set_zoom_around(self, widget_point: QPointF, zoom: float) -> None:
        image_point = self._to_image(widget_point)
        self._view_zoom = max(0.1, min(12.0, zoom))
        scale = self._scale()
        image_w = max(1, self._image.width())
        image_h = max(1, self._image.height())
        base_left = (self.width() - image_w * scale) / 2.0
        base_top = (self.height() - image_h * scale) / 2.0
        self._view_pan = QPointF(
            widget_point.x() - image_point.x() * scale - base_left,
            widget_point.y() - image_point.y() * scale - base_top,
        )
        self.update()

    def reset_view(self) -> None:
        self._view_zoom = 1.0
        self._view_pan = QPointF(0, 0)
        self.update()

    def _nearest_line_handle(self, widget_point: QPointF) -> tuple[int, str]:
        best: tuple[float, int, str] | None = None
        for index, entry in enumerate(self.entries):
            state = self.entry_state(entry)
            for mode in ('line_start', 'line_end'):
                point = state.get(mode)
                if not isinstance(point, QPointF):
                    continue
                widget_handle = self._to_widget(point)
                distance = math.hypot(widget_point.x() - widget_handle.x(), widget_point.y() - widget_handle.y())
                candidate = (distance, index, mode)
                if best is None or candidate < best:
                    best = candidate
        return (best[1], best[2]) if best is not None else (0, 'line_end')

    def _hit_test_line(self, widget_point: QPointF) -> tuple[int, str] | None:
        best: tuple[float, int, str] | None = None
        for index, entry in enumerate(self.entries):
            state = self.entry_state(entry)
            for mode in ('line_start', 'line_end'):
                point = state.get(mode)
                if not isinstance(point, QPointF):
                    continue
                widget_handle = self._to_widget(point)
                distance = math.hypot(widget_point.x() - widget_handle.x(), widget_point.y() - widget_handle.y())
                if distance <= 14:
                    candidate = (distance, index, mode)
                    if best is None or candidate < best:
                        best = candidate
        return (best[1], best[2]) if best is not None else None

    def _nearest_text_entry(self, widget_point: QPointF) -> int:
        best: tuple[float, int] | None = None
        for index, entry in enumerate(self.entries):
            state = self.entry_state(entry)
            center = state.get('text_center')
            if not isinstance(center, QPointF):
                continue
            widget_center = self._to_widget(center)
            distance = math.hypot(widget_point.x() - widget_center.x(), widget_point.y() - widget_center.y())
            candidate = (distance, index)
            if best is None or candidate < best:
                best = candidate
        return best[1] if best is not None else 0

    def _hit_test_text(self, widget_point: QPointF) -> int | None:
        best: tuple[float, int] | None = None
        for index, entry in enumerate(self.entries):
            state = self.entry_state(entry)
            center = state.get('text_center')
            if not isinstance(center, QPointF):
                continue
            widget_center = self._to_widget(center)
            distance = math.hypot(widget_point.x() - widget_center.x(), widget_point.y() - widget_center.y())
            if distance <= 28:
                candidate = (distance, index)
                if best is None or candidate < best:
                    best = candidate
        return best[1] if best is not None else None

    def mousePressEvent(self, event) -> None:
        if event.button() in {Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton}:
            self.setFocus()
            self._panning = True
            self._pan_start_point = QPointF(event.position())
            self._pan_start_offset = QPointF(self._view_pan)
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        self.setFocus()
        self._press_widget_point = QPointF(event.position())
        self._drag_started = False
        if self.interaction_mode == 'text':
            hit_index = self._hit_test_text(event.position())
            self.active_index = hit_index if hit_index is not None else self._nearest_text_entry(event.position())
            self._pending_drag_entry_index = hit_index
            self._pending_drag_mode = 'text' if hit_index is not None else None
            self.changed.emit()
            self.update()
            event.accept()
            return
        hit = self._hit_test_line(event.position())
        if hit is None:
            self.active_index, _mode = self._nearest_line_handle(event.position())
            self._pending_drag_entry_index = None
            self._pending_drag_mode = None
        else:
            self.active_index, self._pending_drag_mode = hit
            self._pending_drag_entry_index = self.active_index
        self.changed.emit()
        self.update()
        event.accept()

    def mouseMoveEvent(self, event) -> None:
        if self._panning and self._pan_start_point is not None and (
            event.buttons() & (Qt.MouseButton.RightButton | Qt.MouseButton.MiddleButton)
        ):
            delta = event.position() - self._pan_start_point
            self._view_pan = QPointF(self._pan_start_offset.x() + delta.x(), self._pan_start_offset.y() + delta.y())
            self.update()
            event.accept()
            return
        if (
            self._pending_drag_entry_index is not None
            and self._pending_drag_mode is not None
            and self._press_widget_point is not None
            and event.buttons() & Qt.MouseButton.LeftButton
        ):
            distance = math.hypot(
                event.position().x() - self._press_widget_point.x(),
                event.position().y() - self._press_widget_point.y(),
            )
            if not self._drag_started:
                if distance < self._drag_threshold:
                    event.accept()
                    return
                self.push_undo_snapshot()
                self._drag_started = True
                self._drag_entry_index = self._pending_drag_entry_index
                self._drag_mode = self._pending_drag_mode
        if self._drag_entry_index is not None and self._drag_mode is not None and event.buttons() & Qt.MouseButton.LeftButton:
            image_point = self._to_image(event.position())
            state = self.entry_state(self.entries[self._drag_entry_index])
            if self._drag_mode in {'line_start', 'line_end'}:
                state[self._drag_mode] = image_point
            elif self._drag_mode == 'text':
                state['text_center'] = image_point
            state['changed'] = True
            self.changed.emit()
            self.update()
            event.accept()
            return
        if self.interaction_mode == 'line':
            self.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.setCursor(Qt.CursorShape.SizeAllCursor)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() in {Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton} and self._panning:
            self._panning = False
            self._pan_start_point = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton and self._drag_mode is not None:
            self._drag_entry_index = None
            self._drag_mode = None
            self._pending_drag_entry_index = None
            self._pending_drag_mode = None
            self._press_widget_point = None
            self._drag_started = False
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton and self._pending_drag_entry_index is not None:
            self._pending_drag_entry_index = None
            self._pending_drag_mode = None
            self._press_widget_point = None
            self._drag_started = False
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event) -> None:
        delta = event.angleDelta().y()
        if delta == 0:
            super().wheelEvent(event)
            return
        steps = delta / 120.0
        factor = 1.15 ** steps
        self.set_zoom_around(event.position(), self._view_zoom * factor)
        event.accept()

    def apply_uniform_rotation(self, rotation: float) -> None:
        rotation = normalized_rotation(rotation)
        self.push_undo_snapshot()
        for entry in self.entries:
            self.set_entry_measurement(entry, rotation=rotation)
        self.changed.emit()
        self.update()

    def apply_uniform_font_size(self, font_size: float) -> None:
        font_size = max(1.0, float(font_size))
        self.push_undo_snapshot()
        for entry in self.entries:
            self.set_entry_measurement(entry, font_size=font_size)
        self.changed.emit()
        self.update()

    def set_entry_measurement(
        self,
        entry: dict[str, object],
        *,
        font_size: float | None = None,
        rotation: float | None = None,
    ) -> None:
        state = self.entry_state(entry)
        center = state.get('text_center')
        if not isinstance(center, QPointF):
            center = entry.get('center') if isinstance(entry.get('center'), QPointF) else QPointF(0, 0)
            state['text_center'] = QPointF(center)
        next_font_size = self.font_size_for_entry(entry) if font_size is None else max(1.0, float(font_size))
        next_rotation = self.rotation_for_entry(entry) if rotation is None else normalized_rotation(float(rotation))
        line_start, line_end = line_from_measurement(center, next_font_size, next_rotation)
        state['line_start'] = line_start
        state['line_end'] = line_end
        state['changed'] = True

    def result_updates(self) -> dict[int, dict[str, object]]:
        results: dict[int, dict[str, object]] = {}
        for entry in self.entries:
            item_index = entry.get('item_index')
            crop_origin = entry.get('crop_origin')
            state = self.entry_state(entry)
            text_center = state.get('text_center')
            if not isinstance(item_index, int) or not isinstance(crop_origin, QPointF) or not isinstance(text_center, QPointF):
                continue
            center = QPointF(crop_origin.x() + text_center.x(), crop_origin.y() + text_center.y())
            results[item_index] = {
                'font-size': max(1, int(round(self.font_size_for_entry(entry)))),
                'rotation': self.rotation_for_entry(entry),
                'center_pixel': [center.x(), center.y()],
            }
        return results

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        painter.fillRect(self.rect(), QColor(18, 20, 22))
        display_rect = self._display_rect()
        painter.drawPixmap(display_rect, self._pixmap, QRectF(self._pixmap.rect()))
        painter.setPen(QPen(QColor(75, 80, 86), 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(display_rect)
        for index, entry in enumerate(self.entries):
            self._draw_text_preview(painter, entry, active=index == self.active_index)
        for index, entry in enumerate(self.entries):
            self._draw_measurement_line(painter, entry, index, active=index == self.active_index)
        painter.end()

    def _draw_measurement_line(self, painter: QPainter, entry: dict[str, object], index: int, *, active: bool) -> None:
        state = self.entry_state(entry)
        line_start = state.get('line_start')
        line_end = state.get('line_end')
        text_center = state.get('text_center')
        if not isinstance(line_start, QPointF) or not isinstance(line_end, QPointF) or not isinstance(text_center, QPointF):
            return
        start = self._to_widget(line_start)
        end = self._to_widget(line_end)
        line_color = QColor(255, 230, 80) if active else QColor(255, 230, 80, 145)
        painter.setPen(QPen(line_color, 3 if active else 2))
        painter.drawLine(start, end)
        if self.interaction_mode == 'line':
            painter.setBrush(QColor(255, 245, 170) if active else QColor(255, 245, 170, 150))
            painter.setPen(QPen(QColor(35, 30, 10), 1))
            for point in (start, end):
                painter.drawEllipse(point, 7 if active else 5, 7 if active else 5)
        center = self._to_widget(text_center)
        if self.interaction_mode == 'text':
            if active:
                painter.setBrush(QColor(80, 180, 255))
                painter.setPen(QPen(QColor(10, 45, 70), 2))
            else:
                painter.setBrush(QColor(80, 180, 255, 135))
                painter.setPen(QPen(QColor(10, 45, 70, 150), 1))
            painter.drawEllipse(center, 6 if active else 5, 6 if active else 5)
    def _draw_text_preview(self, painter: QPainter, entry: dict[str, object], *, active: bool) -> None:
        item = entry.get('item')
        text = str(item.get('text') or '').strip() if isinstance(item, dict) else ''
        if not text:
            return
        state = self.entry_state(entry)
        center = state.get('text_center')
        if not isinstance(center, QPointF):
            return
        scale = self._scale()
        font_size = max(1.0, self.font_size_for_entry(entry) * scale)
        widget_center = self._to_widget(center)
        painter.save()
        painter.translate(widget_center)
        painter.rotate(-self.rotation_for_entry(entry))
        font = QFont(str(entry.get('font_family') or 'Helvetica Neue'))
        font.setPixelSize(max(1, int(round(font_size))))
        font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
        painter.setFont(font)
        stroke_px = max(0, int(round(float(entry.get('stroke_weight') or 0) * scale)))
        orientation = str(entry.get('orientation') or 'vertical')
        color = entry.get('color')
        stroke_color = entry.get('stroke_color')
        if not isinstance(color, QColor):
            color = QColor(0, 0, 0)
        if not isinstance(stroke_color, QColor):
            stroke_color = QColor(255, 255, 255)
        painter.setOpacity(self.text_preview_opacity)
        if orientation == 'horizontal':
            self._draw_horizontal_text(painter, text, stroke_px, color, stroke_color)
        else:
            self._draw_vertical_text(painter, text, stroke_px, color, stroke_color)
        painter.setOpacity(1.0)
        if active:
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor(80, 180, 255), 1))
            metrics = painter.fontMetrics()
            painter.drawRect(metrics.boundingRect(text).adjusted(-4, -4, 4, 4))
        painter.restore()

    def _draw_text_at(
        self,
        painter: QPainter,
        x: float,
        y: float,
        text: str,
        stroke_px: int,
        color: QColor,
        stroke_color: QColor,
    ) -> None:
        if stroke_px > 0:
            painter.setPen(QPen(stroke_color, 1))
            for dx in range(-stroke_px, stroke_px + 1):
                for dy in range(-stroke_px, stroke_px + 1):
                    if dx == 0 and dy == 0:
                        continue
                    if dx * dx + dy * dy <= stroke_px * stroke_px:
                        painter.drawText(QPointF(x + dx, y + dy), text)
        painter.setPen(QPen(color, 1))
        painter.drawText(QPointF(x, y), text)

    def _draw_horizontal_text(
        self,
        painter: QPainter,
        text: str,
        stroke_px: int,
        color: QColor,
        stroke_color: QColor,
    ) -> None:
        metrics = painter.fontMetrics()
        lines = text.replace('\\n', '\n').splitlines() or ['']
        line_h = metrics.height()
        total_h = line_h * len(lines)
        y = -total_h / 2.0 + metrics.ascent()
        for line in lines:
            x = -metrics.horizontalAdvance(line) / 2.0
            self._draw_text_at(painter, x, y, line, stroke_px, color, stroke_color)
            y += line_h

    def _draw_vertical_text(
        self,
        painter: QPainter,
        text: str,
        stroke_px: int,
        color: QColor,
        stroke_color: QColor,
    ) -> None:
        metrics = painter.fontMetrics()
        lines = text.replace('\\n', '\n').splitlines() or ['']
        line_h = metrics.height()
        column_w = max(1, metrics.maxWidth())
        max_chars = max((len(line) for line in lines), default=1)
        total_w = column_w * len(lines)
        for column_index, line in enumerate(lines):
            x = total_w / 2.0 - column_w * (column_index + 0.5)
            y = -line_h * max_chars / 2.0 + metrics.ascent()
            for char in line:
                self._draw_text_at(painter, x - metrics.horizontalAdvance(char) / 2.0, y, char, stroke_px, color, stroke_color)
                y += line_h


class BtMeasurementDialog(QDialog):
    def __init__(
        self,
        image: QImage,
        *,
        entries: list[dict[str, object]],
        display_scale: float | None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle('測量角度')
        self.resize(860, 680)
        self.settings = QSettings('comic-text-detector', 'ctd-overlay-processor')
        self.canvas = BtMeasurementMultiCanvas(
            image,
            entries=entries,
            display_scale=display_scale,
            parent=self,
        )

        self.info_label = QLabel()
        self.info_label.setWordWrap(True)
        self.line_mode_button = QPushButton('測量線')
        self.line_mode_button.setCheckable(True)
        self.line_mode_button.setChecked(True)
        self.text_mode_button = QPushButton('移動文字')
        self.text_mode_button.setCheckable(True)
        self.mode_group = QButtonGroup(self)
        self.mode_group.setExclusive(True)
        self.mode_group.addButton(self.line_mode_button)
        self.mode_group.addButton(self.text_mode_button)
        self.line_mode_button.clicked.connect(lambda _checked=False: self.set_interaction_mode('line'))
        self.text_mode_button.clicked.connect(lambda _checked=False: self.set_interaction_mode('text'))
        self.undo_button = QPushButton('撤銷')
        self.redo_button = QPushButton('重做')

        self.uniform_rotation_spin = QDoubleSpinBox()
        self.uniform_rotation_spin.setRange(-180.0, 180.0)
        self.uniform_rotation_spin.setDecimals(2)
        self.uniform_rotation_spin.setSingleStep(1.0)
        self.uniform_rotation_spin.setSuffix(' deg')
        self.apply_rotation_all_button = QPushButton('套用角度到全部')
        self.uniform_font_size_spin = QSpinBox()
        self.uniform_font_size_spin.setRange(1, 999)
        self.uniform_font_size_spin.setSuffix(' px')
        self.apply_font_size_all_button = QPushButton('套用字級到全部')
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(5, 100)
        self.opacity_slider.setValue(self._saved_text_opacity_percent())
        self.opacity_value_label = QLabel()
        self.canvas.set_text_preview_opacity(self.opacity_slider.value() / 100.0)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText('確認')
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText('取消')
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        mode_row = QHBoxLayout()
        mode_row.addWidget(self.line_mode_button)
        mode_row.addWidget(self.text_mode_button)
        mode_row.addWidget(self.undo_button)
        mode_row.addWidget(self.redo_button)
        mode_row.addStretch(1)
        layout.addLayout(mode_row)
        batch_row = QHBoxLayout()
        batch_row.addWidget(QLabel('統一角度'))
        batch_row.addWidget(self.uniform_rotation_spin)
        batch_row.addWidget(self.apply_rotation_all_button)
        batch_row.addSpacing(12)
        batch_row.addWidget(QLabel('統一字級'))
        batch_row.addWidget(self.uniform_font_size_spin)
        batch_row.addWidget(self.apply_font_size_all_button)
        batch_row.addStretch(1)
        layout.addLayout(batch_row)
        opacity_row = QHBoxLayout()
        opacity_row.addWidget(QLabel('文字透明度'))
        opacity_row.addWidget(self.opacity_slider)
        opacity_row.addWidget(self.opacity_value_label)
        layout.addLayout(opacity_row)
        layout.addWidget(self.canvas, 1)
        layout.addWidget(self.info_label)
        layout.addWidget(buttons)
        self.undo_button.clicked.connect(self.undo)
        self.redo_button.clicked.connect(self.redo)
        self.apply_rotation_all_button.clicked.connect(self.apply_uniform_rotation_to_all)
        self.apply_font_size_all_button.clicked.connect(self.apply_uniform_font_size_to_all)
        self.opacity_slider.valueChanged.connect(self.handle_opacity_changed)
        self.canvas.changed.connect(self.update_info)
        self.handle_opacity_changed(self.opacity_slider.value())
        self.update_info()
        self.resize(self.sizeHint())

    def _saved_text_opacity_percent(self) -> int:
        value = self.settings.value('bt_measurement/text_preview_opacity', 100)
        try:
            return max(5, min(100, int(value)))
        except (TypeError, ValueError):
            return 100

    def event(self, event) -> bool:
        if (
            event.type() == QEvent.Type.KeyPress
            and event.key() == Qt.Key.Key_Tab
            and not event.modifiers()
        ):
            self.toggle_interaction_mode()
            event.accept()
            return True
        return super().event(event)

    def focusNextPrevChild(self, next: bool) -> bool:
        self.toggle_interaction_mode()
        return True

    def keyPressEvent(self, event: QKeyEvent) -> None:
        command_modifier = event.modifiers() & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier)
        if command_modifier and event.key() == Qt.Key.Key_Z:
            if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                self.redo()
            else:
                self.undo()
            event.accept()
            return
        if command_modifier and event.key() == Qt.Key.Key_Y:
            self.redo()
            event.accept()
            return
        if event.key() == Qt.Key.Key_Tab and not event.modifiers():
            self.toggle_interaction_mode()
            event.accept()
            return
        super().keyPressEvent(event)

    def toggle_interaction_mode(self) -> None:
        self.canvas.toggle_interaction_mode()
        self.sync_mode_buttons()
        self.update_info()

    def set_interaction_mode(self, mode: str) -> None:
        self.canvas.set_interaction_mode(mode)
        self.sync_mode_buttons()
        self.update_info()

    def sync_mode_buttons(self) -> None:
        self.line_mode_button.setChecked(self.canvas.interaction_mode == 'line')
        self.text_mode_button.setChecked(self.canvas.interaction_mode == 'text')

    def undo(self) -> None:
        if self.canvas.undo():
            self.update_info()

    def redo(self) -> None:
        if self.canvas.redo():
            self.update_info()

    def apply_uniform_rotation_to_all(self) -> None:
        self.canvas.apply_uniform_rotation(self.uniform_rotation_spin.value())

    def apply_uniform_font_size_to_all(self) -> None:
        self.canvas.apply_uniform_font_size(max(1, int(self.uniform_font_size_spin.value())))

    def handle_opacity_changed(self, value: int) -> None:
        value = max(5, min(100, int(value)))
        self.opacity_value_label.setText(f'{value}%')
        self.settings.setValue('bt_measurement/text_preview_opacity', value)
        self.canvas.set_text_preview_opacity(value / 100.0)

    def update_info(self) -> None:
        mode_text = '測量線' if self.canvas.interaction_mode == 'line' else '移動文字'
        active_entry = self.canvas.active_entry()
        item_index = active_entry.get('item_index', '-') if active_entry is not None else '-'
        self.undo_button.setEnabled(self.canvas.can_undo())
        self.redo_button.setEnabled(self.canvas.can_redo())
        self.uniform_rotation_spin.blockSignals(True)
        self.uniform_font_size_spin.blockSignals(True)
        self.uniform_rotation_spin.setValue(self.canvas.active_rotation())
        self.uniform_font_size_spin.setValue(max(1, int(round(self.canvas.active_font_size()))))
        self.uniform_rotation_spin.blockSignals(False)
        self.uniform_font_size_spin.blockSignals(False)
        self.info_label.setText(
            f'模式：{mode_text}    '
            f'目前：{self.canvas.active_index + 1}/{len(self.canvas.entries)} item={item_index}    '
            f'字體大小：{int(round(self.canvas.active_font_size()))} px    '
            f'旋轉：{self.canvas.active_rotation():g} deg'
        )

    def result_updates(self) -> dict[int, dict[str, object]]:
        return self.canvas.result_updates()
