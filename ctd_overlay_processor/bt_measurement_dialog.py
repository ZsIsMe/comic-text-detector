from __future__ import annotations

import math

import cv2
import numpy as np

from PySide6.QtCore import QEvent, QPointF, QRectF, QSize, QSettings, Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QKeyEvent, QPainter, QPen, QPixmap, QPolygonF
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


def qimage_to_grayscale_array(image: QImage | None) -> np.ndarray | None:
    if image is None or image.isNull():
        return None
    gray = image.convertToFormat(QImage.Format.Format_Grayscale8)
    ptr = gray.constBits()
    arr = np.frombuffer(ptr, dtype=np.uint8).reshape(gray.height(), gray.bytesPerLine())
    return arr[:, :gray.width()].copy()


def normalized_axis_angle(value: float) -> float:
    angle = float(value)
    while angle >= 90.0:
        angle -= 180.0
    while angle < -90.0:
        angle += 180.0
    return round(angle, 2)


def detect_minimum_text_rectangle(
    mask: np.ndarray,
    roi: tuple[int, int, int, int],
    *,
    selection_polygon: list[list[float]] | None = None,
    threshold: int = 10,
) -> dict[str, object] | None:
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    height, width = mask.shape[:2]
    x1, y1, x2, y2 = roi
    x1 = max(0, min(int(x1), width - 1))
    y1 = max(0, min(int(y1), height - 1))
    x2 = max(x1 + 1, min(int(x2), width))
    y2 = max(y1 + 1, min(int(y2), height))
    binary = (mask[y1:y2, x1:x2] > threshold).astype(np.uint8)
    if selection_polygon:
        polygon = np.asarray(selection_polygon, dtype=np.float64).reshape(-1, 2)
        polygon[:, 0] -= x1
        polygon[:, 1] -= y1
        selection_mask = np.zeros_like(binary)
        cv2.fillConvexPoly(selection_mask, np.round(polygon).astype(np.int32), 1)
        binary &= selection_mask
    if int(binary.sum()) < 8:
        return None

    component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if component_count > 1:
        component_areas = stats[1:, cv2.CC_STAT_AREA]
        largest_area = int(component_areas.max()) if component_areas.size else 0
        min_area = max(3, int(round(largest_area * 0.004)))
        keep_labels = [
            label
            for label in range(1, component_count)
            if int(stats[label, cv2.CC_STAT_AREA]) >= min_area
        ]
        if keep_labels:
            binary = np.isin(labels, keep_labels).astype(np.uint8)

    ys, xs = np.where(binary > 0)
    if len(xs) < 8:
        return None
    points = np.column_stack((xs + x1, ys + y1)).astype(np.float32)
    rect = cv2.minAreaRect(points)
    box = cv2.boxPoints(rect).astype(np.float64)
    edges = []
    for index in range(4):
        start = box[index]
        end = box[(index + 1) % 4]
        vector = end - start
        edges.append((float(np.linalg.norm(vector)), vector))
    long_length, long_vector = max(edges, key=lambda item: item[0])
    short_length = min(length for length, _vector in edges)
    if long_length < 1.0 or short_length < 1.0:
        return None
    axis_angle = normalized_axis_angle(math.degrees(math.atan2(-long_vector[1], long_vector[0])))
    return {
        'roi': [x1, y1, x2, y2],
        'box': [[float(point[0]), float(point[1])] for point in box],
        'center': [float(rect[0][0]), float(rect[0][1])],
        'long_side': float(long_length),
        'short_side': float(short_length),
        'axis_angle': axis_angle,
        'pixel_count': int(len(xs)),
    }


class BtMeasurementMultiCanvas(QWidget):
    changed = Signal()

    def __init__(
        self,
        image: QImage,
        *,
        entries: list[dict[str, object]],
        mask: QImage | None = None,
        display_scale: float | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setMouseTracking(True)
        self._image = image.convertToFormat(QImage.Format.Format_RGBA8888)
        self._pixmap = QPixmap.fromImage(self._image)
        self._mask = qimage_to_grayscale_array(mask)
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
        self._detection_band_start: QPointF | None = None
        self._detection_band_end: QPointF | None = None
        self._detection_half_width = 30.0
        self._detection_drag_mode: str | None = None
        self._detection_drag_press: QPointF | None = None
        self._detection_original_start: QPointF | None = None
        self._detection_original_end: QPointF | None = None
        self._detection_original_half_width = 30.0
        self._detection_result: dict[str, object] | None = None
        self._detection_error: str | None = None
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
        if mode not in {'line', 'text', 'detect'}:
            return
        self.interaction_mode = mode
        self._drag_entry_index = None
        self._drag_mode = None
        self.update()

    def toggle_interaction_mode(self) -> None:
        self.set_interaction_mode('text' if self.interaction_mode == 'line' else 'line')

    def detection_result(self) -> dict[str, object] | None:
        return self._detection_result

    def detection_error(self) -> str | None:
        return self._detection_error

    def clear_detection(self) -> None:
        self._detection_band_start = None
        self._detection_band_end = None
        self._detection_drag_mode = None
        self._detection_drag_press = None
        self._detection_result = None
        self._detection_error = None
        self.changed.emit()
        self.update()

    def _detection_band_geometry(self) -> dict[str, object] | None:
        start = self._detection_band_start
        end = self._detection_band_end
        if not isinstance(start, QPointF) or not isinstance(end, QPointF):
            return None
        dx = end.x() - start.x()
        dy = end.y() - start.y()
        length = math.hypot(dx, dy)
        if length < 1.0:
            return None
        ux = dx / length
        uy = dy / length
        nx = -uy
        ny = ux
        half_width = max(4.0, float(self._detection_half_width))
        midpoint = QPointF((start.x() + end.x()) / 2.0, (start.y() + end.y()) / 2.0)
        points = [
            QPointF(start.x() + nx * half_width, start.y() + ny * half_width),
            QPointF(end.x() + nx * half_width, end.y() + ny * half_width),
            QPointF(end.x() - nx * half_width, end.y() - ny * half_width),
            QPointF(start.x() - nx * half_width, start.y() - ny * half_width),
        ]
        return {
            'start': start,
            'end': end,
            'midpoint': midpoint,
            'length': length,
            'unit': (ux, uy),
            'normal': (nx, ny),
            'half_width': half_width,
            'points': points,
            'width_handles': [
                QPointF(midpoint.x() + nx * half_width, midpoint.y() + ny * half_width),
                QPointF(midpoint.x() - nx * half_width, midpoint.y() - ny * half_width),
            ],
        }

    def _hit_test_detection_band(self, widget_point: QPointF) -> str | None:
        geometry = self._detection_band_geometry()
        if geometry is None:
            return None
        for name in ('start', 'end'):
            point = geometry[name]
            if isinstance(point, QPointF):
                handle = self._to_widget(point)
                if math.hypot(widget_point.x() - handle.x(), widget_point.y() - handle.y()) <= 14:
                    return name
        width_handles = geometry.get('width_handles')
        if isinstance(width_handles, list):
            for handle_point in width_handles:
                if not isinstance(handle_point, QPointF):
                    continue
                handle = self._to_widget(handle_point)
                if math.hypot(widget_point.x() - handle.x(), widget_point.y() - handle.y()) <= 14:
                    return 'width'
        image_point = self._to_image(widget_point)
        start = geometry['start']
        unit = geometry['unit']
        normal = geometry['normal']
        if not isinstance(start, QPointF) or not isinstance(unit, tuple) or not isinstance(normal, tuple):
            return None
        rel_x = image_point.x() - start.x()
        rel_y = image_point.y() - start.y()
        along = rel_x * unit[0] + rel_y * unit[1]
        across = rel_x * normal[0] + rel_y * normal[1]
        if 0.0 <= along <= float(geometry['length']) and abs(across) <= float(geometry['half_width']):
            return 'move'
        return None

    def _run_detection(self) -> None:
        self._detection_result = None
        self._detection_error = None
        if self._mask is None:
            self._detection_error = '目前頁面沒有可用的 CTD mask。'
            return
        geometry = self._detection_band_geometry()
        if geometry is None:
            self._detection_error = '框選區域太小。'
            return
        points = geometry.get('points')
        if not isinstance(points, list) or len(points) != 4:
            self._detection_error = '無法建立帶狀選區。'
            return
        x_values = [point.x() for point in points if isinstance(point, QPointF)]
        y_values = [point.y() for point in points if isinstance(point, QPointF)]
        if len(x_values) != 4 or len(y_values) != 4:
            self._detection_error = '無法建立帶狀選區。'
            return
        result = detect_minimum_text_rectangle(
            self._mask,
            (
                int(math.floor(min(x_values))),
                int(math.floor(min(y_values))),
                int(math.ceil(max(x_values))),
                int(math.ceil(max(y_values))),
            ),
            selection_polygon=[[point.x(), point.y()] for point in points],
        )
        if result is None:
            self._detection_error = '框選範圍內沒有找到足夠的文字 mask 像素。'
            return
        self._detection_result = result

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
        if self.interaction_mode == 'detect':
            image_point = self._to_image(event.position())
            hit_mode = self._hit_test_detection_band(event.position())
            self._detection_drag_press = QPointF(image_point)
            self._detection_original_start = (
                QPointF(self._detection_band_start)
                if isinstance(self._detection_band_start, QPointF)
                else None
            )
            self._detection_original_end = (
                QPointF(self._detection_band_end)
                if isinstance(self._detection_band_end, QPointF)
                else None
            )
            self._detection_original_half_width = self._detection_half_width
            if hit_mode is None:
                self._detection_band_start = QPointF(image_point)
                self._detection_band_end = QPointF(image_point)
                self._detection_half_width = max(16.0, self.active_font_size() * 0.75)
                self._detection_drag_mode = 'new'
            else:
                self._detection_drag_mode = hit_mode
            self._detection_result = None
            self._detection_error = None
            self.setCursor(Qt.CursorShape.CrossCursor)
            self.changed.emit()
            self.update()
            event.accept()
            return
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
            self.interaction_mode == 'detect'
            and self._detection_drag_mode is not None
            and event.buttons() & Qt.MouseButton.LeftButton
        ):
            image_point = self._to_image(event.position())
            mode = self._detection_drag_mode
            if mode in {'new', 'end'}:
                self._detection_band_end = QPointF(image_point)
            elif mode == 'start':
                self._detection_band_start = QPointF(image_point)
            elif mode == 'width':
                geometry = self._detection_band_geometry()
                if geometry is not None:
                    midpoint = geometry.get('midpoint')
                    normal = geometry.get('normal')
                    if isinstance(midpoint, QPointF) and isinstance(normal, tuple):
                        offset_x = image_point.x() - midpoint.x()
                        offset_y = image_point.y() - midpoint.y()
                        self._detection_half_width = max(4.0, abs(offset_x * normal[0] + offset_y * normal[1]))
            elif (
                mode == 'move'
                and isinstance(self._detection_drag_press, QPointF)
                and isinstance(self._detection_original_start, QPointF)
                and isinstance(self._detection_original_end, QPointF)
            ):
                dx = image_point.x() - self._detection_drag_press.x()
                dy = image_point.y() - self._detection_drag_press.y()
                self._detection_band_start = QPointF(
                    self._detection_original_start.x() + dx,
                    self._detection_original_start.y() + dy,
                )
                self._detection_band_end = QPointF(
                    self._detection_original_end.x() + dx,
                    self._detection_original_end.y() + dy,
                )
            self._detection_result = None
            self._detection_error = None
            self.changed.emit()
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
        if self.interaction_mode in {'line', 'detect'}:
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
        if event.button() == Qt.MouseButton.LeftButton and self.interaction_mode == 'detect' and self._detection_drag_mode is not None:
            self._detection_drag_mode = None
            self._detection_drag_press = None
            self._detection_original_start = None
            self._detection_original_end = None
            self._run_detection()
            self.changed.emit()
            self.update()
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
        if self.interaction_mode != 'detect':
            for index, entry in enumerate(self.entries):
                self._draw_text_preview(painter, entry, active=index == self.active_index)
            for index, entry in enumerate(self.entries):
                self._draw_measurement_line(painter, entry, index, active=index == self.active_index)
        self._draw_detection(painter)
        painter.end()

    def _draw_detection(self, painter: QPainter) -> None:
        geometry = self._detection_band_geometry()
        if geometry is not None:
            band_points = geometry.get('points')
            widget_points = [
                self._to_widget(point)
                for point in band_points
                if isinstance(point, QPointF)
            ] if isinstance(band_points, list) else []
            if len(widget_points) == 4:
                painter.setBrush(QColor(70, 170, 255, 28))
                painter.setPen(QPen(QColor(70, 170, 255), 2, Qt.PenStyle.DashLine))
                painter.drawPolygon(QPolygonF(widget_points))
            start = geometry.get('start')
            end = geometry.get('end')
            if isinstance(start, QPointF) and isinstance(end, QPointF):
                painter.setPen(QPen(QColor(70, 170, 255, 180), 2))
                painter.drawLine(self._to_widget(start), self._to_widget(end))
                painter.setBrush(QColor(180, 225, 255))
                painter.setPen(QPen(QColor(20, 75, 120), 1))
                for point in (start, end):
                    painter.drawEllipse(self._to_widget(point), 7, 7)
            width_handles = geometry.get('width_handles')
            if isinstance(width_handles, list):
                painter.setBrush(QColor(255, 210, 80))
                painter.setPen(QPen(QColor(95, 65, 5), 1))
                for point in width_handles:
                    if not isinstance(point, QPointF):
                        continue
                    widget_point = self._to_widget(point)
                    painter.drawRect(QRectF(widget_point.x() - 6, widget_point.y() - 6, 12, 12))
        result = self._detection_result
        box = result.get('box') if isinstance(result, dict) else None
        if not isinstance(box, list) or len(box) != 4:
            return
        points = []
        for value in box:
            if not isinstance(value, list) or len(value) != 2:
                return
            points.append(self._to_widget(QPointF(float(value[0]), float(value[1]))))
        painter.setBrush(QColor(80, 255, 140, 24))
        painter.setPen(QPen(QColor(80, 255, 140), 3))
        for index in range(4):
            painter.drawLine(points[index], points[(index + 1) % 4])

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
        mask: QImage | None = None,
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
            mask=mask,
            display_scale=display_scale,
            parent=self,
        )
        self.single_entry_mode = len(entries) == 1

        self.info_label = QLabel()
        self.info_label.setWordWrap(True)
        self.detection_result_label = QLabel('單字測量：沿單字方向拖出帶狀選區。')
        self.detection_result_label.setWordWrap(True)
        self.detection_result_label.setMinimumHeight(42)
        self.detection_result_label.setStyleSheet(
            'QLabel {'
            ' background: #16271d;'
            ' color: #8dffac;'
            ' border: 1px solid #3b8f55;'
            ' border-radius: 5px;'
            ' padding: 7px 10px;'
            ' font-weight: 600;'
            '}'
        )
        self.line_mode_button = QPushButton('測量線')
        self.line_mode_button.setCheckable(True)
        self.line_mode_button.setChecked(True)
        self.text_mode_button = QPushButton('移動文字')
        self.text_mode_button.setCheckable(True)
        self.detect_mode_button = QPushButton('單字測量')
        self.detect_mode_button.setCheckable(True)
        self.detect_mode_button.setToolTip('沿單字方向拖出帶狀選區，識別最小旋轉外接矩形、字體大小與角度')
        self.clear_detection_button = QPushButton('清除單字測量')
        self.detect_mode_button.setVisible(self.single_entry_mode)
        self.clear_detection_button.setVisible(self.single_entry_mode)
        self.detection_result_label.setVisible(self.single_entry_mode)
        self.mode_group = QButtonGroup(self)
        self.mode_group.setExclusive(True)
        self.mode_group.addButton(self.line_mode_button)
        self.mode_group.addButton(self.text_mode_button)
        self.mode_group.addButton(self.detect_mode_button)
        self.line_mode_button.clicked.connect(lambda _checked=False: self.set_interaction_mode('line'))
        self.text_mode_button.clicked.connect(lambda _checked=False: self.set_interaction_mode('text'))
        self.detect_mode_button.clicked.connect(lambda _checked=False: self.set_interaction_mode('detect'))
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

        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText('確認')
        self.buttons.button(QDialogButtonBox.StandardButton.Cancel).setText('取消')
        self.buttons.accepted.connect(self.handle_accept)
        self.buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        mode_row = QHBoxLayout()
        mode_row.addWidget(self.line_mode_button)
        mode_row.addWidget(self.text_mode_button)
        mode_row.addWidget(self.detect_mode_button)
        mode_row.addWidget(self.clear_detection_button)
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
        layout.addWidget(self.detection_result_label)
        layout.addWidget(self.canvas, 1)
        layout.addWidget(self.info_label)
        layout.addWidget(self.buttons)
        self.undo_button.clicked.connect(self.undo)
        self.redo_button.clicked.connect(self.redo)
        self.clear_detection_button.clicked.connect(self.canvas.clear_detection)
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
        if mode == 'detect' and not self.single_entry_mode:
            return
        self.canvas.set_interaction_mode(mode)
        self.sync_mode_buttons()
        self.update_info()

    def sync_mode_buttons(self) -> None:
        self.line_mode_button.setChecked(self.canvas.interaction_mode == 'line')
        self.text_mode_button.setChecked(self.canvas.interaction_mode == 'text')
        self.detect_mode_button.setChecked(self.canvas.interaction_mode == 'detect')

    def detected_single_char_measurement(self) -> dict[str, object] | None:
        if not self.single_entry_mode:
            return None
        detection = self.canvas.detection_result()
        active_entry = self.canvas.active_entry()
        if detection is None or active_entry is None:
            return None
        item_index = active_entry.get('item_index')
        if not isinstance(item_index, int):
            return None
        axis_angle = float(detection.get('axis_angle') or 0)
        long_side = float(detection.get('long_side') or 0)
        if long_side <= 0:
            return None
        orientation = str(active_entry.get('orientation') or 'vertical')
        rotation = normalized_axis_angle(
            axis_angle if orientation == 'horizontal' else axis_angle - 90.0,
        )
        return {
            'item_index': item_index,
            'font-size': max(1, int(math.floor(long_side + 0.5))),
            'rotation': rotation,
            'axis_angle': axis_angle,
            'long_side': long_side,
            'short_side': float(detection.get('short_side') or 0),
            'orientation': orientation,
        }

    def handle_accept(self) -> None:
        if self.canvas.interaction_mode == 'detect' and self.detected_single_char_measurement() is None:
            self.detection_result_label.setText('單字測量：請先框選單字並完成識別，再點擊確認。')
            return
        self.accept()

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
        mode_names = {
            'line': '測量線',
            'text': '移動文字',
            'detect': '單字測量',
        }
        mode_text = mode_names.get(self.canvas.interaction_mode, self.canvas.interaction_mode)
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
        info = (
            f'模式：{mode_text}    '
            f'目前：{self.canvas.active_index + 1}/{len(self.canvas.entries)} item={item_index}    '
            f'字體大小：{int(round(self.canvas.active_font_size()))} px    '
            f'旋轉：{self.canvas.active_rotation():g} deg'
        )
        detection = self.canvas.detection_result()
        if detection is not None:
            axis_angle = float(detection.get('axis_angle') or 0)
            long_side = float(detection.get('long_side') or 0)
            short_side = float(detection.get('short_side') or 0)
            orientation = str(active_entry.get('orientation') or 'vertical') if active_entry is not None else 'vertical'
            detected_rotation = normalized_axis_angle(
                axis_angle if orientation == 'horizontal' else axis_angle - 90.0,
            )
            detected_font_size = max(1, int(math.floor(long_side + 0.5)))
            orientation_text = '橫排' if orientation == 'horizontal' else '直排'
            info += (
                f'\n單字測量（待確認）：字體大小 {detected_font_size} px    短邊 {short_side:.1f} px    '
                f'長邊 {long_side:.1f} px    長軸角度 {axis_angle:g} deg    '
                f'按{orientation_text}換算旋轉 {detected_rotation:g} deg'
            )
            self.detection_result_label.setText(
                f'單字測量　字體大小：{detected_font_size} px　短邊：{short_side:.1f} px　'
                f'長邊：{long_side:.1f} px　長軸角度：{axis_angle:g}°　'
                f'{orientation_text}旋轉：{detected_rotation:g}°　點擊確認後套用'
            )
        elif self.canvas.detection_error():
            info += f'\n識別結果：{self.canvas.detection_error()}'
            self.detection_result_label.setText(f'單字測量失敗：{self.canvas.detection_error()}')
        elif self.canvas.interaction_mode == 'detect':
            info += '\n沿文字方向拖出中心線；圓點調整長度與方向，黃色方塊調整寬度，拖動帶內區域可整體移動。'
            self.detection_result_label.setText('單字測量：沿單字方向拖出中心線，再用黃色方塊調整選區寬度。')
        else:
            self.detection_result_label.setText('單字測量：點擊「單字測量」後框選一個單字。')
        self.info_label.setText(info)

    def result_updates(self) -> dict[int, dict[str, object]]:
        if self.canvas.interaction_mode == 'detect':
            measurement = self.detected_single_char_measurement()
            if measurement is None:
                return {}
            item_index = int(measurement['item_index'])
            return {
                item_index: {
                    'font-size': int(measurement['font-size']),
                    'rotation': float(measurement['rotation']),
                },
            }
        return self.canvas.result_updates()
