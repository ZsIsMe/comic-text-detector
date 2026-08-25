"""Viewer dialogs and font-size utility widgets."""

from __future__ import annotations

import traceback

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


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


def positive_int(value, default: int, *, minimum: int = 1) -> int:
    try:
        number = int(round(float(value)))
    except (TypeError, ValueError):
        return default
    return number if number >= minimum else default


class RangeSlider(QWidget):
    rangeChanged = Signal(int, int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._minimum = 1
        self._maximum = 100
        self._lower = 1
        self._upper = 100
        self._active_handle: str | None = None
        self.setMinimumHeight(44)
        self.setMouseTracking(True)

    def sizeHint(self) -> QSize:
        return QSize(260, 44)

    def setRange(self, minimum: int, maximum: int) -> None:
        self._minimum = int(minimum)
        self._maximum = max(self._minimum, int(maximum))
        self.setValues(self._lower, self._upper)

    def setValues(self, lower: int, upper: int) -> None:
        lower = max(self._minimum, min(int(lower), self._maximum))
        upper = max(self._minimum, min(int(upper), self._maximum))
        if lower > upper:
            lower, upper = upper, lower
        changed = lower != self._lower or upper != self._upper
        self._lower = lower
        self._upper = upper
        self.update()
        if changed:
            self.rangeChanged.emit(self._lower, self._upper)

    def lowerValue(self) -> int:
        return self._lower

    def upperValue(self) -> int:
        return self._upper

    def _groove_rect(self) -> QRectF:
        margin = 16
        y = self.height() / 2.0 - 3
        return QRectF(margin, y, max(1, self.width() - margin * 2), 6)

    def _value_to_x(self, value: int) -> float:
        groove = self._groove_rect()
        if self._maximum <= self._minimum:
            return groove.left()
        ratio = (value - self._minimum) / (self._maximum - self._minimum)
        return groove.left() + ratio * groove.width()

    def _x_to_value(self, x: float) -> int:
        groove = self._groove_rect()
        if groove.width() <= 0 or self._maximum <= self._minimum:
            return self._minimum
        ratio = max(0.0, min(1.0, (x - groove.left()) / groove.width()))
        return int(round(self._minimum + ratio * (self._maximum - self._minimum)))

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        groove = self._groove_rect()
        lower_x = self._value_to_x(self._lower)
        upper_x = self._value_to_x(self._upper)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(58, 63, 70))
        painter.drawRoundedRect(groove, 3, 3)
        painter.setBrush(QColor(80, 160, 255))
        painter.drawRoundedRect(QRectF(lower_x, groove.top(), upper_x - lower_x, groove.height()), 3, 3)
        for x, active in ((lower_x, self._active_handle == 'lower'), (upper_x, self._active_handle == 'upper')):
            painter.setBrush(QColor(245, 248, 252) if active else QColor(218, 224, 230))
            painter.setPen(QPen(QColor(22, 26, 30), 1))
            painter.drawEllipse(QPointF(x, groove.center().y()), 8, 8)
        painter.end()

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        x = event.position().x()
        lower_distance = abs(x - self._value_to_x(self._lower))
        upper_distance = abs(x - self._value_to_x(self._upper))
        self._active_handle = 'lower' if lower_distance <= upper_distance else 'upper'
        self._move_active_handle(x)
        event.accept()

    def mouseMoveEvent(self, event) -> None:
        if self._active_handle is not None and event.buttons() & Qt.MouseButton.LeftButton:
            self._move_active_handle(event.position().x())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._active_handle is not None:
            self._active_handle = None
            self.update()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _move_active_handle(self, x: float) -> None:
        value = self._x_to_value(x)
        if self._active_handle == 'lower':
            self.setValues(min(value, self._upper), self._upper)
        elif self._active_handle == 'upper':
            self.setValues(self._lower, max(value, self._lower))


class FontSizeRegularizeDialog(QDialog):
    def __init__(self, counts: dict[int, int], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle('規整字體大小')
        self.counts = {int(size): int(count) for size, count in counts.items() if int(size) > 0 and int(count) > 0}
        sizes = sorted(self.counts)
        minimum = sizes[0]
        maximum = sizes[-1]

        self.range_slider = RangeSlider(self)
        self.range_slider.setRange(minimum, maximum)
        self.range_slider.setValues(minimum, maximum)
        self.range_label = QLabel()
        self.preview_label = QLabel()
        self.preview_label.setWordWrap(True)
        self.target_spin = QSpinBox()
        self.target_spin.setRange(1, 999)
        self.target_spin.setSuffix(' px')
        mode_size = max(sizes, key=lambda size: (self.counts[size], -size))
        self.target_spin.setValue(mode_size)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText('確認')
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText('取消')
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)
        layout.addWidget(self.range_label)
        layout.addWidget(self.range_slider)
        target_row = QHBoxLayout()
        target_row.addWidget(QLabel('改為'))
        target_row.addWidget(self.target_spin)
        target_row.addStretch(1)
        layout.addLayout(target_row)
        layout.addWidget(self.preview_label)
        layout.addWidget(buttons)

        self.range_slider.rangeChanged.connect(self.update_preview)
        self.target_spin.valueChanged.connect(self.update_preview)
        self.update_preview()
        self.resize(420, self.sizeHint().height())

    def selected_range(self) -> tuple[int, int]:
        return self.range_slider.lowerValue(), self.range_slider.upperValue()

    def target_size(self) -> int:
        return int(self.target_spin.value())

    def affected_count(self) -> int:
        lower, upper = self.selected_range()
        return sum(count for size, count in self.counts.items() if lower <= size <= upper)

    def changed_count(self) -> int:
        lower, upper = self.selected_range()
        target = self.target_size()
        return sum(count for size, count in self.counts.items() if lower <= size <= upper and size != target)

    def update_preview(self, *_args) -> None:
        lower, upper = self.selected_range()
        target = self.target_size()
        affected_sizes = [size for size in sorted(self.counts) if lower <= size <= upper]
        changed_sizes = [size for size in affected_sizes if size != target]
        affected = sum(self.counts[size] for size in affected_sizes)
        changed = sum(self.counts[size] for size in changed_sizes)
        self.range_label.setText(f'範圍：{lower} - {upper} px')
        size_text = ', '.join(str(size) for size in changed_sizes) if changed_sizes else '無'
        self.preview_label.setText(
            f'將範圍內 {affected} 條 _bt 字體大小改為 {target} px；'
            f'實際會修改 {changed} 條。涉及字級：{size_text}'
        )
