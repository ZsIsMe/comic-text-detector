"""Popover and HTML overlay widgets for the CTC overlay viewer."""

from __future__ import annotations

import json

from PySide6.QtCore import QEvent, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPixmap, QTextCursor
from PySide6.QtWidgets import QFrame, QLabel, QPlainTextEdit, QVBoxLayout

try:
    from PySide6.QtWebEngineWidgets import QWebEngineView
except (ImportError, ModuleNotFoundError):
    QWebEngineView = None

try:
    from .viewer_widgets import ImageView
except ImportError:
    from viewer_widgets import ImageView


class BtMatchPopover(QFrame):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName('btMatchPopover')
        self.setWindowFlags(Qt.WindowType.ToolTip)
        self.setStyleSheet(
            'QFrame#btMatchPopover {'
            '  background: rgba(20, 22, 24, 230);'
            '  border: 1px solid rgba(255, 235, 150, 210);'
            '  border-radius: 4px;'
            '}'
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(0)
        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumSize(180, 96)
        self.preview_label.setMouseTracking(True)
        self.preview_label.setStyleSheet('background: #101214;')
        self.preview_label.installEventFilter(self)
        self._char_regions: list[tuple[QRectF, str]] = []
        self._base_pixmap = QPixmap()
        self._hover_region_index: int | None = None
        layout.addWidget(self.preview_label)
        self.hide()

    def set_content(
        self,
        preview: QPixmap | None,
        char_regions: list[tuple[QRectF, str]] | None = None,
    ) -> None:
        self._char_regions = char_regions or []
        self._hover_region_index = None
        if preview is None or preview.isNull():
            self._base_pixmap = QPixmap()
            self.preview_label.setMinimumSize(180, 96)
            self.preview_label.setMaximumSize(16777215, 16777215)
            self.preview_label.setText('')
            self.preview_label.setPixmap(QPixmap())
        else:
            self._base_pixmap = preview
            self.preview_label.setText('')
            self.preview_label.setFixedSize(preview.size())
            self.preview_label.setPixmap(preview)
        self.adjustSize()

    def eventFilter(self, watched, event) -> bool:
        if watched is self.preview_label and event.type() == QEvent.Type.MouseMove:
            point = event.position()
            for index, (rect, text) in enumerate(self._char_regions):
                if rect.contains(point):
                    if self._hover_region_index != index:
                        self._hover_region_index = index
                        self._render_hover_label(rect, text)
                    return False
            if self._hover_region_index is not None:
                self._hover_region_index = None
                self.preview_label.setPixmap(self._base_pixmap)
        elif watched is self.preview_label and event.type() == QEvent.Type.Leave:
            self._hover_region_index = None
            self.preview_label.setPixmap(self._base_pixmap)
        return super().eventFilter(watched, event)

    def _render_hover_label(self, rect: QRectF, text: str) -> None:
        if self._base_pixmap.isNull() or not text:
            return
        pixmap = QPixmap(self._base_pixmap)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        font = QFont('Helvetica Neue', 22)
        font.setBold(True)
        painter.setFont(font)
        metrics = painter.fontMetrics()
        pad_x = 8
        pad_y = 5
        text_w = metrics.horizontalAdvance(text)
        text_h = metrics.ascent() + metrics.descent()
        label_w = text_w + pad_x * 2
        label_h = text_h + pad_y * 2
        x = int(round(rect.center().x() - label_w / 2))
        y = int(round(rect.top() - label_h - 5))
        if y < 2:
            y = int(round(rect.bottom() + 5))
        x = max(2, min(x, max(2, pixmap.width() - label_w - 2)))
        y = max(2, min(y, max(2, pixmap.height() - label_h - 2)))
        painter.setPen(QPen(QColor(25, 25, 25), 1))
        painter.setBrush(QColor(255, 255, 255, 245))
        painter.drawRect(QRectF(x, y, label_w, label_h))
        painter.setPen(QPen(QColor(0, 0, 0), 1))
        painter.drawText(QRectF(x, y, label_w, label_h), Qt.AlignmentFlag.AlignCenter, text)
        painter.end()


class BtTextEditPopover(QFrame):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName('btTextEditPopover')
        self.setWindowFlags(Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(
            'QFrame#btTextEditPopover {'
            '  background: rgba(20, 22, 24, 245);'
            '  border: 1px solid rgba(255, 235, 150, 230);'
            '  border-radius: 5px;'
            '}'
            'QPlainTextEdit {'
            '  background: #101214;'
            '  color: #f4f5f6;'
            '  border: 1px solid #555b63;'
            '  border-radius: 3px;'
            '  padding: 5px;'
            '}'
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)
        self.hint_label = QLabel('文字（輸入即時保存到目前編輯狀態）')
        self.hint_label.setStyleSheet('color: #d9dcdf; border: none;')
        self.text_edit = QPlainTextEdit()
        self.text_edit.setPlaceholderText('輸入文字')
        self.text_edit.setFixedSize(280, 108)
        layout.addWidget(self.hint_label)
        layout.addWidget(self.text_edit)
        self.hide()

    def set_text(self, text: str) -> None:
        self.text_edit.blockSignals(True)
        self.text_edit.setPlainText(text)
        self.text_edit.blockSignals(False)
        self.text_edit.moveCursor(QTextCursor.MoveOperation.End)


class HtmlTextOverlay:
    BASE_HTML = '''<!doctype html>
<meta charset="utf-8">
<style>
  html, body {
margin: 0;
width: 100%;
height: 100%;
overflow: hidden;
background: transparent;
  }
  #overlay {
position: relative;
width: 100%;
height: 100%;
overflow: hidden;
  }
  .text-item {
position: absolute;
transform: translate(-50%, -50%) rotate(0deg);
transform-origin: center center;
font-weight: 500;
white-space: pre;
width: auto;
text-align: left;
line-height: 125%;
letter-spacing: 0;
  }
  .vertical-text {
writing-mode: vertical-rl;
text-orientation: mixed;
  }
</style>
<div id="overlay"></div>
<script>
  const overlay = document.getElementById('overlay');
  const nodes = new Map();

  function applyItem(el, item) {
el.className = item.vertical ? 'text-item vertical-text' : 'text-item';
if (el.__text !== item.text) {
  el.textContent = item.text;
  el.__text = item.text;
}
const style = el.style;
style.left = item.x + 'px';
style.top = item.y + 'px';
style.transform = 'translate(-50%, -50%) rotate(' + (-item.rotation) + 'deg)';
style.fontSize = item.fontSize + 'px';
style.fontFamily = item.fontFamily;
style.color = item.color;
style.textShadow = item.textShadow || '';
  }

  window.updateItems = function(items) {
const live = new Set();
for (const item of items) {
  const id = String(item.id);
  live.add(id);
  let el = nodes.get(id);
  if (!el) {
    el = document.createElement('div');
    nodes.set(id, el);
    overlay.appendChild(el);
  }
  applyItem(el, item);
}
for (const [id, el] of nodes) {
  if (!live.has(id)) {
    el.remove();
    nodes.delete(id);
  }
}
  };
</script>'''

    def __init__(self, view: ImageView) -> None:
        self.view = view
        self.web_view = QWebEngineView(view.viewport()) if QWebEngineView is not None else None
        self._ready = False
        self._pending_items: list[dict[str, object]] | None = None
        self._last_payload = ''
        self._suspended = False
        if self.web_view is None:
            return
        self.web_view.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.web_view.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.web_view.setStyleSheet('background: transparent;')
        self.web_view.page().setBackgroundColor(Qt.GlobalColor.transparent)
        self.web_view.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)
        self.web_view.setGeometry(view.viewport().rect())
        self.web_view.loadFinished.connect(self._handle_load_finished)
        self.web_view.setHtml(self.BASE_HTML)
        self.web_view.hide()

    def set_items(self, items: list[dict[str, object]]) -> None:
        if self.web_view is None:
            return
        self._pending_items = items
        if self._suspended:
            return
        self.web_view.setVisible(bool(items))
        if items:
            self.web_view.raise_()
        if not self._ready:
            return
        payload = json.dumps(items, ensure_ascii=False, separators=(',', ':'))
        if payload == self._last_payload:
            return
        self._last_payload = payload
        self.web_view.page().runJavaScript(f'window.updateItems({payload});')

    def suspend(self) -> None:
        if self.web_view is None:
            return
        self._suspended = True
        self.web_view.hide()

    def resume(self) -> None:
        if self.web_view is None:
            return
        self._suspended = False
        if self._pending_items is not None:
            self.set_items(self._pending_items)

    def hide(self) -> None:
        if self.web_view is None:
            return
        self._suspended = False
        self._pending_items = []
        self._last_payload = ''
        if self._ready:
            self.web_view.page().runJavaScript('window.updateItems([]);')
        self.web_view.hide()

    def update_geometry(self) -> None:
        if self.web_view is None:
            return
        self.web_view.setGeometry(self.view.viewport().rect())
        if self._pending_items:
            self.web_view.raise_()

    def _handle_load_finished(self, ok: bool) -> None:
        self._ready = ok
        if ok and self._pending_items is not None:
            self.set_items(self._pending_items)
