"""BT image, HTML overlay and drag-preview rendering for the CTC viewer."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QApplication

try:
    from .viewer_dialogs import positive_int
except ImportError:
    from viewer_dialogs import positive_int


class ViewerBTRenderMixin:
    def bt_inpainted_overlay_path(self, page_name: str) -> Path | None:
        if self.processor is None:
            return None
        stem = Path(page_name).stem
        candidates = [
            self.processor.ctd_dir / 'inpainted' / f'{stem}.png',
            self.processor.image_dir / 'inpainted' / f'{stem}.png',
        ]
        for path in candidates:
            if path.is_file():
                return path
        return None

    def load_bt_source_image(self, page_name: str) -> QImage | None:
        if self.processor is None:
            return None
        image_path = self.processor.image_dir / page_name
        if not image_path.is_file() and self.page is not None:
            image_path = self.page.image_path
        if self._bt_cached_page_name == page_name and self._bt_cached_source_image is not None:
            return self._bt_cached_source_image.copy()
        image = QImage(str(image_path))
        if image.isNull():
            return None
        image = image.convertToFormat(QImage.Format.Format_RGBA8888)
        self._bt_cached_page_name = page_name
        self._bt_cached_source_image = image.copy()
        self._bt_cached_base_image = None
        self._bt_cached_base_show_inpainted = None
        return image

    def load_bt_base_image(self, page_name: str) -> QImage | None:
        if (
            self._bt_cached_page_name == page_name
            and self._bt_cached_base_image is not None
            and self._bt_cached_base_show_inpainted == self.show_bt_inpainted
        ):
            return self._bt_cached_base_image.copy()
        image = self.load_bt_source_image(page_name)
        if image is None:
            return None
        if not self.show_bt_inpainted:
            self._bt_cached_base_image = image.copy()
            self._bt_cached_base_show_inpainted = False
            return image
        overlay_path = self.bt_inpainted_overlay_path(page_name)
        if overlay_path is None:
            self._bt_cached_base_image = image.copy()
            self._bt_cached_base_show_inpainted = True
            return image
        overlay = QImage(str(overlay_path))
        if overlay.isNull():
            self._bt_cached_base_image = image.copy()
            self._bt_cached_base_show_inpainted = True
            return image
        overlay = overlay.convertToFormat(QImage.Format.Format_RGBA8888)
        if overlay.size() != image.size():
            overlay = overlay.scaled(
                image.size(),
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        painter = QPainter(image)
        painter.drawImage(0, 0, overlay)
        painter.end()
        self._bt_cached_base_image = image.copy()
        self._bt_cached_base_show_inpainted = True
        return image

    def render_bt_page(self, *_, refit: bool = True) -> None:
        if self.processor is None:
            return
        page_name = self.current_page_name()
        if not page_name:
            return
        base_key = (page_name, bool(self.show_bt_inpainted))
        base_changed = (
            self._bt_displayed_base_key != base_key
            or self.bt_view.current_pixmap().isNull()
        )
        if base_changed:
            image = self.load_bt_base_image(page_name)
            if image is None:
                return
            self.bt_annotation_item.set_image_size(image.width(), image.height())
            self.bt_view.set_pixmap(QPixmap.fromImage(image), fit=refit)
            self._bt_displayed_base_key = base_key
        else:
            self.bt_annotation_item.set_image_size(
                self.bt_view.current_pixmap().width(),
                self.bt_view.current_pixmap().height(),
            )
            if refit and not self.bt_view.sceneRect().isEmpty():
                self.bt_view._zoom = 1.0
                self.bt_view.resetTransform()
                self.bt_view.fitInView(
                    self.bt_view.sceneRect(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                )
                self.bt_view.viewportChanged.emit(self.bt_view)
        self.bt_annotation_item.update()
        self.update_navigator()
        self.schedule_bt_html_overlay_update()

    def update_bt_html_overlay(self) -> None:
        overlay = getattr(self, 'bt_html_overlay', None)
        if overlay is None:
            return
        if self._bt_drag_active:
            overlay.suspend()
            return
        overlay.resume()
        overlay.update_geometry()
        items = self.build_bt_html_items()
        if items:
            overlay.set_items(items)
        else:
            overlay.hide()

    def schedule_bt_html_overlay_update(self) -> None:
        if getattr(self, 'bt_html_overlay', None) is None:
            return
        if getattr(self, '_bt_html_overlay_update_scheduled', False):
            return
        self._bt_html_overlay_update_scheduled = True
        QTimer.singleShot(16, self._flush_bt_html_overlay_update)

    def _flush_bt_html_overlay_update(self) -> None:
        self._bt_html_overlay_update_scheduled = False
        self.update_bt_html_overlay()

    def build_bt_html_items(self) -> list[dict[str, object]]:
        if self.bt_data is None:
            return []
        viewport_size = self.bt_view.viewport().size()
        width = max(1, viewport_size.width())
        height = max(1, viewport_size.height())
        scale = max(0.01, min(abs(self.bt_view.transform().m11()), abs(self.bt_view.transform().m22())))
        items = []
        for index, item in enumerate(self.bt_items_for_page()):
            text = str(item.get('text') or '').strip()
            if not text:
                continue
            center = self.bt_center_pixel_from_item(item)
            if center is None:
                continue
            point = self.bt_view.mapFromScene(QPointF(center[0], center[1]))
            if point.x() < -width or point.x() > width * 2 or point.y() < -height or point.y() > height * 2:
                continue
            font_size = max(1, min(999, positive_int(item.get('font-size'), 40)))
            display_font_size = max(1, font_size * scale)
            color_name = self.bt_css_text_color(item)
            stroke_weight, stroke_color_name = self.bt_css_stroke(item, font_size, scale, color_name)
            font_family = self.bt_html_font_family(item)
            orientation = str(item.get('orientation') or 'vertical')
            items.append({
                'id': str(item.get('index', index)),
                'x': round(float(point.x()), 3),
                'y': round(float(point.y()), 3),
                'rotation': self.normalized_rotation(item.get('rotation')),
                'text': self.prepare_html_bt_text(text, orientation),
                'fontSize': round(float(display_font_size), 3),
                'fontFamily': font_family,
                'color': color_name,
                'textShadow': self.bt_text_shadow_css(stroke_weight, stroke_color_name),
                'vertical': orientation == 'vertical',
            })
        return items

    def bt_css_text_color(self, item: dict[str, Any]) -> str:
        color = str(item.get('color') or '#000000').strip()
        if not color:
            return '#000000'
        if color.lower() == 'black':
            return '#000000'
        if color.lower() == 'white':
            return '#FFFFFF'
        return color if color.startswith('#') else f'#{color}'

    def bt_css_stroke(self, item: dict[str, Any], font_size: int, scale: float, color_name: str) -> tuple[float, str]:
        stroke_weight = max(0.0, float(item.get('stroke-weight') or 0) * scale)
        stroke_color = str(item.get('stroke-color') or '').strip()
        if not stroke_color:
            stroke_color = '#000000' if color_name.lower() in {'#ffffff', 'white'} else '#FFFFFF'
        if stroke_color.lower() == 'black':
            stroke_color = '#000000'
        elif stroke_color.lower() == 'white':
            stroke_color = '#FFFFFF'
        elif stroke_color and not stroke_color.startswith('#'):
            stroke_color = f'#{stroke_color}'
        return min(stroke_weight, max(1.0, float(font_size) * scale / 2.0)), stroke_color

    def prepare_html_bt_text(self, text: str, orientation: str) -> str:
        prepared = text.replace('\\n', '\n').replace('\r\n', '\n').replace('\r', '\n')
        prepared = prepared.translate(str.maketrans({
            '「': '｢',
            '」': '｣',
            '“': '‶',
            '”': '〟',
        }))
        prepared = prepared.translate(str.maketrans({
            '!': '！',
            '"': '＂',
            '#': '＃',
            '$': '＄',
            '%': '％',
            '&': '＆',
            "'": '＇',
            '(': '（',
            ')': '）',
            '*': '＊',
            '+': '＋',
            ',': '，',
            '-': '－',
            '.': '．',
            '/': '／',
            ':': '：',
            ';': '；',
            '<': '＜',
            '=': '＝',
            '>': '＞',
            '?': '？',
            '@': '＠',
            '[': '［',
            '\\': '＼',
            ']': '］',
            '^': '＾',
            '_': '＿',
            '`': '｀',
            '{': '｛',
            '|': '｜',
            '}': '｝',
            '~': '～',
        }))
        if orientation == 'vertical':
            prepared = ''.join(
                chr(ord(char) + 0xFEE0) if '0' <= char <= '9' else char
                for char in prepared
            )
        return prepared

    def bt_html_font_family(self, item: dict[str, Any]) -> str:
        font = str(item.get('font') or '').strip()
        fallback = '"Noto Sans TC", "Hiragino Sans", "PingFang TC", "PingFang SC", sans-serif'
        if not font:
            return fallback
        escaped = font.replace('"', '\\"')
        return f'"{escaped}", {fallback}'

    def bt_center_pixel_from_item(
        self,
        item: dict[str, Any],
    ) -> tuple[float, float] | None:
        xyxy = self.bt_xyxy_from_item(item)
        if xyxy is not None:
            x1, y1, x2, y2 = xyxy
            return (x1 + x2) / 2.0, (y1 + y2) / 2.0
        try:
            x = float(item.get('x'))
            y = float(item.get('y'))
        except (TypeError, ValueError):
            x = y = None
        if x is not None and y is not None:
            pixmap = self.bt_view.pixmap_item.pixmap()
            image_width = pixmap.width()
            image_height = pixmap.height()
            return x * image_width, y * image_height
        return None

    def bt_text_shadow_css(self, stroke_weight: float, stroke_color_name: str) -> str:
        if stroke_weight <= 0:
            return ''
        shadows = []
        for angle in range(0, 360, 45):
            rad = np.deg2rad(angle)
            x = round(float(np.cos(rad) * stroke_weight), 2)
            y = round(float(np.sin(rad) * stroke_weight), 2)
            shadows.append(f'{x}px {y}px 0 {stroke_color_name}')
        for angle in range(22, 360, 45):
            rad = np.deg2rad(angle)
            x = round(float(np.cos(rad) * stroke_weight), 2)
            y = round(float(np.sin(rad) * stroke_weight), 2)
            shadows.append(f'{x}px {y}px 0 {stroke_color_name}')
        return 'text-shadow:' + ','.join(shadows) + ';'

    def _draw_bt_items(self, painter: QPainter, image_width: int, image_height: int) -> None:
        items = self.bt_items_for_page()
        if not items:
            font = QFont('Helvetica', 24)
            font.setBold(True)
            painter.setFont(font)
            painter.setPen(QPen(QColor(245, 245, 245), 1))
            painter.drawText(
                QRectF(0, 0, image_width, image_height),
                Qt.AlignmentFlag.AlignCenter,
                '未載入 _bt.json',
            )
            return

        selected_indices = set(self.selected_bt_indices_list())
        multiple_selected = len(selected_indices) > 1
        dragging = self._bt_drag_active
        for index, item in enumerate(items):
            xyxy = self.bt_xyxy_from_item(item)
            if xyxy is None:
                continue
            x1, y1, x2, y2 = xyxy
            selected = index in selected_indices
            if dragging and not selected:
                continue
            frame_color = QColor(255, 236, 150, 210)
            if selected:
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.setPen(QPen(frame_color, 2))
                painter.drawRect(QRectF(x1, y1, max(1, x2 - x1), max(1, y2 - y1)))
            if selected and index == self.selected_bt_index and not multiple_selected:
                painter.setBrush(frame_color)
                for hx, hy in (
                    (x1, y1), ((x1 + x2) / 2, y1), (x2, y1),
                    (x1, (y1 + y2) / 2), (x2, (y1 + y2) / 2),
                    (x1, y2), ((x1 + x2) / 2, y2), (x2, y2),
                ):
                    painter.drawRect(QRectF(hx - 3, hy - 3, 6, 6))
            if self._bt_drag_active and selected:
                center_x = (x1 + x2) / 2.0
                center_y = (y1 + y2) / 2.0
                center_color = QColor(255, 92, 92, 235) if index == self.selected_bt_index else QColor(80, 210, 255, 230)
                painter.setBrush(center_color)
                painter.setPen(QPen(center_color, 2))
                painter.drawEllipse(QPointF(center_x, center_y), 4, 4)
                painter.drawLine(QPointF(center_x - 11, center_y), QPointF(center_x + 11, center_y))
                painter.drawLine(QPointF(center_x, center_y - 11), QPointF(center_x, center_y + 11))
            if not dragging and self.show_bt_font_labels_check.isChecked():
                self.draw_bt_font_label(
                    painter,
                    QRectF(x1, y1, x2 - x1, y2 - y1),
                    self.bt_font_label(item),
                    image_width,
                    image_height,
                )

    def draw_bt_font_label(
        self,
        painter: QPainter,
        item_rect: QRectF,
        label: str,
        image_width: int,
        image_height: int,
    ) -> None:
        if not label:
            return
        font = QFont('Helvetica', max(16, min(30, (image_width // 100) * 2)))
        font.setBold(True)
        painter.setFont(font)
        metrics = painter.fontMetrics()
        text_rect = metrics.boundingRect(label)
        pad = 8
        label_w = text_rect.width() + pad * 2
        label_h = text_rect.height() + pad * 2
        x = min(max(int(round(item_rect.right() + 10)), 2), max(2, image_width - label_w - 2))
        y = min(max(int(round(item_rect.bottom() + 10)), label_h + 2), max(label_h + 2, image_height - 2))
        painter.fillRect(QRectF(x, y - label_h, label_w, label_h), QColor(255, 255, 255, 230))
        painter.setPen(QPen(QColor(20, 20, 20), 1))
        painter.drawText(QPointF(x + pad, y - pad), label)

    def hit_test_bt_item(self, x: float, y: float) -> tuple[int | None, str | None]:
        handles = (
            ('tl', -1, -1), ('t', 0, -1), ('tr', 1, -1),
            ('l', -1, 0), ('r', 1, 0),
            ('bl', -1, 1), ('b', 0, 1), ('br', 1, 1),
        )
        tolerance = 8.0
        matches = []
        for index, item in enumerate(self.bt_items_for_page()):
            xyxy = self.bt_xyxy_from_item(item)
            if xyxy is None:
                continue
            x1, y1, x2, y2 = xyxy
            for mode, sx, sy in handles:
                hx = (x1 + x2) / 2 if sx == 0 else (x1 if sx < 0 else x2)
                hy = (y1 + y2) / 2 if sy == 0 else (y1 if sy < 0 else y2)
                if abs(x - hx) <= tolerance and abs(y - hy) <= tolerance:
                    matches.append((0, index, mode))
            if x1 <= x <= x2 and y1 <= y <= y2:
                area = max(0, x2 - x1) * max(0, y2 - y1)
                matches.append((area, index, 'move'))
        if not matches:
            return None, None
        _, index, mode = min(matches, key=lambda item: item[0])
        return index, mode

    def _schedule_bt_drag_render(self) -> None:
        if not self._bt_drag_render_timer.isActive():
            self._bt_drag_render_timer.start()

    def _render_bt_drag_preview(self) -> None:
        if self._bt_drag_active:
            self.bt_annotation_item.update()

    def handle_bt_mouse_press(self, x: float, y: float) -> None:
        self._bt_drag_active = False
        self._bt_drag_render_timer.stop()
        index, mode = self.hit_test_bt_item(x, y)
        self.bt_view.set_background_pan_enabled(index is None)
        modifiers = QApplication.keyboardModifiers()
        toggle_selection = bool(
            modifiers & (Qt.KeyboardModifier.MetaModifier | Qt.KeyboardModifier.ControlModifier)
        )
        if toggle_selection and index is not None:
            selected = set(self.selected_bt_indices_list())
            if index in selected and len(selected) > 1:
                selected.remove(index)
                active_index = self.active_bt_index_from_selection(selected)
            else:
                selected.add(index)
                active_index = index
            self.set_bt_selection(selected, active_index=active_index)
            self._bt_drag_mode = None
            self._bt_drag_start = None
            self._bt_drag_original = None
            self._bt_drag_original_item = None
            self._bt_drag_original_items = None
            self._bt_drag_original_xyxys = {}
            self._bt_drag_indices = []
            self._bt_drag_temporary = False
            return
        if index is None:
            self.select_bt_item(None)
        elif index in self.selected_bt_indices_list() and self.has_multiple_bt_selection():
            self.set_bt_selection(set(self.selected_bt_indices_list()), active_index=index)
        else:
            self.select_bt_item(index)
        item = self.selected_bt_item()
        xyxy = self.bt_xyxy_from_item(item) if item is not None else None
        if item is None or xyxy is None or mode is None:
            self._bt_drag_mode = None
            self._bt_drag_start = None
            self._bt_drag_original = None
            self._bt_drag_original_item = None
            self._bt_drag_original_items = None
            self._bt_drag_original_xyxys = {}
            self._bt_drag_indices = []
            self._bt_drag_temporary = False
            self._popover_bt_item = None
            self.bt_match_popover.hide()
            return
        temporary = bool(QApplication.keyboardModifiers() & Qt.KeyboardModifier.AltModifier)
        drag_indices = self.selected_bt_indices_list()
        if self.has_multiple_bt_selection():
            mode = 'move'
            drag_indices = [
                selected_index for selected_index in drag_indices
                if self.bt_xyxy_from_item(self.bt_items_for_page()[selected_index]) is not None
            ]
        self._bt_drag_mode = 'move' if temporary else mode
        self._bt_drag_start = (x, y)
        self._bt_drag_original = xyxy
        self._bt_drag_original_item = copy.deepcopy(item)
        self._bt_drag_indices = drag_indices
        self._bt_drag_original_xyxys = {}
        items = self.bt_items_for_page()
        self._bt_drag_original_items = {
            selected_index: copy.deepcopy(items[selected_index])
            for selected_index in drag_indices
            if 0 <= selected_index < len(items) and isinstance(items[selected_index], dict)
        }
        for selected_index in drag_indices:
            if 0 <= selected_index < len(items):
                selected_xyxy = self.bt_xyxy_from_item(items[selected_index])
                if selected_xyxy is not None:
                    self._bt_drag_original_xyxys[selected_index] = selected_xyxy
        self._bt_drag_temporary = temporary
        self._bt_drag_active = True
        if self.bt_html_overlay is not None:
            self.bt_html_overlay.suspend()
        self.show_bt_match_popover(item)

    def update_bt_view_cursor(self, x: float, y: float) -> None:
        self._bt_cursor_image_pos = (float(x), float(y))
        index, _ = self.hit_test_bt_item(x, y)
        self.bt_view.set_background_pan_enabled(index is None)
        self.update_action_state()

    def clear_bt_view_cursor(self) -> None:
        self._bt_cursor_image_pos = None
        self.bt_view.set_background_pan_enabled(False)
        self.update_action_state()

    def handle_bt_mouse_drag(self, x: float, y: float) -> None:
        item = self.selected_bt_item()
        if item is None or self._bt_drag_mode is None or self._bt_drag_start is None or self._bt_drag_original is None:
            return
        dx = int(round(x - self._bt_drag_start[0]))
        dy = int(round(y - self._bt_drag_start[1]))
        items = self.bt_items_for_page()
        if self.has_multiple_bt_selection() and self._bt_drag_mode == 'move':
            for index, xyxy in self._bt_drag_original_xyxys.items():
                if 0 <= index < len(items):
                    x1, y1, x2, y2 = xyxy
                    self.set_bt_xyxy(items[index], self.clamp_xyxy((x1 + dx, y1 + dy, x2 + dx, y2 + dy)))
            self._schedule_bt_drag_render()
            return
        x1, y1, x2, y2 = self._bt_drag_original
        if self._bt_drag_mode == 'move':
            new_xyxy = (x1 + dx, y1 + dy, x2 + dx, y2 + dy)
        else:
            nx1, ny1, nx2, ny2 = x1, y1, x2, y2
            if 'l' in self._bt_drag_mode:
                nx1 += dx
            if 'r' in self._bt_drag_mode:
                nx2 += dx
            if 't' in self._bt_drag_mode:
                ny1 += dy
            if 'b' in self._bt_drag_mode:
                ny2 += dy
            new_xyxy = (nx1, ny1, nx2, ny2)
        self.set_bt_xyxy(item, self.clamp_xyxy(new_xyxy))
        self._schedule_bt_drag_render()

    def handle_bt_mouse_release(self, x: float, y: float) -> None:
        self._bt_drag_render_timer.stop()
        self._bt_drag_active = False
        item = self.selected_bt_item()
        if (
            self._bt_drag_temporary
            and (self._bt_drag_original_item is not None or self._bt_drag_original_items is not None)
            and self.selected_bt_index is not None
        ):
            page_name = self.current_page_name()
            items = self.bt_items_for_page(page_name)
            if self._bt_drag_original_items is not None:
                for index, original_item in self._bt_drag_original_items.items():
                    if 0 <= index < len(items):
                        items[index] = copy.deepcopy(original_item)
                self.populate_box_editor_for_selection()
                self.update_bt_item_list()
                self.render_bt_page(refit=False)
                self.status_label.setText('臨時移動結束，已回到原位。')
            self._bt_drag_mode = None
            self._bt_drag_start = None
            self._bt_drag_original = None
            self._bt_drag_original_item = None
            self._bt_drag_original_items = None
            self._bt_drag_original_xyxys = {}
            self._bt_drag_indices = []
            self._bt_drag_temporary = False
            return
        if (
            self.has_multiple_bt_selection()
            and self._bt_drag_original_items is not None
            and self._bt_drag_original_xyxys
        ):
            page_name = self.current_page_name()
            items = self.bt_items_for_page(page_name)
            changed = False
            for index, original_xyxy in self._bt_drag_original_xyxys.items():
                if 0 <= index < len(items):
                    xyxy = self.bt_xyxy_from_item(items[index])
                    if xyxy is not None and xyxy != original_xyxy:
                        items[index]['match_status'] = 'manual'
                        changed = True
            if changed:
                self.push_bt_changes_undo_snapshot(
                    '移動 _bt 多選框',
                    self._bt_drag_original_items,
                    self.selected_bt_index,
                    self.selected_bt_indices,
                )
                self.mark_bt_dirty()
                self.populate_box_editor_for_selection()
                self.update_bt_item_list()
                self.status_label.setText(f'已移動 {len(self._bt_drag_original_xyxys)} 條 _bt 條目，尚未保存。')
            self.render_bt_page(refit=False)
            self._bt_drag_mode = None
            self._bt_drag_start = None
            self._bt_drag_original = None
            self._bt_drag_original_item = None
            self._bt_drag_original_items = None
            self._bt_drag_original_xyxys = {}
            self._bt_drag_indices = []
            self._bt_drag_temporary = False
            return
        if item is not None and self._bt_drag_original is not None and self._bt_drag_original_item is not None:
            xyxy = self.bt_xyxy_from_item(item)
            if xyxy is not None and xyxy != self._bt_drag_original:
                page_name = self.current_page_name()
                if page_name is not None and self.selected_bt_index is not None:
                    self.bt_undo_stack.append({
                        'page_name': page_name,
                        'item_index': self.selected_bt_index,
                        'item': self._bt_drag_original_item,
                        'description': '移動/調整 _bt 框',
                    })
                self.mark_bt_dirty()
                item['match_status'] = 'manual'
                self.populate_box_editor_from_bt(item)
                self.update_bt_item_list()
                self.status_label.setText('已修改 _bt 框，尚未保存。')
        self.render_bt_page(refit=False)
        self._bt_drag_mode = None
        self._bt_drag_start = None
        self._bt_drag_original = None
        self._bt_drag_original_item = None
        self._bt_drag_original_items = None
        self._bt_drag_original_xyxys = {}
        self._bt_drag_indices = []
        self._bt_drag_temporary = False
