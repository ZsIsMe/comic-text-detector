"""_bt selection, preview and editor controls for the CTC viewer."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap

try:
    from .viewer_dialogs import positive_int
except ImportError:
    from viewer_dialogs import positive_int


class ViewerBTEditorMixin:
    def selected_box(self) -> BoxOverlay | None:
        if self.page is None or self.selected_box_index is None:
            return None
        if self.selected_box_index < 0 or self.selected_box_index >= len(self.page.boxes):
            return None
        return self.page.boxes[self.selected_box_index]

    def set_box_editor_enabled(self, enabled: bool) -> None:
        for widget in (
            self.bt_text_edit,
            self.font_size_spin,
            self.orientation_vertical_button,
            self.orientation_horizontal_button,
            self.rotation_spin,
            self.copy_bt_button,
            self.measure_preview_button,
            self.color_black_button,
            self.color_white_button,
            self.stroke_weight_spin,
            self.text_has_stroke_check,
            self.need_inpaint_check,
        ):
            widget.setEnabled(enabled)
        if not enabled:
            self.box_editor_title.setText('未選擇 _bt 條目')
            self.copy_bt_button.setText('複製文字框')
            self.bt_text_popover.hide()
            previous_updating = self._updating_editor
            self._updating_editor = True
            self.bt_text_edit.clear()
            self.rotation_spin.setValue(0.0)
            self._updating_editor = previous_updating
        self.update_bt_item_list()
        self.update_action_state()

    def selection_mixed_bt_fields(self, selected_items: list[tuple[int, dict[str, Any]]]) -> list[str]:
        if len(selected_items) <= 1:
            return []
        getters = (
            ('字體', lambda item: positive_int(item.get('font-size'), 40)),
            ('旋轉', lambda item: self.normalized_rotation(item.get('rotation'))),
            ('方向', lambda item: item.get('orientation') or 'vertical'),
            ('顏色', lambda item: self.bt_text_color(item)),
            ('描邊', lambda item: positive_int(item.get('stroke-weight'), 0)),
            ('修復', lambda item: item.get('need_inpaint') is True),
        )
        mixed: list[str] = []
        for label, getter in getters:
            values = [getter(item) for _, item in selected_items]
            if any(value != values[0] for value in values[1:]):
                mixed.append(label)
        return mixed

    def populate_box_editor_for_selection(self) -> None:
        selected_items = self.selected_bt_items()
        if not selected_items:
            self.set_box_editor_enabled(False)
            return
        active_item = self.selected_bt_item() or selected_items[0][1]
        self.populate_box_editor_from_bt(active_item)
        if len(selected_items) <= 1:
            self.bt_text_edit.setEnabled(True)
            self.measure_preview_button.setEnabled(True)
            self.copy_bt_button.setText('複製文字框')
            self.show_bt_text_popover(active_item)
            return
        self._updating_editor = True
        self.bt_text_edit.setPlainText('多選時不批量修改文字')
        self._updating_editor = False
        mixed = self.selection_mixed_bt_fields(selected_items)
        suffix = f'；混合：{", ".join(mixed)}' if mixed else ''
        active_index = self.selected_bt_index if self.selected_bt_index is not None else selected_items[0][0]
        self.box_editor_title.setText(f'已選擇 {len(selected_items)} 條；active={active_index + 1}{suffix}')
        self.bt_text_edit.setEnabled(False)
        self.measure_preview_button.setEnabled(True)
        self.copy_bt_button.setText('複製選中文字框（⌘/Ctrl+D）')
        self.bt_match_popover.hide()
        self._popover_bt_item = None
        self.bt_text_popover.hide()

    def set_bt_selection(
        self,
        indices: set[int] | list[int] | None,
        *,
        active_index: int | None = None,
        center: bool = False,
        sync_list: bool = True,
    ) -> None:
        self._commit_bt_editor_preview()
        items = self.bt_items_for_page()
        valid_indices = {
            index for index in (indices or set())
            if 0 <= index < len(items) and isinstance(items[index], dict)
        }
        active_index = self.active_bt_index_from_selection(valid_indices, active_index)
        if active_index is None:
            self.selected_bt_index = None
            self.selected_bt_indices.clear()
            self._popover_bt_item = None
            self.bt_match_popover.hide()
            self.set_box_editor_enabled(False)
            if sync_list:
                self.update_bt_item_list()
            self.render_bt_page(refit=False)
            return
        valid_indices.add(active_index)
        self.selected_bt_index = active_index
        self.selected_bt_indices = valid_indices
        self.populate_box_editor_for_selection()
        if sync_list:
            self.update_bt_item_list()
        self.render_bt_page(refit=False)
        if len(valid_indices) > 1:
            self._popover_bt_item = None
            self.bt_match_popover.hide()
        if center:
            self.center_views_on_bt_item(items[active_index])

    def select_bt_item(self, index: int | None, *, center: bool = False) -> None:
        self.set_bt_selection({index} if index is not None else set(), active_index=index, center=center)

    def clear_current_bt_focus(self) -> None:
        """Clear the current _bt selection without changing its contents."""
        if not self.selected_bt_indices_list():
            return
        self.select_bt_item(None)
        self.bt_view.setFocus()
        self.status_label.setText('已取消目前文字框的選取。')

    def center_views_on_bt_item(self, item: dict[str, Any]) -> None:
        xyxy = self.bt_xyxy_from_item(item)
        if xyxy is not None:
            x1, y1, x2, y2 = xyxy
            self.center_views_on((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            return
        pixmap = self.bt_view.pixmap_item.pixmap()
        if pixmap.isNull():
            return
        center = self.bt_center_pixel_from_item(item)
        if center is not None:
            self.center_views_on(center[0], center[1])

    def measure_box_for_bt_item(self, item: dict[str, Any]) -> BoxOverlay | None:
        if self.page is None:
            return None
        measure_index = item.get('match_measure_item_index')
        if isinstance(measure_index, int):
            for box in self.page.boxes:
                if box.measure_item_index == measure_index:
                    return box
        source_index = item.get('match_source_block_index')
        if isinstance(source_index, int):
            for box in self.page.boxes:
                if box.source_block_index == source_index:
                    return box
        return None

    def box_preview_content(
        self,
        box: BoxOverlay | None,
        *,
        char_box: dict | None = None,
    ) -> tuple[QPixmap, list[tuple[QRectF, str]]] | None:
        if box is None or self.page is None:
            return None
        image = QImage(str(self.page.image_path)).convertToFormat(QImage.Format.Format_RGBA8888)
        if image.isNull():
            return None
        x1, y1, x2, y2 = box.xyxy_pixel
        pad = max(2, int(round(max(x2 - x1, y2 - y1) * 0.04)))
        crop_x1 = max(0, x1 - pad)
        crop_y1 = max(0, y1 - pad)
        crop_x2 = min(image.width(), x2 + pad)
        crop_y2 = min(image.height(), y2 + pad)
        crop_rect = QRectF(crop_x1, crop_y1, crop_x2 - crop_x1, crop_y2 - crop_y1).toRect()
        if crop_rect.isEmpty():
            return None
        crop = image.copy(crop_rect)
        char_regions: list[tuple[QRectF, str]] = []
        for char_item in self.page.char_boxes:
            bbox = char_item.get('bbox')
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            cx1, cy1, cx2, cy2 = [float(value) for value in bbox]
            if cx2 < x1 or cx1 > x2 or cy2 < y1 or cy1 > y2:
                continue
            rect = QRectF(cx1 - crop_x1, cy1 - crop_y1, cx2 - cx1, cy2 - cy1)
            label = char_box_label(char_item) or 'W-H-'
            char_regions.append((rect, label))

        highlight_rect: QRectF | None = None
        if char_box is not None:
            bbox = char_box.get('bbox')
            if isinstance(bbox, list) and len(bbox) == 4:
                cx1, cy1, cx2, cy2 = [float(value) for value in bbox]
                highlight_rect = QRectF(cx1 - crop_x1, cy1 - crop_y1, cx2 - cx1, cy2 - cy1)
        pixmap = QPixmap.fromImage(crop)
        target_w = 460
        target_h = 360
        scaled = pixmap.scaled(
            target_w,
            target_h,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        scale_x = scaled.width() / max(1, crop.width())
        scale_y = scaled.height() / max(1, crop.height())
        scaled_regions = [
            (
                QRectF(
                    rect.x() * scale_x,
                    rect.y() * scale_y,
                    rect.width() * scale_x,
                    rect.height() * scale_y,
                ),
                tooltip,
            )
            for rect, tooltip in char_regions
        ]
        scaled_highlight = None
        if highlight_rect is not None:
            scaled_highlight = QRectF(
                highlight_rect.x() * scale_x,
                highlight_rect.y() * scale_y,
                highlight_rect.width() * scale_x,
                highlight_rect.height() * scale_y,
            )

        painter = QPainter(scaled)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        painter.setBrush(QColor(245, 170, 35, 32))
        painter.setPen(QPen(QColor(245, 170, 35), 2))
        for rect, _tooltip in scaled_regions:
            painter.drawRect(rect)
        if scaled_highlight is not None:
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor(255, 40, 120), 3))
            painter.drawRect(scaled_highlight)

        label_font = QFont('Helvetica Neue', 13)
        label_font.setWeight(QFont.Weight.DemiBold)
        label_font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
        painter.setFont(label_font)
        metrics = painter.fontMetrics()
        painter.setPen(QPen(QColor(74, 42, 12), 1))
        for rect, label in scaled_regions:
            text_w = metrics.horizontalAdvance(label)
            text_h = metrics.ascent() + metrics.descent()
            label_x = int(round(rect.center().x() - text_w / 2))
            label_y = int(round(rect.top() - 3))
            if label_y - text_h < 2:
                label_y = int(round(rect.bottom() + text_h + 3))
            label_x = max(2, min(label_x, max(2, scaled.width() - text_w - 2)))
            label_y = max(text_h + 2, min(label_y, scaled.height() - 2))
            painter.drawText(QPointF(label_x, label_y), label)
        painter.end()
        return scaled, scaled_regions

    def show_bt_match_popover(self, item: dict[str, Any]) -> None:
        if not self.match_popover_enabled():
            self._popover_bt_item = None
            self.bt_match_popover.hide()
            return
        self._popover_bt_item = item
        box = self.measure_box_for_bt_item(item)
        if box is None:
            self.bt_match_popover.hide()
            return
        content = self.box_preview_content(box)
        if content is None:
            self.bt_match_popover.hide()
            return
        preview, regions = content
        if preview.isNull():
            self.bt_match_popover.hide()
            return
        self.bt_match_popover.set_content(preview, regions)
        self.position_bt_match_popover(item)
        self.bt_match_popover.show()
        self.bt_match_popover.raise_()

    def position_bt_match_popover(self, item: dict[str, Any]) -> None:
        xyxy = self.bt_xyxy_from_item(item)
        if xyxy is None:
            return
        x1, y1, x2, y2 = xyxy
        view_rect = self.bt_view.viewport().rect()
        p1 = self.bt_view.mapFromScene(QPointF(x1, y1))
        p2 = self.bt_view.mapFromScene(QPointF(x2, y2))
        selected_rect = QRectF(p1, p2).normalized().adjusted(-8, -8, 8, 8)
        popover_size = self.bt_match_popover.sizeHint()
        gap = 12
        margin = 12
        min_x = margin
        max_x = max(margin, view_rect.width() - popover_size.width() - margin)
        min_y = margin
        max_y = max(margin, view_rect.height() - popover_size.height() - margin)
        x = int(round(selected_rect.center().x() - popover_size.width() / 2))
        x = max(min_x, min(x, max_x))

        top_y = int(round(selected_rect.top() - popover_size.height() - gap))
        bottom_y = int(round(selected_rect.bottom() + gap))
        if top_y >= min_y:
            y = top_y
        elif bottom_y + popover_size.height() <= view_rect.height() - margin:
            y = bottom_y
        else:
            space_above = selected_rect.top() - margin
            space_below = view_rect.height() - selected_rect.bottom() - margin
            y = top_y if space_above >= space_below else bottom_y
            y = max(min_y, min(y, max_y))
        global_pos = self.bt_view.viewport().mapToGlobal(QPointF(x, y).toPoint())
        self.bt_match_popover.move(global_pos)

    def update_popover_with_char_box(self, item: dict) -> None:
        if not self.match_popover_enabled():
            self.bt_match_popover.hide()
            return
        source_index = item.get('source_block_index', '-')
        box = None
        try:
            source_number = int(source_index)
        except (TypeError, ValueError):
            source_number = None
        if self.page is not None and source_number is not None:
            for candidate in self.page.boxes:
                if candidate.source_block_index == source_number:
                    box = candidate
                    break
        if box is None and self._popover_bt_item is not None:
            box = self.measure_box_for_bt_item(self._popover_bt_item)
        content = self.box_preview_content(box, char_box=item)
        if content is None:
            self.bt_match_popover.hide()
            return
        preview, regions = content
        if preview.isNull():
            self.bt_match_popover.hide()
            return
        self.bt_match_popover.set_content(preview, regions)
        if self._popover_bt_item is not None:
            self.position_bt_match_popover(self._popover_bt_item)
        else:
            self.position_char_popover(item)
        self.bt_match_popover.show()
        self.bt_match_popover.raise_()

    def position_char_popover(self, item: dict) -> None:
        bbox = item.get('bbox')
        if not isinstance(bbox, list) or len(bbox) != 4:
            return
        x1, y1, x2, y2 = [float(value) for value in bbox]
        view_rect = self.view.viewport().rect()
        p1 = self.view.mapFromScene(QPointF(x1, y1))
        p2 = self.view.mapFromScene(QPointF(x2, y2))
        hover_rect = QRectF(p1, p2).normalized().adjusted(-8, -8, 8, 8)
        popover_size = self.bt_match_popover.sizeHint()
        gap = 12
        x = hover_rect.right() + gap
        y = hover_rect.top()
        if x + popover_size.width() > view_rect.width() - gap:
            x = hover_rect.left() - popover_size.width() - gap
        if y + popover_size.height() > view_rect.height() - gap:
            y = view_rect.height() - popover_size.height() - gap
        x = max(gap, min(x, max(gap, view_rect.width() - popover_size.width() - gap)))
        y = max(gap, min(y, max(gap, view_rect.height() - popover_size.height() - gap)))
        global_pos = self.view.viewport().mapToGlobal(QPointF(x, y).toPoint())
        self.bt_match_popover.move(global_pos)

    def select_box(self, index: int | None) -> None:
        if self.page is None or index is None or index < 0 or index >= len(self.page.boxes):
            self.selected_box_index = None
            self.render_current_page(refit=False)
            return

        self.selected_box_index = index
        self.render_current_page(refit=False)

    def editor_orientation(self) -> str:
        return 'horizontal' if self.orientation_horizontal_button.isChecked() else 'vertical'

    def set_editor_orientation(self, orientation: object) -> None:
        self.orientation_horizontal_button.setChecked(orientation == 'horizontal')
        self.orientation_vertical_button.setChecked(orientation != 'horizontal')

    def editor_text_color(self) -> str:
        return 'white' if self.color_white_button.isChecked() else 'black'

    def set_editor_text_color(self, color: object) -> None:
        self.color_white_button.setChecked(color == 'white')
        self.color_black_button.setChecked(color != 'white')

    def populate_box_editor(self, box: BoxOverlay) -> None:
        self._updating_editor = True
        self.set_box_editor_enabled(True)
        x1, y1, x2, y2 = box.xyxy_pixel
        self.box_editor_title.setText(
            f'區塊 {box.source_block_index}  框：{x1},{y1},{x2},{y2}'
        )
        self.font_size_spin.setValue(max(1, int(round(float(box.font_size or 1)))))
        self.set_editor_text_color(box.text_color or 'black')
        self.text_has_stroke_check.setChecked(box.text_has_stroke is True)
        self.need_inpaint_check.setChecked(box.need_inpaint is True)
        self._updating_editor = False

    def populate_box_editor_from_bt(self, item: dict[str, Any]) -> None:
        self._updating_editor = True
        self.set_box_editor_enabled(True)
        xyxy = self.bt_xyxy_from_item(item)
        box_text = '無框' if xyxy is None else ','.join(str(v) for v in xyxy)
        self.box_editor_title.setText(
            f'index={item.get("index", "-")}  groupId={item.get("groupId", "-")}  框：{box_text}'
        )
        self.bt_text_edit.setPlainText(str(item.get('text') or ''))
        self.font_size_spin.setValue(positive_int(item.get('font-size'), 40))
        self.set_editor_orientation(item.get('orientation') or 'vertical')
        self.rotation_spin.setValue(self.normalized_rotation(item.get('rotation')))
        self.set_editor_text_color(self.bt_text_color(item))
        stroke_weight = int(round(float(item.get('stroke-weight') or 0)))
        self.stroke_weight_spin.setValue(max(0, min(99, stroke_weight)))
        self.text_has_stroke_check.setChecked(stroke_weight > 0)
        self.need_inpaint_check.setChecked(item.get('need_inpaint') is True)
        self._updating_editor = False

    def selected_box_updates_from_editor(self) -> dict[str, object] | None:
        if self._updating_editor or self.selected_bt_item() is None:
            return None
        color = self.editor_text_color()
        stroke_weight = int(self.stroke_weight_spin.value())
        if self.text_has_stroke_check.isChecked() and stroke_weight <= 0:
            stroke_weight = max(1, int(np.ceil(float(self.font_size_spin.value()) / 8.0)))
        if self.has_multiple_bt_selection():
            sender = self.sender()
            if sender is self.font_size_spin:
                return {'font-size': int(self.font_size_spin.value())}
            if sender in (self.orientation_vertical_button, self.orientation_horizontal_button):
                return {'orientation': self.editor_orientation()}
            if sender is self.rotation_spin:
                return {'rotation': self.normalized_rotation(self.rotation_spin.value())}
            if sender in (self.color_black_button, self.color_white_button):
                return {
                    'color': '#FFFFFF' if color == 'white' else '#000000',
                    'stroke-color': '#000000' if color == 'white' else '#FFFFFF',
                }
            if sender is self.stroke_weight_spin:
                return {'stroke-weight': stroke_weight}
            if sender is self.text_has_stroke_check:
                return {'stroke-weight': stroke_weight if self.text_has_stroke_check.isChecked() else 0}
            if sender is self.need_inpaint_check:
                return {'need_inpaint': self.need_inpaint_check.isChecked()}
            return None
        return {
            'text': self.bt_text_edit.toPlainText(),
            'font-size': int(self.font_size_spin.value()),
            'orientation': self.editor_orientation(),
            'rotation': self.normalized_rotation(self.rotation_spin.value()),
            'color': '#FFFFFF' if color == 'white' else '#000000',
            'stroke-color': '#000000' if color == 'white' else '#FFFFFF',
            'stroke-weight': stroke_weight,
            'need_inpaint': self.need_inpaint_check.isChecked(),
        }
