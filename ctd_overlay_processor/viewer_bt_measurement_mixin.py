"""Measurement and live typography preview for the CTC viewer."""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QApplication, QDialog

try:
    from .bt_measurement_dialog import BtMeasurementDialog
    from .viewer_dialogs import positive_int
except ImportError:
    from bt_measurement_dialog import BtMeasurementDialog
    from viewer_dialogs import positive_int


class ViewerBTMeasurementMixin:
    def open_bt_measurement_dialog(self) -> None:
        selected_indices = self.selected_bt_indices_list()
        page_name = self.current_page_name()
        if not selected_indices or page_name is None:
            self.status_label.setText('請先選擇一條 _bt 文字，再打開測量角度。')
            return
        image = self.load_bt_source_image(page_name)
        if image is None or image.isNull():
            self.status_label.setText('無法讀取目前頁面的圖片。')
            return
        view_crop_rect, view_display_scale = self.measurement_view_crop_rect(image)
        items = self.bt_items_for_page(page_name)
        targets: list[tuple[int, dict[str, Any], tuple[float, float], tuple[int, int, int, int] | None]] = []
        target_rects: list[QRectF] = []
        centers: list[QPointF] = []
        for index in selected_indices:
            if index < 0 or index >= len(items) or not isinstance(items[index], dict):
                continue
            item = items[index]
            center = self.bt_center_pixel_from_item(item)
            if center is None:
                continue
            xyxy = self.bt_xyxy_from_item(item)
            centers.append(QPointF(center[0], center[1]))
            if xyxy is not None:
                x1, y1, x2, y2 = xyxy
                target_rects.append(QRectF(x1, y1, max(1, x2 - x1), max(1, y2 - y1)))
            else:
                target_rects.append(QRectF(center[0] - 25, center[1] - 25, 50, 50))
            targets.append((index, item, center, xyxy))
        if not targets:
            self.status_label.setText('無法建立測量角度裁切圖。')
            return
        crop_rect, display_scale = self.measurement_multi_crop_rect(
            image,
            target_rects,
            centers,
            view_crop_rect,
            view_display_scale,
        )
        crop = image.copy(crop_rect.toRect())
        if crop.isNull():
            self.status_label.setText('無法建立測量角度裁切圖。')
            return
        mask_crop = None
        if self.processor is not None:
            mask_path = self.processor.mask_dir / f'{Path(page_name).stem}.png'
            mask_image = QImage(str(mask_path))
            if not mask_image.isNull():
                mask_crop = mask_image.copy(crop_rect.toRect())
        entries: list[dict[str, object]] = []
        for index, item, center, _xyxy in targets:
            local_center = QPointF(center[0] - crop_rect.left(), center[1] - crop_rect.top())
            font_size = positive_int(item.get('font-size'), 40)
            entries.append({
                'item_index': index,
                'item': item,
                'crop_origin': QPointF(crop_rect.left(), crop_rect.top()),
                'center': local_center,
                'font_size': font_size,
                'rotation': self.normalized_rotation(item.get('rotation')),
                'display_scale': display_scale,
                'orientation': str(item.get('orientation') or 'vertical'),
                'color': self.qcolor_from_bt_value(item.get('color'), QColor(0, 0, 0)),
                'stroke_color': self.qcolor_from_bt_value(item.get('stroke-color'), QColor(255, 255, 255)),
                'stroke_weight': max(0.0, float(item.get('stroke-weight') or 0)),
                'font_family': str(item.get('font') or '').strip() or 'Helvetica Neue',
            })
        if not entries:
            self.status_label.setText('無法建立測量角度裁切圖。')
            return
        dialog = BtMeasurementDialog(
            crop,
            entries=entries,
            mask=mask_crop,
            display_scale=display_scale,
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        single_char_measurement = dialog.canvas.interaction_mode == 'detect'
        results = dialog.result_updates()
        if not results:
            return
        if single_char_measurement:
            apply_status = '已套用單字測量的字體大小與角度，尚未保存。'
        else:
            apply_status = (
                f'已套用 {len(results)} 條測量角度，尚未保存。'
                if len(results) > 1
                else '已套用測量角度，尚未保存。'
            )
        changed = self.apply_bt_updates_to_indices(
            list(results.keys()),
            lambda index, item: self.measurement_updates_for_item(
                item,
                results[index],
                image.width(),
                image.height(),
            ) if index in results else {},
            status=apply_status,
        )
        active_result = results.get(self.selected_bt_index if self.selected_bt_index is not None else selected_indices[0])
        if changed and active_result is not None:
            center_pixel = active_result.get('center_pixel')
            if isinstance(center_pixel, list) and len(center_pixel) == 2:
                self.center_views_on(float(center_pixel[0]), float(center_pixel[1]))

    def qcolor_from_bt_value(self, value: object, fallback: QColor) -> QColor:
        text = str(value or '').strip()
        if not text:
            return QColor(fallback)
        if text.lower() == 'black':
            return QColor(0, 0, 0)
        if text.lower() == 'white':
            return QColor(255, 255, 255)
        color = QColor(text if text.startswith('#') else f'#{text}')
        return color if color.isValid() else QColor(fallback)

    def measurement_multi_crop_rect(
        self,
        image: QImage,
        target_rects: list[QRectF],
        centers: list[QPointF],
        view_crop_rect: QRectF | None,
        view_display_scale: float | None,
    ) -> tuple[QRectF, float | None]:
        image_rect = QRectF(0, 0, image.width(), image.height())
        if view_crop_rect is not None and centers and all(view_crop_rect.contains(center) for center in centers):
            return view_crop_rect.intersected(image_rect), view_display_scale
        if not target_rects:
            return image_rect, None
        union = QRectF(target_rects[0])
        for rect in target_rects[1:]:
            union = union.united(rect)
        pad = max(96, int(round(max(union.width(), union.height()) * 0.45)))
        desired = union.adjusted(-pad, -pad, pad, pad)
        crop_x1 = max(0, int(math.floor(desired.left())))
        crop_y1 = max(0, int(math.floor(desired.top())))
        crop_x2 = min(image.width(), int(math.ceil(desired.right())))
        crop_y2 = min(image.height(), int(math.ceil(desired.bottom())))
        if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
            return image_rect, None
        return QRectF(crop_x1, crop_y1, crop_x2 - crop_x1, crop_y2 - crop_y1), None

    def measurement_view_crop_rect(self, image: QImage) -> tuple[QRectF | None, float | None]:
        if self.bt_view.sceneRect().isEmpty():
            return None, None
        viewport_rect = self.bt_view.viewport().rect()
        if viewport_rect.isEmpty():
            return None, None
        top_left = self.bt_view.mapToScene(viewport_rect.topLeft())
        bottom_right = self.bt_view.mapToScene(viewport_rect.bottomRight())
        visible = QRectF(top_left, bottom_right).normalized()
        image_rect = QRectF(0, 0, image.width(), image.height())
        crop = visible.intersected(image_rect)
        if crop.isEmpty():
            return None, None
        crop_x1 = max(0, int(math.floor(crop.left())))
        crop_y1 = max(0, int(math.floor(crop.top())))
        crop_x2 = min(image.width(), int(math.ceil(crop.right())))
        crop_y2 = min(image.height(), int(math.ceil(crop.bottom())))
        if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
            return None, None
        transform = self.bt_view.transform()
        display_scale = max(0.01, min(abs(transform.m11()), abs(transform.m22())))
        return QRectF(crop_x1, crop_y1, crop_x2 - crop_x1, crop_y2 - crop_y1), display_scale

    def measurement_crop_rect(
        self,
        image: QImage,
        center: tuple[float, float],
        xyxy: tuple[int, int, int, int] | None,
    ) -> QRectF:
        if xyxy is not None:
            x1, y1, x2, y2 = xyxy
            width = max(1, x2 - x1)
            height = max(1, y2 - y1)
            pad = max(96, int(round(max(width, height) * 1.6)))
            crop_x1 = x1 - pad
            crop_y1 = y1 - pad
            crop_x2 = x2 + pad
            crop_y2 = y2 + pad
        else:
            crop_w = min(image.width(), 640)
            crop_h = min(image.height(), 640)
            crop_x1 = int(round(center[0] - crop_w / 2.0))
            crop_y1 = int(round(center[1] - crop_h / 2.0))
            crop_x2 = crop_x1 + crop_w
            crop_y2 = crop_y1 + crop_h
        crop_x1 = max(0, crop_x1)
        crop_y1 = max(0, crop_y1)
        crop_x2 = min(image.width(), max(crop_x1 + 1, crop_x2))
        crop_y2 = min(image.height(), max(crop_y1 + 1, crop_y2))
        return QRectF(crop_x1, crop_y1, crop_x2 - crop_x1, crop_y2 - crop_y1)

    def measurement_updates_for_item(
        self,
        item: dict[str, Any],
        result: dict[str, object],
        image_width: int,
        image_height: int,
    ) -> dict[str, object]:
        center_values = result.get('center_pixel')
        if not isinstance(center_values, list) or len(center_values) != 2:
            return {
                'font-size': result['font-size'],
                'rotation': result['rotation'],
            }
        cx = float(center_values[0])
        cy = float(center_values[1])
        updates: dict[str, object] = {
            'font-size': result['font-size'],
            'rotation': result['rotation'],
        }
        xyxy = self.bt_xyxy_from_item(item)
        if xyxy is not None:
            x1, y1, x2, y2 = xyxy
            width = max(1, x2 - x1)
            height = max(1, y2 - y1)
            half_w = width / 2.0
            half_h = height / 2.0
            cx = max(half_w, min(cx, max(half_w, image_width - half_w)))
            cy = max(half_h, min(cy, max(half_h, image_height - half_h)))
            updates['xyxy_pixel'] = [
                int(round(cx - width / 2.0)),
                int(round(cy - height / 2.0)),
                int(round(cx + width / 2.0)),
                int(round(cy + height / 2.0)),
            ]
        else:
            updates['x'] = max(0.0, min(1.0, cx / max(1, image_width)))
            updates['y'] = max(0.0, min(1.0, cy / max(1, image_height)))
        return updates

    def measure_box_updates_for_bt(self, box: BoxOverlay) -> dict[str, object]:
        item = self.selected_bt_item() or {}
        updates: dict[str, object] = {
            'xyxy_pixel': list(box.xyxy_pixel),
            'orientation': box.orientation or item.get('orientation') or 'vertical',
            'match_status': 'manual',
            'match_source_block_index': box.source_block_index,
        }
        if box.measure_item_index is not None:
            updates['match_measure_item_index'] = box.measure_item_index
        if box.center_normalized is not None:
            updates['x'] = box.center_normalized[0]
            updates['y'] = box.center_normalized[1]

        if box.font_size is not None:
            font_size = max(1, int(round(float(box.font_size))))
            updates['font-size'] = font_size
        else:
            font_size = positive_int(
                item.get('font-size'),
                positive_int(self.font_size_spin.value(), 40),
            )

        color = str(box.text_color or self.bt_text_color(item)).lower()
        if color not in {'black', 'white'}:
            color = 'black'
        updates['color'] = '#FFFFFF' if color == 'white' else '#000000'
        updates['stroke-color'] = '#000000' if color == 'white' else '#FFFFFF'
        needs_stroke = box.text_has_stroke is True or box.need_inpaint is True
        updates['stroke-weight'] = int(np.ceil(font_size / 8.0)) if needs_stroke else 0
        updates['need_inpaint'] = box.need_inpaint is True
        return updates

    def apply_measure_box_to_selected_bt(self, box: BoxOverlay) -> bool:
        if self.has_multiple_bt_selection():
            self.status_label.setText('右側 measure 框套用只支持單個 _bt 條目；請先取消多選。')
            return False
        item = self.selected_bt_item()
        if item is None:
            self.status_label.setText('請先在左側選擇一條 _bt，再點右側 measure 框套用。')
            return False
        entry_index = item.get('index', self.selected_bt_index)
        source_index = box.source_block_index
        status = f'已將右側 measure 區塊 {source_index} 套用到左側 _bt index={entry_index}，尚未保存。'
        changed = self.apply_selected_box_updates(
            self.measure_box_updates_for_bt(box),
            status=status,
        )
        if not changed:
            self.status_label.setText(f'右側 measure 區塊 {source_index} 與左側 _bt 目前內容相同。')
        return changed

    def apply_editor_changes_to_selected_box(self) -> None:
        updates = self.selected_box_updates_from_editor()
        if updates is None:
            return
        refresh_editor = QApplication.focusWidget() is not self.bt_text_edit
        if self.has_multiple_bt_selection():
            count = len(self.selected_bt_indices_list())
            self.apply_bt_updates_to_indices(
                self.selected_bt_indices_list(),
                lambda _index, _item: dict(updates),
                status=f'已批量修改 {count} 條 _bt 條目，尚未保存。',
                refresh_editor=refresh_editor,
            )
        else:
            self.apply_selected_box_updates(
                updates,
                status='已修改當前 _bt 條目，尚未保存。',
                refresh_editor=refresh_editor,
            )
        if QApplication.focusWidget() is self.bt_text_edit:
            self.sync_bt_text_popover_from_item()

    def _begin_bt_editor_preview(self) -> None:
        if self._bt_editor_preview_snapshot is not None:
            return
        page_name = self.current_page_name()
        if page_name is None:
            return
        items = self.bt_items_for_page(page_name)
        indices = self.selected_bt_indices_list()
        self._bt_editor_preview_snapshot = {
            'page_name': page_name,
            'items': {
                index: copy.deepcopy(items[index])
                for index in indices
                if 0 <= index < len(items) and isinstance(items[index], dict)
            },
            'selected_index': self.selected_bt_index,
            'selected_indices': set(self.selected_bt_indices),
        }

    def _schedule_bt_editor_preview_render(self) -> None:
        if not self._bt_editor_preview_render_timer.isActive():
            self._bt_editor_preview_render_timer.start()

    def _render_bt_editor_preview(self) -> None:
        if self._bt_editor_preview_snapshot is None:
            return
        self.render_bt_page(refit=False)

    def _commit_bt_editor_preview(self, *_args: object) -> None:
        render_pending = self._bt_editor_preview_render_timer.isActive()
        self._bt_editor_preview_render_timer.stop()
        self._bt_editor_preview_commit_timer.stop()
        snapshot = self._bt_editor_preview_snapshot
        self._bt_editor_preview_snapshot = None
        if snapshot is None:
            return

        page_name = snapshot.get('page_name')
        if not isinstance(page_name, str) or page_name != self.current_page_name():
            self._bt_editor_preview_status = None
            return
        items = self.bt_items_for_page(page_name)
        before_items = snapshot.get('items')
        if not isinstance(before_items, dict):
            self._bt_editor_preview_status = None
            return
        changed_before_items = {
            index: item
            for index, item in before_items.items()
            if isinstance(index, int)
            and 0 <= index < len(items)
            and isinstance(item, dict)
            and items[index] != item
        }
        if not changed_before_items:
            self._bt_editor_preview_status = None
            return

        selected_index = snapshot.get('selected_index')
        selected_indices = snapshot.get('selected_indices')
        self.push_bt_changes_undo_snapshot(
            self._bt_editor_preview_status or '修改 _bt 字體/旋轉',
            changed_before_items,
            selected_index if isinstance(selected_index, int) else None,
            selected_indices if isinstance(selected_indices, set) else None,
        )
        self.mark_bt_dirty()
        self.update_bt_item_list()
        self.update_action_state()
        self.status_label.setText(self._bt_editor_preview_status or '已修改 _bt，尚未保存。')
        self._bt_editor_preview_status = None
        if render_pending:
            self.render_bt_page(refit=False)

    def _preview_bt_updates_to_indices(
        self,
        indices: list[int] | set[int],
        update_builder,
        *,
        status: str,
    ) -> bool:
        page_name = self.current_page_name()
        if page_name is None:
            return False
        items = self.bt_items_for_page(page_name)
        target_indices = [
            index for index in sorted(set(indices))
            if 0 <= index < len(items) and isinstance(items[index], dict)
        ]
        if not target_indices:
            return False

        changes: list[tuple[int, dict[str, object]]] = []
        for index in target_indices:
            updates = update_builder(index, items[index])
            if not updates:
                continue
            normalized_updates = self.normalized_bt_updates(updates)
            if self.updates_change_item(items[index], normalized_updates):
                changes.append((index, normalized_updates))
        if not changes:
            return False

        self._begin_bt_editor_preview()
        for index, normalized_updates in changes:
            items[index].update(normalized_updates)
        self._bt_editor_preview_status = status
        self._schedule_bt_editor_preview_render()
        self._bt_editor_preview_commit_timer.start()
        return True

    def _sync_typography_controls_from_preview(self) -> None:
        item = self.selected_bt_item()
        if item is None:
            return
        self._updating_editor = True
        self.font_size_spin.setValue(positive_int(item.get('font-size'), 40))
        self.rotation_spin.setValue(self.normalized_rotation(item.get('rotation')))
        self._updating_editor = False

    def handle_typography_control_changed(self, value: object) -> None:
        if self._updating_editor:
            return
        selected_indices = self.selected_bt_indices_list()
        if not selected_indices:
            return
        sender = self.sender()
        if sender is self.font_size_spin:
            size = max(1, min(999, int(value)))
            self._preview_bt_updates_to_indices(
                selected_indices,
                lambda _index, _item: {'font-size': size},
                status=f'已把 {len(selected_indices)} 條 _bt 字體大小預覽為 {size}px，尚未保存。',
            )
        elif sender is self.rotation_spin:
            rotation = self.normalized_rotation(value)
            self._preview_bt_updates_to_indices(
                selected_indices,
                lambda _index, _item: {'rotation': rotation},
                status=f'已把 {len(selected_indices)} 條 _bt 旋轉預覽為 {rotation:g} 度，尚未保存。',
            )
