"""_bt mutation, undo and box interaction for the CTC viewer."""

from __future__ import annotations

import copy
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

try:
    from .processor import BoxOverlay, normalized_center_from_xyxy, tuple_center, xyxy_from_item
    from .viewer_dialogs import positive_int, show_exception_details
    from .viewer_render_utils import qimage_size
except ImportError:
    from processor import BoxOverlay, normalized_center_from_xyxy, tuple_center, xyxy_from_item
    from viewer_dialogs import positive_int, show_exception_details
    from viewer_render_utils import qimage_size


class ViewerBTUpdateMixin:
    def apply_font_size_from_table(self, row: int, column: int) -> None:
        selected_indices = self.selected_bt_indices_list()
        if not selected_indices:
            self.status_label.setText('請先在左側選擇一條 _bt 文字，再點右側字級。')
            return
        item = self.font_size_table.item(row, 0)
        if item is None:
            return
        try:
            size = int(item.text())
        except ValueError:
            return
        if len(selected_indices) > 1:
            self.apply_bt_updates_to_indices(
                selected_indices,
                lambda _index, _item: {'font-size': size},
                status=f'已把 {len(selected_indices)} 條 _bt 條目字體大小改為 {size}，尚未保存。',
            )
        else:
            self.apply_selected_box_updates({'font-size': size}, status=f'已把當前 _bt 條目字體大小改為 {size}，尚未保存。')

    def nudge_selected_font_size(self, delta: int) -> None:
        selected_indices = self.selected_bt_indices_list()
        if not selected_indices:
            self.status_label.setText('請先選擇一條 _bt 文字，再使用字體大小快捷鍵。')
            return
        sign = '+' if delta > 0 else ''
        changed = self._preview_bt_updates_to_indices(
            selected_indices,
            lambda _index, item: {
                'font-size': max(1, min(999, positive_int(item.get('font-size'), 40) + delta))
            },
            status=f'已將 {len(selected_indices)} 條 _bt 條目字體大小各自 {sign}{delta}，尚未保存。',
        )
        if changed:
            self._sync_typography_controls_from_preview()

    def nudge_selected_rotation(self, delta: float) -> None:
        selected_indices = self.selected_bt_indices_list()
        if not selected_indices:
            self.status_label.setText('請先選擇一條 _bt 文字，再使用旋轉快捷鍵。')
            return
        sign = '+' if delta > 0 else ''
        changed = self._preview_bt_updates_to_indices(
            selected_indices,
            lambda _index, item: {
                'rotation': self.normalized_rotation(self.normalized_rotation(item.get('rotation')) + delta)
            },
            status=f'已將 {len(selected_indices)} 條 _bt 條目各自旋轉 {sign}{delta:g} 度，尚未保存。',
        )
        if changed:
            self._sync_typography_controls_from_preview()

    def nudge_selected_box_position(self, dx: int, dy: int) -> None:
        selected_indices = self.selected_bt_indices_list()
        if not selected_indices:
            self.status_label.setText('請先選擇一條 _bt 文字，再使用方向鍵移動。')
            return
        move_text = f'{dx:+d},{dy:+d}'
        if len(selected_indices) > 1:
            self.apply_bt_updates_to_indices(
                selected_indices,
                lambda _index, item: self.move_bt_item_updates(item, dx, dy),
                status=f'已用方向鍵移動 {len(selected_indices)} 條 _bt 條目 {move_text}，尚未保存。',
            )
            return
        item = self.selected_bt_item()
        if item is None:
            return
        xyxy = self.bt_xyxy_from_item(item)
        if xyxy is None:
            return
        x1, y1, x2, y2 = xyxy
        new_xyxy = self.clamp_xyxy((x1 + dx, y1 + dy, x2 + dx, y2 + dy))
        if new_xyxy == xyxy:
            return
        self.apply_selected_box_updates(
            {'xyxy_pixel': list(new_xyxy), 'match_status': 'manual'},
            status=f'已用方向鍵移動當前 _bt 條目 {move_text}，尚未保存。',
        )

    def move_bt_item_updates(self, item: dict[str, Any], dx: int, dy: int) -> dict[str, object]:
        xyxy = self.bt_xyxy_from_item(item)
        if xyxy is None:
            return {}
        x1, y1, x2, y2 = xyxy
        new_xyxy = self.clamp_xyxy((x1 + dx, y1 + dy, x2 + dx, y2 + dy))
        if new_xyxy == xyxy:
            return {}
        return {'xyxy_pixel': list(new_xyxy), 'match_status': 'manual'}

    def copy_selected_box(self) -> None:
        if self.bt_data is None:
            return
        page_name = self.current_page_name()
        selected_indices = self.selected_bt_indices_list()
        if not selected_indices or page_name is None:
            self.status_label.setText('請先選擇一條 _bt 文字，再複製。')
            return
        items = self.bt_items_for_page(page_name)
        before_items = copy.deepcopy(items)
        before_selected = self.selected_bt_index
        before_selected_indices = set(self.selected_bt_indices)
        next_index = self.next_bt_index_for_page(items)
        new_items: list[dict[str, Any]] = []
        for source_index in selected_indices:
            if source_index < 0 or source_index >= len(items) or not isinstance(items[source_index], dict):
                continue
            new_item = copy.deepcopy(items[source_index])
            new_item['index'] = next_index
            next_index += 1
            new_item['match_status'] = 'manual'
            xyxy = self.bt_xyxy_from_item(new_item)
            if xyxy is not None:
                x1, y1, x2, y2 = xyxy
                self.set_bt_xyxy(new_item, self.clamp_xyxy((x1 + 16, y1 + 16, x2 + 16, y2 + 16)))
            else:
                try:
                    new_item['x'] = min(1.0, max(0.0, float(new_item.get('x')) + 0.01))
                    new_item['y'] = min(1.0, max(0.0, float(new_item.get('y')) + 0.01))
                except (TypeError, ValueError):
                    pass
            new_items.append(new_item)
        if not new_items:
            return
        insert_at = max(selected_indices) + 1
        for offset, new_item in enumerate(new_items):
            items.insert(insert_at + offset, new_item)
        new_selection = set(range(insert_at, insert_at + len(new_items)))
        self.selected_bt_indices = new_selection
        self.selected_bt_index = insert_at
        self.push_bt_items_undo_snapshot('複製 _bt 條目', before_items, before_selected, before_selected_indices)
        self.mark_bt_dirty()
        self.populate_box_editor_for_selection()
        self.update_bt_item_list()
        self.render_bt_page(refit=False)
        if len(new_items) > 1:
            self.status_label.setText(f'已複製 {len(new_items)} 條 _bt 條目，尚未保存。')
        else:
            self.status_label.setText('已複製 _bt 條目，尚未保存。')

    def add_empty_bt_box(self) -> None:
        if self.bt_data is None:
            self.status_label.setText('請先打開 _bt.json，再新增空文案。')
            return
        page_name = self.current_page_name()
        cursor = self._bt_cursor_image_pos
        if page_name is None or cursor is None:
            self.status_label.setText('請先將鼠標移到左側圖片上，再新增空文案。')
            return
        items = self.bt_items_for_page(page_name)
        image_size = qimage_size(self.page.image_path) if self.page is not None else None
        if image_size is None:
            self.status_label.setText('目前頁面沒有可用的圖片尺寸。')
            return

        before_items = copy.deepcopy(items)
        before_selected = self.selected_bt_index
        before_selected_indices = set(self.selected_bt_indices)
        center_x = int(round(cursor[0]))
        center_y = int(round(cursor[1]))
        half_size = 15
        xyxy = self.clamp_xyxy(
            (
                center_x - half_size,
                center_y - half_size,
                center_x + half_size,
                center_y + half_size,
            )
        )
        clipboard_text = QApplication.clipboard().text()
        initial_text = clipboard_text if clipboard_text.strip() else ''
        new_item: dict[str, Any] = {
            'index': self.next_bt_index_for_page(items),
            'text': initial_text,
            'font-size': 30,
            'orientation': 'vertical',
            'rotation': 0,
            'color': '#000000',
            'stroke-color': '#FFFFFF',
            'stroke-weight': 4,
            'need_inpaint': False,
            'match_status': 'manual',
            'xyxy_pixel': list(xyxy),
        }
        self.set_bt_xyxy(new_item, xyxy)
        items.append(new_item)
        new_index = len(items) - 1
        self.selected_bt_index = new_index
        self.selected_bt_indices = {new_index}
        self.push_bt_items_undo_snapshot(
            '新增空文案',
            before_items,
            before_selected,
            before_selected_indices,
        )
        self.mark_bt_dirty()
        self.populate_box_editor_for_selection()
        self.update_bt_item_list()
        self.render_bt_page(refit=False)
        if initial_text:
            self.status_label.setText('已在鼠標位置新增文案並帶入系統剪貼簿文字，尚未保存。')
        else:
            self.status_label.setText('已在鼠標位置新增空文案，尚未保存。')

    def delete_selected_box(self) -> None:
        if self.bt_data is None:
            return
        selected_indices = self.selected_bt_indices_list()
        if not selected_indices:
            self.status_label.setText('請先選擇一條 _bt 文字，再刪除。')
            return
        page_name = self.current_page_name()
        items = self.bt_items_for_page(page_name)
        if page_name is None:
            return
        before_items = copy.deepcopy(items)
        before_selected = self.selected_bt_index
        before_selected_indices = set(self.selected_bt_indices)
        for index in sorted(selected_indices, reverse=True):
            if 0 <= index < len(items):
                del items[index]
        self.selected_bt_index = None
        self.selected_bt_indices.clear()
        self.push_bt_items_undo_snapshot('刪除 _bt 條目', before_items, before_selected, before_selected_indices)
        self.mark_bt_dirty()
        self.set_box_editor_enabled(False)
        self.update_bt_item_list()
        self.render_bt_page(refit=False)
        if len(selected_indices) > 1:
            self.status_label.setText(f'已刪除 {len(selected_indices)} 條 _bt 條目，尚未保存。')
        else:
            self.status_label.setText('已刪除 _bt 條目，尚未保存。')

    def update_action_state(self) -> None:
        can_save = self.bt_data is not None and self.bt_dirty
        self.save_button.setEnabled(can_save)
        if self.save_action is not None:
            self.save_action.setEnabled(can_save)
        if self.undo_action is not None:
            self.undo_action.setEnabled(bool(self.bt_undo_stack))
        has_pages = bool(self.page_names)
        if self.prev_page_action is not None:
            self.prev_page_action.setEnabled(has_pages and self.page_list.currentRow() > 0)
        if self.next_page_action is not None:
            self.next_page_action.setEnabled(has_pages and self.page_list.currentRow() < len(self.page_names) - 1)
        has_selected_bt = bool(self.selected_bt_indices_list())
        if self.increase_font_action is not None:
            self.increase_font_action.setEnabled(has_selected_bt)
        if self.decrease_font_action is not None:
            self.decrease_font_action.setEnabled(has_selected_bt)
        if self.increase_font_10_action is not None:
            self.increase_font_10_action.setEnabled(has_selected_bt)
        if self.decrease_font_10_action is not None:
            self.decrease_font_10_action.setEnabled(has_selected_bt)
        if self.rotate_counterclockwise_action is not None:
            self.rotate_counterclockwise_action.setEnabled(has_selected_bt)
        if self.rotate_clockwise_action is not None:
            self.rotate_clockwise_action.setEnabled(has_selected_bt)
        if self.rotate_counterclockwise_5_action is not None:
            self.rotate_counterclockwise_5_action.setEnabled(has_selected_bt)
        if self.rotate_clockwise_5_action is not None:
            self.rotate_clockwise_5_action.setEnabled(has_selected_bt)
        if self.copy_box_action is not None:
            self.copy_box_action.setEnabled(has_selected_bt)
        if hasattr(self, 'copy_to_clipboard_button'):
            self.copy_to_clipboard_button.setEnabled(has_selected_bt)
        if self.copy_to_memory_action is not None:
            self.copy_to_memory_action.setEnabled(has_selected_bt)
        has_memory_clipboard_item = self.memory_bt_clipboard_item is not None
        can_paste_from_memory = (
            self.bt_data is not None
            and self.current_page_name() is not None
            and self._bt_cursor_image_pos is not None
            and has_memory_clipboard_item
        )
        if self.paste_from_memory_action is not None:
            self.paste_from_memory_action.setEnabled(can_paste_from_memory)
        if hasattr(self, 'delete_clipboard_button'):
            list_item = self.bt_clipboard_list.currentItem()
            list_index = list_item.data(Qt.ItemDataRole.UserRole) if list_item is not None else None
            self.delete_clipboard_button.setEnabled(isinstance(list_index, int))
        if self.measure_angle_action is not None:
            self.measure_angle_action.setEnabled(has_selected_bt)
        can_add_empty_box = (
            self.bt_data is not None
            and self.current_page_name() is not None
            and self._bt_cursor_image_pos is not None
        )
        if self.add_empty_box_action is not None:
            self.add_empty_box_action.setEnabled(can_add_empty_box)
        if hasattr(self, 'add_empty_bt_button'):
            self.add_empty_bt_button.setEnabled(can_add_empty_box)
        if self.delete_box_action is not None:
            self.delete_box_action.setEnabled(has_selected_bt)
        for action in self.move_box_actions:
            action.setEnabled(has_selected_bt)
        suffix = ' *' if self.bt_dirty else ''
        self.setWindowTitle(f'CTD / MEO BT 編輯器{suffix}')

    def mark_measure_dirty(self) -> None:
        self.measure_dirty = False
        self.update_action_state()

    def save_pending_changes(self, *_, auto: bool = False) -> bool:
        self._commit_bt_editor_preview()
        if self.bt_data is None:
            if not auto:
                self.status_label.setText('目前沒有載入 _bt.json。')
            return True
        if not self.bt_dirty:
            if not auto:
                self.status_label.setText('目前沒有需要保存的 _bt 修改。')
            return True
        try:
            self.save_bt_json()
        except Exception as exc:
            show_exception_details(self, '保存失敗', '無法寫入 _bt.json。下方是完整可複製的出錯信息。', exc)
            return False
        self.status_label.setText(f'已保存：{self.bt_path}')
        return True

    def build_box_updates(self, box: BoxOverlay, updates: dict[str, object]) -> dict[str, object]:
        result = dict(updates)
        if 'xyxy_pixel' in result:
            xyxy = tuple(int(v) for v in result['xyxy_pixel'])
            result['xyxy_pixel'] = list(self.clamp_xyxy(xyxy))
            image_size = qimage_size(self.page.image_path) if self.page is not None else None
            center = normalized_center_from_xyxy(tuple(result['xyxy_pixel']), image_size)
            if center is not None:
                result['center_normalized'] = list(center)
        return result

    def values_equal(self, left, right) -> bool:
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            return abs(float(left) - float(right)) < 1e-6
        return left == right

    def updates_change_item(self, item: dict, updates: dict[str, object]) -> bool:
        return any(not self.values_equal(item.get(key), value) for key, value in updates.items())

    def normalized_bt_updates(self, updates: dict[str, object]) -> dict[str, object]:
        normalized_updates = dict(updates)
        if 'xyxy_pixel' in normalized_updates:
            xyxy = tuple(int(v) for v in normalized_updates['xyxy_pixel'])
            normalized_updates['xyxy_pixel'] = list(self.clamp_xyxy(xyxy))
        if 'rotation' in normalized_updates:
            normalized_updates['rotation'] = self.normalized_rotation(normalized_updates['rotation'])
        return normalized_updates

    def apply_bt_updates_to_indices(
        self,
        indices: list[int] | set[int],
        update_builder,
        *,
        status: str = '已修改，尚未保存。',
        refresh_editor: bool = True,
    ) -> bool:
        self._commit_bt_editor_preview()
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

        before_items = {
            index: copy.deepcopy(items[index])
            for index, _updates in changes
            if 0 <= index < len(items) and isinstance(items[index], dict)
        }
        before_selected = self.selected_bt_index
        before_selected_indices = set(self.selected_bt_indices)
        self.push_bt_changes_undo_snapshot(status, before_items, before_selected, before_selected_indices)
        for index, normalized_updates in changes:
            item = items[index]
            if 'xyxy_pixel' in normalized_updates:
                self.set_bt_xyxy(item, tuple(int(v) for v in normalized_updates.pop('xyxy_pixel')))
            item.update(normalized_updates)
        self.mark_bt_dirty()
        if refresh_editor:
            self.populate_box_editor_for_selection()
        self.update_bt_item_list()
        self.render_bt_page(refit=False)
        self.status_label.setText(status)
        return True

    def sync_box_from_measure_item(self, box: BoxOverlay, item: dict) -> None:
        xyxy = xyxy_from_item(item)
        if xyxy is not None:
            box.xyxy_pixel = xyxy
        image_size = qimage_size(self.page.image_path) if self.page is not None else None
        box.center_normalized = tuple_center(item.get('center_normalized')) or normalized_center_from_xyxy(box.xyxy_pixel, image_size)
        box.font_size = float(item['font_size']) if item.get('font_size') is not None else None
        box.text_color = item.get('text_color')
        box.text_has_stroke = (
            bool(item.get('text_has_stroke'))
            if item.get('text_has_stroke') is not None
            else None
        )
        box.need_inpaint = (
            bool(item.get('need_inpaint'))
            if item.get('need_inpaint') is not None
            else None
        )
        box.raw_measure = dict(item)

    def apply_selected_box_updates(
        self,
        updates: dict[str, object],
        *,
        status: str = '已修改，尚未保存。',
        refresh_editor: bool = True,
    ) -> bool:
        if self.selected_bt_item() is None or self.selected_bt_index is None:
            return False
        return self.apply_bt_updates_to_indices(
            [self.selected_bt_index],
            lambda _index, _item: dict(updates),
            status=status,
            refresh_editor=refresh_editor,
        )

    def undo_last_edit(self) -> None:
        if self.bt_data is None or not self.bt_undo_stack:
            self.status_label.setText('沒有可撤銷的 _bt 修改。')
            self.update_action_state()
            return
        entry = self.bt_undo_stack.pop()
        page_name = str(entry.get('page_name') or '')
        if self.page is None or page_name != self.page.page_name:
            self.status_label.setText('撤銷只支持當前頁；已忽略其它頁面的撤銷記錄。')
            self.update_action_state()
            return
        if isinstance(entry.get('changes'), list):
            items = self.bt_items_for_page(page_name)
            for change in entry['changes']:
                if not isinstance(change, dict):
                    continue
                item_index = change.get('index')
                before_item = change.get('item')
                if (
                    isinstance(item_index, int)
                    and 0 <= item_index < len(items)
                    and isinstance(before_item, dict)
                ):
                    items[item_index] = copy.deepcopy(before_item)
            selected_index = entry.get('selected_index')
            selected_indices = entry.get('selected_indices')
            restored_indices = {
                index for index in selected_indices
                if isinstance(index, int) and 0 <= index < len(items)
            } if isinstance(selected_indices, list) else set()
            if isinstance(selected_index, int) and 0 <= selected_index < len(items):
                restored_indices.add(selected_index)
                self.selected_bt_index = selected_index
            else:
                self.selected_bt_index = self.active_bt_index_from_selection(restored_indices)
            self.selected_bt_indices = restored_indices
            self.mark_bt_dirty()
            if self.selected_bt_items():
                self.populate_box_editor_for_selection()
            else:
                self.set_box_editor_enabled(False)
            self.update_bt_item_list()
            self.render_bt_page(refit=False)
            self.status_label.setText('已撤銷上一個 _bt 修改，尚未保存。')
            return
        if isinstance(entry.get('items'), list):
            items = self.bt_items_for_page(page_name)
            items[:] = copy.deepcopy(entry.get('items') or [])
            selected_index = entry.get('selected_index')
            selected_indices = entry.get('selected_indices')
            restored_indices = {
                index for index in selected_indices
                if isinstance(index, int) and 0 <= index < len(items)
            } if isinstance(selected_indices, list) else set()
            if isinstance(selected_index, int) and 0 <= selected_index < len(items):
                restored_indices.add(selected_index)
                self.selected_bt_index = selected_index
            else:
                self.selected_bt_index = self.active_bt_index_from_selection(restored_indices)
            self.selected_bt_indices = restored_indices
            self.mark_bt_dirty()
            if self.selected_bt_items():
                self.populate_box_editor_for_selection()
            else:
                self.set_box_editor_enabled(False)
            self.update_bt_item_list()
            self.render_bt_page(refit=False)
            self.status_label.setText('已撤銷上一個 _bt 修改，尚未保存。')
            return
        item = copy.deepcopy(entry.get('item') or {})
        item_index = entry.get('item_index')
        items = self.bt_items_for_page(page_name)
        if isinstance(item_index, int) and 0 <= item_index < len(items):
            items[item_index] = item
        else:
            insert_index = item_index if isinstance(item_index, int) else len(items)
            items.insert(max(0, min(insert_index, len(items))), item)
        self.selected_bt_index = item_index if isinstance(item_index, int) else None
        self.selected_bt_indices = {self.selected_bt_index} if self.selected_bt_index is not None else set()
        self.mark_bt_dirty()
        if self.selected_bt_item() is not None:
            self.populate_box_editor_from_bt(self.selected_bt_item())
        self.update_bt_item_list()
        self.render_bt_page(refit=False)
        self.status_label.setText('已撤銷上一個 _bt 修改，尚未保存。')

    def refresh_current_page_from_measure(
        self,
        *,
        selected_source_index: int | None = None,
        status: str | None = None,
        refit: bool = False,
    ) -> None:
        if self.processor is None or self.page is None:
            self.update_action_state()
            return
        page_name = self.page.page_name
        if selected_source_index is None:
            box = self.selected_box()
            selected_source_index = box.source_block_index if box is not None else None
        image_size = qimage_size(self.processor.image_dir / page_name)
        self.page = self.processor.load_page(page_name, image_size=image_size)
        self.selected_box_index = None
        if selected_source_index is not None:
            for index, box in enumerate(self.page.boxes):
                if box.source_block_index == selected_source_index:
                    self.selected_box_index = index
                    break
        if self.selected_box_index is not None:
            pass
        else:
            self.selected_box_index = None
        self.update_font_size_list()
        self.render_current_page(refit=refit)
        if status:
            self.status_label.setText(status)
        self.update_action_state()

    def hit_test_box(self, x: float, y: float) -> tuple[int | None, str | None]:
        if self.page is None:
            return None, None
        handles = (
            ('tl', -1, -1), ('t', 0, -1), ('tr', 1, -1),
            ('l', -1, 0), ('r', 1, 0),
            ('bl', -1, 1), ('b', 0, 1), ('br', 1, 1),
        )
        tolerance = 8.0
        matches = []
        for index, box in enumerate(self.page.boxes):
            x1, y1, x2, y2 = box.xyxy_pixel
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

    def handle_image_mouse_press(self, x: float, y: float) -> None:
        index, _mode = self.hit_test_box(x, y)
        self.select_box(index)
        self._box_drag_mode = None
        self._box_drag_start = None
        self._box_drag_original = None
        self._box_drag_temporary = False
        if index is not None:
            box = self.page.boxes[index] if self.page is not None else None
            if box is not None and self.selected_bt_item() is not None:
                self.apply_measure_box_to_selected_bt(box)
            else:
                self.status_label.setText('已選中 CTD measure 框；請先在左側選擇 _bt 條目後再點右側套用。')

    def handle_image_mouse_drag(self, x: float, y: float) -> None:
        return

    def handle_image_mouse_release(self, x: float, y: float) -> None:
        self._box_drag_mode = None
        self._box_drag_start = None
        self._box_drag_original = None
        self._box_drag_temporary = False

    def clamp_xyxy(self, xyxy: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        if self.page is None:
            return xyxy
        image_size = self._bt_cached_image_size or qimage_size(self.page.image_path)
        width, height = image_size or (10_000, 10_000)
        x1, y1, x2, y2 = xyxy
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(1, min(x2, width))
        y2 = max(1, min(y2, height))
        if x2 - x1 < 4:
            if self._box_drag_mode and 'l' in self._box_drag_mode:
                x1 = max(0, x2 - 4)
            else:
                x2 = min(width, x1 + 4)
        if y2 - y1 < 4:
            if self._box_drag_mode and 't' in self._box_drag_mode:
                y1 = max(0, y2 - 4)
            else:
                y2 = min(height, y1 + 4)
        return x1, y1, x2, y2
