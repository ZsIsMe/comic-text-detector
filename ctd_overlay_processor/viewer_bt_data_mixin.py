"""_bt data, selection, clipboard and font-size list operations for the CTC viewer."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import QDialog, QFileDialog, QMessageBox, QTableWidgetItem

try:
    from .processor import normalized_center_from_xyxy
    from .viewer_dialogs import FontSizeRegularizeDialog, even_font_size, positive_int, show_exception_details
    from .viewer_render_utils import qimage_size
except ImportError:
    from processor import normalized_center_from_xyxy
    from viewer_dialogs import FontSizeRegularizeDialog, even_font_size, positive_int, show_exception_details
    from viewer_render_utils import qimage_size


class ViewerBTDataMixin:
    def load_bt_clipboard_items(self) -> list[dict[str, Any]]:
        """Load the app-wide text-box clipboard without trusting stored data."""
        raw_value = self.settings.value('bt_clipboard/items', '[]', str)
        try:
            decoded = json.loads(raw_value) if isinstance(raw_value, str) else raw_value
        except (TypeError, ValueError, json.JSONDecodeError):
            return []
        if not isinstance(decoded, list):
            return []
        return [copy.deepcopy(item) for item in decoded if isinstance(item, dict)][:50]

    def save_bt_clipboard_items(self) -> None:
        self.settings.setValue(
            'bt_clipboard/items',
            json.dumps(self.bt_clipboard_items[:50], ensure_ascii=False, separators=(',', ':')),
        )

    def bt_clipboard_item_label(self, item: dict[str, Any]) -> str:
        text = ' '.join(str(item.get('text') or '').split())
        if len(text) > 32:
            text = f'{text[:32]}...'
        if not text:
            text = '(空文字)'
        font_size = item.get('font-size', item.get('font_size', '?'))
        orientation = '直' if str(item.get('orientation') or 'vertical') == 'vertical' else '橫'
        return f'[{orientation} {font_size}px] {text}'

    def update_bt_clipboard_list(self) -> None:
        if not hasattr(self, 'bt_clipboard_list'):
            return
        self.bt_clipboard_list.blockSignals(True)
        self.bt_clipboard_list.clear()
        if not self.bt_clipboard_items:
            self.bt_clipboard_list.addItem('剪貼簿目前是空的')
            self.bt_clipboard_list.item(0).setFlags(Qt.ItemFlag.NoItemFlags)
        else:
            for index, item in enumerate(self.bt_clipboard_items):
                active_prefix = '● ' if index == self.active_bt_clipboard_index else '○ '
                self.bt_clipboard_list.addItem(f'{active_prefix}{self.bt_clipboard_item_label(item)}')
                list_item = self.bt_clipboard_list.item(index)
                list_item.setData(Qt.ItemDataRole.UserRole, index)
                list_item.setToolTip('單擊即可複製為目前剪貼簿內容。')
            if self.active_bt_clipboard_index is not None and 0 <= self.active_bt_clipboard_index < len(self.bt_clipboard_items):
                self.bt_clipboard_list.setCurrentRow(self.active_bt_clipboard_index)
            else:
                self.bt_clipboard_list.setCurrentRow(-1)
        self.bt_clipboard_list.blockSignals(False)
        self.update_action_state()

    def active_bt_clipboard_item(self) -> dict[str, Any] | None:
        if self.active_bt_clipboard_index is None or not 0 <= self.active_bt_clipboard_index < len(self.bt_clipboard_items):
            return None
        return self.bt_clipboard_items[self.active_bt_clipboard_index]

    def activate_bt_clipboard_item(self, index: int, *, announce: bool = True) -> None:
        if not 0 <= index < len(self.bt_clipboard_items):
            return
        self.active_bt_clipboard_index = index
        self.memory_bt_clipboard_item = copy.deepcopy(self.bt_clipboard_items[index])
        self.update_bt_clipboard_list()
        if announce:
            self.status_label.setText('已暫存複製剪貼簿項目；可用 F2 在游標位置貼上。')

    def copy_bt_clipboard_list_item(self, list_item) -> None:
        index = list_item.data(Qt.ItemDataRole.UserRole) if list_item is not None else None
        if isinstance(index, int):
            self.activate_bt_clipboard_item(index)

    def copy_selected_box_to_clipboard(self) -> None:
        item = self.selected_bt_item()
        if item is None:
            self.status_label.setText('請先選擇一條 _bt 文字，再加入持久剪貼簿列表。')
            return
        self.bt_clipboard_items.insert(0, copy.deepcopy(item))
        self.bt_clipboard_items = self.bt_clipboard_items[:50]
        self.save_bt_clipboard_items()
        self.update_bt_clipboard_list()
        self.status_label.setText('已將目前文字框加入持久剪貼簿列表。')

    def copy_selected_box_to_memory(self) -> None:
        item = self.selected_bt_item()
        if item is None:
            return
        self.memory_bt_clipboard_item = copy.deepcopy(item)
        self.active_bt_clipboard_index = None
        self.update_bt_clipboard_list()
        self.status_label.setText('已暫存複製目前文字框；可用 F2 在游標位置貼上。')

    def paste_box_from_memory(self) -> None:
        if self.bt_data is None:
            return
        page_name = self.current_page_name()
        source_item = self.memory_bt_clipboard_item
        if page_name is None or source_item is None:
            return
        cursor = self._bt_cursor_image_pos
        if cursor is None:
            return
        items = self.bt_items_for_page(page_name)
        before_items = copy.deepcopy(items)
        before_selected = self.selected_bt_index
        before_selected_indices = set(self.selected_bt_indices)
        new_item = copy.deepcopy(source_item)
        new_item['index'] = self.next_bt_index_for_page(items)
        new_item['match_status'] = 'manual'
        xyxy = self.bt_xyxy_from_item(new_item)
        if xyxy is not None:
            x1, y1, x2, y2 = xyxy
            width, height = x2 - x1, y2 - y1
            center_x, center_y = (int(round(cursor[0])), int(round(cursor[1])))
            xyxy = (center_x - width // 2, center_y - height // 2, center_x - width // 2 + width, center_y - height // 2 + height)
            self.set_bt_xyxy(new_item, self.clamp_xyxy(xyxy))
        else:
            center_x, center_y = (int(round(cursor[0])), int(round(cursor[1])))
            self.set_bt_xyxy(new_item, self.clamp_xyxy((center_x - 25, center_y - 25, center_x + 25, center_y + 25)))
        items.append(new_item)
        new_index = len(items) - 1
        self.selected_bt_index = new_index
        self.selected_bt_indices = {new_index}
        self.push_bt_items_undo_snapshot(
            '從文字框剪貼簿建立 _bt 條目',
            before_items,
            before_selected,
            before_selected_indices,
        )
        self.mark_bt_dirty()
        self.populate_box_editor_for_selection()
        self.update_bt_item_list()
        self.render_bt_page(refit=False)
        self.status_label.setText('已在游標位置貼上暫存文字框，尚未保存。')

    def delete_selected_clipboard_item(self) -> None:
        list_item = self.bt_clipboard_list.currentItem() if hasattr(self, 'bt_clipboard_list') else None
        index = list_item.data(Qt.ItemDataRole.UserRole) if list_item is not None else None
        if not isinstance(index, int) or not 0 <= index < len(self.bt_clipboard_items):
            self.status_label.setText('請先選擇一個剪貼簿文字框。')
            return
        del self.bt_clipboard_items[index]
        if self.active_bt_clipboard_index == index:
            self.active_bt_clipboard_index = None
        elif self.active_bt_clipboard_index is not None and self.active_bt_clipboard_index > index:
            self.active_bt_clipboard_index -= 1
        self.save_bt_clipboard_items()
        self.update_bt_clipboard_list()
        self.status_label.setText('已從全域剪貼簿刪除文字框。')

    def current_page_name(self) -> str | None:
        if self.page is not None:
            return self.page.page_name
        if 0 <= self.current_page_row < len(self.page_names):
            return self.page_names[self.current_page_row]
        return None

    def bt_items_for_page(self, page_name: str | None = None) -> list[dict[str, Any]]:
        if self.bt_data is None:
            return []
        page_name = page_name or self.current_page_name()
        if not page_name:
            return []
        items = self.bt_data.get('transMap', {}).get(page_name, [])
        return items if isinstance(items, list) else []

    def _bt_total_count(self) -> int:
        if self.bt_data is None:
            return 0
        trans_map = self.bt_data.get('transMap', {})
        if not isinstance(trans_map, dict):
            return 0
        return sum(len(items) for items in trans_map.values() if isinstance(items, list))

    def _bt_remaining_count_from_current_page(self) -> int:
        if self.bt_data is None or not self.page_names:
            return 0
        row = self.current_page_row
        if row < 0:
            page_name = self.current_page_name()
            row = self.page_names.index(page_name) if page_name in self.page_names else 0
        return sum(len(self.bt_items_for_page(page_name)) for page_name in self.page_names[row:])

    def update_bt_stats_label(self) -> None:
        if not hasattr(self, 'bt_stats_label'):
            return
        if self.bt_data is None:
            self.bt_stats_label.setText('_bt 統計：未載入')
            return
        current_count = len(self.bt_items_for_page())
        remaining_count = self._bt_remaining_count_from_current_page()
        self.bt_stats_label.setText(
            f'_bt 總條數：{self._bt_total_count()}，'
            f'當前頁：{current_count}，'
            f'當前頁起剩餘：{remaining_count}'
        )

    def selected_bt_item(self) -> dict[str, Any] | None:
        items = self.bt_items_for_page()
        if self.selected_bt_index is None or self.selected_bt_index < 0 or self.selected_bt_index >= len(items):
            return None
        item = items[self.selected_bt_index]
        return item if isinstance(item, dict) else None

    def selected_bt_indices_list(self) -> list[int]:
        items = self.bt_items_for_page()
        valid = [
            index for index in sorted(self.selected_bt_indices)
            if 0 <= index < len(items) and isinstance(items[index], dict)
        ]
        if self.selected_bt_index is not None and self.selected_bt_index not in valid:
            if 0 <= self.selected_bt_index < len(items) and isinstance(items[self.selected_bt_index], dict):
                valid.append(self.selected_bt_index)
        return sorted(valid)

    def selected_bt_items(self) -> list[tuple[int, dict[str, Any]]]:
        items = self.bt_items_for_page()
        return [
            (index, items[index])
            for index in self.selected_bt_indices_list()
            if isinstance(items[index], dict)
        ]

    def has_multiple_bt_selection(self) -> bool:
        return len(self.selected_bt_indices_list()) > 1

    def active_bt_index_from_selection(self, indices: set[int] | list[int], fallback: int | None = None) -> int | None:
        items = self.bt_items_for_page()
        valid = sorted(index for index in indices if 0 <= index < len(items) and isinstance(items[index], dict))
        if fallback is not None and fallback in valid:
            return fallback
        if self.selected_bt_index in valid:
            return self.selected_bt_index
        return valid[0] if valid else None

    def bt_item_status(self, item: dict[str, Any]) -> str:
        raw_status = str(item.get('match_status') or '').lower()
        if raw_status == 'manual':
            return '手動'
        if raw_status == 'auto':
            return '自動'
        if raw_status == 'unmatched':
            return '未匹配'
        if raw_status in {'duplicate', 'fallback'}:
            return '待確認'
        if self.bt_item_needs_review(item):
            return '待確認'
        return '自動'

    def bt_item_needs_review(self, item: dict[str, Any]) -> bool:
        raw_status = str(item.get('match_status') or '').lower()
        if raw_status in {'manual', 'auto'}:
            return False
        if raw_status in {'unmatched', 'duplicate', 'fallback'}:
            return True
        xyxy = self.bt_xyxy_from_item(item)
        if xyxy is None:
            return True
        x1, y1, x2, y2 = xyxy
        return abs((x2 - x1) - 50) <= 1 and abs((y2 - y1) - 50) <= 1

    def bt_item_list_label(self, index: int, item: dict[str, Any]) -> str:
        status = self.bt_item_status(item)
        text = ' '.join(str(item.get('text') or '').split())
        if len(text) > 28:
            text = f'{text[:28]}...'
        if not text:
            text = '(空文字)'
        entry_index = item.get('index', index + 1)
        group_id = item.get('groupId', '-')
        return f'{index + 1:03d} [{status}] index={entry_index} g={group_id}  {text}'

    def update_bt_item_list(self) -> None:
        if not hasattr(self, 'bt_item_list'):
            return
        self.bt_item_list.blockSignals(True)
        self.bt_item_list.clear()
        items = self.bt_items_for_page()
        if self.bt_data is None:
            self.bt_item_list.addItem('未載入 _bt.json')
            self.bt_item_list.item(0).setFlags(Qt.ItemFlag.NoItemFlags)
        elif not items:
            self.bt_item_list.addItem('本頁沒有 _bt 條目')
            self.bt_item_list.item(0).setFlags(Qt.ItemFlag.NoItemFlags)
        else:
            for index, item in enumerate(items):
                label = self.bt_item_list_label(index, item if isinstance(item, dict) else {})
                self.bt_item_list.addItem(label)
                list_item = self.bt_item_list.item(index)
                list_item.setData(Qt.ItemDataRole.UserRole, index)
                status = self.bt_item_status(item if isinstance(item, dict) else {})
                if status in {'未匹配', '待確認'}:
                    list_item.setBackground(QBrush(QColor(255, 236, 194)))
                    list_item.setForeground(QBrush(QColor(92, 50, 0)))
                    list_item.setToolTip('需要人工確認：可先選左側此條，再點右側 measure 框套用。')
                elif status == '手動':
                    list_item.setBackground(QBrush(QColor(214, 245, 223)))
                    list_item.setForeground(QBrush(QColor(18, 92, 50)))
                    list_item.setToolTip('已手動套用 measure 框。')
        selected_indices = set(self.selected_bt_indices_list())
        if self.selected_bt_index is not None:
            selected_indices.add(self.selected_bt_index)
        active_index = self.active_bt_index_from_selection(selected_indices, self.selected_bt_index)
        for index in selected_indices:
            if 0 <= index < self.bt_item_list.count():
                self.bt_item_list.item(index).setSelected(True)
        if active_index is not None and 0 <= active_index < len(items):
            self.bt_item_list.setCurrentRow(active_index)
        elif not selected_indices:
            self.bt_item_list.clearSelection()
            self.bt_item_list.setCurrentRow(-1)
        self.bt_item_list.blockSignals(False)
        self.update_bt_stats_label()

    def handle_bt_item_row_changed(self, row: int) -> None:
        item = self.bt_item_list.item(row)
        if item is None:
            return
        index = item.data(Qt.ItemDataRole.UserRole)
        if not isinstance(index, int):
            self.select_bt_item(None)
            return
        selected = {
            selected_item.data(Qt.ItemDataRole.UserRole)
            for selected_item in self.bt_item_list.selectedItems()
        }
        selected_indices = {value for value in selected if isinstance(value, int)}
        if not selected_indices:
            selected_indices = {index}
        self.set_bt_selection(selected_indices, active_index=index, center=True, sync_list=False)

    def handle_bt_item_selection_changed(self) -> None:
        selected = {
            selected_item.data(Qt.ItemDataRole.UserRole)
            for selected_item in self.bt_item_list.selectedItems()
        }
        selected_indices = {value for value in selected if isinstance(value, int)}
        current_item = self.bt_item_list.currentItem()
        current_index = current_item.data(Qt.ItemDataRole.UserRole) if current_item is not None else None
        active_index = current_index if isinstance(current_index, int) else None
        self.set_bt_selection(selected_indices, active_index=active_index, sync_list=False)

    def load_bt_json_path(self, path: Path, *, remember: bool = True) -> None:
        self._commit_bt_editor_preview()
        path = path.expanduser().resolve()
        data = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(data, dict) or not isinstance(data.get('transMap'), dict):
            raise ValueError('不是有效的 _bt/MEO JSON：缺少 transMap。')
        self.bt_data = data
        self.bt_path = path
        self.bt_dirty = False
        self.selected_bt_index = None
        self.selected_bt_indices.clear()
        self.bt_undo_stack.clear()
        self.bt_path_label.setText(str(path))
        self.set_box_editor_enabled(False)
        self.update_bt_item_list()
        self.render_bt_page(refit=True)
        self.update_action_state()
        if remember:
            self._save_bt_mapping(path)

    def open_bt_json(self) -> None:
        start_dir = self.current_image_dir or self._last_existing_image_dir() or str(Path.home())
        path_text, _ = QFileDialog.getOpenFileName(self, '打開 _bt.json', start_dir, 'JSON (*.json)')
        if not path_text:
            return
        try:
            self.load_bt_json_path(Path(path_text).expanduser().resolve())
        except Exception as exc:
            show_exception_details(self, '打開失敗', '無法打開 _bt.json。下方是完整可複製的出錯信息。', exc)

    def save_bt_json(self) -> None:
        self._commit_bt_editor_preview()
        if self.bt_data is None or self.bt_path is None:
            return
        self.bt_path.write_text(
            json.dumps(self.bt_data, ensure_ascii=False, indent=2) + '\n',
            encoding='utf-8',
        )
        self.bt_dirty = False
        self.update_action_state()
        self.schedule_bt_html_overlay_update()

    def mark_bt_dirty(self) -> None:
        self.bt_dirty = True
        self.update_action_state()

    def bt_xyxy_from_item(self, item: dict[str, Any]) -> tuple[int, int, int, int] | None:
        xyxy = item.get('xyxy_pixel')
        if isinstance(xyxy, list) and len(xyxy) == 4:
            return tuple(int(round(float(v))) for v in xyxy)
        page_name = self.current_page_name()
        if not page_name or self.processor is None:
            return None
        image_size = qimage_size(self.processor.image_dir / page_name)
        if image_size is None:
            return None
        width, height = image_size
        try:
            cx = float(item.get('x')) * width
            cy = float(item.get('y')) * height
        except (TypeError, ValueError):
            return None
        half = 25
        return int(round(cx - half)), int(round(cy - half)), int(round(cx + half)), int(round(cy + half))

    def set_bt_xyxy(self, item: dict[str, Any], xyxy: tuple[int, int, int, int]) -> None:
        item['xyxy_pixel'] = list(xyxy)
        if self.page is not None:
            image_size = self._bt_cached_image_size or qimage_size(self.page.image_path)
            center = normalized_center_from_xyxy(xyxy, image_size)
            if center is not None:
                item['x'] = center[0]
                item['y'] = center[1]

    def bt_text_color(self, item: dict[str, Any]) -> str:
        color = str(item.get('color') or '#000000').lower()
        return 'white' if color in {'#ffffff', 'ffffff', 'white'} else 'black'

    def normalized_rotation(self, value: object) -> float:
        try:
            angle = float(value)
        except (TypeError, ValueError):
            return 0.0
        while angle > 180.0:
            angle -= 360.0
        while angle <= -180.0:
            angle += 360.0
        return round(angle, 2)

    def bt_font_label(self, item: dict[str, Any]) -> str:
        orientation = str(item.get('orientation') or 'vertical')
        direction = 'H' if orientation == 'horizontal' else 'V'
        font_size = positive_int(item.get('font-size'), 0)
        parts = [f'{font_size}{direction}' if font_size > 0 else direction]
        color_name = self.bt_text_color(item)
        parts.append('白' if color_name == 'white' else '黑')
        if positive_int(item.get('stroke-weight'), 0) > 0:
            parts.append('描邊')
        return ','.join(parts)

    def set_bt_text_color(self, item: dict[str, Any], value: str) -> None:
        if value == 'white':
            item['color'] = '#FFFFFF'
            item['stroke-color'] = '#000000'
        else:
            item['color'] = '#000000'
            item['stroke-color'] = '#FFFFFF'

    def push_bt_undo(self, description: str) -> None:
        item = self.selected_bt_item()
        page_name = self.current_page_name()
        if item is None or page_name is None or self.selected_bt_index is None:
            return
        self.bt_undo_stack.append({
            'page_name': page_name,
            'item_index': self.selected_bt_index,
            'item': copy.deepcopy(item),
            'description': description,
        })
        if len(self.bt_undo_stack) > 200:
            self.bt_undo_stack = self.bt_undo_stack[-200:]

    def push_bt_items_undo_snapshot(
        self,
        description: str,
        items: list[dict[str, Any]],
        selected_index: int | None,
        selected_indices: set[int] | list[int] | None = None,
    ) -> None:
        page_name = self.current_page_name()
        if page_name is None:
            return
        self.bt_undo_stack.append({
            'page_name': page_name,
            'items': copy.deepcopy(items),
            'selected_index': selected_index,
            'selected_indices': sorted(selected_indices or []),
            'description': description,
        })
        if len(self.bt_undo_stack) > 200:
            self.bt_undo_stack = self.bt_undo_stack[-200:]

    def push_bt_changes_undo_snapshot(
        self,
        description: str,
        before_items: dict[int, dict[str, Any]],
        selected_index: int | None,
        selected_indices: set[int] | list[int] | None = None,
    ) -> None:
        page_name = self.current_page_name()
        if page_name is None or not before_items:
            return
        self.bt_undo_stack.append({
            'page_name': page_name,
            'changes': [
                {'index': index, 'item': item}
                for index, item in sorted(before_items.items())
            ],
            'selected_index': selected_index,
            'selected_indices': sorted(selected_indices or []),
            'description': description,
        })
        if len(self.bt_undo_stack) > 200:
            self.bt_undo_stack = self.bt_undo_stack[-200:]

    def next_bt_index_for_page(self, items: list[dict[str, Any]]) -> int:
        indexes = []
        for item in items:
            try:
                indexes.append(int(item.get('index')))
            except (AttributeError, TypeError, ValueError):
                continue
        return max(indexes, default=0) + 1

    def font_size_counts(self) -> dict[int, int]:
        if self.processor is None:
            return {}
        counts = {}
        for items in (self.processor.measure.get('pages') or {}).values():
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict) or item.get('font_size') is None:
                    continue
                try:
                    size = int(round(float(item['font_size'])))
                except (TypeError, ValueError):
                    continue
                if size > 0:
                    counts[size] = counts.get(size, 0) + 1
        return counts

    def bt_font_size_counts(self) -> dict[int, int]:
        if self.bt_data is None:
            return {}
        counts: dict[int, int] = {}
        pages = self.bt_data.get('transMap') or {}
        if not isinstance(pages, dict):
            return counts
        for items in pages.values():
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                size = positive_int(item.get('font-size'), 0)
                if size > 0:
                    counts[size] = counts.get(size, 0) + 1
        return counts

    def open_regularize_font_size_dialog(self) -> None:
        if self.bt_data is None:
            self.status_label.setText('請先打開 _bt.json，再規整字體大小。')
            return
        counts = self.bt_font_size_counts()
        if not counts:
            self.status_label.setText('_bt.json 中沒有可規整的字體大小。')
            return
        dialog = FontSizeRegularizeDialog(counts, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        lower, upper = dialog.selected_range()
        target = dialog.target_size()
        self.regularize_bt_font_sizes(lower, upper, target)

    def regularize_bt_font_sizes(self, lower: int, upper: int, target: int) -> None:
        if self.bt_data is None:
            return
        lower, upper = sorted((int(lower), int(upper)))
        target = max(1, int(target))
        pages = self.bt_data.get('transMap') or {}
        if not isinstance(pages, dict):
            return
        changed = 0
        affected = 0
        for items in pages.values():
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                size = positive_int(item.get('font-size'), 0)
                if lower <= size <= upper:
                    affected += 1
                    if size != target:
                        item['font-size'] = target
                        changed += 1
        if changed <= 0:
            self.status_label.setText(f'{lower}-{upper} px 範圍內沒有需要改為 {target} px 的 _bt 條目。')
            return
        self.mark_bt_dirty()
        if self.selected_bt_items():
            self.populate_box_editor_for_selection()
        else:
            self.set_box_editor_enabled(False)
        self.update_bt_item_list()
        self.update_font_size_list()
        self.render_bt_page(refit=False)
        self.update_action_state()
        self.status_label.setText(
            f'已將 {lower}-{upper} px 範圍內 {affected} 條 _bt 中的 {changed} 條改為 {target} px，尚未保存。'
        )

    def even_font_size_preview(self) -> tuple[dict[int, int], int]:
        if self.processor is None:
            return {}, 0
        counts = {}
        changed = 0
        for items in (self.processor.measure.get('pages') or {}).values():
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict) or item.get('font_size') is None:
                    continue
                new_size = even_font_size(item.get('font_size'))
                if new_size is None:
                    continue
                try:
                    old_size = int(round(float(item['font_size'])))
                except (TypeError, ValueError):
                    old_size = new_size
                if new_size != old_size:
                    changed += 1
                counts[new_size] = counts.get(new_size, 0) + 1
        return counts, changed

    def format_font_size_counts(self, counts: dict[int, int]) -> str:
        if not counts:
            return '沒有可預覽的字級資料。'
        lines = ['字級  數目', '----------']
        for size in sorted(counts):
            lines.append(f'{size:>4}  {counts[size]:>4}')
        return '\n'.join(lines)

    def preview_even_font_sizes(self) -> None:
        QMessageBox.information(self, '功能已移除', 'ctd/measure.json 生成後不再支持批量修改字體大小。')

    def apply_even_font_sizes(self) -> None:
        return

    def current_page_font_sizes(self) -> set[int]:
        if self.page is None:
            return set()
        sizes = set()
        for box in self.page.boxes:
            if box.font_size is None:
                continue
            size = int(round(float(box.font_size)))
            if size > 0:
                sizes.add(size)
        return sizes

    def update_font_size_list(self) -> None:
        current_sizes = self.current_page_font_sizes()
        counts = self.font_size_counts()
        self.font_size_table.setRowCount(0)
        for row, size in enumerate(sorted(counts)):
            self.font_size_table.insertRow(row)
            size_item = QTableWidgetItem(str(size))
            count_item = QTableWidgetItem(str(counts[size]))
            for item in (size_item, count_item):
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            if size in current_sizes:
                for item in (size_item, count_item):
                    item.setBackground(QBrush(QColor(255, 220, 92)))
                    item.setForeground(QBrush(QColor(0, 0, 0)))
                    item.setToolTip('目前頁面使用')
            else:
                for item in (size_item, count_item):
                    item.setForeground(QBrush(QColor(225, 229, 233)))
            self.font_size_table.setItem(row, 0, size_item)
            self.font_size_table.setItem(row, 1, count_item)
        self.font_size_table.resizeRowsToContents()
