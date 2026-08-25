"""Folder, page and viewport coordination for the CTC viewer."""

from __future__ import annotations

import hashlib
from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QPixmap

try:
    from .processor import CtdOverlayProcessor
    from .viewer_dialogs import show_exception_details
    from .viewer_widgets import ImageView
except ImportError:
    from processor import CtdOverlayProcessor
    from viewer_dialogs import show_exception_details
    from viewer_widgets import ImageView


class ViewerSessionMixin:
    def _last_existing_image_dir(self) -> str | None:
        value = self.settings.value('last_image_dir', '', str)
        if not value:
            return None
        path = Path(value).expanduser()
        if not path.is_dir():
            return None
        return str(path.resolve())

    def _save_last_image_dir(self, image_dir: str) -> None:
        path = Path(image_dir).expanduser()
        if path.is_dir():
            self.settings.setValue('last_image_dir', str(path.resolve()))

    def _bt_mapping_key(self, image_dir: str | None = None) -> str | None:
        image_dir = image_dir or self.current_image_dir
        if not image_dir:
            return None
        path = Path(image_dir).expanduser()
        try:
            resolved = path.resolve()
        except OSError:
            return None
        digest = hashlib.sha1(str(resolved).encode('utf-8')).hexdigest()
        return f'bt_json_for_folder/{digest}'

    def _save_bt_mapping(self, bt_path: Path, image_dir: str | None = None) -> None:
        key = self._bt_mapping_key(image_dir)
        if key is None:
            return
        self.settings.setValue(key, str(bt_path.expanduser().resolve()))

    def _mapped_bt_path(self, image_dir: str | None = None) -> Path | None:
        key = self._bt_mapping_key(image_dir)
        if key is None:
            return None
        value = self.settings.value(key, '', str)
        if not value:
            return None
        path = Path(value).expanduser()
        if path.is_file():
            return path.resolve()
        self.settings.remove(key)
        return None

    def _autoload_mapped_bt_json(self) -> None:
        path = self._mapped_bt_path()
        if path is None:
            return
        try:
            self.load_bt_json_path(path, remember=False)
            self.status_label.setText(f'{self.status_label.text()}\n已自動載入：{path.name}')
        except Exception as exc:
            key = self._bt_mapping_key()
            if key is not None:
                self.settings.remove(key)
            show_exception_details(
                self,
                '自動打開 _bt.json 失敗',
                f'已找到此資料夾記錄的 _bt.json，但無法打開：\n{path}',
                exc,
            )

    def load_folder(self, image_dir: str) -> None:
        self._commit_bt_editor_preview()
        if self.bt_dirty and not self.save_pending_changes(auto=True):
            return
        image_dir = str(Path(image_dir).expanduser().resolve())
        self.current_image_dir = image_dir
        self._clear_bt_image_cache()
        self.clear_hover_char_box(render=False)
        self._bt_cursor_image_pos = None
        self.page = None
        self.selected_box_index = None
        self.selected_bt_index = None
        self.bt_data = None
        self.bt_path = None
        self.bt_dirty = False
        self.bt_undo_stack.clear()
        self.bt_path_label.setText('尚未載入 _bt.json')
        self.update_bt_item_list()
        self.undo_stack.clear()
        self.measure_dirty = False
        self.current_page_row = -1
        self.update_action_state()
        self.set_box_editor_enabled(False)
        try:
            self.processor = CtdOverlayProcessor(image_dir)
            self.page_names = self.processor.page_names()
        except Exception as exc:
            show_exception_details(self, '載入失敗', '無法載入資料夾。下方是完整可複製的出錯信息。', exc)
            return

        self._save_last_image_dir(image_dir)
        self.page_list.clear()
        self.page_list.addItems(self.page_names)
        self.update_font_size_list()
        summary = self.processor.summary()
        self.status_label.setText(self._summary_text(summary))
        if self.page_names:
            self.page_list.setCurrentRow(0)
            self._autoload_mapped_bt_json()
        else:
            self.page = None
            self.current_page_row = -1
            self.view.set_pixmap(QPixmap(), fit=False)
            self.bt_view.set_pixmap(QPixmap(), fit=False)
            self.update_font_size_list()
            self.status_label.setText(f'{self.status_label.text()}\n此資料夾沒有可顯示的圖片。')

    def handle_page_row_changed(self, row: int) -> None:
        if row == self.current_page_row:
            return
        previous_row = self.current_page_row
        if self.bt_dirty and not self.save_pending_changes(auto=True):
            self.page_list.blockSignals(True)
            self.page_list.setCurrentRow(previous_row)
            self.page_list.blockSignals(False)
            return
        self.undo_stack.clear()
        self.update_action_state()
        self.load_page_at_row(row)

    def go_to_relative_page(self, delta: int) -> None:
        if not self.page_names:
            return
        current = self.page_list.currentRow()
        if current < 0:
            current = self.current_page_row
        target = max(0, min(current + delta, len(self.page_names) - 1))
        if target != current:
            self.page_list.setCurrentRow(target)

    def fit_both_views(self) -> None:
        if not hasattr(self, 'bt_view') or not hasattr(self, 'view'):
            return
        if self._fitting_views:
            return
        self._fitting_views = True
        views = (self.bt_view,) if self.is_right_panel_collapsed() else (self.bt_view, self.view)
        for view in views:
            if view.sceneRect().isValid() and not view.sceneRect().isEmpty():
                view.fitInView(view.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self._fitting_views = False
        if self.is_right_panel_collapsed():
            self.sync_viewports_from(self.bt_view)
        else:
            self.sync_viewports_from(self.view)
        self.update_navigator()
        self.schedule_bt_html_overlay_update()

    def sync_viewports_from(self, source: ImageView) -> None:
        if self._syncing_views or self._fitting_views:
            return
        self._pending_viewport_source = source
        if not self._viewport_sync_timer.isActive():
            self._viewport_sync_timer.start()

    def _flush_viewport_sync(self) -> None:
        source = self._pending_viewport_source
        self._pending_viewport_source = None
        if source is None or self._syncing_views or self._fitting_views:
            return
        target = self.view if source is self.bt_view else self.bt_view
        if source.sceneRect().isEmpty() or target.sceneRect().isEmpty():
            return
        self._syncing_views = True
        target.setTransform(source.transform())
        center = source.mapToScene(source.viewport().rect().center())
        target.centerOn(center)
        self._syncing_views = False
        self.update_navigator()
        self.schedule_bt_html_overlay_update()
        if self._popover_bt_item is not None and self.bt_match_popover.isVisible():
            self.position_bt_match_popover(self._popover_bt_item)
        item = self.selected_bt_item()
        if item is not None and self.bt_text_popover.isVisible():
            self.position_bt_text_popover(item)

    def center_views_on(self, x: float, y: float) -> None:
        self._syncing_views = True
        center = QPointF(x, y)
        self.bt_view.centerOn(center)
        self.view.centerOn(center)
        self._syncing_views = False
        self.update_navigator()
        self.schedule_bt_html_overlay_update()

    def update_navigator(self) -> None:
        if not hasattr(self, 'navigator'):
            return
        source_view = self.bt_view if self.is_right_panel_collapsed() else self.view
        pixmap = source_view.current_pixmap()
        if pixmap.isNull():
            self.navigator.set_navigator_state(QPixmap(), (0, 0), QRectF())
            return
        visible_polygon = source_view.mapToScene(source_view.viewport().rect())
        visible_rect = visible_polygon.boundingRect()
        self.navigator.set_navigator_state(
            pixmap,
            (pixmap.width(), pixmap.height()),
            visible_rect,
        )

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if not hasattr(self, '_fitting_views'):
            return
        self.fit_both_views()
        self.schedule_bt_html_overlay_update()
