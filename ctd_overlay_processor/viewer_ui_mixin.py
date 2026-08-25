"""UI construction and panel behavior for the CTC viewer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import QPointF, QRectF, Qt, QSettings
from PySide6.QtGui import QAction, QFont, QKeySequence
from PySide6.QtWidgets import (
    QFileDialog,
    QCheckBox,
    QDockWidget,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

try:
    from .viewer_widgets import ImageView
except ImportError:
    from viewer_widgets import ImageView


class ViewerUIMixin:
    def _build_toolbar(self) -> None:
        toolbar = QToolBar('主工具列')
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        open_action = QAction('選擇資料夾', self)
        open_action.triggered.connect(self.choose_folder)
        toolbar.addAction(open_action)

        generate_action = QAction('生成/更新 CTD', self)
        generate_action.triggered.connect(self.generate_ctd)
        toolbar.addAction(generate_action)

        import_labelplus_action = QAction('導入 LabelPlus txt', self)
        import_labelplus_action.triggered.connect(self.import_labelplus_txt)
        toolbar.addAction(import_labelplus_action)

        open_bt_action = QAction('打開 _bt.json', self)
        open_bt_action.triggered.connect(self.open_bt_json)
        toolbar.addAction(open_bt_action)

        shortcuts_action = QAction('快捷鍵說明', self)
        shortcuts_action.setToolTip('查看 _bt.json 編輯器的全部快捷鍵')
        shortcuts_action.triggered.connect(self.show_shortcuts_dialog)
        toolbar.addAction(shortcuts_action)

        fit_action = QAction('適合視窗', self)
        fit_action.triggered.connect(self.fit_both_views)
        toolbar.addAction(fit_action)

        self.toggle_right_panel_action = QAction('收起右側', self)
        self.toggle_right_panel_action.setCheckable(True)
        self.toggle_right_panel_action.setToolTip('展開/收起右邊圖片和最右功能區')
        self.toggle_right_panel_action.triggered.connect(self.toggle_right_panel)
        toolbar.addAction(self.toggle_right_panel_action)

        toolbar.addWidget(self.show_popover_check)
        toolbar.addWidget(self.show_bt_font_labels_check)
        toggle_popover_action = QAction('切換預覽浮窗', self)
        toggle_popover_action.setShortcuts([QKeySequence('Meta+P'), QKeySequence('Ctrl+P')])
        toggle_popover_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        toggle_popover_action.triggered.connect(self.toggle_match_popover_enabled)
        self.addAction(toggle_popover_action)

        self.prev_page_action = QAction('上一頁', self)
        self.prev_page_action.setShortcut(QKeySequence(Qt.Key.Key_PageUp))
        self.prev_page_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        self.prev_page_action.triggered.connect(lambda: self.go_to_relative_page(-1))
        toolbar.addAction(self.prev_page_action)

        self.next_page_action = QAction('下一頁', self)
        self.next_page_action.setShortcut(QKeySequence(Qt.Key.Key_PageDown))
        self.next_page_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        self.next_page_action.triggered.connect(lambda: self.go_to_relative_page(1))
        toolbar.addAction(self.next_page_action)

        self.save_action = QAction('保存 _bt', self)
        self.save_action.setShortcuts([QKeySequence('Meta+S'), QKeySequence('Ctrl+S')])
        self.save_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        self.save_action.triggered.connect(self.save_pending_changes)
        self.save_action.setEnabled(False)
        toolbar.addAction(self.save_action)

        self.undo_action = QAction('撤銷', self)
        self.undo_action.setShortcuts([QKeySequence('Meta+Z'), QKeySequence('Ctrl+Z')])
        self.undo_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        self.undo_action.triggered.connect(self.undo_last_edit)
        self.undo_action.setEnabled(False)
        toolbar.addAction(self.undo_action)

        self.increase_font_action = QAction('字體+2', self)
        self.increase_font_action.setShortcuts([
            QKeySequence('Meta++'),
            QKeySequence('Meta+='),
            QKeySequence('Ctrl++'),
            QKeySequence('Ctrl+='),
        ])
        self.increase_font_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        self.increase_font_action.triggered.connect(lambda: self.nudge_selected_font_size(2))
        self.increase_font_action.setEnabled(False)
        self.addAction(self.increase_font_action)

        self.decrease_font_action = QAction('字體-2', self)
        self.decrease_font_action.setShortcuts([QKeySequence('Meta+-'), QKeySequence('Ctrl+-')])
        self.decrease_font_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        self.decrease_font_action.triggered.connect(lambda: self.nudge_selected_font_size(-2))
        self.decrease_font_action.setEnabled(False)
        self.addAction(self.decrease_font_action)

        self.increase_font_10_action = QAction('字體+10', self)
        self.increase_font_10_action.setShortcuts([
            QKeySequence('Meta+Alt++'),
            QKeySequence('Meta+Alt+='),
            QKeySequence('Ctrl+Alt++'),
            QKeySequence('Ctrl+Alt+='),
        ])
        self.increase_font_10_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        self.increase_font_10_action.triggered.connect(lambda: self.nudge_selected_font_size(10))
        self.increase_font_10_action.setEnabled(False)
        self.addAction(self.increase_font_10_action)

        self.decrease_font_10_action = QAction('字體-10', self)
        self.decrease_font_10_action.setShortcuts([
            QKeySequence('Meta+Alt+-'),
            QKeySequence('Ctrl+Alt+-'),
        ])
        self.decrease_font_10_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        self.decrease_font_10_action.triggered.connect(lambda: self.nudge_selected_font_size(-10))
        self.decrease_font_10_action.setEnabled(False)
        self.addAction(self.decrease_font_10_action)

        self.rotate_counterclockwise_action = QAction('逆時針+1', self)
        self.rotate_counterclockwise_action.setShortcuts([QKeySequence('Meta+['), QKeySequence('Ctrl+[')])
        self.rotate_counterclockwise_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        self.rotate_counterclockwise_action.triggered.connect(lambda: self.nudge_selected_rotation(1.0))
        self.rotate_counterclockwise_action.setEnabled(False)
        self.addAction(self.rotate_counterclockwise_action)

        self.rotate_clockwise_action = QAction('順時針-1', self)
        self.rotate_clockwise_action.setShortcuts([QKeySequence('Meta+]'), QKeySequence('Ctrl+]')])
        self.rotate_clockwise_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        self.rotate_clockwise_action.triggered.connect(lambda: self.nudge_selected_rotation(-1.0))
        self.rotate_clockwise_action.setEnabled(False)
        self.addAction(self.rotate_clockwise_action)

        self.rotate_counterclockwise_5_action = QAction('逆時針+5', self)
        self.rotate_counterclockwise_5_action.setShortcuts([
            QKeySequence('Meta+Alt+['),
            QKeySequence('Ctrl+Alt+['),
        ])
        self.rotate_counterclockwise_5_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        self.rotate_counterclockwise_5_action.triggered.connect(lambda: self.nudge_selected_rotation(5.0))
        self.rotate_counterclockwise_5_action.setEnabled(False)
        self.addAction(self.rotate_counterclockwise_5_action)

        self.rotate_clockwise_5_action = QAction('順時針-5', self)
        self.rotate_clockwise_5_action.setShortcuts([
            QKeySequence('Meta+Alt+]'),
            QKeySequence('Ctrl+Alt+]'),
        ])
        self.rotate_clockwise_5_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        self.rotate_clockwise_5_action.triggered.connect(lambda: self.nudge_selected_rotation(-5.0))
        self.rotate_clockwise_5_action.setEnabled(False)
        self.addAction(self.rotate_clockwise_5_action)

        self.copy_box_action = QAction('複製文字框', self)
        self.copy_box_action.setShortcuts([QKeySequence('Meta+D'), QKeySequence('Ctrl+D')])
        self.copy_box_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        self.copy_box_action.setToolTip('複製當前文字框（Command/Ctrl+D）')
        self.copy_box_action.triggered.connect(self.copy_selected_box)
        self.copy_box_action.setEnabled(False)
        self.addAction(self.copy_box_action)

        self.copy_to_memory_action = QAction('暫存複製文字框', self)
        self.copy_to_memory_action.setShortcut(QKeySequence('F1'))
        self.copy_to_memory_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        self.copy_to_memory_action.setToolTip('暫存複製目前文字框（F1）')
        self.copy_to_memory_action.triggered.connect(self.copy_selected_box_to_memory)
        self.copy_to_memory_action.setEnabled(False)
        self.addAction(self.copy_to_memory_action)

        self.paste_from_memory_action = QAction('在游標位置暫存貼上文字框', self)
        self.paste_from_memory_action.setShortcut(QKeySequence('F2'))
        self.paste_from_memory_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        self.paste_from_memory_action.setToolTip('在游標位置貼上暫存文字框（F2）')
        self.paste_from_memory_action.triggered.connect(self.paste_box_from_memory)
        self.paste_from_memory_action.setEnabled(False)
        self.addAction(self.paste_from_memory_action)

        self.clear_bt_selection_action = QAction('取消選取文字框', self)
        self.clear_bt_selection_action.setShortcut(QKeySequence(Qt.Key.Key_Escape))
        self.clear_bt_selection_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        self.clear_bt_selection_action.setToolTip('取消目前文字框的選取與編輯焦點（Esc）')
        self.clear_bt_selection_action.triggered.connect(self.clear_current_bt_focus)
        self.addAction(self.clear_bt_selection_action)

        self.measure_angle_action = QAction('測量角度', self)
        self.measure_angle_action.setShortcuts([QKeySequence('Meta+M'), QKeySequence('Ctrl+M')])
        self.measure_angle_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        self.measure_angle_action.setToolTip('打開測量角度（Command/Ctrl+M）')
        self.measure_angle_action.triggered.connect(self.open_bt_measurement_dialog)
        self.measure_angle_action.setEnabled(False)
        self.addAction(self.measure_angle_action)

        self.add_empty_box_action = QAction('新增空文案', self)
        self.add_empty_box_action.setShortcuts([QKeySequence('Meta+N'), QKeySequence('Ctrl+N')])
        self.add_empty_box_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        self.add_empty_box_action.setToolTip('在鼠標位置新增預設空文案（Command/Ctrl+N）')
        self.add_empty_box_action.triggered.connect(self.add_empty_bt_box)
        self.add_empty_box_action.setEnabled(False)
        self.addAction(self.add_empty_box_action)

        self.delete_box_action = QAction('刪除文字框', self)
        self.delete_box_action.setShortcuts([QKeySequence(Qt.Key.Key_Delete), QKeySequence(Qt.Key.Key_Backspace)])
        self.delete_box_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        self.delete_box_action.triggered.connect(self.delete_selected_box)
        self.delete_box_action.setEnabled(False)
        self.addAction(self.delete_box_action)

        move_shortcuts = (
            ('左移', Qt.Key.Key_Left, -1, 0, 1),
            ('右移', Qt.Key.Key_Right, 1, 0, 1),
            ('上移', Qt.Key.Key_Up, 0, -1, 1),
            ('下移', Qt.Key.Key_Down, 0, 1, 1),
            ('左移10', Qt.KeyboardModifier.ShiftModifier | Qt.Key.Key_Left, -1, 0, 10),
            ('右移10', Qt.KeyboardModifier.ShiftModifier | Qt.Key.Key_Right, 1, 0, 10),
            ('上移10', Qt.KeyboardModifier.ShiftModifier | Qt.Key.Key_Up, 0, -1, 10),
            ('下移10', Qt.KeyboardModifier.ShiftModifier | Qt.Key.Key_Down, 0, 1, 10),
            ('左移50', Qt.KeyboardModifier.ShiftModifier | Qt.KeyboardModifier.ControlModifier | Qt.Key.Key_Left, -1, 0, 50),
            ('右移50', Qt.KeyboardModifier.ShiftModifier | Qt.KeyboardModifier.ControlModifier | Qt.Key.Key_Right, 1, 0, 50),
            ('上移50', Qt.KeyboardModifier.ShiftModifier | Qt.KeyboardModifier.ControlModifier | Qt.Key.Key_Up, 0, -1, 50),
            ('下移50', Qt.KeyboardModifier.ShiftModifier | Qt.KeyboardModifier.ControlModifier | Qt.Key.Key_Down, 0, 1, 50),
            ('左移50（Mac）', Qt.KeyboardModifier.ShiftModifier | Qt.KeyboardModifier.MetaModifier | Qt.Key.Key_Left, -1, 0, 50),
            ('右移50（Mac）', Qt.KeyboardModifier.ShiftModifier | Qt.KeyboardModifier.MetaModifier | Qt.Key.Key_Right, 1, 0, 50),
            ('上移50（Mac）', Qt.KeyboardModifier.ShiftModifier | Qt.KeyboardModifier.MetaModifier | Qt.Key.Key_Up, 0, -1, 50),
            ('下移50（Mac）', Qt.KeyboardModifier.ShiftModifier | Qt.KeyboardModifier.MetaModifier | Qt.Key.Key_Down, 0, 1, 50),
        )
        for label, key, dx, dy, step in move_shortcuts:
            action = QAction(label, self)
            action.setShortcut(QKeySequence(key))
            action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
            action.triggered.connect(lambda checked=False, dx=dx, dy=dy, step=step: self.nudge_selected_box_position(dx * step, dy * step))
            self.move_box_actions.append(action)
            self.addAction(action)

    def _build_side_panel(self) -> None:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        layout.addWidget(QLabel('資料狀態'))
        layout.addWidget(self.status_label)
        layout.addWidget(QLabel('頁面'))
        self.page_list.setMinimumHeight(150)
        layout.addWidget(self.page_list, 1)
        layout.addWidget(QLabel('_bt.json'))
        layout.addWidget(self.bt_path_label)
        self.open_bt_button.clicked.connect(self.open_bt_json)
        layout.addWidget(self.open_bt_button)
        layout.addWidget(self.bt_stats_label)
        layout.addWidget(QLabel('當前頁 _bt 條目'))
        self.bt_item_list.setMinimumHeight(260)
        self.bt_item_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        layout.addWidget(self.bt_item_list, 2)
        layout.addWidget(QLabel('游標資訊'))
        layout.addWidget(self.char_info_label)
        layout.addWidget(QLabel('導航器'))
        layout.addWidget(self.navigator)

        dock = QDockWidget('頁面 / _bt 條目', self)
        dock.setWidget(panel)
        dock.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)

    def _build_center_tools_panel(self) -> QWidget:
        panel = QWidget()
        panel.setMinimumWidth(280)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        title = QLabel('字級快捷列表')
        title.setToolTip('點擊字級即可套用到目前選中的 _bt 條目。')
        list_font = QFont('Menlo', 13)
        list_font.setBold(True)
        list_font.setStyleHint(QFont.StyleHint.Monospace)
        self.font_size_table.setFont(list_font)
        self.font_size_table.setColumnCount(2)
        self.font_size_table.setHorizontalHeaderLabels(['字級', '數目'])
        self.font_size_table.verticalHeader().setVisible(False)
        self.font_size_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.font_size_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.font_size_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.font_size_table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.font_size_table.setAlternatingRowColors(True)
        self.font_size_table.setShowGrid(False)
        self.font_size_table.horizontalHeader().setStretchLastSection(True)
        self.font_size_table.horizontalHeader().setDefaultAlignment(Qt.AlignmentFlag.AlignCenter)
        self.font_size_table.setColumnWidth(0, 86)
        self.font_size_table.setMinimumHeight(340)
        self.font_size_table.setStyleSheet(
            'QTableWidget {'
            '  background: #101214;'
            '  alternate-background-color: #181a1d;'
            '  color: #e9ecef;'
            '  border: 1px solid #33383d;'
            '}'
            'QHeaderView::section {'
            '  background: #2a2e33;'
            '  color: #f4f6f8;'
            '  border: 0;'
            '  border-bottom: 1px solid #454b52;'
            '  padding: 6px 4px;'
            '  font-weight: 700;'
            '}'
            'QTableWidget::item {'
            '  padding: 5px 8px;'
            '}'
        )
        self.even_font_button.clicked.connect(self.preview_even_font_sizes)
        self.even_font_button.hide()
        self.regularize_font_button.clicked.connect(self.open_regularize_font_size_dialog)

        self.font_size_spin.setRange(1, 999)
        self.font_size_spin.setSuffix(' px')
        self.orientation_vertical_button.setChecked(True)
        self.rotation_spin.setRange(-180.0, 180.0)
        self.rotation_spin.setDecimals(2)
        self.rotation_spin.setSingleStep(1.0)
        self.rotation_spin.setSuffix(' deg')
        self.color_black_button.setChecked(True)
        self.stroke_weight_spin.setRange(0, 99)
        self.stroke_weight_spin.setSuffix(' px')

        choice_style = (
            'QPushButton { padding: 4px 8px; min-height: 24px; }'
            'QPushButton:checked { background: #4a5057; border: 1px solid #7c8791; }'
        )
        for button in (
            self.orientation_vertical_button,
            self.orientation_horizontal_button,
            self.color_black_button,
            self.color_white_button,
        ):
            button.setStyleSheet(choice_style)

        typography_row = QWidget()
        typography_layout = QHBoxLayout(typography_row)
        typography_layout.setContentsMargins(0, 0, 0, 0)
        typography_layout.setSpacing(6)
        typography_layout.addWidget(QLabel('字級'))
        typography_layout.addWidget(self.font_size_spin, 1)
        typography_layout.addWidget(QLabel('方向'))
        typography_layout.addWidget(self.orientation_horizontal_button)
        typography_layout.addWidget(self.orientation_vertical_button)

        rotation_row = QWidget()
        rotation_layout = QHBoxLayout(rotation_row)
        rotation_layout.setContentsMargins(0, 0, 0, 0)
        rotation_layout.setSpacing(6)
        rotation_layout.addWidget(QLabel('角度'))
        rotation_layout.addWidget(self.rotation_spin, 1)

        appearance_row = QWidget()
        appearance_layout = QHBoxLayout(appearance_row)
        appearance_layout.setContentsMargins(0, 0, 0, 0)
        appearance_layout.setSpacing(6)
        appearance_layout.addWidget(QLabel('顏色'))
        appearance_layout.addWidget(self.color_black_button)
        appearance_layout.addWidget(self.color_white_button)
        appearance_layout.addWidget(QLabel('描邊'))
        appearance_layout.addWidget(self.stroke_weight_spin, 1)

        appearance_actions_row = QWidget()
        appearance_actions_layout = QHBoxLayout(appearance_actions_row)
        appearance_actions_layout.setContentsMargins(0, 0, 0, 0)
        appearance_actions_layout.setSpacing(8)
        appearance_actions_layout.addWidget(self.text_has_stroke_check)
        appearance_actions_layout.addWidget(self.need_inpaint_check)
        appearance_actions_layout.addStretch(1)
        appearance_actions_layout.addWidget(self.save_button)

        box_actions_row = QWidget()
        box_actions_layout = QHBoxLayout(box_actions_row)
        box_actions_layout.setContentsMargins(0, 0, 0, 0)
        box_actions_layout.setSpacing(8)
        box_actions_layout.addWidget(self.copy_bt_button, 1)
        box_actions_layout.addWidget(self.add_empty_bt_button, 1)

        clipboard_actions_row = QWidget()
        clipboard_actions_layout = QHBoxLayout(clipboard_actions_row)
        clipboard_actions_layout.setContentsMargins(0, 0, 0, 0)
        clipboard_actions_layout.setSpacing(8)
        clipboard_actions_layout.addWidget(self.copy_to_clipboard_button, 1)
        clipboard_actions_layout.addWidget(self.delete_clipboard_button, 1)

        layout.addWidget(QLabel('文字'))
        layout.addWidget(self.bt_text_edit)
        layout.addWidget(typography_row)
        layout.addWidget(rotation_row)
        layout.addWidget(box_actions_row)
        layout.addWidget(self.measure_preview_button)
        self.save_button.clicked.connect(self.save_pending_changes)
        layout.addWidget(appearance_row)
        layout.addWidget(appearance_actions_row)
        layout.addWidget(title)
        layout.addWidget(QLabel('文字框剪貼簿（跨專案保留）'))
        self.bt_clipboard_list.setMinimumHeight(130)
        self.bt_clipboard_list.setToolTip('單擊一項即可暫存複製它；可用 F2 貼上。')
        layout.addWidget(self.bt_clipboard_list)
        layout.addWidget(clipboard_actions_row)
        layout.addWidget(self.regularize_font_button)
        layout.addWidget(self.font_size_table, 1)
        layout.addWidget(self.even_font_button)
        self.set_box_editor_enabled(False)
        self.update_bt_clipboard_list()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(panel)
        scroll.setMinimumWidth(280)
        return scroll

    def _build_layer_panel(self) -> None:
        panel = QWidget()
        panel.setMinimumWidth(220)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        layout.addWidget(QLabel('圖層'))
        for widget in (
            self.show_mask,
            self.show_npz_smoothed,
            self.show_npz_outer,
            self.show_block_boxes,
            self.show_align_boxes,
            self.show_line_polygons,
            self.show_char_boxes,
            self.show_font_labels,
        ):
            layout.addWidget(widget)

        opacity_row = QHBoxLayout()
        opacity_row.addWidget(QLabel('疊圖透明度'))
        opacity_row.addWidget(self.opacity_slider)
        layout.addLayout(opacity_row)

        layout.addWidget(QLabel('流程'))
        reload_button = QPushButton('重新載入目前頁')
        reload_button.clicked.connect(self.reload_current_page)
        layout.addWidget(reload_button)
        layout.addWidget(QLabel('字級計算方法'))
        self.ocr_aligned_font_radio.setToolTip(
            '使用 mit48 整行 OCR 對齊可靠文字，再依字型墨跡緩存逐字計算字級。'
        )
        self.char_box_font_radio.setToolTip(
            '使用原有遮罩切分單字框的字級計算方法，不執行 mit48 OCR。'
        )
        self.ocr_aligned_font_radio.toggled.connect(self.auto_calibrate_device_combo.setEnabled)
        layout.addWidget(self.ocr_aligned_font_radio)
        layout.addWidget(self.char_box_font_radio)
        layout.addWidget(self.auto_calibrate_device_combo)
        self.generate_button.clicked.connect(self.generate_ctd)
        layout.addWidget(self.generate_button)
        layout.addWidget(self.font_calibration_progress_label)
        self.edit_measure_button.clicked.connect(self.open_measure_editor)
        layout.addWidget(self.edit_measure_button)
        self.import_labelplus_button.clicked.connect(self.import_labelplus_txt)
        layout.addWidget(self.import_labelplus_button)
        layout.addStretch(1)

        dock = QDockWidget('圖層 / 流程', self)
        dock.setWidget(panel)
        dock.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        self.right_layer_dock = dock
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

    def _connect_signals(self) -> None:
        self.page_list.currentRowChanged.connect(self.handle_page_row_changed)
        self.bt_item_list.currentRowChanged.connect(self.handle_bt_item_row_changed)
        self.bt_item_list.itemSelectionChanged.connect(self.handle_bt_item_selection_changed)
        for widget in (
            self.show_mask,
            self.show_npz_smoothed,
            self.show_npz_outer,
            self.show_block_boxes,
            self.show_align_boxes,
            self.show_line_polygons,
            self.show_char_boxes,
            self.show_font_labels,
        ):
            widget.stateChanged.connect(self.handle_layer_checkbox_changed)
        self.opacity_slider.valueChanged.connect(self.render_current_page)
        self.view.imageMouseMoved.connect(self.update_hover_char_box)
        self.view.imageMouseLeft.connect(self.clear_hover_char_box)
        self.view.imageMousePressed.connect(self.handle_image_mouse_press)
        self.bt_view.imageMousePressed.connect(self.handle_bt_mouse_press)
        self.bt_view.imageMouseMoved.connect(self.update_bt_view_cursor)
        self.bt_view.imageMouseLeft.connect(self.clear_bt_view_cursor)
        self.bt_view.imageMouseDragged.connect(self.handle_bt_mouse_drag)
        self.bt_view.imageMouseReleased.connect(self.handle_bt_mouse_release)
        self.bt_view.fontSizeWheelRequested.connect(self.nudge_selected_font_size)
        self.view.fontSizeWheelRequested.connect(self.nudge_selected_font_size)
        self.bt_view.viewportChanged.connect(self.sync_viewports_from)
        self.view.viewportChanged.connect(self.sync_viewports_from)
        self.navigator.navigateRequested.connect(self.center_views_on)
        self.show_popover_check.stateChanged.connect(self.handle_match_popover_setting_changed)
        self.show_bt_font_labels_check.stateChanged.connect(self.handle_bt_font_label_setting_changed)
        self.bt_text_edit.textChanged.connect(self.apply_editor_changes_to_selected_box)
        self.bt_text_popover.text_edit.textChanged.connect(self.apply_bt_text_popover_changes)
        self.font_size_spin.valueChanged.connect(self.handle_typography_control_changed)
        self.orientation_vertical_button.clicked.connect(self.apply_editor_changes_to_selected_box)
        self.orientation_horizontal_button.clicked.connect(self.apply_editor_changes_to_selected_box)
        self.rotation_spin.valueChanged.connect(self.handle_typography_control_changed)
        self.font_size_spin.editingFinished.connect(self._commit_bt_editor_preview)
        self.rotation_spin.editingFinished.connect(self._commit_bt_editor_preview)
        self.copy_bt_button.clicked.connect(self.copy_selected_box)
        self.copy_to_clipboard_button.clicked.connect(self.copy_selected_box_to_clipboard)
        self.delete_clipboard_button.clicked.connect(self.delete_selected_clipboard_item)
        self.bt_clipboard_list.itemClicked.connect(self.copy_bt_clipboard_list_item)
        self.measure_preview_button.clicked.connect(self.open_bt_measurement_dialog)
        self.add_empty_bt_button.clicked.connect(self.add_empty_bt_box)
        self.color_black_button.clicked.connect(self.apply_editor_changes_to_selected_box)
        self.color_white_button.clicked.connect(self.apply_editor_changes_to_selected_box)
        self.stroke_weight_spin.valueChanged.connect(self.apply_editor_changes_to_selected_box)
        self.text_has_stroke_check.stateChanged.connect(self.apply_editor_changes_to_selected_box)
        self.need_inpaint_check.stateChanged.connect(self.apply_editor_changes_to_selected_box)
        self.font_size_table.cellClicked.connect(self.apply_font_size_from_table)

    def choose_folder(self) -> None:
        start_dir = self.current_image_dir or self._last_existing_image_dir() or str(Path.home())
        folder = QFileDialog.getExistingDirectory(self, '選擇包含原圖和 ctd 的圖片資料夾', start_dir)
        if folder:
            self.load_folder(folder)

    def _layer_checkboxes(self) -> tuple[tuple[str, QCheckBox], ...]:
        return (
            ('show_mask', self.show_mask),
            ('show_npz_smoothed', self.show_npz_smoothed),
            ('show_npz_outer', self.show_npz_outer),
            ('show_block_boxes', self.show_block_boxes),
            ('show_align_boxes', self.show_align_boxes),
            ('show_line_polygons', self.show_line_polygons),
            ('show_char_boxes', self.show_char_boxes),
            ('show_font_labels', self.show_font_labels),
        )

    def _settings_bool(self, key: str, default: bool) -> bool:
        value = self.settings.value(key, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in ('1', 'true', 'yes', 'on'):
                return True
            if lowered in ('0', 'false', 'no', 'off'):
                return False
        return default

    def _restore_layer_checkbox_states(self) -> None:
        for name, checkbox in self._layer_checkboxes():
            default = name in self.DEFAULT_LAYER_CHECKED
            checkbox.setChecked(self._settings_bool(f'layers/{name}', default))

    def _save_layer_checkbox_states(self) -> None:
        for name, checkbox in self._layer_checkboxes():
            self.settings.setValue(f'layers/{name}', checkbox.isChecked())

    def handle_layer_checkbox_changed(self, *_: object) -> None:
        self._save_layer_checkbox_states()
        self.render_current_page()

    def handle_bt_font_label_setting_changed(self, *_: object) -> None:
        self.settings.setValue('ui/show_bt_font_labels', self.show_bt_font_labels_check.isChecked())
        self.render_bt_page(refit=False)

    def match_popover_enabled(self) -> bool:
        return self.show_popover_check.isChecked() and not self.has_multiple_bt_selection()

    def show_bt_text_popover(self, item: dict[str, Any]) -> None:
        if self.has_multiple_bt_selection() or self.selected_bt_item() is None:
            self.bt_text_popover.hide()
            return
        self.bt_text_popover.set_text(str(item.get('text') or ''))
        self.position_bt_text_popover(item)
        self.bt_text_popover.show()
        self.bt_text_popover.raise_()

    def position_bt_text_popover(self, item: dict[str, Any]) -> None:
        xyxy = self.bt_xyxy_from_item(item)
        if xyxy is None:
            self.bt_text_popover.hide()
            return
        x1, y1, x2, y2 = xyxy
        p1 = self.bt_view.mapFromScene(QPointF(x1, y1))
        p2 = self.bt_view.mapFromScene(QPointF(x2, y2))
        selected_rect = QRectF(p1, p2).normalized().adjusted(-8, -8, 8, 8)
        popover_size = self.bt_text_popover.sizeHint()
        gap = 12
        left_local = QPointF(
            selected_rect.left() - gap - popover_size.width(),
            selected_rect.center().y() - popover_size.height() / 2,
        )
        global_pos = self.bt_view.viewport().mapToGlobal(left_local.toPoint())
        self.bt_text_popover.move(global_pos)

    def apply_bt_text_popover_changes(self) -> None:
        if self._updating_editor or self.has_multiple_bt_selection():
            return
        item = self.selected_bt_item()
        if item is None:
            return
        text = self.bt_text_popover.text_edit.toPlainText()
        changed = self.apply_selected_box_updates(
            {'text': text},
            status='已修改當前 _bt 文字，尚未保存。',
            refresh_editor=False,
        )
        if not changed:
            return
        self._updating_editor = True
        self.bt_text_edit.setPlainText(text)
        self._updating_editor = False
        self.position_bt_text_popover(item)

    def sync_bt_text_popover_from_item(self) -> None:
        if not self.bt_text_popover.isVisible() or self.has_multiple_bt_selection():
            return
        item = self.selected_bt_item()
        if item is not None:
            self.bt_text_popover.set_text(str(item.get('text') or ''))
            self.position_bt_text_popover(item)

    def handle_match_popover_setting_changed(self, *_: object) -> None:
        enabled = self.show_popover_check.isChecked()
        self.settings.setValue('ui/show_match_popover', enabled)
        if not self.match_popover_enabled():
            self.bt_match_popover.hide()
            return
        item = self.selected_bt_item()
        if item is not None:
            self.show_bt_match_popover(item)

    def toggle_match_popover_enabled(self, *_: object) -> None:
        self.show_popover_check.setChecked(not self.show_popover_check.isChecked())

    def restore_right_panel_state(self) -> None:
        collapsed = self._settings_bool('right_panel/collapsed', False)
        width = self.settings.value('right_panel/width', self._last_right_splitter_width)
        try:
            self._last_right_splitter_width = max(160, int(width))
        except (TypeError, ValueError):
            self._last_right_splitter_width = 520
        self.set_right_panel_collapsed(collapsed, remember=False)

    def toggle_right_panel(self, checked: bool = False) -> None:
        self.set_right_panel_collapsed(checked, remember=True)

    def set_right_panel_collapsed(self, collapsed: bool, *, remember: bool) -> None:
        if not hasattr(self, 'main_splitter'):
            return
        sizes = self.main_splitter.sizes()
        if len(sizes) < 3:
            return

        if collapsed:
            if sizes[2] > 0:
                self._last_right_splitter_width = sizes[2]
            sizes[0] = max(160, sizes[0] + sizes[2])
            sizes[2] = 0
            self.main_splitter.setSizes(sizes)
            self.view.hide()
            if self.right_layer_dock is not None:
                self.right_layer_dock.hide()
            label = '展開右側'
        else:
            self.view.show()
            if self.right_layer_dock is not None:
                self.right_layer_dock.show()
            restored_width = max(160, self._last_right_splitter_width)
            if sizes[2] <= 0:
                sizes[2] = restored_width
                sizes[0] = max(160, sizes[0] - restored_width)
            self.main_splitter.setSizes(sizes)
            label = '收起右側'

        if self.toggle_right_panel_action is not None:
            self.toggle_right_panel_action.blockSignals(True)
            self.toggle_right_panel_action.setChecked(collapsed)
            self.toggle_right_panel_action.setText(label)
            self.toggle_right_panel_action.blockSignals(False)
        if remember:
            self.settings.setValue('right_panel/collapsed', collapsed)
            if self._last_right_splitter_width > 0:
                self.settings.setValue('right_panel/width', self._last_right_splitter_width)
        self.bt_match_popover.hide()
        self.fit_both_views()

    def is_right_panel_collapsed(self) -> bool:
        return hasattr(self, 'view') and not self.view.isVisible()
