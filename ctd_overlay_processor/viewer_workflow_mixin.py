"""CTC generation, calibration and page workflow for the viewer."""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QProcess
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QVBoxLayout,
)

try:
    from .font_size_calibration import DEFAULT_FONT_SIZE_BASE, DEFAULT_FONT_SIZE_STEP
    from .labelplus_pipeline import build_bt_from_labelplus_txt
    from .processor import CtdOverlayProcessor
    from .viewer_dialogs import compact_px, show_error_details, show_exception_details
    from .viewer_render_utils import qimage_size
except ImportError:
    from font_size_calibration import DEFAULT_FONT_SIZE_BASE, DEFAULT_FONT_SIZE_STEP
    from labelplus_pipeline import build_bt_from_labelplus_txt
    from processor import CtdOverlayProcessor
    from viewer_dialogs import compact_px, show_error_details, show_exception_details
    from viewer_render_utils import qimage_size


class ViewerWorkflowMixin:
    def _summary_text(self, summary: dict) -> str:
        if summary.get('has_overlay_data'):
            return (
                f'已載入 {summary["pages"]} 頁，'
                f'{summary["boxes"]} 個區塊，{summary["lines"]} 條文字行。'
            )
        issues = summary.get('overlay_data_issues') or []
        missing = summary.get('missing_required_data') or []
        issue_text = '、'.join(issues) if issues else '、'.join(Path(path).name for path in missing)
        return (
            f'已載入 {summary["source_images"]} 張原圖，但尚未找到完整 CTD 疊圖資料。'
            f'\n問題：{issue_text or "ctd 資料不完整"}'
            '\n請按「生成/更新 CTD」建立資料，或選擇已產生 ctd/ 的資料夾。'
        )

    def import_labelplus_txt(self) -> None:
        if self.processor is None:
            QMessageBox.information(self, '尚未選擇資料夾', '請先選擇包含原圖的圖片資料夾。')
            return
        if not self.processor.ctd_measure_path.is_file():
            QMessageBox.information(self, '缺少 measure.json', '請先生成 CTD，確保 ctd/measure.json 已存在。')
            return

        start_dir = self.current_image_dir or self._last_existing_image_dir() or str(Path.home())
        txt_path, _ = QFileDialog.getOpenFileName(
            self,
            '導入 LabelPlus txt',
            start_dir,
            'LabelPlus/Text (*.txt);;All Files (*)',
        )
        if not txt_path:
            return

        try:
            result = build_bt_from_labelplus_txt(
                txt_path,
                self.processor.ctd_measure_path,
                self.processor.image_dir,
            )
        except Exception as exc:
            show_exception_details(
                self,
                '導入失敗',
                '無法由 LabelPlus txt 生成 _meo_bt.json。下方是完整可複製的出錯信息。',
                exc,
            )
            return

        filtered_path = result.get('filtered_path')
        filtered_text = f'\n{filtered_path}' if filtered_path is not None else ''
        message = (
            '已生成：\n'
            f'{result["meo_path"]}'
            f'{filtered_text}\n'
            f'{result["bt_path"]}\n\n'
            f'頁數：{result["pages"]}，條目：{result["labels"]}\n'
            f'未匹配：{result["unmatched_pages"]} 頁，{result["unmatched_labels"]} 條'
        )
        self.status_label.setText(f'已生成 {Path(result["bt_path"]).name}')
        self.load_bt_json_path(Path(result['bt_path']))
        QMessageBox.information(self, '生成完成', message)

    def generate_ctd(self) -> None:
        if not self.current_image_dir:
            QMessageBox.information(self, '尚未選擇資料夾', '請先選擇包含原圖的圖片資料夾。')
            return
        if self.detect_process is not None or self.font_calibration_process is not None:
            QMessageBox.information(self, '正在處理', 'CTD 資料正在生成中，請稍候。')
            return

        script = Path(__file__).resolve().parent / 'run_detection.py'
        if not script.is_file():
            QMessageBox.critical(self, '找不到生成器', f'找不到：\n{script}')
            return

        calibration_settings = self._ask_generated_font_calibration_settings()
        if calibration_settings is None:
            return
        self.generated_default_font_size, self.generated_font_size_step = calibration_settings
        self.generated_font_size_method = (
            'ocr_aligned' if self.ocr_aligned_font_radio.isChecked() else 'char_box'
        )
        self._capture_generated_calibration_backup()

        self.generate_button.setEnabled(False)
        method_label = (
            'OCR 對齊逐字計算'
            if self.generated_font_size_method == 'ocr_aligned'
            else '單字框計算'
        )
        self.status_label.setText(
            f'正在生成 CTD 資料（{method_label}），'
            f'字級候選基準 {self.generated_default_font_size:.1f}、'
            f'Step {self.generated_font_size_step:.1f}...'
        )
        process = QProcess(self)
        args = [
            str(script),
            self.current_image_dir,
            '--font-size-calculation-method',
            self.generated_font_size_method,
            '--default-font-size',
            f'{self.generated_default_font_size:.1f}',
            '--font-size-step',
            f'{self.generated_font_size_step:.1f}',
        ]
        self.detect_command = [sys.executable, *args]
        self.detect_output_chunks = []
        process.setProgram(sys.executable)
        process.setArguments(args)
        process.setWorkingDirectory(str(Path(__file__).resolve().parents[1]))
        process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        process.readyReadStandardOutput.connect(self._read_detection_output)
        process.errorOccurred.connect(self._detection_process_error)
        process.finished.connect(self._detection_finished)
        self.detect_process = process
        process.start()

    def _ask_generated_font_calibration_settings(self) -> tuple[float, float] | None:
        dialog = QDialog(self)
        dialog.setWindowTitle('生成／更新 CTD 字級設定')
        layout = QVBoxLayout(dialog)
        if self.ocr_aligned_font_radio.isChecked():
            description_text = (
                'OCR 對齊逐字計算會先排除不可靠字元，再從「預設字級＋Step」形成的候選中選擇總誤差最低者。'
            )
        else:
            description_text = (
                '單字框計算沿用原有遮罩切框與段落字級估算，最後依「預設字級＋Step」選擇最接近的候選字級。'
            )
        description = QLabel(description_text)
        description.setWordWrap(True)
        layout.addWidget(description)

        default_row = QHBoxLayout()
        default_row.addWidget(QLabel('預設字級'))
        default_spin = QDoubleSpinBox()
        default_spin.setRange(0.1, 999.0)
        default_spin.setDecimals(1)
        default_spin.setSingleStep(0.1)
        try:
            saved_default = float(self.settings.value('font_calibration/default_font_size', DEFAULT_FONT_SIZE_BASE))
        except (TypeError, ValueError):
            saved_default = DEFAULT_FONT_SIZE_BASE
        default_spin.setValue(max(0.1, min(999.0, saved_default)))
        default_row.addWidget(default_spin, 1)
        layout.addLayout(default_row)

        step_row = QHBoxLayout()
        step_row.addWidget(QLabel('Step'))
        step_spin = QDoubleSpinBox()
        step_spin.setRange(0.1, 999.0)
        step_spin.setDecimals(1)
        step_spin.setSingleStep(0.1)
        try:
            saved_step = float(self.settings.value('font_calibration/font_size_step', DEFAULT_FONT_SIZE_STEP))
        except (TypeError, ValueError):
            saved_step = DEFAULT_FONT_SIZE_STEP
        step_spin.setValue(max(0.1, min(999.0, saved_step)))
        step_row.addWidget(step_spin, 1)
        layout.addLayout(step_row)

        preview = QLabel()
        preview.setWordWrap(True)

        def update_preview() -> None:
            base = float(default_spin.value())
            step = float(step_spin.value())
            preview.setText(
                '候選示例：'
                + '、'.join(f'{base + offset * step:.1f}' for offset in range(-2, 3) if base + offset * step > 0)
            )

        default_spin.valueChanged.connect(update_preview)
        step_spin.valueChanged.connect(update_preview)
        update_preview()
        layout.addWidget(preview)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText('開始生成')
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText('取消')
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None
        default_font_size = round(float(default_spin.value()), 1)
        font_size_step = round(float(step_spin.value()), 1)
        self.settings.setValue('font_calibration/default_font_size', default_font_size)
        self.settings.setValue('font_calibration/font_size_step', font_size_step)
        return default_font_size, font_size_step

    def _generated_calibration_paths(self) -> tuple[Path, ...]:
        if not self.current_image_dir:
            return ()
        ctd_dir = Path(self.current_image_dir) / 'ctd'
        return (
            ctd_dir / 'measure.json',
            ctd_dir / 'measure.debug.json',
            ctd_dir / 'measure_ocr.json',
        )

    def _capture_generated_calibration_backup(self) -> None:
        self.generated_calibration_backup = {
            path: path.read_bytes() if path.is_file() else None
            for path in self._generated_calibration_paths()
        }

    def _restore_generated_calibration_backup(self) -> None:
        backup = self.generated_calibration_backup
        self.generated_calibration_backup = None
        if backup is None:
            return
        for path, content in backup.items():
            if content is None:
                path.unlink(missing_ok=True)
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)

    def _clear_generated_calibration_backup(self) -> None:
        self.generated_calibration_backup = None

    def _read_detection_output(self) -> None:
        if self.detect_process is None:
            return
        text = bytes(self.detect_process.readAllStandardOutput()).decode('utf-8', errors='replace')
        if text:
            self.detect_output_chunks.append(text)
            lines = text.strip().splitlines()
            if lines:
                self.status_label.setText(lines[-1])

    def _detection_process_error(self, error) -> None:
        process = self.detect_process
        command_text = ' '.join(self.detect_command)
        details = (
            '【命令】\n'
            f'{command_text}\n\n'
            '【工作目錄】\n'
            f'{Path(__file__).resolve().parents[1]}\n\n'
            '【Python】\n'
            f'{sys.executable}\n\n'
            '【QProcess 錯誤】\n'
            f'{error}\n\n'
            '【完整輸出】\n'
            f'{"".join(self.detect_output_chunks).strip() or "(沒有輸出)"}'
        )
        show_error_details(
            self,
            '生成 CTD 失敗',
            '生成程序無法啟動或執行中斷。下方是完整可複製的出錯信息。',
            details,
        )
        self.generate_button.setEnabled(True)
        self.status_label.setText('生成 CTD 失敗。')
        self._restore_generated_calibration_backup()
        self.detect_process = None
        if process is not None:
            process.deleteLater()

    def _detection_finished(self, exit_code: int, exit_status) -> None:
        process = self.detect_process
        self.detect_process = None
        if process is None:
            return
        trailing_output = bytes(process.readAllStandardOutput()).decode('utf-8', errors='replace')
        if trailing_output:
            self.detect_output_chunks.append(trailing_output)
        output = ''.join(self.detect_output_chunks).strip()
        if exit_code != 0:
            command_text = ' '.join(self.detect_command)
            details = (
                '【命令】\n'
                f'{command_text}\n\n'
                '【工作目錄】\n'
                f'{Path(__file__).resolve().parents[1]}\n\n'
                '【Python】\n'
                f'{sys.executable}\n\n'
                '【退出碼】\n'
                f'{exit_code}\n\n'
                '【完整輸出】\n'
                f'{output or "(沒有輸出)"}'
            )
            show_error_details(
                self,
                '生成 CTD 失敗',
                f'生成失敗，退出碼：{exit_code}。\n下方是完整可複製的出錯信息。',
                details,
            )
            self.status_label.setText('生成 CTD 失敗。')
            self.generate_button.setEnabled(True)
            self._restore_generated_calibration_backup()
            return
        if self.generated_font_size_method == 'ocr_aligned':
            self.status_label.setText('CTD 資料生成完成，正在啟動 mit48 字級校準...')
            self._start_generated_font_calibration()
            return
        self._clear_generated_calibration_backup()
        self.generate_button.setEnabled(True)
        self.status_label.setText('CTD 與單字框字級計算完成，正在重新載入...')
        if self.current_image_dir:
            self.load_folder(self.current_image_dir)

    def _start_generated_font_calibration(self) -> None:
        if not self.current_image_dir:
            self.generate_button.setEnabled(True)
            self._restore_generated_calibration_backup()
            return
        project_root = Path(__file__).resolve().parents[1]
        script = project_root / 'measure_ocr.py'
        if not script.is_file():
            self.generate_button.setEnabled(True)
            self._restore_generated_calibration_backup()
            QMessageBox.critical(self, '找不到 OCR 腳本', f'找不到：\n{script}')
            return
        device = str(self.auto_calibrate_device_combo.currentData() or 'cpu')
        args = [
            '-u',
            str(script),
            self.current_image_dir,
            '--device',
            device,
            '--apply-font-sizes',
            '--default-font-size',
            f'{self.generated_default_font_size:.1f}',
            '--font-size-step',
            f'{self.generated_font_size_step:.1f}',
        ]
        process = QProcess(self)
        process.setProgram(sys.executable)
        process.setArguments(args)
        process.setWorkingDirectory(str(project_root))
        process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        process.readyReadStandardOutput.connect(self._read_font_calibration_output)
        process.errorOccurred.connect(self._font_calibration_process_error)
        process.finished.connect(self._font_calibration_finished)
        self.font_calibration_process = process
        self.font_calibration_command = [sys.executable, *args]
        self.font_calibration_output_chunks = []
        progress_text = f'字級 OCR：正在以 {device.upper()} 載入 mit48 模型...'
        self.status_label.setText(progress_text)
        self.font_calibration_progress_label.setText(progress_text)
        self.font_calibration_progress_label.show()
        process.start()

    def _read_font_calibration_output(self) -> None:
        if self.font_calibration_process is None:
            return
        text = bytes(self.font_calibration_process.readAllStandardOutput()).decode('utf-8', errors='replace')
        if not text:
            return
        self.font_calibration_output_chunks.append(text)
        lines = text.strip().splitlines()
        if lines:
            progress_text = lines[-1]
            self.status_label.setText(progress_text)
            self.font_calibration_progress_label.setText(progress_text)
            self.font_calibration_progress_label.show()

    def _font_calibration_process_error(self, error) -> None:
        process = self.font_calibration_process
        command_text = ' '.join(self.font_calibration_command)
        details = (
            '【命令】\n'
            f'{command_text}\n\n'
            '【QProcess 錯誤】\n'
            f'{error}\n\n'
            '【完整輸出】\n'
            f'{"".join(self.font_calibration_output_chunks).strip() or "(沒有輸出)"}'
        )
        show_error_details(
            self,
            '自動字級校準失敗',
            'CTD 已生成，但 mit48 字級校準程序中斷。可稍後在 measure 編輯器中重新執行。',
            details,
        )
        self.font_calibration_process = None
        self.generate_button.setEnabled(True)
        self.font_calibration_progress_label.hide()
        self._restore_generated_calibration_backup()
        self.status_label.setText('OCR 字級校準失敗，已恢復執行前的字級結果。')
        if process is not None:
            process.deleteLater()
        if self.current_image_dir:
            self.load_folder(self.current_image_dir)

    def _font_calibration_finished(self, exit_code: int, exit_status) -> None:
        process = self.font_calibration_process
        self.font_calibration_process = None
        if process is None:
            return
        trailing_output = bytes(process.readAllStandardOutput()).decode('utf-8', errors='replace')
        if trailing_output:
            self.font_calibration_output_chunks.append(trailing_output)
        process.deleteLater()
        if exit_code != 0:
            details = (
                '【命令】\n'
                f'{" ".join(self.font_calibration_command)}\n\n'
                '【退出碼】\n'
                f'{exit_code}\n\n'
                '【完整輸出】\n'
                f'{"".join(self.font_calibration_output_chunks).strip() or "(沒有輸出)"}'
            )
            show_error_details(
                self,
                '自動字級校準失敗',
                'CTD 已生成，但 mit48 字級校準失敗。可稍後在 measure 編輯器中重新執行。',
                details,
            )
            self._restore_generated_calibration_backup()
            self.status_label.setText('OCR 字級校準失敗，已恢復執行前的字級結果。')
        else:
            self._clear_generated_calibration_backup()
            self.status_label.setText('CTD 與字型緩存字級校準完成，正在重新載入...')
        self.generate_button.setEnabled(True)
        self.font_calibration_progress_label.hide()
        if self.current_image_dir:
            self.load_folder(self.current_image_dir)

    def load_page_at_row(self, row: int) -> None:
        if row < 0 or row >= len(self.page_names) or self.processor is None:
            return
        self._commit_bt_editor_preview()
        page_name = self.page_names[row]
        try:
            self._clear_bt_image_cache()
            self.clear_hover_char_box(render=False)
            self.selected_box_index = None
            self.selected_bt_index = None
            self.selected_bt_indices.clear()
            self.show_bt_inpainted = True
            self.set_box_editor_enabled(False)
            image_size = qimage_size(self.processor.image_dir / page_name)
            self._bt_cached_image_size = image_size
            self.page = self.processor.load_page(page_name, image_size=image_size)
            self.current_page_row = row
            self.update_font_size_list()
            self.render_current_page(refit=False)
            self.select_bt_item(None)
            self.update_bt_item_list()
            self.render_bt_page(refit=False)
            self.fit_both_views()
            self.schedule_bt_html_overlay_update()
        except Exception as exc:
            show_exception_details(self, '頁面載入失敗', '無法載入此頁。下方是完整可複製的出錯信息。', exc)

    def update_hover_char_box(self, x: float, y: float) -> None:
        if self.page is None or not self.show_char_boxes.isChecked():
            self.clear_hover_char_box()
            return

        matches = []
        for item in self.page.char_boxes:
            bbox = item.get('bbox')
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = [float(value) for value in bbox]
            if x1 <= x <= x2 and y1 <= y <= y2:
                matches.append((max(0.0, x2 - x1) * max(0.0, y2 - y1), item))

        if not matches:
            self.clear_hover_char_box()
            return

        item = min(matches, key=lambda entry: entry[0])[1]
        if item is self.hover_char_box:
            return
        self.hover_char_box = item
        self.char_info_label.setText(self._char_info_text(item))
        self.update_popover_with_char_box(item)
        self.render_current_page(refit=False)

    def clear_hover_char_box(self, render: bool = True) -> None:
        had_hover = self.hover_char_box is not None
        self.hover_char_box = None
        self.char_info_label.setText('游標單字框：未選中')
        if self._popover_bt_item is not None and self.bt_match_popover.isVisible():
            self.show_bt_match_popover(self._popover_bt_item)
        elif self.bt_match_popover.isVisible():
            self.bt_match_popover.hide()
        if render and had_hover:
            self.render_current_page(refit=False)

    def _char_info_text(self, item: dict) -> str:
        bbox = item.get('bbox')
        width_text = compact_px(item.get('width')) or '-'
        height_text = compact_px(item.get('height')) or '-'
        source_index = item.get('source_block_index', '-')
        line_index = item.get('line_index', '-')
        bbox_text = ', '.join(str(int(round(float(value)))) for value in bbox) if isinstance(bbox, list) else '-'
        try:
            estimated_font_size = float(item.get('estimated_font_size', item.get('calculated_font_size')))
            font_size_text = f'{estimated_font_size:.1f}' if estimated_font_size > 0 else '-'
        except (TypeError, ValueError):
            font_size_text = '-'
        return (
            '游標單字框：\n'
            f'寬：{width_text}px  高：{height_text}px  單字字級：{font_size_text}px\n'
            f'區塊：{source_index}  行：{line_index}\n'
            f'bbox：{bbox_text}'
        )

    def reload_current_page(self) -> None:
        row = self.page_list.currentRow()
        if row >= 0:
            self.load_page_at_row(row)

    def open_measure_editor(self) -> None:
        if self.processor is None:
            QMessageBox.information(self, '尚未載入資料夾', '請先選擇包含 ctd/measure.json 的圖片資料夾。')
            return
        if not self.processor.measure_path.is_file():
            QMessageBox.information(self, '缺少 measure.json', '請先生成 CTD，確保 ctd/measure.json 已存在。')
            return
        try:
            from .measure_editor import MeasureEditorWindow
        except ImportError:
            from measure_editor import MeasureEditorWindow
        editor = MeasureEditorWindow(self.processor, self.current_page_name(), self)
        editor.saved.connect(self.handle_measure_editor_saved)
        editor.destroyed.connect(lambda _=None, editor=editor: self.forget_measure_editor(editor))
        self.measure_editor_windows.append(editor)
        editor.show()

    def forget_measure_editor(self, editor: QMainWindow) -> None:
        if editor in self.measure_editor_windows:
            self.measure_editor_windows.remove(editor)

    def handle_measure_editor_saved(self) -> None:
        if self.current_image_dir:
            try:
                current_name = self.current_page_name()
                self.processor = CtdOverlayProcessor(self.current_image_dir)
                self.page_names = self.processor.page_names()
                if current_name in self.page_names:
                    self.load_page_at_row(self.page_names.index(current_name))
                elif self.page_names:
                    self.load_page_at_row(max(0, min(self.current_page_row, len(self.page_names) - 1)))
            except Exception as exc:
                show_exception_details(self, '重新載入失敗', 'measure.json 已保存，但主視窗重新載入失敗。', exc)
