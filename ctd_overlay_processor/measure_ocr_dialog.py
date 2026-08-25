#!/usr/bin/env python3
"""OCR progress dialog for ctd/measure.json boxes."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from PySide6.QtCore import QProcess, QSettings, QTimer, Qt, Signal
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

try:
    from .font_size_calibration import (
        DEFAULT_FONT_SIZE_BASE,
        DEFAULT_FONT_SIZE_STEP,
        calibrate_ocr_output,
    )
    from .processor import CtdOverlayProcessor
except ImportError:
    from font_size_calibration import (
        DEFAULT_FONT_SIZE_BASE,
        DEFAULT_FONT_SIZE_STEP,
        calibrate_ocr_output,
    )
    from processor import CtdOverlayProcessor


class MeasureOcrDialog(QDialog):
    completed = Signal(str)
    applyRequested = Signal(object)

    def __init__(
        self,
        processor: CtdOverlayProcessor,
        current_page_name: str | None,
        current_source_block_index: int | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle('逐字 OCR 校準字級')
        self.resize(760, 540)
        self.processor = processor
        self.settings = QSettings('comic-text-detector', 'ctd-overlay-processor')
        self.current_page_name = current_page_name
        self.current_source_block_index = current_source_block_index
        self.process: QProcess | None = None
        self.output_chunks: list[str] = []
        self.command: list[str] = []
        self._stopping = False
        self.calibrated_output: dict = {}
        self.proposed_updates: dict[str, dict[int, float]] = {}
        self.active_default_font_size = DEFAULT_FONT_SIZE_BASE
        self.active_font_size_step = DEFAULT_FONT_SIZE_STEP

        self.scope_combo = QComboBox()
        if current_page_name and current_source_block_index is not None:
            self.scope_combo.addItem(
                f'當前文字框：{current_page_name} / block {current_source_block_index}',
                'item',
            )
        if current_page_name:
            self.scope_combo.addItem(f'當前頁：{current_page_name}', 'current')
        self.scope_combo.addItem('全部頁面', 'all')
        self.device_combo = QComboBox()
        self.device_combo.addItem('CPU（目前可用）', 'cpu')
        self.device_combo.addItem('MPS（不可用時自動改 CPU）', 'mps')
        self.device_combo.addItem('CUDA', 'cuda')
        self.default_font_size_spin = QDoubleSpinBox()
        self.default_font_size_spin.setRange(0.1, 999.0)
        self.default_font_size_spin.setDecimals(1)
        self.default_font_size_spin.setSingleStep(0.1)
        self.font_size_step_spin = QDoubleSpinBox()
        self.font_size_step_spin.setRange(0.1, 999.0)
        self.font_size_step_spin.setDecimals(1)
        self.font_size_step_spin.setSingleStep(0.1)
        try:
            saved_default = float(self.settings.value('font_calibration/default_font_size', DEFAULT_FONT_SIZE_BASE))
        except (TypeError, ValueError):
            saved_default = DEFAULT_FONT_SIZE_BASE
        try:
            saved_step = float(self.settings.value('font_calibration/font_size_step', DEFAULT_FONT_SIZE_STEP))
        except (TypeError, ValueError):
            saved_step = DEFAULT_FONT_SIZE_STEP
        self.default_font_size_spin.setValue(max(0.1, min(999.0, saved_default)))
        self.font_size_step_spin.setValue(max(0.1, min(999.0, saved_step)))
        self.candidate_preview_label = QLabel()
        self.candidate_preview_label.setWordWrap(True)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.status_label = QLabel('尚未開始逐字 OCR。整行識別後逐字獨立定位，失敗不影響同行其他字。')
        self.status_label.setWordWrap(True)
        self.command_label = QLabel('')
        self.command_label.setWordWrap(True)
        self.log_edit = QPlainTextEdit()
        self.log_edit.setReadOnly(True)
        self.result_table = QTableWidget()
        self.result_table.setColumnCount(7)
        self.result_table.setHorizontalHeaderLabels(
            ['頁面', 'Block', 'OCR', '原字級', '建議', '有效字', '狀態'],
        )
        self.result_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.result_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.result_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.result_table.horizontalHeader().setStretchLastSection(True)
        self.result_table.hide()
        self.start_button = QPushButton('開始逐字 OCR')
        self.stop_button = QPushButton('停止 OCR')
        self.apply_button = QPushButton('套用可靠字級')
        self.close_button = QPushButton('關閉')
        self.stop_button.setEnabled(False)
        self.apply_button.setEnabled(False)

        self._build_ui()
        self.start_button.clicked.connect(self.start_ocr)
        self.stop_button.clicked.connect(self.stop_ocr)
        self.apply_button.clicked.connect(self.apply_results)
        self.close_button.clicked.connect(self.close)
        self.default_font_size_spin.valueChanged.connect(self._update_candidate_preview)
        self.font_size_step_spin.valueChanged.connect(self._update_candidate_preview)
        self._update_candidate_preview()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        scope_row = QHBoxLayout()
        scope_row.addWidget(QLabel('範圍'))
        scope_row.addWidget(self.scope_combo, 1)
        layout.addLayout(scope_row)

        device_row = QHBoxLayout()
        device_row.addWidget(QLabel('設備'))
        device_row.addWidget(self.device_combo, 1)
        layout.addLayout(device_row)

        calibration_row = QHBoxLayout()
        calibration_row.addWidget(QLabel('預設字級'))
        calibration_row.addWidget(self.default_font_size_spin, 1)
        calibration_row.addWidget(QLabel('Step'))
        calibration_row.addWidget(self.font_size_step_spin, 1)
        layout.addLayout(calibration_row)
        layout.addWidget(self.candidate_preview_label)

        layout.addWidget(self.progress_bar)
        layout.addWidget(self.status_label)
        layout.addWidget(self.command_label)
        layout.addWidget(self.log_edit, 1)
        layout.addWidget(self.result_table, 1)

        button_row = QHBoxLayout()
        button_row.addStretch(1)
        button_row.addWidget(self.start_button)
        button_row.addWidget(self.stop_button)
        button_row.addWidget(self.apply_button)
        button_row.addWidget(self.close_button)
        layout.addLayout(button_row)

    def _update_candidate_preview(self) -> None:
        base = float(self.default_font_size_spin.value())
        step = float(self.font_size_step_spin.value())
        candidates = [
            base + offset * step
            for offset in range(-2, 3)
            if base + offset * step > 0
        ]
        self.candidate_preview_label.setText(
            '候選示例：' + '、'.join(f'{candidate:.1f}' for candidate in candidates),
        )

    def start_ocr(self) -> None:
        if self.process is not None:
            return
        project_root = Path(__file__).resolve().parents[1]
        script = project_root / 'measure_ocr.py'
        if not script.is_file():
            QMessageBox.critical(self, '找不到 OCR 腳本', f'找不到：\n{script}')
            return
        if not self.processor.measure_path.is_file():
            QMessageBox.information(self, '缺少 measure.json', '請先生成或保存 ctd/measure.json。')
            return

        self.active_default_font_size = round(float(self.default_font_size_spin.value()), 1)
        self.active_font_size_step = round(float(self.font_size_step_spin.value()), 1)
        self.settings.setValue('font_calibration/default_font_size', self.active_default_font_size)
        self.settings.setValue('font_calibration/font_size_step', self.active_font_size_step)

        args = ['-u', str(script), str(self.processor.image_dir)]
        scope = self.scope_combo.currentData()
        if scope in {'item', 'current'} and self.current_page_name:
            args.extend(['--page', self.current_page_name])
        if scope == 'item' and self.current_source_block_index is not None:
            args.extend(['--source-block-index', str(self.current_source_block_index)])
        args.extend([
            '--device',
            self.device_combo.currentData() or 'cpu',
            '--default-font-size',
            f'{self.active_default_font_size:.1f}',
            '--font-size-step',
            f'{self.active_font_size_step:.1f}',
        ])
        python = self._ocr_python(project_root)
        self.command = [str(python), *args]
        self.output_chunks = []
        self._stopping = False
        self.progress_bar.setRange(0, 0)
        self.status_label.setText('正在啟動 mit48px CTC 逐字 OCR 模型...')
        self.command_label.setText('命令：' + ' '.join(self.command))
        self.log_edit.clear()
        self.result_table.clearContents()
        self.result_table.setRowCount(0)
        self.result_table.hide()
        self.apply_button.setEnabled(False)
        self.calibrated_output = {}
        self.proposed_updates = {}

        process = QProcess(self)
        process.setProgram(str(python))
        process.setArguments(args)
        process.setWorkingDirectory(str(project_root))
        process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        process.readyReadStandardOutput.connect(self._read_output)
        process.errorOccurred.connect(self._process_error)
        process.finished.connect(self._process_finished)
        self.process = process
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.close_button.setEnabled(False)
        self.default_font_size_spin.setEnabled(False)
        self.font_size_step_spin.setEnabled(False)
        process.start()

    def _ocr_python(self, project_root: Path) -> Path:
        venv_python = project_root / '.venv' / 'bin' / 'python'
        if venv_python.is_file():
            return venv_python
        return Path(sys.executable)

    def stop_ocr(self) -> None:
        if self.process is None:
            return
        self._stopping = True
        self.status_label.setText('正在停止 OCR...')
        self.stop_button.setEnabled(False)
        self.process.terminate()
        QTimer.singleShot(3000, self._kill_if_running)

    def _kill_if_running(self) -> None:
        if self.process is not None and self.process.state() != QProcess.ProcessState.NotRunning:
            self.process.kill()

    def _read_output(self) -> None:
        if self.process is None:
            return
        text = bytes(self.process.readAllStandardOutput()).decode('utf-8', errors='replace')
        if not text:
            return
        self.output_chunks.append(text)
        self.log_edit.moveCursor(QTextCursor.MoveOperation.End)
        self.log_edit.insertPlainText(text)
        self.log_edit.moveCursor(QTextCursor.MoveOperation.End)
        for line in text.splitlines():
            self._update_progress_from_line(line)

    def _update_progress_from_line(self, line: str) -> None:
        match = re.search(r'\[(\d+)/(\d+)\]\s+(OCR|完成)\s+(.+)', line)
        if not match:
            if line.strip():
                self.status_label.setText(line.strip())
            return
        page_index = int(match.group(1))
        total_pages = int(match.group(2))
        phase = match.group(3)
        detail = match.group(4).strip()
        self.progress_bar.setRange(0, total_pages)
        value = page_index if phase == '完成' else max(0, page_index - 1)
        self.progress_bar.setValue(value)
        self.status_label.setText(f'{phase} {page_index}/{total_pages}：{detail}')

    def _process_error(self, error) -> None:
        details = self._details_text(f'QProcess 錯誤：{error}')
        QMessageBox.critical(self, 'OCR 啟動失敗', details)
        self._reset_after_finish('OCR 啟動失敗。')

    def _process_finished(self, exit_code: int, exit_status) -> None:
        process = self.process
        if process is not None:
            trailing = bytes(process.readAllStandardOutput()).decode('utf-8', errors='replace')
            if trailing:
                self.output_chunks.append(trailing)
                self.log_edit.moveCursor(QTextCursor.MoveOperation.End)
                self.log_edit.insertPlainText(trailing)
                self.log_edit.moveCursor(QTextCursor.MoveOperation.End)
                for line in trailing.splitlines():
                    self._update_progress_from_line(line)
            process.deleteLater()
        self.process = None
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.close_button.setEnabled(True)
        self.default_font_size_spin.setEnabled(True)
        self.font_size_step_spin.setEnabled(True)

        if self._stopping:
            self.status_label.setText('OCR 已停止。')
            return
        if exit_code != 0:
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(0)
            self.status_label.setText(f'OCR 失敗，退出碼：{exit_code}。')
            QMessageBox.critical(self, 'OCR 失敗', self._details_text(f'退出碼：{exit_code}'))
            return

        output_path = self.processor.ctd_dir / 'measure_ocr.json'
        try:
            output = json.loads(output_path.read_text(encoding='utf-8'))
            ready_count = calibrate_ocr_output(
                output,
                default_font_size=self.active_default_font_size,
                font_size_step=self.active_font_size_step,
            )
            output_path.write_text(
                json.dumps(output, ensure_ascii=False, indent=2) + '\n',
                encoding='utf-8',
            )
            self.calibrated_output = output
            self._show_calibration_results(output)
        except Exception as exc:
            self.status_label.setText(f'OCR 完成，但字級校準失敗：{exc}')
            QMessageBox.warning(self, '字級校準失敗', str(exc))
            self.completed.emit(str(output_path))
            return
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(1)
        self.status_label.setText(f'OCR 完成，可套用 {ready_count} 個可靠字級：{output_path}')
        self.completed.emit(str(output_path))

    def _show_calibration_results(self, output: dict) -> None:
        status_text = {
            'ready': '可套用',
            'no_reliable_characters': '沒有可靠字元',
            'too_few_reliable_characters': '可靠字元不足',
            'suggestion_too_far_from_detected': '與檢測字級差距過大',
        }
        rows = []
        updates: dict[str, dict[int, float]] = {}
        for page_name, items in (output.get('pages') or {}).items():
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                fit = item.get('font_fit') or {}
                status = str(fit.get('status') or 'unknown')
                suggested = fit.get('suggested_font_size_float', fit.get('suggested_font_size'))
                item_index = item.get('measure_item_index')
                if (
                    status == 'ready'
                    and isinstance(suggested, (int, float))
                    and not isinstance(suggested, bool)
                    and isinstance(item_index, int)
                ):
                    updates.setdefault(str(page_name), {})[item_index] = round(float(suggested), 1)
                rows.append((page_name, item, fit, status_text.get(status, status)))

        self.proposed_updates = updates
        self.result_table.setRowCount(len(rows))
        for row, (page_name, item, fit, display_status) in enumerate(rows):
            values = [
                page_name,
                str(item.get('source_block_index', '-')),
                str(item.get('ocr_text') or '').replace('\n', ' / '),
                f'{float(fit.get("original_font_size") or item.get("font_size") or 0):.1f}',
                (
                    f'{float(fit.get("suggested_font_size_float")):.1f}'
                    if fit.get('suggested_font_size_float') is not None
                    else '-'
                ),
                str(fit.get('accepted_character_count') or 0),
                display_status,
            ]
            for column, value in enumerate(values):
                table_item = QTableWidgetItem(value)
                if column in {1, 3, 4, 5}:
                    table_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.result_table.setItem(row, column, table_item)
        self.log_edit.hide()
        self.command_label.hide()
        self.result_table.show()
        self.apply_button.setEnabled(any(updates.values()))

    def apply_results(self) -> None:
        if not any(self.proposed_updates.values()):
            return
        self.applyRequested.emit(self.proposed_updates)
        self.apply_button.setEnabled(False)
        self.status_label.setText('可靠字級已套用到編輯器，請在主視窗預覽並保存。')

    def _reset_after_finish(self, message: str) -> None:
        process = self.process
        self.process = None
        if process is not None:
            process.deleteLater()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.close_button.setEnabled(True)
        self.default_font_size_spin.setEnabled(True)
        self.font_size_step_spin.setEnabled(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.status_label.setText(message)

    def _details_text(self, summary: str) -> str:
        return (
            f'{summary}\n\n'
            '命令：\n'
            f'{" ".join(self.command) if self.command else "(尚未建立命令)"}\n\n'
            '輸出：\n'
            f'{"".join(self.output_chunks).strip() or "(沒有輸出)"}'
        )

    def closeEvent(self, event) -> None:
        if self.process is None:
            event.accept()
            return
        result = QMessageBox.question(
            self,
            'OCR 尚在運行',
            'OCR 尚在運行，要停止後關閉嗎？',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if result != QMessageBox.StandardButton.Yes:
            event.ignore()
            return
        self.stop_ocr()
        event.ignore()
