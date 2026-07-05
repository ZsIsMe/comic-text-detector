#!/usr/bin/env python3
"""OCR progress dialog for ctd/measure.json boxes."""

from __future__ import annotations

import re
import sys
from pathlib import Path

from PySide6.QtCore import QProcess, QTimer, Signal
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)

try:
    from .processor import CtdOverlayProcessor
except ImportError:
    from processor import CtdOverlayProcessor


class MeasureOcrDialog(QDialog):
    completed = Signal(str)

    def __init__(
        self,
        processor: CtdOverlayProcessor,
        current_page_name: str | None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle('OCR 圖片文本')
        self.resize(760, 540)
        self.processor = processor
        self.current_page_name = current_page_name
        self.process: QProcess | None = None
        self.output_chunks: list[str] = []
        self.command: list[str] = []
        self._stopping = False

        self.scope_combo = QComboBox()
        if current_page_name:
            self.scope_combo.addItem(f'當前頁：{current_page_name}', 'current')
        self.scope_combo.addItem('全部頁面', 'all')
        self.device_combo = QComboBox()
        self.device_combo.addItem('MPS（Apple GPU）', 'mps')
        self.device_combo.addItem('CPU（相容）', 'cpu')
        self.device_combo.addItem('CUDA', 'cuda')
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.status_label = QLabel('尚未開始 OCR。')
        self.status_label.setWordWrap(True)
        self.command_label = QLabel('')
        self.command_label.setWordWrap(True)
        self.log_edit = QPlainTextEdit()
        self.log_edit.setReadOnly(True)
        self.start_button = QPushButton('開始 OCR')
        self.stop_button = QPushButton('停止 OCR')
        self.close_button = QPushButton('關閉')
        self.stop_button.setEnabled(False)

        self._build_ui()
        self.start_button.clicked.connect(self.start_ocr)
        self.stop_button.clicked.connect(self.stop_ocr)
        self.close_button.clicked.connect(self.close)

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

        layout.addWidget(self.progress_bar)
        layout.addWidget(self.status_label)
        layout.addWidget(self.command_label)
        layout.addWidget(self.log_edit, 1)

        button_row = QHBoxLayout()
        button_row.addStretch(1)
        button_row.addWidget(self.start_button)
        button_row.addWidget(self.stop_button)
        button_row.addWidget(self.close_button)
        layout.addLayout(button_row)

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

        args = ['-u', str(script), str(self.processor.image_dir)]
        if self.scope_combo.currentData() == 'current' and self.current_page_name:
            args.extend(['--page', self.current_page_name])
        args.extend(['--device', self.device_combo.currentData() or 'mps'])
        python = self._ocr_python(project_root)
        self.command = [str(python), *args]
        self.output_chunks = []
        self._stopping = False
        self.progress_bar.setRange(0, 0)
        self.status_label.setText('正在啟動 OCR 模型...')
        self.command_label.setText('命令：' + ' '.join(self.command))
        self.log_edit.clear()

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
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(1)
        self.status_label.setText(f'OCR 完成：{output_path}')
        self.completed.emit(str(output_path))

    def _reset_after_finish(self, message: str) -> None:
        process = self.process
        self.process = None
        if process is not None:
            process.deleteLater()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.close_button.setEnabled(True)
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
