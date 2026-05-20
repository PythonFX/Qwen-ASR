#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PyQt5 GUI for SRT subtitle generation.

This file ONLY handles UI state and user interactions.  All business logic
lives in asr_service.py — the GUI never imports audio / asr / pipeline
modules directly.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from typing import Optional

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from asr_service import ASRService

_SETTINGS_PATH = Path.home() / ".qwen_asr_gui.json"
_MAX_RECENT_CODES = 20


# ======================================================================
# Worker thread — bridges ASRService into the Qt signal/slot world
# ======================================================================


class ASRWorker(QThread):
    progress = pyqtSignal(str, float)  # (stage, percent 0-100)
    finished = pyqtSignal(str)  # output srt path
    error = pyqtSignal(str)  # error message

    def __init__(
        self,
        service: ASRService,
        root_dir: str,
        code: str,
        options: dict | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.service = service
        self.root_dir = root_dir
        self.code = code
        self.options = options or {}

    def run(self) -> None:
        try:
            video_path, srt_path = self.service.find_video_by_code(
                self.root_dir, self.code,
            )
            self.service.generate_srt(
                video_path,
                srt_path,
                progress_callback=self._emit_progress,
                **self.options,
            )
            self.finished.emit(str(srt_path))
        except Exception as exc:
            self.error.emit(str(exc))

    def _emit_progress(self, stage: str, percent: float) -> None:
        self.progress.emit(stage, percent)


# ======================================================================
# Main window
# ======================================================================


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.service = ASRService()
        self.worker: Optional[ASRWorker] = None
        self.queue: list[dict] = []
        self._task_start: float = 0.0
        self._init_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _init_ui(self) -> None:
        self.setWindowTitle("SRT字幕生成器")
        self.setMinimumSize(640, 460)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(10)

        # --- Row 1: root directory ---
        row1 = QHBoxLayout()
        lbl1 = QLabel("根目录:")
        lbl1.setFixedWidth(50)
        row1.addWidget(lbl1)
        self.root_dir_input = QLineEdit("/Volumes/XSK/-素人-/")
        self.root_dir_input.setPlaceholderText("包含番号文件夹的根目录")
        row1.addWidget(self.root_dir_input, stretch=1)
        browse_btn = QPushButton("浏览")
        browse_btn.setFixedWidth(60)
        browse_btn.clicked.connect(self._browse_dir)
        row1.addWidget(browse_btn)
        root.addLayout(row1)

        # --- Row 2: code ---
        row2 = QHBoxLayout()
        lbl2 = QLabel("番号:")
        lbl2.setFixedWidth(50)
        row2.addWidget(lbl2)
        self.code_input = QComboBox()
        self.code_input.setEditable(True)
        self.code_input.setFixedWidth(180)
        self.code_input.setPlaceholderText("如 HUNTC-377")
        self.code_input.lineEdit().returnPressed.connect(self._on_generate)
        row2.addWidget(self.code_input)
        row2.addStretch()
        root.addLayout(row2)

        # --- Row 2b: engine + options ---
        row_opt = QHBoxLayout()
        row_opt.addWidget(QLabel("ASR引擎:"))
        self.engine_combo = QComboBox()
        self.engine_combo.addItem("Qwen", "qwen")
        self.engine_combo.addItem("Parakeet", "parakeet")
        self.engine_combo.setFixedWidth(120)
        self.engine_combo.currentIndexChanged.connect(self._on_engine_changed)
        row_opt.addWidget(self.engine_combo)
        row_opt.addSpacing(20)
        self.chk_keep_middle = QCheckBox("保留中间文件")
        self.chk_no_vad = QCheckBox("禁用VAD")
        self.chk_builtin_ts = QCheckBox("内置时间戳")
        self._on_engine_changed()  # set initial state
        row_opt.addWidget(self.chk_keep_middle)
        row_opt.addWidget(self.chk_no_vad)
        row_opt.addWidget(self.chk_builtin_ts)
        row_opt.addStretch()
        root.addLayout(row_opt)

        # --- Row 2c: numeric options ---
        row_params = QHBoxLayout()

        row_params.addWidget(QLabel("语言:"))
        self.lang_combo = QComboBox()
        self.lang_combo.addItems(["Japanese", "Chinese", "English", "Korean", "Cantonese"])
        self.lang_combo.setFixedWidth(100)
        row_params.addWidget(self.lang_combo)

        row_params.addSpacing(15)
        row_params.addWidget(QLabel("Workers:"))
        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(1, 2)
        self.workers_spin.setValue(1)
        self.workers_spin.setFixedWidth(60)
        row_params.addWidget(self.workers_spin)

        row_params.addSpacing(15)
        row_params.addWidget(QLabel("断句阈值:"))
        self.pause_spin = QDoubleSpinBox()
        self.pause_spin.setRange(0.1, 3.0)
        self.pause_spin.setValue(0.65)
        self.pause_spin.setSingleStep(0.05)
        self.pause_spin.setFixedWidth(80)
        self.pause_spin.setSuffix("s")
        row_params.addWidget(self.pause_spin)

        row_params.addSpacing(15)
        row_params.addWidget(QLabel("字幕延时:"))
        self.delay_spin = QDoubleSpinBox()
        self.delay_spin.setRange(0.0, 3.0)
        self.delay_spin.setValue(0.5)
        self.delay_spin.setSingleStep(0.1)
        self.delay_spin.setFixedWidth(80)
        self.delay_spin.setSuffix("s")
        row_params.addWidget(self.delay_spin)

        row_params.addStretch()
        root.addLayout(row_params)

        # --- Row 3: progress bar ---
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(28)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p%  %v")
        root.addWidget(self.progress_bar)

        # --- Row 4: status ---
        self.status_label = QLabel("就绪")
        root.addWidget(self.status_label)

        # --- Row 5: buttons ---
        row5 = QHBoxLayout()
        self.generate_btn = QPushButton("生成字幕")
        self.generate_btn.setFixedWidth(100)
        self.generate_btn.clicked.connect(self._on_generate)
        row5.addWidget(self.generate_btn)

        self.queue_btn = QPushButton("排队")
        self.queue_btn.setFixedWidth(80)
        self.queue_btn.clicked.connect(self._on_queue)
        row5.addWidget(self.queue_btn)
        row5.addStretch()
        root.addLayout(row5)

        # --- Log ---
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setFixedHeight(120)
        root.addWidget(self.log_box)

        # --- Task queue (compact, at bottom) ---
        row_q = QHBoxLayout()
        row_q.addWidget(QLabel("队列:"))
        self.queue_list = QListWidget()
        self.queue_list.setFixedHeight(64)
        row_q.addWidget(self.queue_list, stretch=1)
        root.addLayout(row_q)

        self._load_settings()

    # ------------------------------------------------------------------
    # Settings persistence
    # ------------------------------------------------------------------

    def _load_settings(self) -> None:
        try:
            data = json.loads(_SETTINGS_PATH.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return

        if data.get("root_dir"):
            self.root_dir_input.setText(data["root_dir"])
        for code in data.get("recent_codes", []):
            self.code_input.addItem(code)

    def _save_settings(self) -> None:
        data = {
            "root_dir": self.root_dir_input.text().strip(),
            "recent_codes": [
                self.code_input.itemText(i)
                for i in range(self.code_input.count())
            ],
        }
        _SETTINGS_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2))

    def _push_recent_code(self, code: str) -> None:
        idx = self.code_input.findText(code)
        if idx >= 0:
            self.code_input.removeItem(idx)
        self.code_input.insertItem(0, code)
        while self.code_input.count() > _MAX_RECENT_CODES:
            self.code_input.removeItem(self.code_input.count() - 1)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log(self, msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        self.log_box.append(f"[{ts}] {msg}")

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_engine_changed(self) -> None:
        has_builtin = self.engine_combo.currentData() == "parakeet"
        self.chk_builtin_ts.setEnabled(has_builtin)
        if not has_builtin:
            self.chk_builtin_ts.setChecked(False)

    def _browse_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "选择根目录")
        if d:
            self.root_dir_input.setText(d)

    def _on_generate(self) -> None:
        root_dir = self.root_dir_input.text().strip()
        code = self.code_input.currentText().strip()
        if not root_dir or not code:
            self.status_label.setText("请填写根目录和番号")
            return
        self._push_recent_code(code)
        self._start_task(root_dir, code)

    def _on_queue(self) -> None:
        root_dir = self.root_dir_input.text().strip()
        code = self.code_input.currentText().strip()
        if not root_dir or not code:
            self.status_label.setText("请填写根目录和番号")
            return

        self._push_recent_code(code)
        self.queue.append({"root_dir": root_dir, "code": code})
        self._refresh_queue()
        self._log(f"排队: {code}")
        self.code_input.clearEditText()
        self.code_input.setFocus()

        if self.worker is None:
            self._process_next()

    # ------------------------------------------------------------------
    # Task lifecycle
    # ------------------------------------------------------------------

    def _start_task(self, root_dir: str, code: str) -> None:
        self.generate_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self._task_start = time.time()
        self.status_label.setText(f"正在处理: {code}")
        self._log(f"开始: {code}")

        options = {
            "engine_type": self.engine_combo.currentData(),
            "keep_middle": self.chk_keep_middle.isChecked(),
            "no_vad": self.chk_no_vad.isChecked(),
            "aligner": "none" if self.chk_builtin_ts.isChecked() else "qwen",
            "language": self.lang_combo.currentText(),
            "workers": self.workers_spin.value(),
            "pause_threshold": self.pause_spin.value(),
            "sub_display_delay": self.delay_spin.value(),
        }

        self.worker = ASRWorker(
            self.service, root_dir, code, options=options, parent=self,
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_task_finished)
        self.worker.error.connect(self._on_task_error)
        self.worker.start()

    @staticmethod
    def _stage_key(stage: str) -> str:
        return re.sub(r'\s*\[\d+/\d+\]', '', stage)

    def _on_progress(self, stage: str, percent: float) -> None:
        self.progress_bar.setValue(int(percent))
        self.status_label.setText(stage)
        key = self._stage_key(stage)
        if key != getattr(self, "_last_stage", None):
            self._log(stage)
            self._last_stage = key

    def _on_task_finished(self, output_path: str) -> None:
        code = self.worker.code if self.worker else ""
        elapsed = time.time() - self._task_start
        self.status_label.setText(f"完成: {code} ({elapsed:.1f}s)")
        self.progress_bar.setValue(100)
        self._log(f"完成: {code} — {elapsed:.1f}s")
        self.worker = None
        self._process_next()

    def _on_task_error(self, msg: str) -> None:
        code = self.worker.code if self.worker else ""
        self.status_label.setText(f"错误: {code}")
        self._log(f"错误 ({code}): {msg}")
        QMessageBox.warning(self, f"错误 — {code}", msg)
        self.worker = None
        self._process_next()

    def _process_next(self) -> None:
        if self.queue:
            task = self.queue.pop(0)
            self._refresh_queue()
            self._start_task(task["root_dir"], task["code"])
        else:
            self.generate_btn.setEnabled(True)

    def _refresh_queue(self) -> None:
        self.queue_list.clear()
        for i, task in enumerate(self.queue, start=1):
            self.queue_list.addItem(f"{i}. {task['code']}")

    # ------------------------------------------------------------------
    # Window close
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:  # noqa: N802
        self._save_settings()
        if self.worker and self.worker.isRunning():
            self.service.cancel()
            self.worker.wait(5000)
        event.accept()


# ======================================================================
# Entry point
# ======================================================================


def main() -> None:
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
