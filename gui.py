#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PyQt5 GUI for SRT subtitle generation.

This file ONLY handles UI state and user interactions.  All business logic
lives in asr_service.py — the GUI never imports audio / asr / pipeline
modules directly.
"""

from __future__ import annotations

import sys
from typing import Optional

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from asr_service import ASRService


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
        parent=None,
    ):
        super().__init__(parent)
        self.service = service
        self.root_dir = root_dir
        self.code = code

    def run(self) -> None:
        try:
            video_path, srt_path = self.service.find_video_by_code(
                self.root_dir, self.code,
            )
            self.service.generate_srt(
                video_path,
                srt_path,
                progress_callback=self._emit_progress,
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
        self.queue: list[dict] = []  # [{'root_dir': str, 'code': str}, ...]
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
        self.code_input = QLineEdit()
        self.code_input.setFixedWidth(180)
        self.code_input.setPlaceholderText("如 HUNTC-377")
        self.code_input.returnPressed.connect(self._on_generate)
        row2.addWidget(self.code_input)
        row2.addStretch()
        root.addLayout(row2)

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

        # --- Row 6: task queue ---
        lbl_q = QLabel("任务队列:")
        root.addWidget(lbl_q)
        self.queue_list = QListWidget()
        root.addWidget(self.queue_list, stretch=1)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _browse_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "选择根目录")
        if d:
            self.root_dir_input.setText(d)

    def _on_generate(self) -> None:
        root_dir = self.root_dir_input.text().strip()
        code = self.code_input.text().strip()
        if not root_dir or not code:
            self.status_label.setText("请填写根目录和番号")
            return
        self._start_task(root_dir, code)

    def _on_queue(self) -> None:
        root_dir = self.root_dir_input.text().strip()
        code = self.code_input.text().strip()
        if not root_dir or not code:
            self.status_label.setText("请填写根目录和番号")
            return

        self.queue.append({"root_dir": root_dir, "code": code})
        self._refresh_queue()
        self.code_input.clear()
        self.code_input.setFocus()

        if self.worker is None:
            self._process_next()
        else:
            self.status_label.setText(f"已添加到队列: {code}")

    # ------------------------------------------------------------------
    # Task lifecycle
    # ------------------------------------------------------------------

    def _start_task(self, root_dir: str, code: str) -> None:
        self.generate_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status_label.setText(f"正在处理: {code} ...")

        self.worker = ASRWorker(self.service, root_dir, code, parent=self)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_task_finished)
        self.worker.error.connect(self._on_task_error)
        self.worker.start()

    def _on_progress(self, stage: str, percent: float) -> None:
        self.progress_bar.setValue(int(percent))
        self.status_label.setText(stage)

    def _on_task_finished(self, output_path: str) -> None:
        code = self.worker.code if self.worker else ""
        self.status_label.setText(f"完成: {code} -> {output_path}")
        self.progress_bar.setValue(100)
        self.worker = None
        self._process_next()

    def _on_task_error(self, msg: str) -> None:
        code = self.worker.code if self.worker else ""
        self.status_label.setText(f"错误 ({code}): {msg}")
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
