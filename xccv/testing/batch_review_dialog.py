"""O19+O20（V0.7）：批量结果复判与回流对话框（Qt 层）。

纯逻辑在 batch_review.py（状态持久化/建议/回流计划与执行），本模块只做
表格展示与按钮接线。非模态窗口，宿主测试窗口持有单实例；offscreen 自动
化只走纯逻辑函数，模态交互（回流确认弹窗/目录选择）属真机验证项。
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PyQt5.QtWidgets import (
    QDialog, QFileDialog, QHBoxLayout, QHeaderView, QLabel, QMessageBox,
    QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout,
)

from .batch_review import (
    STATUS_FP, STATUS_MISSED, STATUS_OK, STATUS_UNCERTAIN, STATUS_UNJUDGED,
    execute_reflow, load_review, plan_reflow, review_stats, save_review,
    suggest_status,
)

_STAT_ORDER = (STATUS_OK, STATUS_MISSED, STATUS_FP, STATUS_UNCERTAIN,
               STATUS_UNJUDGED)


class BatchReviewDialog(QDialog):
    """批量结果复判（正常/漏检/过检/存疑）与一键回流——非模态。"""

    def __init__(self, results: list, match_stats: dict | None,
                 review_path, dataset_dir=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("批量结果复判与回流")
        self.setModal(False)
        self._review_path = Path(review_path)
        self._dataset_dir = Path(dataset_dir) if dataset_dir else None
        self._review = dict(load_review(self._review_path))

        entries = (match_stats or {}).get("per_image") or []
        by_path = {str(e.get("path")): e for e in entries}
        self._entries: list[dict] = []
        for r in results or []:
            base = dict(by_path.get(str(r.get("path")))
                        or {"path": str(r.get("path")), "gt": None,
                            "missed": None, "fp": None, "matched_all": None})
            base["detections"] = len(r.get("detections") or [])
            base["result"] = r
            self._entries.append(base)

        lay = QVBoxLayout(self)
        self.lbl_stats = QLabel("")
        lay.addWidget(self.lbl_stats)
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(
            ["文件", "检出", "GT", "漏检", "过检", "建议", "状态"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        lay.addWidget(self.table)

        btns = QHBoxLayout()
        for text, status in (("标记正常", STATUS_OK), ("标记漏检", STATUS_MISSED),
                             ("标记过检", STATUS_FP), ("标记存疑", STATUS_UNCERTAIN),
                             ("清除标记", STATUS_UNJUDGED)):
            btn = QPushButton(text)
            btn.clicked.connect(lambda _, s=status: self._mark_selected(s))
            btns.addWidget(btn)
        lay.addLayout(btns)

        btns2 = QHBoxLayout()
        self.btn_reflow = QPushButton("回流所选到训练集（待标注）")
        self.btn_reflow.setToolTip(
            "把选中图片复制进数据集目录的「待标注」子目录：有标注的图带原"
            "标注（人工修正漏检），无标注的图带当前模型预测框（labelme 格式）")
        self.btn_reflow.clicked.connect(self._reflow_selected)
        btns2.addWidget(self.btn_reflow)
        btns2.addStretch(1)
        btn_close = QPushButton("关闭")
        btn_close.clicked.connect(self.close)
        btns2.addWidget(btn_close)
        lay.addLayout(btns2)
        self._reload_table()

    # ---- 状态模型 ----
    def _effective(self, entry: dict) -> str:
        rec = self._review.get(str(entry.get("path"))) or {}
        return rec.get("status") or suggest_status(entry)

    def _reload_table(self):
        t = self.table
        t.setRowCount(len(self._entries))
        for r, e in enumerate(self._entries):
            gt = e.get("gt")
            vals = [Path(str(e.get("path"))).name,
                    str(e.get("detections", "")),
                    "-" if gt is None else str(gt),
                    "-" if e.get("missed") is None else str(e["missed"]),
                    "-" if e.get("fp") is None else str(e["fp"]),
                    suggest_status(e), self._effective(e)]
            for c, text in enumerate(vals):
                t.setItem(r, c, QTableWidgetItem(text))
        self._refresh_stats()

    def _refresh_stats(self):
        stats = review_stats(self._review, self._entries)
        total = stats["_total"]
        parts = [f"{s} {stats[s]['count']}（{stats[s]['rate']:.0%}）"
                 for s in _STAT_ORDER]
        self.lbl_stats.setText(f"共 {total} 张：　" + "　".join(parts))

    def _selected_entries(self) -> list[dict]:
        rows = sorted({i.row() for i in self.table.selectedIndexes()})
        return [self._entries[r] for r in rows if 0 <= r < len(self._entries)]

    def _mark_selected(self, status: str):
        for e in self._selected_entries():
            if status == STATUS_UNJUDGED:
                # 清除标记=回到"建议即状态"：删除覆盖记录
                self._review.pop(str(e.get("path")), None)
                continue
            self._review[str(e.get("path"))] = {
                "status": status,
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "source": "manual"}
        try:
            save_review(self._review_path, self._review)
        except Exception:  # noqa: BLE001  落盘失败不阻塞界面状态
            pass
        self._reload_table()

    def _reflow_selected(self):
        sel = self._selected_entries()
        if not sel:
            QMessageBox.information(self, "回流", "请先在表格中选中要回流的图片")
            return
        dataset_dir = self._dataset_dir
        if dataset_dir is None or not Path(dataset_dir).is_dir():
            picked = QFileDialog.getExistingDirectory(
                self, "选择训练数据集目录（图片将复制到其「待标注」子目录）")
            if not picked:
                return
            self._dataset_dir = Path(picked)
            dataset_dir = self._dataset_dir
        actions = [a for a in (plan_reflow(e, e.get("result") or {},
                                           dataset_dir) for e in sel) if a]
        if not actions:
            QMessageBox.warning(self, "回流", "所选图片不存在或已失效")
            return
        copied, errors = execute_reflow(actions)
        msg = f"已回流 {copied}/{len(actions)} 张 → {dataset_dir / '待标注'}"
        if errors:
            msg += "\n失败：\n" + "\n".join(errors[:5])
        QMessageBox.information(self, "回流完成", msg)


from xccv.qt_slots import apply_auto_slot_module  # noqa: E402

apply_auto_slot_module(globals())
