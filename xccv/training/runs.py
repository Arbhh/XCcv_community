"""训练历史记录：扫描项目模型储存目录，还原各次训练的时间与状态。

每次训练输出到独立子目录（时间戳_模型名），训练完成后引擎会写入
metrics.json（含开始/结束时间、时长、状态与参数快照）；
据此在界面上还原历史记录列表与单次训练的详情。
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from ..engine.epoch_weights import EPOCH_DIR_NAME
from ..engine.io_utils import read_json

STATUS_TEXT = {
    "completed": "已完成",
    "interrupted": "已中断",
    "failed": "失败",
    "running": "训练中",
    "unknown": "未知",
}

# 状态在列表中的显示颜色（浅色主题）
STATUS_COLOR = {
    "completed": "#2e8b57",
    "running": "#11809c",
    "interrupted": "#e8853d",
    "failed": "#e45756",
    "unknown": "#64748b",
}


@dataclass
class RunRecord:
    """一次训练的历史条目（对应模型储存目录下的一个子目录）。"""

    run_dir: Path
    status: str = "unknown"
    start_time: str = ""
    end_time: str = ""
    duration_sec: int | None = None
    epochs_done: int = 0
    epochs_planned: int = 0
    metrics: dict = field(default_factory=dict)  # metrics.json 原文（缺失时为空）

    @property
    def progress(self) -> float:
        if self.status == "completed":
            return 1.0
        if self.epochs_planned > 0:
            return min(1.0, self.epochs_done / self.epochs_planned)
        return 0.0

    @property
    def status_text(self) -> str:
        return STATUS_TEXT.get(self.status, self.status)

    @property
    def model_name(self) -> str:
        """从目录名后缀还原模型名（20260904_102300_yolo11n → yolo11n）。"""
        parts = self.run_dir.name.split("_", 2)
        return parts[2] if len(parts) == 3 else ""


def _time_from_dirname(name: str) -> str:
    try:
        return datetime.strptime(name[:15], "%Y%m%d_%H%M%S").strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return ""


def load_run(run_dir: Path) -> RunRecord | None:
    """读取一次训练记录；目录为空（尚无任何产出）时返回 None。"""
    run_dir = Path(run_dir)
    rec = RunRecord(run_dir=run_dir, start_time=_time_from_dirname(run_dir.name))
    metrics_path = run_dir / "metrics.json"
    if metrics_path.is_file():
        try:
            m = read_json(metrics_path)
        except Exception:
            m = None  # 指标文件损坏：按失败留档
        if m is None:
            rec.status = "failed"
            return rec
        rec.metrics = m
        rec.status = m.get("status", "completed")
        rec.start_time = m.get("start_time") or rec.start_time
        rec.end_time = m.get("end_time", "")
        rec.duration_sec = m.get("duration_sec")
        history = m.get("history") or []
        rec.epochs_done = history[-1].get("epoch", len(history)) if history else 0
        rec.epochs_planned = m.get("epochs_planned") or (m.get("config") or {}).get("epochs", 0)
        return rec
    if (run_dir / "best.pt").is_file():
        rec.status = "interrupted"  # 有最优权重但没有指标：训练被中途停止
        return rec
    if any(run_dir.iterdir()):
        rec.status = "failed"       # 有残留文件但没有任何产出：训练失败
        return rec
    return None


def scan_runs(model_dir) -> list[RunRecord]:
    """按时间倒序（目录名含时间戳前缀）列出全部训练记录。

    跳过轮次权重共用文件夹（epoch_weights，非训练记录）与隐藏目录。
    """
    root = Path(model_dir)
    if not root.is_dir():
        return []
    runs = []
    for d in root.iterdir():
        if d.is_dir() and not d.name.startswith(".") and d.name != EPOCH_DIR_NAME:
            rec = load_run(d)
            if rec is not None:
                runs.append(rec)
    return sorted(runs, key=lambda r: r.run_dir.name, reverse=True)


def format_duration(seconds: int | float | None) -> str:
    """时长展示：x时x分x秒 / x分x秒。"""
    if seconds is None:
        return "—"
    total = int(round(seconds))
    h, rest = divmod(total, 3600)
    m, s = divmod(rest, 60)
    return f"{h}时{m:02d}分{s:02d}秒" if h else f"{m}分{s:02d}秒"
