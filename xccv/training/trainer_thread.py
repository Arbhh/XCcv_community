"""训练线程：后台跑引擎训练，通过 Qt 信号把进度发给界面。"""

import logging
import time
from pathlib import Path

from PyQt5.QtCore import QThread, pyqtSignal

from .dataset_stats import collect_stats
from .project import Project
from ..engine.dataset.base import TaskType

log = logging.getLogger(__name__)


def _make_reporter(thread: "TrainThread"):
    """创建把引擎回调转成线程信号的 reporter（torch 延迟导入，T1-4：
    基类 TrainReporter 来自 engine.models，类定义必须后置到训练启动时）。"""
    from ..engine.models import TrainReporter

    class _QtReporter(TrainReporter):
        """把引擎回调转成线程信号；同时承载界面发来的停止请求。"""

        def __init__(self, thread_: "TrainThread"):
            super().__init__()
            self._thread = thread_

        def on_epoch(self, epoch: int, logs: dict) -> None:
            self._thread.epoch_changed.emit(epoch, dict(logs))

        def on_train_progress(self, epoch: int, batch_index: int, batch_total: int) -> None:
            self._thread.batch_progress.emit(epoch, batch_index, batch_total)

    return _QtReporter(thread)


class TrainThread(QThread):
    """一次训练任务。界面只与信号交互，不直接调用引擎。"""

    epoch_changed = pyqtSignal(int, dict)
    batch_progress = pyqtSignal(int, int, int)  # (epoch, 批序, 批总数)：轮内进度
    train_finished = pyqtSignal(bool, str, dict)  # (成功, 错误信息, 最终指标)

    def __init__(self, project: Project, run_dir, parent=None):
        super().__init__(parent)
        self.project = project
        self.run_dir = Path(run_dir)
        self._reporter = _make_reporter(self)
        self._t0 = time.time()

    def stop(self) -> None:
        """请求停止：分类/R-CNN 家族当前轮结束后生效；YOLO 家族不可中断。"""
        self._reporter.stop_requested = True

    def _resume_model_path(self):
        """解析迁移学习起始权重，返回其路径（无迁移时返回 None）。

        优先级：项目 transfer_file（界面"迁移学习"文件选择，支持 .pt）>
        旧版模型名前缀引用（"pruned:<文件名>" / "run:<记录目录>/<权重文件>"）。
        路径越界（../ 等）一律拒绝。
        """
        transfer = (getattr(self.project, "transfer_file", "") or "").strip()
        if transfer:
            path = Path(transfer)
            if not path.is_file():
                raise FileNotFoundError(f"迁移学习权重文件不存在：{path}")
            return path
        name = self.project.model_name or ""
        if name.startswith("pruned:"):
            from .project import PRUNED_DIR_NAME

            path = (Path(self.project.model_dir) / PRUNED_DIR_NAME
                    / name.split(":", 1)[1])
        elif name.startswith("run:"):
            rel = name.split(":", 1)[1]
            if "/" not in rel and "\\" not in rel:
                raise ValueError(f"模型引用不合法：{name}")
            path = Path(self.project.model_dir) / rel
        else:
            return None
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"模型权重文件不存在：{path}")
        # 安全校验：引用必须落在项目模型储存路径之内
        try:
            path.resolve().relative_to(Path(self.project.model_dir).resolve())
        except ValueError:
            raise ValueError(f"模型引用越出项目文件夹，已拒绝：{name}")
        return path

    def run(self) -> None:
        try:
            project = self.project
            task = project.task_type
            log.info("训练开始: 项目=%s 模型=%s 输出=%s", project.name,
                     project.model_name, self.run_dir)
            from ..engine.models import TrainConfig, create_model  # torch 延迟导入（T1-4）

            stats = collect_stats(task, project.dataset_dir)
            if stats.n_images == 0:
                raise ValueError("数据集中没有图片，请检查数据集路径")
            if not stats.classes:
                raise ValueError("数据集中未发现类别，请先完成标注或检查数据集路径")
            resume_path = self._resume_model_path()
            if resume_path is not None:
                # 项目自有的已有权重（剪枝模型或训练记录产物）：直接作为起始模型
                from ..engine.models import load_model

                model = load_model(resume_path, device=project.device)
                model.resume_weights = True
                # A1/A2（V0.7）：异常检测/OCR 模型的 classes 恒为内置占位，
                # 不做数据集类别一致性校验（记忆库/字符集与类别无关）
                if task not in (TaskType.ANOMALY, TaskType.OCR) and \
                        list(model.classes or []) != list(stats.classes):
                    raise ValueError(
                        f"所选模型的类别（{model.classes or []}）与当前数据集"
                        f"（{stats.classes}）不一致，无法直接继续训练；"
                        "请在模型工具中重新生成剪枝模型，或检查数据集类别设置")
                log.info("从已有模型权重恢复训练: %s", resume_path)
            else:
                model = create_model(
                    project.model_name, task, len(stats.classes),
                    device=project.device, use_pretrained=project.use_pretrained,
                )
                if getattr(project, "p2_head", False):
                    # P2 小目标检测头仅 YOLO 家族有效，其他家族无该能力即忽略
                    enable_p2 = getattr(model, "enable_p2", None)
                    if enable_p2 is not None:
                        enable_p2()
                        log.info("已启用 P2 小目标检测头: %s", project.model_name)
                    else:
                        log.info("P2 小目标检测头仅对 YOLO 家族生效，%s 忽略该选项",
                                 project.model_name)
            resize = (project.augment_ops or {}).get("resize") or {}
            resize_w, resize_h = int(resize.get("width", 0) or 0), int(resize.get("height", 0) or 0)
            cfg = TrainConfig(
                task=task,
                dataset_dir=Path(project.dataset_dir),
                output_dir=self.run_dir,
                model_name=project.model_name,
                epochs=project.epochs,
                batch_size=project.batch_size,
                learning_rate=project.learning_rate,
                weight_decay=project.weight_decay,
                val_ratio=project.val_ratio,
                train_ratio=getattr(project, "train_ratio", 0.0),
                eval_interval=project.eval_interval,
                multiply_factor=project.multiply_factor,
                augment_ops=dict(project.augment_ops),
                augment_probs=dict(getattr(project, "augment_probs", None) or {}),
                color_mode=getattr(project, "color_mode", "color"),
                deterministic=bool(getattr(project, "deterministic", False)),
                # 训练输入尺寸与增强 resize 统一（归一化）；未启用 resize 时走模型默认
                image_width=resize_w,
                image_height=resize_h,
                device=project.device,
                workers=project.workers,
                use_pretrained=project.use_pretrained,
                industrial_opt=project.industrial_opt,
                patience=project.patience,
                freeze_backbone=project.freeze_backbone,
                rpn_reduced=project.rpn_reduced,
                use_amp=project.use_amp,
                mask_head_light=project.mask_head_light,
                slim_head=getattr(project, "slim_head", False),
                anchor_small=getattr(project, "anchor_small", False),
                rcnn_optimizer=getattr(project, "rcnn_optimizer", "adamw"),
                rcnn_learning_rate=getattr(project, "rcnn_learning_rate", 0.0),
                rcnn_weight_decay=getattr(project, "rcnn_weight_decay", 0.05),
                tf32=project.tf32,
                train_input=getattr(project, "train_input", 0),
                tiling=bool(getattr(project, "tiling", False)),
                tile_size=int(getattr(project, "tile_size", 640) or 640),
                use_negatives=bool(getattr(project, "use_negatives", True)),
                p2_head=bool(getattr(project, "p2_head", False)
                             and getattr(model, "p2", False)),
                save_epoch_weights=getattr(project, "save_epoch_weights", True),
                resume_from_pruned=resume_path is not None,
            )
            metrics = model.train(cfg, self._reporter)
            log.info("训练完成: 项目=%s 输出=%s 耗时=%.0fs", project.name,
                     self.run_dir, time.time() - self._t0)
            self.train_finished.emit(True, "", metrics)
        except Exception as e:  # 任何失败都要把原因带回界面
            log.error("训练失败: %s: %s", type(e).__name__, e, exc_info=True)
            self.train_finished.emit(False, f"{type(e).__name__}: {e}", {})
