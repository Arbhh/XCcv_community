"""YOLO 内存数据集与训练器：训练数据（图片副本、标注文件）零落盘。

- InMemoryYoloDataset：标签全部在内存构造（复用平台的 shape 转换逻辑），
  不写 label txt、不写 *.cache；图片直接读取数据集原文件；平台的增强策略
  在 get_image_and_label 阶段在线随机应用——增强与标签在同一处同步重建，
  几何算子（翻转/旋转/裁剪）变换图像的同时按变换后标注重建标签字典，
  保证检测/分割标注始终与图像一致；
- XccvYoloTrainer：覆盖 ultralytics 的数据装配（get_dataset / get_dataloader）
  与模型装配（get_model 支持从剪枝模型恢复），训练全程不产生 data.yaml；
- 训练产物（best/last 权重、args.yaml、results.csv）属于输出而非数据，
  仍写入训练输出目录，由 YoloModel 负责加密与清理明文权重。
"""

import logging
import math

import cv2
import numpy as np
import torch
from ultralytics.data.build import build_dataloader
from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.models.yolo.segment.train import SegmentationTrainer
from ultralytics.utils import colorstr
from ultralytics.utils.torch_utils import unwrap_model

from ..dataset.converters import _clipped_bbox, _seg_points
from ..dataset.transforms import maybe_apply_ops
from ..io_utils import imread_unicode

log = logging.getLogger(__name__)


def build_labels_from_shapes(shapes: list, classes: list[str], task: str,
                             w: int, h: int, im_file: str = "") -> dict:
    """把变换后的平台 Shape 列表转成 ultralytics 标签字典（增强后重建用）。"""
    cls_list, boxes, segments = [], [], []
    for sh in shapes:
        ci = classes.index(sh.label) if sh.label in classes else -1
        if ci < 0:
            continue
        seg_pts = _seg_points(sh, w, h) if task == "segmentation" else None
        if seg_pts is not None:
            xs = [p[0] for p in seg_pts]
            ys = [p[1] for p in seg_pts]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            segments.append(np.asarray(seg_pts, dtype=np.float32) / [w, h])
        else:
            bbox = _clipped_bbox(sh, w, h)
            if bbox is None:
                continue
            x1, y1, x2, y2 = bbox
        boxes.append([(x1 + x2) / 2 / w, (y1 + y2) / 2 / h,
                      (x2 - x1) / w, (y2 - y1) / h])
        cls_list.append(ci)
    return {
        "im_file": im_file,
        "shape": (h, w),
        "cls": np.asarray(cls_list, dtype=np.float32).reshape(-1, 1),
        "bboxes": np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
        "segments": segments,
        "keypoints": None,
        "normalized": True,
        "bbox_format": "xywh",
    }


def build_label_dict(sample, classes: list[str], task: str) -> dict:
    """把平台的 Sample 转成 ultralytics 标签字典（归一化 xywh / 多边形）。

    与旧导出逻辑逐字段一致：检测用裁剪外接框；分割用轮廓顶点（矩形转四角点），
    框取轮廓外接框。
    """
    return build_labels_from_shapes(
        sample.shapes, classes, task, sample.width, sample.height,
        im_file=str(sample.image_path))


class InMemoryYoloDataset(YOLODataset):
    """ultralytics YOLODataset 的内存版：标签内存构造、原图直读、在线增强。"""

    def __init__(self, samples, classes: list[str], task: str, augment_ops: dict | None,
                 *args, augment_probs: dict | None = None, **kwargs):
        self._samples = list(samples)
        self._classes = list(classes)
        self._task = task  # 平台任务名：detection / segmentation
        self._ops = dict(augment_ops or {})
        self._probs = dict(augment_probs or {})
        ul_task = {"segmentation": "segment"}.get(task, task)  # ultralytics 任务名
        super().__init__(*args, task=ul_task, **kwargs)

    def get_img_files(self, img_path):  # noqa: ARG002 - 原实现按目录/清单扫描磁盘
        """直接使用样本清单中的原图路径（不扫描、不复制）。"""
        return [str(s.image_path) for s in self._samples]

    def get_labels(self) -> list[dict]:
        """内存构造全部标签（不读盘、不写缓存）。"""
        labels = [build_label_dict(s, self._classes, self._task) for s in self._samples]
        self.im_files = [lb["im_file"] for lb in labels]
        n_boxes = sum(len(lb["bboxes"]) for lb in labels)
        log.info("YOLO 内存标签: %d 张图, %d 个目标, 任务=%s, 在线增强=%s",
                 len(labels), n_boxes, self._task, bool(self._ops))
        return labels

    def _read_original(self, index: int) -> np.ndarray:
        im = imread_unicode(self.im_files[index])
        if im is None:
            raise FileNotFoundError(f"Image Not Found {self.im_files[index]}")
        return im

    def get_image_and_label(self, index: int) -> dict:
        """覆盖父类：平台增强与标签在同一处同步。

        父类先 deepcopy 标签再 load_image 里做增强，几何算子（翻转/旋转/裁剪）
        变换了图像却无法回写标注 → 训练标签错位。这里在取样线程内对同一份
        shapes 先增强、再按变换后标注重建标签字典，图像与标注始终一致。
        """
        from copy import deepcopy

        sample = self._samples[index]
        im = self._read_original(index)
        shapes = list(sample.shapes)
        if self._ops:
            im, new_shapes = maybe_apply_ops(im, shapes, self._ops, probs=self._probs)
        else:
            new_shapes = shapes
        if new_shapes is not shapes:  # 命中几何增强（纯光度算子 shapes 不变，无需重建）
            label = build_labels_from_shapes(
                new_shapes, self._classes, self._task,
                im.shape[1], im.shape[0], im_file=self.im_files[index])
        else:
            label = deepcopy(self.labels[index])
        label.pop("shape", None)  # shape 仅供 rect 使用，与父类一致移除
        h0, w0 = im.shape[:2]
        if self.rect:  # 长边缩放到 imgsz，保持长宽比（val letterbox 由 collate 补边）
            r = self.imgsz / max(h0, w0)
            if r != 1:
                w, h = (min(math.ceil(w0 * r), self.imgsz),
                        min(math.ceil(h0 * r), self.imgsz))
                im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
        elif not (h0 == w0 == self.imgsz):
            im = cv2.resize(im, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
        if im.ndim == 2:
            im = im[..., None]
        label["img"], label["ori_shape"], label["resized_shape"] = im, (h0, w0), im.shape[:2]
        label["ratio_pad"] = (
            label["resized_shape"][0] / label["ori_shape"][0],
            label["resized_shape"][1] / label["ori_shape"][1],
        )
        if self.rect:
            label["rect_shape"] = self.batch_shapes[self.batch[index]]
        return self.update_labels_info(label)

    def load_image(self, i: int, rect_mode: bool = True):
        """读原图并按矩形模式缩放（增强已移入 get_image_and_label 同步标签）。"""
        im = self._read_original(i)
        h0, w0 = im.shape[:2]
        if rect_mode:  # 长边缩放到 imgsz，保持长宽比
            r = self.imgsz / max(h0, w0)
            if r != 1:
                w, h = (min(math.ceil(w0 * r), self.imgsz),
                        min(math.ceil(h0 * r), self.imgsz))
                im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
        elif not (h0 == w0 == self.imgsz):
            im = cv2.resize(im, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
        if im.ndim == 2:
            im = im[..., None]
        return im, (h0, w0), im.shape[:2]


def make_yolo_trainer(task: str):
    """构造内存训练器类：task 为 detect/segment，分别继承对应官方 Trainer。"""
    base = SegmentationTrainer if task == "segmentation" else DetectionTrainer

    class XccvYoloTrainer(base):
        """数据装配内存化 + 剪枝模型恢复训练。"""

        _xccv_data: dict = {}
        _xccv_samples: dict = {}
        _xccv_classes: list[str] = []
        _xccv_task: str = task
        _xccv_ops: dict = {}
        _xccv_probs: dict = {}
        resume_module: torch.nn.Module | None = None  # 剪枝模型恢复时注入

        def get_dataset(self) -> dict:
            """跳过 data.yaml 解析，直接返回内存数据字典。"""
            return self._xccv_data

        def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
            """用内存数据集替代磁盘数据集（dataset_path 参数忽略）。"""
            assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not {mode}."
            gs = max(int(unwrap_model(self.model).stride.max() if self.model else 0), 32)
            split = "train" if mode == "train" else "val"
            dataset = InMemoryYoloDataset(
                self._xccv_samples[split], self._xccv_classes, self._xccv_task,
                self._xccv_ops if mode == "train" else None,
                augment_probs=self._xccv_probs if mode == "train" else None,
                img_path="xccv://in-memory", imgsz=self.args.imgsz,
                batch_size=batch_size, augment=mode == "train", hyp=self.args,
                rect=mode == "val", cache=None, single_cls=False, stride=gs,
                pad=0.0 if mode == "train" else 0.5, prefix=colorstr(f"{mode}: "),
                data=self.data, fraction=1.0,
            )
            shuffle = mode == "train"
            return build_dataloader(
                dataset, batch=batch_size,
                workers=self.args.workers if mode == "train" else self.args.workers * 2,
                shuffle=shuffle, rank=rank,
                drop_last=self.args.compile and mode == "train",
            )

        def get_model(self, cfg=None, weights=None, verbose=True):
            """剪枝模型恢复：直接使用剪后网络（ultralytics 会按 yaml 重建导致剪枝丢失）。"""
            if type(self).resume_module is not None:
                net = type(self).resume_module
                net.nc = self.data["nc"]
                net.names = self.data["names"]
                log.info("从剪枝模型恢复训练: 参数量=%s", f"{sum(p.numel() for p in net.parameters()):,}")
                return net
            return super().get_model(cfg, weights, verbose)

        def final_eval(self):
            """跳过官方收尾验证（其按磁盘数据集独立校验，与内存数据集不符）。

            最终指标由 YoloModel._final_metrics 用内存验证集评估。
            """
            log.info("跳过 ultralytics final_eval（最终指标由平台内存验证集评估）")

    return XccvYoloTrainer


class _ValTrainerShim:
    """validator 复用所需的 trainer 适配（未覆盖的属性全部委托给真实 trainer）。

    模拟训练期调用协议：validator 从 trainer 取 device/data/ema/loss_items 等，
    这里把模型替换为最优权重模块，其余一律透传。
    """

    def __init__(self, trainer, model_module):
        self._trainer = trainer
        self._model_module = model_module
        self.device = trainer.device
        self.data = trainer.data
        self.amp = trainer.amp
        self.args = trainer.args
        self.loss_items = torch.zeros(max(len(trainer.loss_names), 1))
        self.loss = torch.zeros(1)
        self.ema = type("_Ema", (), {"ema": model_module})()

    def __getattr__(self, name):
        return getattr(self._trainer, name)
