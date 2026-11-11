"""YOLO11 检测/分割封装（ultralytics）。

说明：ultralytics 自身每轮训练都会验证，eval_interval 对 YOLO 家族不生效。
平台关闭了 YOLO 内置增强，增强统一由"数据倍增"负责，所见即所得。
"""

import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from ..dataset.base import TaskType
from ..dataset.converters import _clipped_bbox
from ..io_utils import imread_unicode
from ..metrics.detection import detection_confusion_matrix
from .base import BaseModel, Prediction, TrainConfig, TrainReporter
from .modules import NWDCIoUBboxLoss
from .yolo_data import _ValTrainerShim, make_yolo_trainer
from ..offline import WEIGHTS_DIR

log = logging.getLogger(__name__)

try:  # 离线环境关闭遥测上报，避免联网探测
    from ultralytics.utils import SETTINGS as _Y_SETTINGS

    if _Y_SETTINGS.get("sync"):
        _Y_SETTINGS["sync"] = False
except Exception:
    pass


def _clean_yolo_key(k: str) -> str:
    return k.split("/")[-1].replace("(B)", "").replace("(M)", "").replace("(P)", "").strip()


def _masks_to_original(res, orig_hw) -> list:
    """把 ultralytics 输出的实例掩码还原到原图尺寸（bool, (H, W)）。

    res.masks.data 停在网络输入分辨率（letterbox 后的 imgsz，例如 1216），
    直接交给可视化层会与原图 (H, W) 形状不符——IndexError（2026-10-26 测试
    平台 YOLO-seg 崩溃根因，T1-1）。这里按 LetterBox 逆变换先去 padding 再
    缩放回原图，与 boxes 坐标系同源（ultralytics 对 boxes 做了同规则的
    scale_boxes，框无需再变换）。

    返回值遵守 Prediction.masks 契约：**list[np.ndarray(H, W) bool]**（不是
    3 维数组——消费方普遍以 `if pred.masks:` 判空，ndarray 真值歧义会抛
    ValueError，用户回传"YOLO 无法正常推理"即此因，坑 P150）。
    """
    mdata = res.masks.data
    mh, mw = int(mdata.shape[-2]), int(mdata.shape[-1])
    oh, ow = int(orig_hw[0]), int(orig_hw[1])
    if (mh, mw) == (oh, ow):
        return [m for m in (mdata.cpu().numpy() > 0.5)]
    gain = min(mh / oh, mw / ow)
    pad_top = (mh - oh * gain) / 2.0
    pad_left = (mw - ow * gain) / 2.0
    y1 = max(0, int(round(pad_top)))
    x1 = max(0, int(round(pad_left)))
    y2 = min(mh, max(y1 + 1, int(round(pad_top + oh * gain))))
    x2 = min(mw, max(x1 + 1, int(round(pad_left + ow * gain))))
    crop = mdata[:, y1:y2, x1:x2].cpu().numpy().astype(np.float32)
    return [cv2.resize(crop[i], (ow, oh), interpolation=cv2.INTER_LINEAR) > 0.5
            for i in range(crop.shape[0])]


def _lean_yolo_ckpt(ckpt: dict) -> dict:
    """剥离 ultralytics ckpt 中推理无关的数据（optimizer/scaler/训练指标等）。

    原生 best.pt 体积约为网络本身的 2~3 倍。注意两点：
    - 权重可能挂在 ema 键（训练期保存的 best.pt 中 model 可能为 None，
      strip_optimizer 尚未运行）——权重统一归一到 model 键，YOLO() 的
      load_checkpoint 才能取到；
    - 白名单保留加载必需键（epoch/best_fitness/train_args/date/version），
      其余一律不落盘。
    """
    lean = {}
    for k in ("epoch", "best_fitness", "train_args", "date", "version"):
        if k in ckpt:
            lean[k] = ckpt[k]
    model = ckpt.get("model")
    if model is None:
        model = ckpt.get("ema")
    lean["model"] = model
    if model is None:
        raise RuntimeError("权重文件中未找到可用模型（model/ema 均为空）")
    return lean


class YoloModel(BaseModel):
    family = "yolo"
    supported_tasks = (TaskType.DETECTION, TaskType.SEGMENTATION)
    default_image_size = 640

    def __init__(self, size: str, task: TaskType, num_classes: int,
                 device: str = "auto", use_pretrained: bool = True):
        super().__init__(num_classes, device)
        self.size = size
        self.task = TaskType(task)
        base = size if self.task is TaskType.DETECTION else f"{size}-seg"
        if use_pretrained:
            bundled = WEIGHTS_DIR / "ultralytics" / f"{base}.pt"
            # 优先用随包官方预训练权重，保证离线可训练
            self.init_weight = str(bundled) if bundled.is_file() else f"{base}.pt"
        else:
            self.init_weight = f"{base}.yaml"
        self.model: YOLO | None = None

    # ---- 训练 ----
    def fit(self, data, cfg: TrainConfig, reporter: TrainReporter) -> dict:
        # 恢复训练（剪枝模型）时直接用已加载的 YOLO 对象，否则从预训练权重起步
        resume = getattr(self, "resume_weights", False)
        model = self.model if resume else YOLO(self.init_weight)
        self._reporter = reporter
        self._history: list[dict] = []
        self._loss_patch = "off"
        if cfg.industrial_opt:
            # 工业优化：Focaler-WIoU v3 边界框损失 + Varifocal 分类损失
            model.add_callback("on_train_start", self._apply_industrial_loss)
        model.add_callback("on_fit_epoch_end", self._on_epoch_end)
        # validator 有独立 callbacks（on_val_start 不经 trainer）——
        # on_train_start 时 validator 已创建，包装一次 update_metrics
        model.add_callback("on_train_start", self._on_val_start)
        model.add_callback("on_train_epoch_start", self._on_epoch_dets_reset)
        try:
            model.add_callback("on_train_batch_end", self._on_train_batch)
        except Exception:
            pass  # 版本差异：缺少批级回调时退回逐轮曲线
        freeze = 10 if (cfg.freeze_backbone and cfg.use_pretrained and not resume) else 0
        if cfg.image_width > 0 and cfg.image_height > 0:
            imgsz = [cfg.image_height, cfg.image_width]  # ultralytics imgsz 支持 [高, 宽]
        else:
            imgsz = cfg.image_size or self.default_image_size
        log.info("YOLO 训练: 权重=%s 任务=%s imgsz=%s 轮次=%d 设备=%s 恢复训练=%s",
                 "剪枝模型" if resume else self.init_weight, self.task.value,
                 imgsz, cfg.epochs, self.device, resume)

        # 逐轮宽松口径评估的数据源（用户反馈：YOLO 该列恒 0 不正常）
        from types import SimpleNamespace as _NS

        self._val_data = _NS(val_samples=list(data.val_samples),
                             classes=list(data.classes))
        self._val_index = {str(s.image_path): i
                           for i, s in enumerate(data.val_samples)}
        self._epoch_dets = []
        self._det_seen = 0
        # 内存数据装配：零落盘（无 data.yaml / 无标签文件 / 无图片副本）
        trainer_cls = make_yolo_trainer(
            "segmentation" if self.task is TaskType.SEGMENTATION else "detection")
        trainer_cls._xccv_data = {
            "train": "xccv-in-memory", "val": "xccv-in-memory",
            "nc": len(data.classes), "channels": 3,
            "names": {i: n for i, n in enumerate(data.classes)},
        }
        trainer_cls._xccv_samples = {"train": data.train_samples, "val": data.val_samples}
        trainer_cls._xccv_classes = data.classes
        trainer_cls._xccv_ops = data.augment_ops
        trainer_cls._xccv_probs = data.augment_probs
        if resume:
            trainer_cls.resume_module = model.model  # 剪后网络直接参与训练

        model.train(
            trainer=trainer_cls,
            data=str(cfg.dataset_dir),
            epochs=cfg.epochs,
            batch=cfg.batch_size,
            imgsz=imgsz,
            lr0=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            device=self.device,
            seed=cfg.seed,
            workers=cfg.workers,
            plots=False,
            patience=cfg.patience if cfg.patience else 100,
            freeze=freeze,
            amp=bool(cfg.use_amp),
            project=str(cfg.output_dir),
            name="run",
            # 关闭 ultralytics 内置增强，增强由平台策略在线负责
            mosaic=0.0, mixup=0.0, copy_paste=0.0, erasing=0.0, close_mosaic=0,
            degrees=0.0, translate=0.0, scale=0.0, fliplr=0.0, flipud=0.0,
            hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
        )
        try:  # 冻结后的可训练参数量（写入指标，历史记录可见）
            self._trainable = sum(p.numel() for p in model.trainer.model.parameters()
                                  if p.requires_grad)
        except Exception:
            self._trainable = 0
        best = Path(model.trainer.best)
        # 保存最优权重 best.pt。瘦身：ultralytics 原生 ckpt 携带 optimizer 状态
        # 等训练期数据（体积接近网络本身），对推理无影响——
        # 只保留 model/train_args 等加载必需键
        ckpt = torch.load(best, map_location="cpu", weights_only=False)
        plaintext = cfg.output_dir / "best.pt"
        torch.save(_lean_yolo_ckpt(ckpt), plaintext)
        # 加载最优权重进内存，再清理 ultralytics 训练目录里的中间权重
        self.model = YOLO(str(best))
        self.model.model.to(self.device)
        for w in (Path(model.trainer.save_dir) / "weights").glob("*.pt"):
            w.unlink(missing_ok=True)

        # 最终指标：用最优权重在内存验证集上评估（复用训练期 validator）
        final = self._final_metrics(model.trainer)
        final.update(self._loose_metrics(data))
        log.info("YOLO 训练完成: bbox mAP@50=%.4f mAP@50-95=%.4f 掩码 mAP@50=%s",
                 final["map50"], final["map50_95"], final.get("mask_map50", "-"))
        return {
            "best_val_map50": final["map50"],
            "final": final,
            "industrial_opt": cfg.industrial_opt,
            "freeze_backbone": bool(cfg.freeze_backbone and cfg.use_pretrained),
            "trainable_params": getattr(self, "_trainable", 0),
            "loss_patch": self._loss_patch,
            "early_stopped": getattr(getattr(self.model, "trainer", None), "early_stop", False),
            "history": self._history,
            "confusion_matrix": self._confusion_matrix(data),
            "classes": data.classes,
        }

    def _final_metrics(self, trainer) -> dict:
        """用最优权重在内存验证集上评估（复用训练期 validator，不产生任何文件）。"""
        validator = trainer.validator
        module = self.model.model
        # 重载权重（attempt_load_one_weight）的 args 是 dict，而验证期 loss 以
        # 属性访问 hyp.box/hyp.cls 等——换回训练期 namespace
        module.args = trainer.args
        if getattr(module, "criterion", None) is not None:
            # best.pt 里保存的 criterion（训练期 EMA 模型带出）其 proj 等是
            # 普通张量属性（非参数/buffer），.to(device) 不搬运：GPU 训练保存、
            # CPU map_location 重载后 proj 停留 CPU，model.loss 会因 cpu/cuda
            # 混用抛设备不匹配。置空让验证时按当前设备惰性重建。
            module.criterion = None
        shim = _ValTrainerShim(trainer, module)
        validator(shim)
        m = validator.metrics
        ns = m.box
        out = {
            "map50": round(float(ns.map50), 4),  # 对外可视化恒为 bbox 口径
            "map50_95": round(float(ns.map), 4),
            "precision": round(float(ns.mp), 4),
            "recall": round(float(ns.mr), 4),
        }
        if self.task is TaskType.SEGMENTATION and getattr(m, "seg", None) is not None:
            # mask 口径仅作内部参数记录，不进曲线/摘要展示
            out.update(mask_map50=round(float(m.seg.map50), 4),
                       mask_map50_95=round(float(m.seg.map), 4))
        return out

    def _loose_metrics(self, data, predict_fn=None) -> dict:
        """宽松口径（IoU {0.10,0.25,0.50} 三档平均）bbox mAP/mAR——与
        R-CNN 家族同一口径（第三批：YOLO 路径此前不产出 map_loose，
        训练表该列恒 0 误导）。predict_fn(path)->Prediction 可注入
        （逐轮评估用 EMA 权重）；缺省用最优权重 self.model。失败不阻塞。"""
        if predict_fn is None:
            def predict_fn(p):
                return self.model.predict(p, conf=0.05, iou=0.6,
                                          verbose=False,
                                          device=self.device)[0]
        import numpy as np

        try:
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval
        except Exception:  # noqa: BLE001  无 pycocotools 时跳过
            return {}

        def _sane(x: float) -> float:
            return round(max(0.0, float(x)), 4)

        cats = [{"id": i + 1, "name": c} for i, c in enumerate(data.classes)]
        imgs, anns, dets = [], [], []
        ann_id = 1
        try:
            for si, s in enumerate(data.val_samples):
                imgs.append({"id": si, "width": int(s.width),
                             "height": int(s.height)})
                for sh in s.shapes:
                    if sh.label not in data.classes:
                        continue
                    xs = [float(p[0]) for p in sh.points]
                    ys = [float(p[1]) for p in sh.points]
                    x1, y1 = max(0.0, min(xs)), max(0.0, min(ys))
                    x2 = min(float(s.width), max(xs))
                    y2 = min(float(s.height), max(ys))
                    if x2 - x1 < 1 or y2 - y1 < 1:
                        continue
                    anns.append({"id": ann_id, "image_id": si,
                                 "category_id": data.classes.index(sh.label) + 1,
                                 "bbox": [x1, y1, x2 - x1, y2 - y1],
                                 "area": (x2 - x1) * (y2 - y1),
                                 "iscrowd": 0})
                    ann_id += 1
                pred = predict_fn(str(s.image_path))
                for b in pred.boxes:
                    px1, py1, px2, py2 = (float(v) for v in b.xyxy[0])
                    dets.append({"image_id": si,
                                 "category_id": int(b.cls.item()) + 1,
                                 "bbox": [px1, py1, px2 - px1, py2 - py1],
                                 "score": float(b.conf.item())})
        except Exception as e:  # noqa: BLE001
            import traceback as _tb

            log.warning("YOLO 宽松口径评估失败: {}\n{}".format(
                e, _tb.format_exc()))
            return {}
        empty = {"map_loose": 0.0, "map10": 0.0, "map25": 0.0,
                 "mar_loose": 0.0}
        if not anns or not dets:
            return empty
        coco_gt = COCO()
        coco_gt.dataset = {"images": imgs, "annotations": anns,
                           "categories": cats}
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(dets)
        ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
        ev.params.imgIds = list(range(len(imgs)))
        ev.params.iouThrs = np.array([0.10, 0.25, 0.50])
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
        if ev.eval is None:
            return empty
        prec = ev.eval["precision"]   # [T,R,K,A,M]
        rec = ev.eval["recall"]       # [T,K,A,M]

        def _avg(arr) -> float:
            v = arr[arr > -1]
            return float(v.mean()) if v.size else 0.0

        aps = [_avg(prec[t, :, :, 0, -1]) for t in range(3)]
        ars = [_avg(rec[t, :, 0, -1]) for t in range(3)]
        return {"map10": _sane(aps[0]), "map25": _sane(aps[1]),
                "map_loose": _sane(sum(aps) / 3.0),
                "mar_loose": _sane(sum(ars) / 3.0)}

    def _on_epoch_dets_reset(self, trainer):
        """每轮训练开始：清空上一轮的检出累积（宽松口径逐轮隔离）。"""
        self._epoch_dets = []
        self._det_seen = 0

    def _on_val_start(self, trainer):
        """训练开始：包装 validator.update_metrics 收集 NMS 后检出。"""
        try:
            v = getattr(trainer, "validator", None)
            if v is None or getattr(v, "_xccv_wired", False):
                return

            def wrapped(preds, batch, *a, **k):
                try:
                    for si, pred in enumerate(preds):
                        predn = v._prepare_pred(pred)
                        pbatch = v._prepare_batch(si, batch)
                        scaled = v.scale_preds(predn, pbatch)
                        bb = scaled["bboxes"].detach().float().cpu().numpy()
                        cf = scaled["conf"].detach().float().cpu().numpy()
                        cl = scaled["cls"].detach().float().cpu().numpy()
                        im = str(pbatch.get("im_file", ""))
                        gi = self._val_index.get(
                            im, self._det_seen + si)
                        for (x1, y1, x2, y2), conf, c in zip(bb, cf, cl):
                            self._epoch_dets.append({
                                "image_id": int(gi),
                                "category_id": int(c) + 1,
                                "bbox": [float(x1), float(y1),
                                         float(x2 - x1), float(y2 - y1)],
                                "score": float(conf)})
                    self._det_seen += len(preds)
                except Exception as e:  # noqa: BLE001
                    log.warning("验证检出收集失败: %s", e)
                return orig(preds, batch, *a, **k)

            orig = v.update_metrics
            v.update_metrics = wrapped
            v._xccv_wired = True
        except Exception as e:  # noqa: BLE001
            log.warning("validator 包装失败: %s", e)

    def _epoch_loose(self) -> dict:
        """每轮宽松口径（与 R-CNN 表格同频）：复用本轮验证已 NMS 的
        检出（零额外推理）。失败静默跳过（该轮缺列即可，不拖训练）。"""
        try:
            vd = getattr(self, "_val_data", None)
            dets = getattr(self, "_epoch_dets", None)
            if vd is None or not vd.val_samples or dets is None:
                return {}
            import numpy as np

            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            def _sane(x: float) -> float:
                return round(max(0.0, float(x)), 4)

            cats = [{"id": i + 1, "name": c}
                    for i, c in enumerate(vd.classes)]
            imgs, anns = [], []
            ann_id = 1
            for si, s in enumerate(vd.val_samples):
                imgs.append({"id": si, "width": int(s.width),
                             "height": int(s.height)})
                for sh in s.shapes:
                    if sh.label not in vd.classes:
                        continue
                    xs = [float(p[0]) for p in sh.points]
                    ys = [float(p[1]) for p in sh.points]
                    x1, y1 = max(0.0, min(xs)), max(0.0, min(ys))
                    x2 = min(float(s.width), max(xs))
                    y2 = min(float(s.height), max(ys))
                    if x2 - x1 < 1 or y2 - y1 < 1:
                        continue
                    anns.append({"id": ann_id, "image_id": si,
                                 "category_id": vd.classes.index(sh.label) + 1,
                                 "bbox": [x1, y1, x2 - x1, y2 - y1],
                                 "area": (x2 - x1) * (y2 - y1),
                                 "iscrowd": 0})
                    ann_id += 1
            empty = {"map_loose": 0.0, "map10": 0.0, "map25": 0.0,
                     "mar_loose": 0.0}
            if not anns or not dets:
                return empty
            coco_gt = COCO()
            coco_gt.dataset = {"images": imgs, "annotations": anns,
                               "categories": cats}
            coco_gt.createIndex()
            coco_dt = coco_gt.loadRes(list(dets))
            ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
            ev.params.imgIds = list(range(len(imgs)))
            ev.params.iouThrs = np.array([0.10, 0.25, 0.50])
            ev.evaluate()
            ev.accumulate()
            ev.summarize()
            if ev.eval is None:
                return empty
            prec = ev.eval["precision"]   # [T,R,K,A,M]
            rec = ev.eval["recall"]       # [T,K,A,M]

            def _avg(arr) -> float:
                v = arr[arr > -1]
                return float(v.mean()) if v.size else 0.0

            aps = [_avg(prec[t, :, :, 0, -1]) for t in range(3)]
            ars = [_avg(rec[t, :, 0, -1]) for t in range(3)]
            return {"map10": _sane(aps[0]), "map25": _sane(aps[1]),
                    "map_loose": _sane(sum(aps) / 3.0),
                    "mar_loose": _sane(sum(ars) / 3.0)}
        except Exception as e:  # noqa: BLE001
            log.warning("YOLO 逐轮宽松口径评估失败（跳过本轮）: %s", e)
            return {}

    def _apply_industrial_loss(self, trainer):
        """on_train_start：包装 init_criterion（惰性创建点），创建瞬间替换损失组件。

        YOLO（anchor-free）专用工业方案（第三批，用户指令）：仅替换
        bbox_loss 为 NWD-CIoU 混合（微小缺陷友好，见
        modules.NWDCIoUBboxLoss）；cls/DFL 保持 ultralytics 原生——
        与 R-CNN 家族的 DCN+CBAM+Dice 方案完全独立。
        旧 Focaler-WIoU 以 GT 尺寸归一再 exp，微小目标上 loss 爆炸
        （实测 box_loss 42~160），已弃用。
        """
        model = trainer.model
        if getattr(model, "_xccv_patched", False):
            return
        orig_init = model.init_criterion

        def init_criterion():
            criterion = orig_init()
            try:
                device = next(model.parameters()).device
                criterion.bbox_loss = NWDCIoUBboxLoss(
                    getattr(criterion, "reg_max", 16)).to(device)
                self._loss_patch = "applied(NWD-CIoU 混合框损失)"
            except Exception as e:
                self._loss_patch = f"fallback({type(e).__name__}: {e})"
            return criterion

        model.init_criterion = init_criterion
        model._xccv_patched = True

    def _on_train_batch(self, trainer):
        """每个批次结束：上报轮内进度（总体进度条实时百分比的数据源）。"""
        bi = int(getattr(trainer, "batch_i", 0) or 0) + 1
        nb = int(getattr(trainer, "nb", 0) or 0)
        self._reporter.on_train_progress(trainer.epoch + 1, bi, nb)
        loss = getattr(trainer, "loss", None)
        if loss is not None and not torch.isfinite(
                loss.detach().float().sum()):
            # ultralytics 自身会跳过非有限步；这里记录日志便于远程排障
            log.warning("YOLO 第 %d 轮第 %d/%d 批损失非有限（NaN/Inf）",
                        trainer.epoch + 1, bi, max(nb, 1))

    def _on_epoch_end(self, trainer):
        """把 ultralytics 每轮指标转成平台统一格式并上报。

        键名标准化为平台白名单（val_top1 / map50 / map50_95）：
        - 对外可视化口径恒为 bbox：框指标 (B) 用裸键名 map50/map50_95 进曲线；
        - 分割的掩码指标 (M) 加 mask_ 前缀（mask_map50 等），仅作内部参数
          （ultralytics 内部最优权重选择本身按掩码 fitness），不进曲线/摘要。
        训练损失：trainer.metrics 只含验证指标——训练损失在 tloss，需经
        label_loss_items 打标（此前直接读 metrics 导致 loss 曲线恒 0）；
        tloss 分量含 box/cls/dfl 增益，小目标数据集 box_loss 会破百
        （用户实测 160+），按增益归一化回原始损失均值口径（可解释量级）。
        """
        logs, train_loss = {}, 0.0
        seg = self.task is TaskType.SEGMENTATION
        for k, v in trainer.metrics.items():
            if k.startswith("train/"):
                train_loss += float(v)
                continue
            name = _clean_yolo_key(k).lower().replace("-", "_")  # mAP50-95 → map50_95
            if "(M)" in k:
                logs[f"mask_{name}" if seg else name] = round(float(v), 4)
            elif "(B)" in k:
                logs[name] = round(float(v), 4)
        if not train_loss:
            tloss = getattr(trainer, "tloss", None)
            get_label = getattr(trainer, "label_loss_items", None)
            if get_label is not None and tloss is not None:
                try:
                    labeled = get_label(tloss)
                    # 增益归一化：ultralytics 的 tloss 分量已乘 box/cls/dfl 增益
                    # （小目标数据集 box_loss 会因类别得分分母极小而高达百余，
                    # 用户实测曲线破百）。除回各自增益得到原始损失均值口径。
                    args = getattr(trainer, "args", None)
                    gain_by_name = {
                        "box_loss": getattr(args, "box", 7.5),
                        "seg_loss": getattr(args, "box", 7.5),  # 分割 seg 用 box 增益
                        "cls_loss": getattr(args, "cls", 0.5),
                        "dfl_loss": getattr(args, "dfl", 1.5),
                        "pose_loss": getattr(args, "pose", 12.0),
                    }
                    vals = (labeled.values() if isinstance(labeled, dict)
                            else [labeled])
                    names = list(getattr(trainer, "loss_names", []) or [])
                    total = 0.0
                    for i, v in enumerate(vals):
                        gain = 1.0
                        if i < len(names):
                            gain = float(gain_by_name.get(names[i], 1.0)) or 1.0
                        total += float(v) / gain
                    train_loss = total
                except Exception:
                    train_loss = 0.0
        logs["train_loss"] = round(train_loss, 4)
        logs.update(self._epoch_loose())
        self._history.append({"epoch": trainer.epoch + 1, **logs})
        self._reporter.on_epoch(trainer.epoch + 1, logs)
        store = getattr(self, "epoch_store", None)
        if store is not None:
            # 逐轮存档 EMA 权重（与 ultralytics 最优权重同源）
            ema = getattr(trainer, "ema", None)
            module = ema.ema if ema is not None else trainer.model
            store.save(trainer.epoch + 1, {
                "model": module, "epoch": trainer.epoch + 1,
                "train_args": {"task": self.task.value},
            })

    def _confusion_matrix(self, data, conf: float = 0.25, iou: float = 0.5) -> list:
        """在验证集上做 IoU 匹配统计（漏检->背景列，误检->背景行）。

        直接遍历内存验证样本（零落盘后不再读取导出的标注 txt）。
        """
        per_gt, per_pred = [], []
        for s in data.val_samples:
            img = imread_unicode(s.image_path)
            if img is None:
                continue
            h, w = img.shape[:2]
            gt_boxes, gt_labels = [], []
            for sh in s.shapes:
                bbox = _clipped_bbox(sh, w, h)
                ci = data.classes.index(sh.label) if sh.label in data.classes else -1
                if bbox is None or ci < 0:
                    continue
                gt_boxes.append(list(bbox))
                gt_labels.append(ci)
            pred = self.predict(img, conf=conf, iou=iou)
            per_gt.append({"boxes": gt_boxes, "labels": gt_labels})
            per_pred.append({"boxes": pred.boxes, "labels": pred.labels, "scores": pred.scores})
        return detection_confusion_matrix(per_gt, per_pred, len(data.classes), iou).tolist()

    # ---- 推理 ----
    def predict(self, image_bgr: np.ndarray, conf: float = 0.25, iou: float = 0.45) -> Prediction:
        if self.model is None:
            raise RuntimeError("模型尚未训练或加载权重")
        res = self.model.predict(image_bgr, conf=conf, iou=iou,
                                 device=self.device, verbose=False)[0]
        boxes = res.boxes.xyxy.cpu().numpy().tolist() if res.boxes is not None else []
        scores = res.boxes.conf.cpu().numpy().tolist() if res.boxes is not None else []
        labels = res.boxes.cls.cpu().numpy().astype(int).tolist() if res.boxes is not None else []
        masks = None
        if res.masks is not None:
            masks = _masks_to_original(res, image_bgr.shape[:2])
        return Prediction(labels=labels, scores=scores, boxes=boxes, masks=masks)

    # ---- 权重 ----
    def load_weights(self, path: Path) -> None:
        """加载 ultralytics 训练产出的 .pt。"""
        self.model = YOLO(str(path))
        self.model.model.to(self.device)
        names = self.model.names
        self.num_classes = len(names)
        self.classes = list(names.values())
        self.task = TaskType.DETECTION if self.model.task == "detect" else TaskType.SEGMENTATION
