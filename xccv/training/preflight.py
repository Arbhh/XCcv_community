"""训练前预检：参数层面的「提示 / 警告」两档建议 + 智能推荐配置（纯逻辑）。

用户规则（To2.3）：尽量少弹窗——预检结果由训练窗口汇总为一个对话框，
且只有存在「警告」级条目时才弹窗，仅有「提示」时状态栏一句话放行；
必须提供折叠区查看「一键应用」修改了哪些参数。
阻止训练的硬规则只有 rotate90 守卫（用户在 To2.1 明确要求，见
training_window.rotate_guard_decision），本模块的所有条目都不阻止训练。
"""

from dataclasses import dataclass, field

from .data_health import LEVEL_INFO, LEVEL_WARN

# 各模型建议默认学习率（与训练平台 DEFAULT_LR 对齐；此处独立常量避免循环导入）
_RECOMMEND_LR = {
    "resnet18": 0.001, "resnet50": 0.001, "mobilenet_v3_small": 0.001,
    "yolo11n": 0.01, "yolo11s": 0.01,
    "yolo11n-seg": 0.01, "yolo11s-seg": 0.01,
    "fasterrcnn": 0.005, "maskrcnn": 0.005, "maskrcnn18": 0.005,
}
_RCNN_MODELS = {"fasterrcnn", "maskrcnn", "maskrcnn18"}


@dataclass
class Finding:
    """一条预检建议。field/value 给出"一键应用"的目标与值（可为空）。"""

    level: str
    code: str
    message: str
    field: str = ""        # 训练窗口控件键（patience / learning_rate / batch_size / device）
    value: object = None   # 一键应用值
    apply_text: str = ""   # 应用按钮文案；空 = 无一键应用

    @property
    def is_warning(self) -> bool:
        return self.level == LEVEL_WARN


@dataclass
class DatasetFacts:
    """预检/推荐用的轻量数据集事实（训练窗口从样本元信息汇总，不解码图片）。"""

    class_counts: dict = field(default_factory=dict)
    n_images: int = 0
    aspect_ratio: float = 1.0   # max(w/h) / min(w/h)；样本不足时为 1.0
    max_dim_p90: float = 0.0    # P90 图像最长边（T9-3 大图判据；0=未知）


def dataset_facts(samples) -> DatasetFacts:
    """从扫描得到的样本元信息汇总 DatasetFacts（分类样本带 label，
    labelme 样本统计 shapes）。"""
    counts: dict = {}
    aspects: list = []
    for s in samples:
        if getattr(s, "width", 0) > 0 and getattr(s, "height", 0) > 0:
            aspects.append(s.width / s.height)
        labels = ([s.label] if getattr(s, "label", None)
                  else [sh.label for sh in (s.shapes or [])])
        for lb in labels:
            counts[lb] = counts.get(lb, 0) + 1
    ratio = (max(aspects) / min(aspects)) if aspects and min(aspects) > 0 else 1.0
    dims = sorted(max(int(getattr(s, "width", 0) or 0),
                      int(getattr(s, "height", 0) or 0))
                  for s in samples if getattr(s, "width", 0) > 0)
    p90 = float(dims[int(0.9 * (len(dims) - 1))]) if dims else 0.0
    return DatasetFacts(class_counts=counts, n_images=len(samples),
                        aspect_ratio=round(ratio, 3), max_dim_p90=p90)


def param_findings(project, facts: DatasetFacts | None = None,
                   cuda_available: bool = True) -> list:
    """参数合法性/合理性检查。返回 list[Finding]，全部为 提示/警告 两档。"""
    findings: list[Finding] = []
    epochs = int(project.epochs)

    # T9-3 大图判据：R-CNN 家族 + 大图占比高 + 未启用切图 → 提示（不阻止）
    if (facts is not None and facts.max_dim_p90 >= 2000
            and project.model_name in _RCNN_MODELS
            and not getattr(project, "tiling", False)):
        findings.append(Finding(
            LEVEL_INFO, "tiling",
            f"数据集大图占比高（P90 最长边 {facts.max_dim_p90:.0f}px）："
            "整图口径下小目标占比过小易漏检，可考虑启用「切图训练」"
            "（数据与验证分区），推理侧配合测试平台「4K 大图分块推理」"
            "或服务 tiled=true。"))

    if project.patience and project.patience >= epochs > 1:
        findings.append(Finding(
            LEVEL_WARN, "patience", 
            f"早停耐心（{project.patience} 轮）不小于训练轮次（{epochs} 轮），"
            "早停不会触发，等同关闭。",
            field="patience", value=max(1, epochs // 3),
            apply_text=f"设为 {max(1, epochs // 3)} 轮"))

    name = str(project.model_name or "")
    if getattr(project, "p2_head", False):
        if name.startswith("yolo"):
            findings.append(Finding(
                LEVEL_INFO, "p2_vram",
                "已启用 P2 小目标检测头：新增 stride 4 检测层，显存占用约为普通"
                "模型的 2 倍以上（实测 1920 输入下 yolo11n-seg + P2 batch4 约 "
                "13 GiB）——显存不足时优先减小批量或输入尺寸；新增分支从零学，"
                "建议适当增加训练轮次。"))
        else:
            findings.append(Finding(
                LEVEL_INFO, "p2_not_yolo",
                f"P2 小目标检测头仅对 YOLO 家族生效，当前模型 {name} 不受影响。"))

    rec_lr = _RECOMMEND_LR.get(project.model_name)
    if rec_lr and epochs > 0:
        lr = float(project.learning_rate)
        if lr > 0 and (lr >= rec_lr * 10 or lr <= rec_lr / 10):
            findings.append(Finding(
                LEVEL_WARN, "learning_rate",
                f"学习率 {lr:g} 与模型推荐值 {rec_lr:g} 相差 10 倍以上"
                "（迁移学习时可刻意调低；从零训练时易不收敛或发散）。",
                field="learning_rate", value=rec_lr,
                apply_text=f"恢复推荐值 {rec_lr:g}"))

    if float(getattr(project, "val_ratio", 0.0)) <= 0:
        findings.append(Finding(
            LEVEL_INFO, "val_zero",
            "验证集比例为 0：验证集复用训练集，指标曲线可能偏乐观。"))

    task = str(getattr(project, "task", ""))
    if task == "classification" and project.batch_size < 8:
        findings.append(Finding(
            LEVEL_INFO, "batch_small",
            f"批量大小 {project.batch_size} 对分类任务偏小，建议 16~32。",
            field="batch_size", value=16, apply_text="设为 16"))
    if task in ("detection", "segmentation") and project.batch_size > 16:
        findings.append(Finding(
            LEVEL_INFO, "batch_large",
            f"批量大小 {project.batch_size} 对检测/分割偏大，"
            "大分辨率下显存压力大（平台会自动降批量，但训练变慢）。",
            field="batch_size", value=8, apply_text="设为 8"))

    if 0 < epochs < 5:
        findings.append(Finding(
            LEVEL_INFO, "epochs_few",
            f"训练轮次仅 {epochs} 轮，模型可能欠拟合；"
            "小数据集建议 12~30 轮。"))

    resize_on = bool((getattr(project, "augment_ops", None) or {}).get("resize"))
    if not resize_on and facts is not None and facts.aspect_ratio > 1.5 \
            and task in ("detection", "segmentation"):
        findings.append(Finding(
            LEVEL_INFO, "resize_off",
            f"数据集图片长宽比差异较大（约 {facts.aspect_ratio:g}×）且未启用"
            "「调整图像大小」：各图保持原生比例进入训练，建议启用 resize 统一输入。",
            field="resize", value=(640, 640), apply_text="启用 640×640"))

    if not cuda_available:
        if str(project.device) == "cuda":
            findings.append(Finding(
                LEVEL_WARN, "device_no_cuda",
                "训练设备指定为 GPU，但未检测到 NVIDIA GPU，训练将失败或回退 CPU。",
                field="device", value="auto", apply_text="改回自动"))
        if getattr(project, "use_amp", True):
            findings.append(Finding(
                LEVEL_INFO, "amp_cpu",
                "当前无 GPU：AMP 混合精度会自动关闭（CPU 训练不受影响）。"))

    if int(getattr(project, "multiply_factor", 1)) > 10:
        findings.append(Finding(
            LEVEL_INFO, "multiply_high",
            f"数据倍增倍数 {project.multiply_factor} 较大：训练时间近似线性增加，"
            "小数据集建议 2~5 配合增强策略。"))

    return findings


def recommend_settings(project, facts: DatasetFacts | None,
                       cuda_available: bool) -> list:
    """智能推荐配置（宁可不推荐也不能推荐错）。

    返回 [(参数键, 值, 理由), ...]；键与训练窗口控件映射，由界面层应用
    并以 toast（状态栏消息）展示理由。
    """
    recs = []
    task = str(getattr(project, "task", ""))
    max_count = max((facts or DatasetFacts()).class_counts.values(), default=0)
    if 0 < max_count < 500:
        if not getattr(project, "freeze_backbone", False):
            recs.append(("freeze_backbone", True,
                         f"最大类别样本仅 {max_count}（<500）：冻结骨干可降低过拟合"
                         "并提速 40~60%"))
        if not getattr(project, "industrial_opt", True):
            recs.append(("industrial_opt", True,
                         "小样本工业场景建议开启工业优化（注意力 + 改进损失）"))
        recs.append(("augment_probs", 0.7,
                     "小样本场景建议把已启用增强算子的概率提高到 0.7 增加样本多样性"))
    # O7（V0.7）：轮次智能推荐——**保守口径**：只在小数据且当前轮次明显
    # 不足（<20）时升级到 60（早停耐心兜底）。不做"大档推荐"：预设
    # （如 quality=300 轮）是用户的显式选择，推荐不与之打架
    # （f7e 实锤：无条件 epochs 推荐会覆盖预设/污染保守断言）。
    if task in ("detection", "segmentation") and 0 < max_count < 300 \
            and 0 < int(getattr(project, "epochs", 0)) < 20:
        recs.append(("epochs", 60,
                     f"最大类别样本 {max_count}（<300）且当前仅 "
                     f"{int(project.epochs)} 轮：建议 60 轮"
                     "（小样本收敛慢，早停耐心兜底防过训）"))
    if task in ("detection", "segmentation") \
            and project.model_name in _RCNN_MODELS \
            and not getattr(project, "rpn_reduced", False):
        recs.append(("rpn_reduced", True,
                     "R-CNN 家族建议削减 RPN 候选框（工业单图目标少，"
                     "几乎无精度损失且更快更省显存）"))
    if not cuda_available and getattr(project, "use_amp", True):
        recs.append(("use_amp", False, "无 GPU：关闭 AMP 混合精度（CPU 下本不生效）"))
    return recs
