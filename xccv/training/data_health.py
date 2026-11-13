"""数据集体检（用户规则：仅「提示 / 警告」两档，不阻止训练）。

检查项与引擎既有逻辑对齐（体检说"会启用工业损失"就真的会触发）：
- 类别不均衡：最小类实例占整体比例 < imbalance_share（默认 5%，用户 Q3 回执口径）→ 警告；
- 单类样本过少：分类 <20/类、检测/分割 <50 实例/类 → 警告；
- 验证集类别缺失：按当前划分参数模拟抽查 → 提示；
- 未标注图片（labelme 结构缺同名 json）→ 提示；
- 疑似重复图：dHash 64 位汉明距离 ≤ 阈值 → 提示（同款工件多角度拍摄易误报）；
- 尺寸离群：面积偏离中位数 ±N×MAD → 提示。

阈值在训练平台「开发者选项」（齿轮）中可调（dev_options.DevOptions）。
"疑似错标"暂不实现（用户决策 D3）。本模块为纯逻辑，不依赖 Qt。
"""

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from ..engine.dataset import IMAGE_EXTS
from ..engine.dataset.base import TaskType
from ..engine.dataset.converters import split_samples
from ..engine.dataset.ignore import IGNORE_LABEL
from ..engine.dataset.reader import scan_classification_dataset, scan_labelme_dataset
from ..engine.io_utils import imread_unicode
from .dev_options import DevOptions

LEVEL_INFO = "提示"
LEVEL_WARN = "警告"


@dataclass
class HealthIssue:
    """一条体检结论。level 只允许 提示/警告 两档（永不"错误/阻止"）。"""

    level: str
    code: str
    title: str
    detail: str = ""
    suggestion: str = ""

    @property
    def is_warning(self) -> bool:
        return self.level == LEVEL_WARN


@dataclass
class HealthReport:
    """一次体检的完整结果。"""

    issues: list = field(default_factory=list)  # list[HealthIssue]
    n_images: int = 0
    classes: list = field(default_factory=list)

    @property
    def n_warnings(self) -> int:
        return sum(1 for i in self.issues if i.is_warning)

    @property
    def n_infos(self) -> int:
        return sum(1 for i in self.issues if not i.is_warning)


def _dhash(img_bgr) -> int:
    """dHash 64 位：缩到 9×8 灰度，比较每行相邻像素亮度。"""
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (9, 8), interpolation=cv2.INTER_AREA)
    bits = np.packbits((g[:, 1:] > g[:, :-1]).flatten())
    return int.from_bytes(bits.tobytes(), "big")


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def _duplicate_groups(hashes: dict, threshold: int) -> list[list]:
    """按汉明距离 ≤ threshold 把图片聚成组（并查集）。hashes: {路径: dhash}。

    T2-4 分桶加速（原实现全量两两比对 O(n²)，万图目录体检冻结主线程）：
    64 位 dHash 切成 8 个 8bit 块建倒排索引——鸽笼原理保证距离 ≤7 的图片
    对至少有一块完全相同，只需块内两两比对；threshold ≥ 8 时块探测不再
    保证召回，回退全量两两比对（结果等价，仅速度差异）。
    """
    paths = list(hashes)
    parent = list(range(len(paths)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    if 0 <= threshold < 8:
        buckets: dict[tuple, list] = {}
        for idx, p in enumerate(paths):
            h = hashes[p]
            for shift in range(56, -1, -8):  # 8 块 × 8bit
                buckets.setdefault((shift, (h >> shift) & 0xFF), []).append(idx)
        seen: set = set()
        for members in buckets.values():
            if len(members) < 2:
                continue
            for a in range(len(members)):
                for b in range(a + 1, len(members)):
                    i, j = members[a], members[b]
                    key = (i, j) if i < j else (j, i)
                    if key in seen:
                        continue
                    seen.add(key)
                    if _hamming(hashes[paths[i]], hashes[paths[j]]) <= threshold:
                        parent[find(i)] = find(j)
    else:
        for i in range(len(paths)):
            for j in range(i + 1, len(paths)):
                if _hamming(hashes[paths[i]], hashes[paths[j]]) <= threshold:
                    parent[find(i)] = find(j)
    groups: dict[int, list] = {}
    for i, p in enumerate(paths):
        groups.setdefault(find(i), []).append(p)
    return [g for g in groups.values() if len(g) > 1]


def _check_imbalance(counts: dict, opt: DevOptions) -> list:
    """类别不均衡（用户 Q3 回执改占比口径）：最小类实例占整体比例 < 阈值 → 警告。"""
    issues = []
    positive = {c: n for c, n in counts.items() if n > 0}
    total = sum(positive.values())
    if len(positive) >= 2 and total > 0:
        lo = min(positive, key=positive.get)
        share = positive[lo] / total
        if share < opt.imbalance_share:
            hi = max(positive, key=positive.get)
            issues.append(HealthIssue(
                LEVEL_WARN, "imbalance", "类别数量不均衡",
                f"最少类「{lo}」{positive[lo]} 个，仅占整体 {share:.1%}"
                f"（阈值 {opt.imbalance_share:.0%}）；最多类「{hi}」{positive[hi]} 个。",
                "工业场景建议开启「工业场景优化」（会自动启用 Focal 类改进损失）；"
                "可能时补充稀少类别样本。"))
    return issues


def _check_few_samples(counts: dict, task: TaskType, opt: DevOptions) -> list:
    floor = opt.min_cls_samples if task is TaskType.CLASSIFICATION \
        else opt.min_det_instances
    unit = "张样本" if task is TaskType.CLASSIFICATION else "个实例"
    few = {c: n for c, n in counts.items() if 0 < n < floor}
    if few:
        shown = "、".join(f"{c}={n}" for c, n in sorted(few.items())[:6])
        more = f" 等 {len(few)} 类" if len(few) > 6 else ""
        return [HealthIssue(
            LEVEL_WARN, "few_samples", "部分类别样本过少",
            f"建议每类至少 {floor} {unit}，当前偏少：{shown}{more}。",
            "样本过少的类别不易学好；可补充采集，或在训练设置中增大"
            "「数据倍增倍数」并配合数据增强。")]
    return []


def _check_val_coverage(task, samples: list, classes: list, val_ratio: float,
                        seed: int, train_ratio=None) -> list:
    """验证集类别缺失：用与训练一致的划分参数模拟抽查。"""
    if val_ratio <= 0 or not samples or not classes:
        return []
    _, val = split_samples(samples, val_ratio, seed,
                           stratify=task is TaskType.CLASSIFICATION,
                           train_ratio=train_ratio)
    if not val:
        return []
    if task is TaskType.CLASSIFICATION:
        val_counts: dict = {}
        for s in val:
            val_counts[s.label] = val_counts.get(s.label, 0) + 1
    else:
        val_counts = {}
        for s in val:
            for sh in s.shapes:
                val_counts[sh.label] = val_counts.get(sh.label, 0) + 1
    missing = [c for c in classes if val_counts.get(c, 0) == 0]
    if missing:
        return [HealthIssue(
            LEVEL_INFO, "val_missing", "部分类别未进入验证集",
            f"按当前验证集比例（{val_ratio:.0%}）划分后，这些类别没有验证样本："
            f"{'、'.join(missing)}。",
            "验证集看不到该类别时，mAP 指标不覆盖它；可增大验证集比例，"
            "或保持「训练集 100% + 验证集 20%」的重叠划分方式。")]
    return []


def _check_unlabeled(dataset_dir: Path) -> list:
    """labelme 结构：存在图片但没有同名 json → 不参与训练。"""
    n = 0
    for p in Path(dataset_dir).rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS \
                and not p.name.startswith(".") and not p.with_suffix(".json").exists():
            n += 1
    if n:
        return [HealthIssue(
            LEVEL_INFO, "unlabeled", "存在未标注的图片",
            f"{n} 张图片没有同名 .json 标注，不会参与训练。",
            "如需使用请到标注平台补标注；若为背景图/废片可忽略。")]
    return []


def _check_duplicates(dataset_dir: Path, samples: list, opt: DevOptions) -> list:
    paths = sorted({Path(s.image_path) for s in samples})
    if len(paths) > opt.dup_scan_cap:
        return [HealthIssue(
            LEVEL_INFO, "dup_skipped", "重复图检测已跳过",
            f"图片数 {len(paths)} 超过检测上限 {opt.dup_scan_cap}（可在开发者选项调整）。",
            "")]
    hashes = {}
    for p in paths:
        img = imread_unicode(p)  # 中文/Unicode 路径安全读取（cv2.imread 不可用）
        if img is None:
            continue
        hashes[str(p)] = _dhash(img)
    groups = _duplicate_groups(hashes, max(0, opt.dup_hamming))
    if groups:
        total = sum(len(g) for g in groups)
        example = "；".join(
            f"{Path(g[0]).name} ↔ {Path(g[1]).name}" for g in groups[:3])
        more = f" 等 {len(groups)} 组" if len(groups) > 3 else ""
        return [HealthIssue(
            LEVEL_INFO, "duplicates", "疑似重复图",
            f"{total} 张图片高度相似（{example}{more}）。",
            "同款工件多角度拍摄可能被误报；确认冗余后删除副本可降低过拟合风险。")]
    return []


def _check_size_outliers(samples: list, opt: DevOptions) -> list:
    areas = [s.width * s.height for s in samples if s.width > 0 and s.height > 0]
    if len(areas) < 4:
        return []
    med = float(np.median(areas))
    mad = float(np.median(np.abs(np.array(areas) - med)))
    mad_eff = mad if mad > 0 else max(1.0, 0.01 * med)  # 全同尺寸时按 1% 视作离群
    k = opt.size_outlier_mad
    outliers = [(s, a) for s, a in zip(samples, areas)
                if abs(a - med) > k * mad_eff]
    if outliers:
        names = "、".join(Path(s.image_path).name for s, _ in outliers[:5])
        more = f" 等 {len(outliers)} 张" if len(outliers) > 5 else ""
        return [HealthIssue(
            LEVEL_INFO, "size_outlier", "存在尺寸离群的图片",
            f"{len(outliers)} 张图片面积明显偏离中位数（>{k:g}×MAD）：{names}{more}。",
            "极端尺寸可能影响训练稳定性；确认是误拍/废片可删除，"
            "或启用增强策略「调整图像大小」统一输入尺寸。")]
    return []


def _check_ignore_regions(samples: list) -> list:
    """O8（V0.7）：`__ignore__` 忽略区域可见性（信息级）。

    告知用户数据里存在忽略区域、将被近似语义剔除的 GT 数量，避免
    "标注突然变少"的困惑；语义与限制见《XCcv 操作流程手册》。
    """
    from ..engine.dataset.ignore import filter_ignore_shapes

    n_regions = n_dropped = 0
    for s in samples:
        _, nr, nd = filter_ignore_shapes(s.shapes)
        n_regions += nr
        n_dropped += nd
    if not n_regions:
        return []
    return [HealthIssue(
        LEVEL_INFO, "ignore_regions", "存在 __ignore__ 忽略区域",
        f"数据集共有 {n_regions} 个忽略区域；训练/评估时将剔除与其重叠率"
        f">50% 的 GT 实例 {n_dropped} 个（近似语义：区域内背景仍参与学习，"
        "区域内检出不计误检）。",
        "适用于反光、丝纹等界限模糊区域：标注平台画矩形/多边形、标签填 "
        "__ignore__ 即可；若非故意标注请删除该标签。")]


def _check_ok_ratio(dataset_dir: Path, task: TaskType, samples: list) -> list:
    """O6（V0.7）：OK 负样本利用与比例（对照 VIMO「合格:缺陷 ≈ 1:1~1:2」）。

    信息级不阻断；口径与训练侧一致——OK 图 = NEGATIVE_DIR_NAMES 目录内
    无同名 json 的图片（reader.scan_negative_images 同源）。
    """
    if task is TaskType.CLASSIFICATION:
        return []
    from ..engine.dataset.reader import scan_negative_images

    n_ok = len(scan_negative_images(dataset_dir))
    n_defect_img = sum(1 for s in samples if s.shapes)
    if n_ok == 0:
        return [HealthIssue(
            LEVEL_INFO, "ok_missing", "未利用 OK 负样本",
            f"当前 {n_defect_img} 张含标注图片参与训练，OK 背景图 0 张。",
            "工业场景误检风险高：建议建一个名为 OK/良品/正常 的子文件夹"
            "放入良品图（无需标注），训练时保持「负样本训练」勾选。")]
    ratio = n_ok / max(1, n_defect_img)
    if ratio < 0.5 or ratio > 2.0:
        return [HealthIssue(
            LEVEL_INFO, "ok_ratio", "OK 图与缺陷图比例偏离建议区间",
            f"OK 图 {n_ok} 张、含标注缺陷图 {n_defect_img} 张"
            f"（约 {ratio:.1%}）；经验建议区间 OK:缺陷 ≈ 1:1~1:2。",
            "OK 过少防误检收益有限；OK 过多会稀释缺陷样本（训练侧超量"
            "自动按种子抽样、验证侧全保留用于误检率统计）。")]
    return []


def run_health_check(task, dataset_dir, options: DevOptions | None = None,
                     val_ratio: float = 0.2, seed: int = 42, train_ratio=None,
                     scan=None) -> HealthReport:
    """执行数据体检，返回 HealthReport（只含 提示/警告 两档）。

    scan 可传入 (samples, classes[, from_file]) 预扫描结果避免重复读盘。
    """
    opt = options or DevOptions()
    task = TaskType(task)
    dataset_dir = Path(dataset_dir)
    issues: list = []
    is_anomaly = task in (TaskType.ANOMALY, TaskType.OCR)
    # A1/A2：无监督异常检测与 OCR 无实例级类别语义——不均衡/单类数量/
    # 验证覆盖/未标注四项跳过（见下方分支守卫）

    if task is TaskType.OCR and scan is None:
        # OCR 数据仍是 labelme 结构（rectangle + label=文本内容），
        # 不能按分类文件夹扫描（否则体检图片数为 0）
        samples, classes, _ = scan_labelme_dataset(dataset_dir)
        scan = (samples, classes)

    if task is TaskType.CLASSIFICATION or is_anomaly:
        if scan is None:
            samples, classes = scan_classification_dataset(dataset_dir)
        else:
            samples, classes = scan[0], scan[1]
        counts: dict = {c: 0 for c in classes}
        for s in samples:
            counts[s.label] = counts.get(s.label, 0) + 1
    else:
        if scan is None:
            samples, classes, _ = scan_labelme_dataset(dataset_dir)
        else:
            samples, classes = scan[0], scan[1]
        counts = {c: 0 for c in classes}
        for s in samples:
            for sh in s.shapes:
                if sh.label == IGNORE_LABEL:   # O8：忽略区域不是类别，不进样本统计
                    continue
                counts[sh.label] = counts.get(sh.label, 0) + 1

    report = HealthReport(n_images=len(samples), classes=list(classes))

    if not samples:
        issues.append(HealthIssue(
            LEVEL_WARN, "empty", "数据集中没有可用样本",
            f"未在 {dataset_dir} 扫描到"
            + ("分类样本" if task is TaskType.CLASSIFICATION else "已标注图片") + "。",
            "请检查数据集路径与目录结构（检测/分割需图片+同名 .json；"
            "分类按子文件夹归档）。"))
        report.issues = issues
        return report

    if not is_anomaly:
        # 异常检测/OCR：无实例级标注语义——不均衡/单类数量/验证覆盖/未标注四项跳过
        issues += _check_imbalance(counts, opt)
        issues += _check_few_samples(counts, task, opt)
        issues += _check_val_coverage(task, samples, classes, val_ratio, seed,
                                      train_ratio)
        issues += _check_unlabeled(dataset_dir)
    if task in (TaskType.DETECTION, TaskType.SEGMENTATION):
        issues += _check_ok_ratio(dataset_dir, task, samples)
        issues += _check_ignore_regions(samples)  # O8（V0.7）：忽略区域可见性
    issues += _check_duplicates(dataset_dir, samples, opt)
    issues += _check_size_outliers(samples, opt)

    report.issues = issues
    return report
