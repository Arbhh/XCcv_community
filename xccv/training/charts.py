"""训练平台图表（深色主题、随窗口自适应）：
- 类别分布：横向条形图
- 图像尺寸分布：气泡分布图（宽、高坐标上以圆的大小表示该尺寸附近的图片数量）
- 混淆矩阵：热力图
"""

import matplotlib

matplotlib.use("Qt5Agg")

from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import numpy as np
from PyQt5.QtWidgets import QSizePolicy

from ..ui_style import ACCENT, ACCENT2, BG, FG, GRID, MUTED, PANEL

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

_BUCKET = 16  # 尺寸分布的取整粒度（像素）


def _style_axes(ax):
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=11)
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)
    ax.title.set_color(FG)
    ax.grid(color=GRID, alpha=0.4)


class _WheelThroughCanvas(FigureCanvasQTAgg):
    """滚轮事件透传：画布不拦截滚轮，浏览时父级滚动区可直接滚动页面。"""

    def wheelEvent(self, ev):
        ev.ignore()


class DistributionCanvas(_WheelThroughCanvas):
    """数据集可视化：类别分布（条形）+ 尺寸分布（气泡图）。"""

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(8.0, 3.4), constrained_layout=True)
        self.fig.patch.set_facecolor(BG)
        super().__init__(self.fig)
        if parent is not None:
            self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumHeight(230)
        self.ax_bar, self.ax_bubble = self.fig.subplots(1, 2)
        self._style()

    def _style(self):
        _style_axes(self.ax_bar)
        self.ax_bar.set_title("类别分布", fontsize=13)
        _style_axes(self.ax_bubble)
        self.ax_bubble.set_title("图像尺寸分布（圆大小 = 数量）", fontsize=13)
        self.ax_bubble.set_xlabel("宽（像素）", fontsize=10)
        self.ax_bubble.set_ylabel("高（像素）", fontsize=10)

    def update_stats(self, stats) -> None:
        # 类别分布（横向条形，类名多时更易读）
        self.ax_bar.clear()
        self._style()
        names = list(stats.class_counts)
        counts = [stats.class_counts[n] for n in names]
        y = range(len(names))
        self.ax_bar.barh(y, counts, color=ACCENT, height=0.62)
        self.ax_bar.set_yticks(y)
        self.ax_bar.set_yticklabels(names, fontsize=10)
        self.ax_bar.invert_yaxis()
        self.ax_bar.set_xlim(0, max(max(counts, default=0) * 1.18, 1))
        self.ax_bar.set_xlabel("数量", fontsize=10)

        # 尺寸分布（气泡图：坐标从 0 开始，便于直观比较尺寸差距）
        self.ax_bubble.clear()
        self._style()
        if stats.widths:
            max_dim = max(max(stats.widths), max(stats.heights))
            bucket = max(_BUCKET, int(max_dim / 40))  # 聚合粒度随尺寸自适应
            counter: dict = {}
            for w, h in zip(stats.widths, stats.heights):
                key = (max(bucket, round(w / bucket) * bucket),
                       max(bucket, round(h / bucket) * bucket))
                counter[key] = counter.get(key, 0) + 1
            ws = [k[0] for k in counter]
            hs = [k[1] for k in counter]
            cs = np.array([counter[k] for k in counter], dtype=float)
            sizes = 30 + 260 * (cs / cs.max())
            self.ax_bubble.scatter(ws, hs, s=sizes, color=ACCENT2,
                                   alpha=0.6, edgecolors=FG, linewidths=0.6)
            for (w, h), c in counter.items():
                if c >= max(2, cs.max() * 0.5):
                    self.ax_bubble.annotate(int(c), (w, h), fontsize=9,
                                            color=FG, ha="center", va="center")
            # 坐标轴从 0 开始、各自以宽/高的最大值为上限（最大图位于右上角），
            # 对角线从原点连到 (x 最大, y 最大)，横贯整个绘图区
            lim_x = max(ws) * 1.08
            lim_y = max(hs) * 1.08
            self.ax_bubble.set_xlim(0, lim_x)
            self.ax_bubble.set_ylim(0, lim_y)
            self.ax_bubble.plot([0, lim_x], [0, lim_y], linestyle="--",
                                color=MUTED, linewidth=0.8, alpha=0.6)
        self.draw()


class SplitDonutCanvas(_WheelThroughCanvas):
    """O12（V0.7）：训练/验证划分环形图（已标训练/验证 + OK 训练/验证）。"""

    _COLORS = (ACCENT, "#f58518", "#54a24b", "#b279a2")

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(4.6, 3.4), constrained_layout=True)
        self.fig.patch.set_facecolor(BG)
        super().__init__(self.fig)
        if parent is not None:
            self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumHeight(230)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor(PANEL)
        self.ax.text(0.5, 0.5, "暂无数据", ha="center", va="center",
                     color=MUTED, fontsize=11)
        self.ax.set_axis_off()

    def update_split(self, n_labeled: int, n_ok: int, val_ratio: float) -> None:
        """按比例近似展示划分（划分种子细节以体检/日志为准，本图为示意）。"""
        self.ax.clear()
        self.ax.set_facecolor(PANEL)
        val_ratio = min(max(float(val_ratio), 0.0), 0.9)
        segs = []
        if n_labeled:
            segs.append((f"已标训练 {round(n_labeled * (1 - val_ratio))}",
                         n_labeled * (1 - val_ratio)))
            segs.append((f"已标验证 {round(n_labeled * val_ratio)}",
                         n_labeled * val_ratio))
        if n_ok:
            segs.append((f"OK 训练 {round(n_ok * (1 - val_ratio))}",
                         n_ok * (1 - val_ratio)))
            segs.append((f"OK 验证 {round(n_ok * val_ratio)}",
                         n_ok * val_ratio))
        if not segs or sum(v for _, v in segs) <= 0:
            self.ax.text(0.5, 0.5, "暂无数据", ha="center", va="center",
                         color=MUTED, fontsize=11)
            self.ax.set_axis_off()
            self.draw()
            return
        names = [s[0] for s in segs]
        values = [max(s[1], 1e-9) for s in segs]
        wedges, _ = self.ax.pie(values, colors=self._COLORS[:len(values)],
                                startangle=90,
                                wedgeprops=dict(width=0.42, edgecolor=BG))
        self.ax.set_title("训练 / 验证划分", fontsize=13, color=FG)
        self.ax.legend(wedges, names, loc="center left",
                       bbox_to_anchor=(1.0, 0.5), fontsize=9,
                       frameon=False, labelcolor=FG)
        self.ax.set_aspect("equal")
        self.draw()


class ConfusionMatrixCanvas(_WheelThroughCanvas):
    """混淆矩阵热力图（含背景行列，用于检测/分割的漏检误检）。"""

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(5.2, 4.2), constrained_layout=True)
        self.fig.patch.set_facecolor(BG)
        super().__init__(self.fig)
        if parent is not None:
            self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumHeight(260)
        self.ax = self.fig.add_subplot(111)

    def update_matrix(self, matrix, classes) -> None:
        # 必须整图重建：colorbar 是独立 axes，仅 ax.clear() 清不掉，
        # 来回切换训练记录时 colorbar 会逐次累积
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor(PANEL)
        m = np.asarray(matrix, dtype=int)
        labels = list(classes)
        if m.shape[0] == len(classes) + 1:
            labels = labels + ["背景"]
        im = self.ax.imshow(m, cmap="Blues")
        self.ax.set_xticks(range(len(labels)))
        self.ax.set_yticks(range(len(labels)))
        self.ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10, color=FG)
        self.ax.set_yticklabels(labels, fontsize=10, color=FG)
        self.ax.set_xlabel("预测类别", color=FG)
        self.ax.set_ylabel("真实类别", color=FG)
        for spine in self.ax.spines.values():
            spine.set_color(GRID)
        self.ax.tick_params(colors=MUTED)
        vmax = int(m.max()) if m.size else 0
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                v = int(m[i, j])
                if v:
                    self.ax.text(j, i, str(v), ha="center", va="center", fontsize=9,
                                 color="white" if vmax and v > vmax / 2 else "black")
        self.fig.colorbar(im, ax=self.ax, fraction=0.046)
        self.draw()
