"""轮次权重存档：每次训练共用的权重文件夹，逐轮保存、新训练覆盖。

- 目录：<模型储存路径>/epoch_weights，同一项目所有训练共用一个文件夹；
- 训练启动时清空旧文件（新训练直接覆盖旧权重，磁盘占用只保留最近一次训练）；
- 每轮结束把权重内存序列化 → torch.save 写 .pt；
- 产出为标准 .pt 文件，测试平台可直接加载对比。
"""

import logging
import torch
from pathlib import Path

log = logging.getLogger(__name__)

# 共用文件夹名（模型储存路径下）；scan_runs 等目录扫描需跳过它
EPOCH_DIR_NAME = "epoch_weights"


class EpochWeightStore:
    """轮次权重写入器：reset() 清空共用目录，save() 序列化落盘。"""

    def __init__(self, model_dir):
        self.dir = Path(model_dir) / EPOCH_DIR_NAME

    def reset(self) -> None:
        """训练启动时调用：清空上一训练留下的轮次权重（覆盖式）。"""
        self.dir.mkdir(parents=True, exist_ok=True)
        n = 0
        for f in self.dir.iterdir():
            if f.is_file():
                f.unlink()
                n += 1
        if n:
            log.info("轮次权重文件夹已清空旧内容: %s（%s 个文件）", self.dir, n)

    def save(self, epoch: int, payload) -> Path:
        """保存第 epoch 轮权重：内存序列化 → torch.save 写 .pt。

        目录可能被外部清理（用户删除、同步盘冲突）：自动重建，保证训练
        不因轮次存档中断。
        """
        self.dir.mkdir(parents=True, exist_ok=True)
        dst = self.dir / f"epoch_{epoch:03d}.pt"
        torch.save(payload, dst)
        log.debug("轮次权重已保存: %s (%.1f MB)",
                  dst.name, dst.stat().st_size / 1e6)
        return dst
