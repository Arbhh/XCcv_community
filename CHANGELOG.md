# Changelog

本文件记录 XCcv Community 的显著变更。格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [0.6.1-community] - 2026-11-11

首个开源发布版本（经发布前评审后发布）。

### 新增

- 标注平台：集成 labelme 5.8.1（vendored，含本地修改，见 `labelme/MODIFICATIONS.md`），
  矩形/多边形标注、自动保存、类别体系管理；AI 智能标注（S 智能多边形 / B 画笔 / L 单张智能标注 /
  目录批量预标注，基于 SAM / EfficientSAM，权重不随仓库分发）
- 训练平台：项目制管理；分类 / 检测 / 实例分割 10 种内置模型；快速配置与专家模式；
  工业优化模块（注意力 / DCNv2 / Focal 系损失等）；训练加速（AMP / TF32 / 早停）；
  切图训练；实时曲线与数据体检
- 测试平台：单图 / 批量推理，阈值实时调节，大图分块推理，结果导出（统一 JSON / labelme / COCO）
- 流程图模块：36 个 AI 与通用算子，Web 画布编排（LiteGraph.js），`.xvap` 打包，3 个模板
- 缺陷样本合成：泊松融合扩充缺陷样本（本地 FastAPI + 内嵌 Web，仅监听回环地址）
