# XCcv Community — 深度学习视觉模型训练平台（社区版）

XCcv Community 是一个面向工业视觉场景的**离线桌面端**深度学习模型训练平台社区版，
覆盖 **数据标注 → 模型训练 → 测试评估** 全流程，并附 **缺陷样本合成** 工具。
零基础用户通过桌面 UI 即可完成一个视觉模型从数据到权重的完整落地，全程本地运行、数据不出内网。

## 功能总览

| 模块 | 能力 |
|---|---|
| 标注平台 | 集成 labelme 标注器：矩形 / 多边形、自动保存、类别体系管理；**AI 智能标注**（S 键智能多边形、B 键画笔塑形、L 键单张智能标注、目录批量预标注，基于 SAM / EfficientSAM） |
| 训练平台 | 项目制管理；图像分类 / 目标检测 / 实例分割三类任务共 10 种内置模型（ResNet / MobileNetV3 / YOLO11 / Faster R-CNN / Mask R-CNN）；快速配置与专家模式双轨参数；工业优化模块（CBAM/ECA 注意力、DCNv2、Focal 系损失、Dice+BCE 等）；训练加速（冻结骨干、AMP、TF32、早停）；切图训练与小目标锚框自适应；实时训练曲线、数据体检、多训练对比 |
| 测试平台 | 单图 / 批量推理；置信度与 IOU 阈值实时调节；类别筛选；4K 大图分块推理；结果导出为平台统一 JSON / labelme / COCO 格式；批量统计摘要 |
| 流程图 | 与 AI 平台同级的独立模块：内置 **36 个 AI 与通用算子**（模型推理 / 检出过滤 / 裁图与合并 / 逐图执行 / 结果合并 / 逻辑判定 / 输出）；Web 画布拖拽编排，节点库分「AI 算子 / 通用算子」；支持运行预览、保存 JSON、打包 / 解包 `.xvap`、批量运行；附 3 个模板（AI 过滤、级联裁图合并、检测→分割级联） |
| 缺陷样本合成 | 把缺陷贴图以泊松融合自然合成到良品图上，扩充稀缺缺陷样本（独立进程，内嵌 Web 界面） |

> 社区版定位为主体能力的开源阉割版：**保留流程图基础能力**（AI + 通用算子编排，36 节点），
> 但不含传统视觉算子（预处理 / 定位 / 测量）与外部算法桥接。完整版另含：部署推理服务
> （HTTP/WebSocket API 与 C# SDK）、模型资产加密体系、模型工具（结构化剪枝 / 加解密 /
> TensorRT 加速）、完整 64 算子与 HALCON / VisionMaster 桥接。以上能力不在本仓库范围内。

## 环境要求

- Windows 10 / 11（x64）
- Python 3.11（建议使用独立虚拟环境）
- GPU：可选。NVIDIA 显卡可显著加速训练与推理；CPU 亦可运行（推荐配合小模型）
- 磁盘：依赖（PyTorch 等）约需 6~8 GB 空间

## 安装

```bat
:: 1) 创建并激活虚拟环境（Python 3.11）
python -m venv venv
venv\Scripts\activate

:: 2) 安装 PyTorch（按官方源选择 CUDA / CPU 版本，示例为 CUDA 12.8）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

:: 3) 安装其余依赖
pip install -r requirements.txt
```

## 快速开始

> **启动环境（重要）**：主界面与流程画布、缺陷样本合成都依赖 PyQtWebEngine。
> `run_*.bat` 会依次尝试 `XCCV_PY` 环境变量 → 本目录 `venv` → `.venv` → PATH 中的 python，
> 并在启动前校验 QtWebEngine；校验失败会给出中文提示而不是静默打不开窗口。
> 若使用自己的解释器，请确认已安装 `PyQtWebEngine`（见 requirements.txt）。

双击 `run_main.bat`（或 `python -m xccv.app`）打开主界面，按卡片进入各平台：

1. **标注**：新建项目 → 导入图片 → 绘制矩形 / 多边形（可选按 S 使用 AI 智能多边形）；
2. **训练**：选择项目与模型 → 快速配置或专家模式 → 开始训练，实时查看曲线；
3. **测试**：加载项目最新模型（best.pt）→ 单图 / 批量推理 → 导出 JSON / 可视化；
4. **流程图**（左侧导航同级模块）：选择数据路径与流程 JSON →「运行流程」看单图结果，
   或「编辑流程」在画布中拖拽 AI / 通用算子编排，支持打包为 `.xvap` 与批量运行；
5. **缺陷样本合成**（可选）：主界面卡片进入，合成缺陷样本扩充训练集。

命令行入口（等价于主界面卡片）：

```bat
run_annotation.bat   :: 标注平台
run_training.bat     :: 训练平台
run_testing.bat      :: 测试平台
run_flow.bat         :: 流程图（独立窗口；主界面左侧导航亦可进入）
run_defectgen.bat    :: 缺陷样本合成
```

详细上手文档见 [docs/quickstart.md](docs/quickstart.md)。

### 可选：AI 智能标注权重

AI 智能标注（智能多边形 / 画笔 / 智能标注 / 批量预标注）需要 SAM 系列权重，默认不随仓库分发。
请自行从官方渠道获取（如 `sam2.1_t.pt`、`mobile_sam.pt`，均允许商用），放到仓库根目录的
`weights/sam/` 下；缺失时相关功能会给出明确指引，其余功能不受影响。

## 目录结构

```
XCcv-Community/
├── xccv/                # 主包（主界面 / 标注 / 训练 / 测试 / 缺陷样本合成 / 引擎）
│   ├── annotation/      #   标注平台（labelme 宿主 + AI 智能标注）
│   ├── training/        #   训练平台
│   ├── testing/         #   测试平台
│   ├── defectgen/       #   缺陷样本合成（FastAPI + 内嵌 Web）
│   ├── flow_platform.py #   流程图页（与 AI 平台同级；图片树 / 运行 / 批量）
│   ├── modeltools/      #   流程画布宿主（QtWebEngine + LiteGraph）与流程会话/模板
│   ├── engine/          #   纯逻辑引擎（数据管线 / 模型注册表 / 指标 / 结果契约）
│   │   └── flow/        #     流程图引擎（类型化端口 IR / 执行器 / 36 算子）
│   └── assets/          #   图标资源
├── labelme/             # vendored labelme 5.8.1（GPLv3，含本地修改，清单见 labelme/MODIFICATIONS.md）
├── docs/                # 快速上手文档
├── licenses/            # 许可证文本（AGPL-3.0 / labelme GPLv3）
├── requirements.txt
└── run_*.bat            # 各平台启动脚本
```

运行期数据（项目、日志、运行配置）生成于 `platform_data/`，不随仓库分发。

## 许可

- 本项目代码以 **GNU Affero General Public License v3.0（AGPL-3.0）** 发布，见 [LICENSE](LICENSE)。
- `labelme/` 目录为 labelme 5.8.1 的 vendored 副本并含本地修改，遵循 **GPLv3**，
  许可文本见 [licenses/labelme-LICENSE-GPLv3.txt](licenses/labelme-LICENSE-GPLv3.txt)。
- 训练链路使用的 ultralytics（YOLO11 / SAM）为 **AGPL-3.0** 许可；如需闭源商用，
  请自行评估其企业许可方案。
- 第三方依赖清单与许可见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 免责声明

本项目按"现状"提供，不附带任何明示或默示的担保。使用本工具训练的模型用于生产环境前，
请自行完成充分验证。
