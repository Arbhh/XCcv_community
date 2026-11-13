# labelme vendored 副本本地修改清单

本目录为 [labelme](https://github.com/wkentaro/labelme) **v5.8.1**（tag `v5.8.1`）的源码副本，
并包含针对 XCcv Community 平台的本地修改。按 GPLv3 第 5 条（a）项要求，
在此显著声明全部修改。上游版权与许可声明保持完好，许可证文本见
[licenses/labelme-LICENSE-GPLv3.txt](../licenses/labelme-LICENSE-GPLv3.txt)。

修改部分与本仓库其余代码一并按 **AGPL-3.0** 授权（GPLv3 §13 允许 GPLv3 作品并入
AGPL-3.0 组合作品）。

对比基线：上游 tag `v5.8.1` 源码包。下表 `+N/-M` 为 unified diff 的新增/删除行数统计，
可用以下命令复现：

```bash
curl -sL https://github.com/wkentaro/labelme/archive/refs/tags/v5.8.1.tar.gz | tar xz
diff -r labelme-5.8.1/labelme labelme/
```

## 修改的文件（15 个）

| 文件 | 规模 | 修改性质 |
|---|---|---|
| `__init__.py` | +17/-12 | `__appname__` 改为「XCcv 标注平台」；宿主初始化调整 |
| `app.py` | +1029/-184（0.7 追加 +281/-6） | 接入 AI 智能标注（模型选择、参数对话框 `_ai_params` / `_fill_ai_models` 等）；类别归类归档 `_archive_to_class_dir`；文件列表右键菜单；灰度预览 `_apply_grayscale_if_enabled`；0.7 追加 O9 OK 徽标/设为 OK 目录、O10 标注质量要点/AI 复核漏标、AI 等待横幅 |
| `config/__init__.py` | +2/-2 | 配置文件读取显式 `encoding="utf-8"`（Windows 中文路径兼容） |
| `config/default_config.yaml` | +12/-9 | 默认开启自动保存与 AI 多边形/掩码/画笔（`auto_save: true` 等）；快捷键体系调整为 Q/W/E/I/S/B/T/V |
| `label_file.py` | +199/-165 | 标注文件读写适配（默认 `store_data: false` 后的图像数据策略等） |
| `shape.py` | +13/-6 | 未标类形状（画笔直绘）的兜底线色/填充色/顶点色 |
| `utils/image.py` | +128/-82 | 图像编解码工具函数适配 |
| `utils/qt.py` | +103/-78 | Qt 控件构建工具适配（图标、快捷键格式化等） |
| `widgets/ai_prompt_widget.py` | +170/-131 | AI 提示参数面板：新增置信度/IOU 阈值、NMS 参数、文本提示控件（`_ScoreThresholdWidget` / `_IouThresholdWidget` / `_NmsParamsWidget` / `_TextPromptWidget`） |
| `widgets/brightness_contrast_dialog.py` | +86/-57 | 亮度/对比度对话框行为适配 |
| `widgets/canvas.py` | +2142/-953（0.7 追加 +6） | 画布接入 AI 智能多边形与画笔标注（锚点交互 `_ai_add_anchor`、异步编解码 `_ai_decode_async` / `_ai_decode_worker`、结果提交 `_ai_commit_pending` 等 `_ai_*` 系列）；0.7 追加加载失败解锁与嵌入等待提示 |
| `widgets/color_dialog.py` | +36/-27 | 颜色对话框适配 |
| `widgets/file_dialog_preview.py` | +80/-58 | 文件对话框预览适配 |
| `widgets/label_dialog.py` | +243/-215 | 类别对话框适配（类别体系管理等） |
| `widgets/label_list_widget.py` | +182/-133 | 标注列表控件适配 |

## 新增的文件（1 个）

| 文件 | 说明 |
|---|---|
| `language.py` | 中英文语言设置与翻译层：补充官方 `zh_CN.qm` 缺失键，未命中时返回 `None` 以保留 Qt 回退链 |

## 删除的文件（1 个）

| 文件 | 说明 |
|---|---|
| `_automation/bbox_from_text.py` | 上游自动化脚本，未随社区版分发 |

0.7.0 追加修改（2026-11-13）：

- `app.py`：O9 OK/良品目录徽标与右键“设为 OK 目录”；O10 帮助菜单“标注质量要点”“AI 复核当前图（疑似漏标）”；AI 等待顶部横幅。
- `widgets/canvas.py`：AI 模型加载失败时解除 `_ai_loading` 锁；首次点击嵌入未就绪时状态提示。

## 未修改的部分

`icons/`、`translate/`（含 `zh_CN.ts` / `zh_CN.qm`）、`testing.py`、`cli/`、`_automation/polygon_from_mask.py`
及 `utils/_io.py` 等其余文件与上游 v5.8.1 完全一致；`utils/_io.py`、`utils/image.py`、`utils/shape.py`
中的上游版权声明（Copyright (c) Kentaro Wada）均已保留。
