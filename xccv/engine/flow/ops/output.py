"""输出类算子（v0.4.2 用户批注 Q5 建立输出节点；修复循环 3 按用户批注统一）。

端口约定（用户批注：输出节点不应再有输出端口；输入统一，通常 image+result）：
- 全部输出节点**没有输出端口**（流程终点）；
- 保存结果 result_output：image（可选，提供宽高）+ result（可选）→ 统一 JSON 落盘；
- 输出结果 result_view：同输入但**不落盘**——结果进入运行摘要与画布，
  供测试平台/服务直接消费（用户批注：只有保存结果没有输出结果，补齐）；
- 保存图像 image_sink：image + result（可选，叠加蒙版用）；
- 保存表格 csv_sink：result（检出行）+ image（可选）+ number（可选，测量行）
  ——表格节点按用户批注允许多种输入。

落盘位置：params.out_dir（留空 = 会话键 flow_dir 的「流程结果/」子目录，
即流程 JSON 同目录；工作台单跑时为图片所在目录）。文件名 = 会话键
source_name（来源图名）派生或时间戳。

判定 NG 通路：判定类节点（数值判定/布尔运算/多路判定/数量判定）的判定
随运行摘要（verdict）与统一 JSON 的顶层 verdict 字段对外可读。
"""

from datetime import datetime

import numpy as np

from ..ir import Geometry, ImageData, ImageList, MeasureResult, NumberVal, ResultList
from ..node_base import ParamSpec, Port, VisionNode
from ..registry import register_node


def _result_dir(ctx, params) -> "Path":
    from pathlib import Path

    out = str(params.get("out_dir") or "").strip()
    if out:
        return Path(out)
    flow_dir = ctx.get("flow_dir")
    base = Path(flow_dir) if flow_dir else Path.cwd()
    return base / "流程结果"


def _collect_result(v, ctx):
    """统一结果值 → (det_list, cls_list, geo_list, meas_list, verdict)。

    det/cls 经 result_json 与 AI 检出同构；几何/测量进扩展数组；
    判定类结果（布尔/携带 verdict 的检出）折算 verdict。
    """
    from ...result_json import classification_topk, detection_rows

    det_list: list = []
    cls_list: list = []
    geo_list: list = []
    meas_list: list = []
    verdict: bool | None = None
    # 结果列表（逐图结果）逐条展开
    items = list(v.items) if isinstance(v, ResultList) else [v]
    for it in items:
        value = getattr(it, "value", it)
        if isinstance(value, (bool, np.bool_)):
            verdict = bool(value) if verdict is None else (verdict and bool(value))
            continue
        if isinstance(value, (MeasureResult, NumberVal)):
            meas_list.append(value)
            if getattr(value, "passed", None) is False:
                verdict = False   # 任一 NG 即整体 NG（AND 语义）
            continue
        if isinstance(value, Geometry):
            geo_list.append(value)
            continue
        boxes = getattr(value, "boxes", None)
        if boxes is not None:
            extra = getattr(value, "extra", None)
            classes = ((extra.get("classes") if isinstance(extra, dict) else None)
                       or ctx.get("model_classes")
                       or ctx.get("flow_class_names") or [])
            if isinstance(extra, dict) and isinstance(extra.get("verdict"), bool):
                ev = extra["verdict"]
                verdict = ev if verdict is None else (verdict and ev)
            det_list.extend(detection_rows(value, classes))
            cls_list.extend(classification_topk(value, classes))
    return det_list, cls_list, geo_list, meas_list, verdict


@register_node
class ResultOutput(VisionNode):
    """保存结果：把结果按统一 JSON schema 写盘（传统算法流程的出口之一）。

    输入：图像（可选，提供宽高）+ 统一结果（可选：检出/几何/测量/判定/
    逐图结果列表）。结果 JSON 与 AI 检出完全同 schema_version，顶层带
    verdict 判定（接入判定类节点时为 OK/NG），C# 等外部接口可直接消费。
    """

    NAME = "result_output"
    TITLE = "保存结果"
    CATEGORY = "输出"
    INPUTS = (Port("image", "image", required=False, label="图像"),
              Port("result", "result", required=False, label="结果"))
    OUTPUTS = ()
    PARAMS = {
        "out_dir": ParamSpec("out_dir", "输出目录", "path", "", folder=True,
                             tooltip="留空 = 流程文件同目录「流程结果/」；"
                                     "批量运行时由批量设置决定"),
    }

    def run(self, inputs, params, ctx):
        from ...result_json import build_flow_result_json, save_result_json

        det_list, cls_list, geo_list, meas_list, verdict = _collect_result(
            inputs.get("result"), ctx)
        img = inputs.get("image")
        source = {"file": str(ctx.get("source_name") or ""),
                  "path": "", "width": None, "height": None}
        if img is not None and getattr(img.mat, "ndim", 0) >= 2:
            source["width"] = int(img.mat.shape[1])
            source["height"] = int(img.mat.shape[0])
        if det_list or cls_list or geo_list or meas_list or img is not None:
            payload = build_flow_result_json(
                source=source, detections=det_list, classifications=cls_list,
                geometries=geo_list,
                measurements=meas_list, flow_name=str(ctx.get("flow_name") or ""),
                verdict=({"passed": verdict, "source": "flow",
                          "detail": "判定" + ("通过" if verdict else "未通过")}
                         if verdict is not None else None))
        else:
            ctx.warn("保存结果：未连接任何输入，已写出空结果文件")
            payload = build_flow_result_json(
                source=source, flow_name=str(ctx.get("flow_name") or ""),
                error="未连接任何输入")
        out_dir = _result_dir(ctx, params)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = ""
        name = str(ctx.get("source_name") or "")
        if name:
            stem = name.rsplit(".", 1)[0]
        if not stem:
            stem = datetime.now().strftime("result_%Y%m%d_%H%M%S")
        out_path = out_dir / f"{stem}.json"
        save_result_json(out_path, payload)
        ctx.node_log(0, self.NAME, f"结果已保存：{out_path}")
        return {}


def _encode_to(path, mat, fmt: str, quality: int) -> bool:
    """中文路径安全写图（imencode+tofile），jpg 支持质量参数。"""
    import cv2

    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    params = []
    if fmt == "jpg":
        params = [cv2.IMWRITE_JPEG_QUALITY, int(quality)]
    ok, buf = cv2.imencode(f".{fmt}", mat, params)
    if ok:
        buf.tofile(str(path))
    return bool(ok)


def _stem(ctx) -> str:
    name = str(ctx.get("source_name") or "")
    stem = name.rsplit(".", 1)[0] if name else ""
    return stem or datetime.now().strftime("image_%Y%m%d_%H%M%S")


@register_node
class ResultView(VisionNode):
    """输出结果：流程的**不落盘**出口——结果进入运行摘要与测试平台/服务，
    不写任何文件（与「保存结果」成对，用户批注：缺少不落盘的输出节点）。

    输入与「保存结果」一致（图像可选 + 统一结果可选）。运行后：节点日志
    给出摘要；FlowSession 运行摘要带 verdict 判定；服务/测试平台从执行器
    结果直接读取本节点数据。
    """

    NAME = "result_view"
    TITLE = "输出结果"
    CATEGORY = "输出"
    INPUTS = (Port("image", "image", required=False, label="图像"),
              Port("result", "result", required=False, label="结果"))
    OUTPUTS = ()
    PARAMS = {}

    def run(self, inputs, params, ctx):
        det_list, cls_list, geo_list, meas_list, verdict = _collect_result(
            inputs.get("result"), ctx)
        parts = []
        if det_list:
            parts.append(f"检出 {len(det_list)} 个")
        if cls_list:
            parts.append(f"分类 {len(cls_list)} 条")
        if geo_list:
            parts.append(f"几何 {len(geo_list)} 条")
        if meas_list:
            parts.append(f"测量/数值 {len(meas_list)} 条")
        if verdict is not None:
            parts.append("判定 " + ("OK" if verdict else "NG"))
        text = "；".join(parts) if parts else "未连接结果"
        ctx.set("flow_output_summary", text)
        ctx.node_log(0, self.NAME, f"输出结果：{text}")
        return {}


@register_node
class ImageSink(VisionNode):
    """保存图像：把图像（可选叠加检出框蒙版）写盘，中文路径安全（V0.6 T3-5）。

    what=image 保存原图；overlay 把结果输入的检出以半透明蒙版+
    描边叠加后保存（与快照叠加同一画法）。文件名=来源图名+后缀。
    """

    NAME = "image_sink"
    TITLE = "保存图像"
    CATEGORY = "输出"
    INPUTS = (Port("image", "image", label="图像"),
              Port("result", "result", required=False, label="结果"))
    OUTPUTS = ()
    PARAMS = {
        "out_dir": ParamSpec("out_dir", "输出目录", "path", "", folder=True,
                             tooltip="留空 = 流程文件同目录「流程结果/图像/」"),
        "what": ParamSpec("what", "保存内容", "enum", "image", options=["image", "overlay"],
                          tooltip="image=原图；overlay=叠加检出框蒙版"),
        "fmt": ParamSpec("fmt", "格式", "enum", "png", options=["png", "jpg", "bmp"]),
        "quality": ParamSpec("quality", "JPG 质量", "int", 95, range=(1, 100),
                             tooltip="仅 jpg 生效"),
        "name_suffix": ParamSpec("name_suffix", "文件名后缀", "text", "",
                                 tooltip="如 _ok；留空则与来源图同名"),
    }

    def run(self, inputs, params, ctx):
        from pathlib import Path

        img = inputs["image"]
        if img.meta.get("empty"):
            ctx.warn("保存图像：输入为空图（上游无结果），跳过保存")
            return {}
        mat = img.mat
        if params["what"] == "overlay":
            det = inputs.get("result")
            det = getattr(det, "value", det)   # 结果列表取原值
            if det is None or not hasattr(det, "boxes"):
                ctx.warn("保存图像：overlay 需要连接结果输入（检出），已按原图保存")
            else:
                from ..snapshot import overlay_result

                mat = overlay_result(mat, {"detection": det})
        out = str(params.get("out_dir") or "").strip()
        out_dir = Path(out) if out else _result_dir(ctx, params) / "图像"
        fmt = params["fmt"]
        path = out_dir / f"{_stem(ctx)}{params.get('name_suffix') or ''}.{fmt}"
        ok = _encode_to(path, mat, fmt, params["quality"])
        if ok:
            ctx.node_log(0, self.NAME, f"图像已保存：{path}")
        else:
            ctx.warn(f"保存图像：编码失败 {path}")
        return {}


CSV_COLUMNS = ("source", "time", "index", "label", "label_id", "score", "x1", "y1",
               "x2", "y2", "bbox_area", "mask_area", "value", "unit", "passed")


def parse_csv_columns(text) -> list:
    cols = [c.strip() for c in str(text or "").replace("，", ",").split(",") if c.strip()]
    bad = [c for c in cols if c not in CSV_COLUMNS]
    if bad:
        raise ValueError(f"未知列：{', '.join(bad)}（可选：{', '.join(CSV_COLUMNS)}）")
    if not cols:
        raise ValueError("至少需要一列")
    return cols


@register_node
class CsvSink(VisionNode):
    """保存表格：检出/测量/判定按行写 CSV（utf-8-sig，Excel 打开中文不乱码）。

    输入（多种可选，按用户批注表格节点可接多种输入）：结果（检出 → 每检出
    一行；测量/判定 → 汇总行）、数值（测量数值行）、图像（可选，无消费）。
    append=True 追加，文件已存在但表头与当前列配置不一致时**表头自愈**：
    旧文件改名备份（*.bak_时间戳.csv）后以新表头重建，数据不丢。
    """

    NAME = "csv_sink"
    TITLE = "保存表格"
    CATEGORY = "输出"
    INPUTS = (Port("image", "image", required=False, label="图像"),
              Port("result", "result", required=False, label="结果"),
              Port("number", "number", required=False, label="数值"))
    OUTPUTS = ()
    PARAMS = {
        "out_path": ParamSpec("out_path", "CSV 文件", "path", "",
                              tooltip="留空 = 流程文件同目录「流程结果/results.csv」"),
        "columns": ParamSpec("columns", "列", "text", "source,time,label,score,x1,y1,x2,y2",
                             tooltip="逗号分隔，可选：" + ",".join(CSV_COLUMNS)),
        "append": ParamSpec("append", "追加写入", "bool", True, tooltip="关闭则每次覆盖"),
    }

    def validate(self, params):
        errs = super().validate(params)
        try:
            parse_csv_columns(params.get("columns"))
        except ValueError as e:
            errs.append(str(e))
        return errs

    def _rows(self, inputs, ctx) -> list:
        from ...result_json import detection_rows

        base = {"source": str(ctx.get("source_name") or ""),
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        num = inputs.get("number")
        value, unit = "", ""
        rv = inputs.get("result")
        rv = getattr(rv, "value", rv) if isinstance(rv, ResultList) else rv
        if isinstance(rv, MeasureResult):
            value = rv.value if rv.value is not None else rv.values
            unit = rv.unit
            passed = rv.passed
        elif isinstance(rv, bool):
            value, passed = "", rv
        elif isinstance(num, NumberVal):
            value, unit, passed = num.value, num.unit, None
        else:
            passed = None
        common = dict(base, value=value, unit=unit,
                      passed="" if passed is None else bool(passed))
        rows = []
        boxes = getattr(rv, "boxes", None)
        if boxes is not None:
            extra = getattr(rv, "extra", None)
            classes = ((extra or {}).get("classes") if isinstance(extra, dict) else None) \
                or ctx.get("model_classes") or ctx.get("flow_class_names") or []
            for i, r in enumerate(detection_rows(rv, classes)):
                x1, y1, x2, y2 = r["bbox"]
                rows.append(dict(common, index=i, label=r["label"], label_id=r["label_id"],
                                 score=r["score"], x1=x1, y1=y1, x2=x2, y2=y2,
                                 bbox_area=r["bbox_area"], mask_area=r.get("mask_area", "")))
        if not rows and (value != "" or passed is not None):
            rows.append(common)
        return rows

    def run(self, inputs, params, ctx):
        import csv
        from pathlib import Path

        cols = parse_csv_columns(params["columns"])
        out = str(params.get("out_path") or "").strip()
        path = Path(out) if out else _result_dir(ctx, params) / "results.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = self._rows(inputs, ctx)
        if not rows:
            ctx.warn("保存表格：没有可写的数据（未连接输入或无检出），跳过")
            return {}
        mode = "a" if params["append"] else "w"
        write_header = True
        if mode == "a" and path.exists() and path.stat().st_size > 0:
            with open(path, encoding="utf-8-sig", newline="") as f:
                first = next(csv.reader(f), [])
            if first == cols:
                write_header = False
            else:  # 表头自愈：旧文件备份后重建
                bak = path.with_name(
                    f"{path.stem}.bak_{datetime.now():%Y%m%d_%H%M%S}{path.suffix}")
                path.rename(bak)
                ctx.warn(f"保存表格：表头与列配置不一致，旧文件已备份为 {bak.name}")
        with open(path, mode, encoding="utf-8-sig" if write_header else "utf-8",
                  newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(cols)
            for r in rows:
                w.writerow(["" if r.get(c) is None else r.get(c, "") for c in cols])
        ctx.node_log(0, self.NAME, f"表格已写入 {len(rows)} 行：{path}")
        return {}