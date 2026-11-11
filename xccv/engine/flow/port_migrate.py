"""旧端口布局迁移（V0.6 修复循环 3/4：端口统一大改的兼容层）。

背景：连线在流程 JSON 里按【槽位序号】存储（[id, src, src槽, dst, dst槽]）。
修复循环 3 把端口收敛为 image/image_list/result/number 四类并调整了
各节点的端口表——旧图直接加载会把线接错槽；修复循环 4 取消 image_list
端口类型（图像端口统一承载单张/多张）并删除「取列表第k张」节点（槽位
一一对应，无需重排；仅移除 list_pick 节点及其连线）。本模块提供：

- `migrate_ports(flow)`：对缺少当前 "schema" 标记的图按表迁移连线槽位；
  缺失的映射（被移除的便利端口，如 centers/line2/saved）连线删除；
  并按「图像在上、结果在下」重排各节点端口槽位（循环四）；
  schema<3 时删除 list_pick 节点及其连线（并行输入语义下不再需要）。
- 未列入表的节点 = 槽位一一对应（多数节点旧槽 0 的 result 类端口语义
  平移：detection/geometry/measure/bool → result）。

迁移后的图写入 "schema": 3 标记；新保存的图一律带标记（不再重复迁移）。
"""

# 每节点：{"in": {旧输入槽: 新输入槽或 None(删除)}, "out": {旧输出槽: ...}}
# 未列出的旧槽 = 删除（保守：宁删勿错接）。
_PORT_MIGRATE: dict = {
    # ---- 输入端口变化 ----
    "relabel_map":        {"in": {0: 0}, "out": {0: 0}},
    "mask_to_box":        {"in": {0: 0}, "out": {0: 0}},
    "roi_batch_classify": {"in": {0: 0, 1: 1}, "out": {0: 0}},
    "detection_to_measure": {"in": {0: 0}, "out": {0: 0, 2: 1}},   # centers 移除
    "geometry_to_affine": {"in": {0: 0, 1: 1}, "out": {0: 0, 1: 1}},
    "filter_count_check": {"in": {0: 0}, "out": {0: 0, 2: 2}},     # passed 端口移除（verdict 走结果 JSON）
    "measure_distance":   {"in": {0: 0, 1: 1}, "out": {0: 0, 1: 1}},
    "measure_angle":      {"in": {0: 0, 1: 1}, "out": {0: 0, 1: 1}},
    "template_match":     {"out": {0: 0, 1: 1}},                   # centers 移除
    "find_line_pair":     {"out": {0: 0, 2: 1}},                   # line2/angle_diff 移除（并入 result.values）
    "blob_analysis":      {"out": {0: 0}},                         # centers 移除
    "result_output":      {"in": {0: 0, 4: 1}, "out": {}},         # saved 端口移除
    "image_sink":         {"in": {0: 0, 1: 1}, "out": {}},
    "csv_sink":           {"in": {0: 0, 2: 2}, "out": {}},
    "external_op":        {"in": {0: 0}, "out": {0: 0, 1: 1, 2: 2}},
    "model_on_list":      {"in": {0: 0}, "out": {0: 0}},
    "op_on_list":         {"in": {0: 0}, "out": {0: 0}},
    # 结果合并：输入改为成对（图像, 结果, 图像2, 结果2, …）
    "merge_results":      {"in": {1: 0, 0: 1}, "out": {0: 0}},     # 旧 image→图像(0)、results→结果(1)；count 移除
}

# 修复循环 3 移除的节点参数（裁图族区域筛选 → 由「保留/屏蔽区域」过滤节点承担）
_PARAM_DROP: dict = {
    "crop_by_detection": ("region_mode", "regions"),
    "crop_by_detection_all": ("region_mode", "regions"),
}

# 修复循环 4 移除的节点（图内直接删除并断开其连线——并行输入后不再需要中转）
_NODE_DROP: dict = {
    "list_pick",   # 「取列表第k张」：任意节点收到多张自动逐张映射
}

# ---- 修复循环 4 第二级：端口顺序统一「图像在上、结果在下」（含中文标签统一）。
# 槽位重排表，按【修复循环 3 布局】的旧槽 → 新槽；schema==2 的图只需本表，
# schema<2 的图先过上方循环三表再过本表（两级复合）。
_OUT_REORDER: dict = {
    "blob_analysis":        {0: 1, 1: 0},
    "model_detection":      {0: 1, 1: 0},
    "model_instance_seg":   {0: 1, 1: 0},
    "model_classification": {0: 1, 1: 0},
    "find_line":            {0: 1, 1: 2, 2: 0},   # 结果/残差/图像 → 图像/结果/残差
    "find_circle":          {0: 1, 1: 0},
    "find_corner":          {0: 1, 1: 0},
    "template_match":       {0: 1, 1: 2, 2: 0},
    "fit_rectangle":        {0: 1, 1: 2, 2: 0},
    "find_line_pair":       {0: 1, 1: 2, 2: 0},
    "measure_distance":     {0: 1, 1: 2, 2: 0},
    "measure_angle":        {0: 1, 1: 2, 2: 0},
    "geometry_to_affine":   {0: 1, 1: 2, 2: 0},
    "detection_to_measure": {0: 1, 1: 2, 2: 0},
    "relabel_map":          {0: 1, 1: 0},
    "mask_to_box":          {0: 1, 1: 0},
    "roi_batch_classify":   {0: 1, 1: 0},
    "filter_overlap":       {0: 1, 1: 2, 2: 0},
    "filter_topk":          {0: 1, 1: 2, 2: 0},
    "merge_results":        {0: 1, 1: 0},         # 结果/图像 → 图像/结果
    "filter_confidence":    {0: 1, 1: 2, 2: 0},   # 过滤族：结果/被滤除/图像 → 图像/结果/被滤除
    "filter_class":         {0: 1, 1: 2, 2: 0},
    "filter_bbox_size":     {0: 1, 1: 2, 2: 0},
    "filter_bbox_area":     {0: 1, 1: 2, 2: 0},
    "filter_mask_area":     {0: 1, 1: 2, 2: 0},
    "filter_region_block":  {0: 1, 1: 2, 2: 0},
    "filter_region_keep":   {0: 1, 1: 2, 2: 0},
    "filter_count_check":   {0: 1, 1: 2, 2: 3, 3: 0},   # 结果/数量/被滤除/图像 → 图像/结果/数量/被滤除
}
_IN_REORDER: dict = {
    "measure_distance":     {0: 1, 1: 2, 2: 0},   # 结果/结果2/图像 → 图像/结果/结果2
    "measure_angle":        {0: 1, 1: 2, 2: 0},
    "mask_to_box":          {0: 1, 1: 0},
    "detection_to_measure": {0: 1, 1: 0},
    "geometry_to_affine":   {0: 1, 1: 0},
    "relabel_map":          {0: 1, 1: 0},
    "filter_overlap":       {0: 1, 1: 0},
    "filter_topk":          {0: 1, 1: 0},
    "filter_confidence":    {0: 1, 1: 0},         # 过滤族：结果/图像 → 图像/结果
    "filter_class":         {0: 1, 1: 0},
    "filter_bbox_size":     {0: 1, 1: 0},
    "filter_bbox_area":     {0: 1, 1: 0},
    "filter_mask_area":     {0: 1, 1: 0},
    "filter_region_block":  {0: 1, 1: 0},
    "filter_region_keep":   {0: 1, 1: 0},
    "filter_count_check":   {0: 1, 1: 0},
    "result_output":        {0: 1, 1: 0},
    "result_view":          {0: 1, 1: 0},
    "csv_sink":             {0: 1, 1: 0, 2: 2},   # 结果/图像/数值 → 图像/结果/数值
    "condition_branch":     {0: 2, 1: 0, 2: 1},   # 判定/真路/假路 → 真路/假路/判定
}

SCHEMA = 3   # 当前端口布局代数（3=取消 image_list 端口 + 删 list_pick + 图像在上）


def _remap(tab: dict | None, slot: int) -> int | None:
    """单槽重映射；tab=None=槽位不变；越界槽=删除（None）。"""
    if tab is None:
        return slot
    if slot not in tab:
        return None
    return tab[slot]


def migrate_ports(flow: dict) -> dict:
    """旧图连线槽位 → 当前端口布局（浅拷贝节点/连线；不修改入参）。

    schema<2：先过循环三表（端口收敛）再过循环四重排表（两级复合）；
    schema==2：只过循环四重排表 + 删 list_pick。
    """
    if not isinstance(flow, dict) or int(flow.get("schema") or 0) >= SCHEMA:
        return flow
    first_gen = int(flow.get("schema") or 0) < 2
    nodes = flow.get("nodes") or []
    types = {}
    for n in nodes:
        try:
            types[int(n.get("id"))] = str(n.get("type", "")).removeprefix("xccv2/")
        except (TypeError, ValueError):
            continue
    out = dict(flow)
    new_nodes = []
    for n in nodes:
        t = types.get(int(n.get("id") or 0), "")
        if t in _NODE_DROP:      # 已删除的节点：连节点带连线一起移除
            continue
        drop = _PARAM_DROP.get(t)
        if drop and isinstance(n.get("properties"), dict) \
                and any(k in n["properties"] for k in drop):
            n = {**n, "properties": {k: v for k, v in n["properties"].items()
                                     if k not in drop}}
        new_nodes.append(n)
    out["nodes"] = new_nodes
    new_links = []
    for lk in flow.get("links") or []:
        try:
            _lid, src, sslot, dst, dslot = int(lk[0]), int(lk[1]), int(lk[2]), int(lk[3]), int(lk[4])
        except (TypeError, ValueError):
            continue
        if types.get(src, "") in _NODE_DROP or types.get(dst, "") in _NODE_DROP:
            continue
        if first_gen:
            tab = _PORT_MIGRATE.get(types.get(src, ""), {}).get("out")
            sslot = _remap(tab, sslot)
            if sslot is None:
                continue
            tab = _PORT_MIGRATE.get(types.get(dst, ""), {}).get("in")
            dslot = _remap(tab, dslot)
            if dslot is None:
                continue
        sslot = _remap(_OUT_REORDER.get(types.get(src, "")), sslot)
        if sslot is None:
            continue
        dslot = _remap(_IN_REORDER.get(types.get(dst, "")), dslot)
        if dslot is None:
            continue
        new_links.append([_lid, src, sslot, dst, dslot])
    out["links"] = new_links
    out["schema"] = SCHEMA
    return out
