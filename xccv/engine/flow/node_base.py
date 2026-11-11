"""VisionNode 节点基类、端口/参数定义与运行上下文。

节点实现约定：
- run() 返回 {输出端口名: 数据}，禁止修改输入对象（零拷贝纪律，测试断言）；
- run() 内禁止任何 UI 调用/弹窗（UI 反馈一律经 ctx.warn 与快照）；
- PARAMS 是"参数键 → ParamSpec"表：属性面板据此自动渲染（批次十九），
  execute 前按 schema 校验。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from .ir import PORT_TYPES

PARAM_TYPES = ("int", "float", "enum", "bool", "path", "text")


@dataclass(frozen=True)
class Port:
    """节点端口：名称 + 类型（PORT_TYPES 的 key）+ 是否必需。

    label 为画布上的中文显示名（缺省用 name）；连线按槽位序号存储，
    显示名不影响兼容性。accepts 声明额外允许接入的源端口类型（仅个别
    端口使用，如结果合并的「图像」可直接接图像列表，运行期逐张配对）。
    """

    name: str
    type: str
    required: bool = True
    label: str = ""
    accepts: tuple = ()

    def __post_init__(self):
        if self.type not in PORT_TYPES:
            raise ValueError(f"端口 {self.name!r} 的类型 {self.type!r} 不受支持"
                             f"（允许：{', '.join(PORT_TYPES)}；旧图请用新版软件重新保存）")


@dataclass(frozen=True)
class ParamSpec:
    """参数定义（属性面板 schema 的唯一来源，Python 侧下发前端）。"""

    key: str
    label: str                 # 中文标签
    type: str                  # PARAM_TYPES 的 key
    default: object
    range: Optional[tuple] = None      # 数值范围（含端点）
    options: Optional[list] = None     # enum 选项
    tooltip: str = ""
    folder: bool = False               # path 参数：浏览时选目录而非文件

    def __post_init__(self):
        if self.type not in PARAM_TYPES:
            raise ValueError(f"参数 {self.key!r} 类型 {self.type!r} 不在 PARAM_TYPES")

    def validate(self, value) -> Optional[str]:
        """校验一个参数值；返回错误文本或 None（默认 schema 校验）。"""
        if self.type in ("int", "float"):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return f"「{self.label}」应为数值"
            if self.range is not None:
                lo, hi = self.range
                if not (lo <= value <= hi):
                    return f"「{self.label}」应在 {lo} ~ {hi} 内"
            if self.type == "int" and float(value) != int(value):
                return f"「{self.label}」应为整数"
        elif self.type == "enum":
            if value not in (self.options or []):
                return f"「{self.label}」应为 {self.options} 之一"
        elif self.type == "bool":
            if not isinstance(value, bool):
                return f"「{self.label}」应为布尔值"
        elif self.type in ("path", "text"):
            if not isinstance(value, str):
                return f"「{self.label}」应为文本"
        return None


class RunContext:
    """一次运行的上下文：跨节点共享数据、节点会话状态、人话警告与缓存。

    生命周期 = 一次 Executor.run()（缓存为会话内记忆化，不持久化；单节点
    重跑是调试语义）。缓存键含 revision——set() 一次 +1，因此依赖 ctx 注入
    的节点（如 detection_source）在注入变化后自动失效，不会命中过期结果。
    """

    def __init__(self, snapshot_store=None):
        self._shared: dict = {}
        self._cache: dict = {}
        self._revision: int = 0           # 共享键变更代数（缓存失效依据）
        self.warnings: list = []          # 节点追加的警告（快照/结果显示用）
        self.logs: list = []              # 节点级结构化日志（日志捕捉，见 node_log）
        self.snapshot_store = snapshot_store

    def get(self, key: str, default=None):
        return self._shared.get(key, default)

    def set(self, key: str, value) -> None:
        self._shared[key] = value
        self._revision += 1
        parent = getattr(self, "_parent", None)
        while parent is not None:   # 子流程改共享键：主图缓存同样失效（T4-4）
            parent._revision += 1
            parent = getattr(parent, "_parent", None)

    def child(self) -> "RunContext":
        """子流程上下文（自定义复合算子，V0.6 T4-4）：共享键（模型会话缓存、
        flow_dir/source_name 等）与警告列表共用；计算缓存与节点日志独立——
        子图节点 id 与主图互不冲突。"""
        c = RunContext(snapshot_store=None)
        c._shared = self._shared
        c.warnings = self.warnings
        c._parent = self
        return c

    @property
    def revision(self) -> int:
        """共享键变更代数：每次 set() +1（执行器缓存键的失效依据）。"""
        return self._revision

    def cache_get(self, key: str):
        return self._cache.get(key)

    def cache_set(self, key: str, value) -> None:
        self._cache[key] = value

    def cache_clear(self) -> None:
        """清空计算缓存（图重建时调用）；共享键缓存（如模型加载）不受影响。"""
        self._cache.clear()

    def warn(self, message: str) -> None:
        self.warnings.append(str(message))

    def node_log(self, node_id, node: str, message: str, level: str = "INFO") -> None:
        """记录一条节点级日志（含节点 id 与注册名，可精确定位出错节点）。"""
        self.logs.append({"node_id": int(node_id), "node": str(node),
                          "level": str(level), "message": str(message)})


class VisionNode(ABC):
    """流程节点基类。NAME 全局唯一（registry 查重），CATEGORY 用于节点库分组。

    HANDLES_LIST（修复循环 4，并行输入）：默认 False——任一输入收到多张
    集合（ImageList/ResultList）时执行器把节点**逐张自动映射**（节点实现
    只面向单张编程）；原生消费集合的节点（结果合并/逐图推理/逐图执行算子）
    置 True 关闭自动映射，自行展开集合。
    """

    NAME: str = ""             # 全局唯一注册名，小写下划线，如 "gaussian_blur"
    CATEGORY: str = ""         # "图像源"|"预处理"|"定位"|"逻辑"|"AI"（前端分组）
    INPUTS: tuple = ()         # tuple[Port]，槽序即连线 slot 序
    OUTPUTS: tuple = ()
    PARAMS: dict = {}          # {参数键: ParamSpec}
    HANDLES_LIST: bool = False # True=节点原生处理多张集合（关闭执行器自动映射）

    @abstractmethod
    def run(self, inputs: dict, params: dict, ctx: RunContext) -> dict:
        """inputs: 端口名→数据；返回: 输出端口名→数据。禁止修改输入对象。"""

    def validate(self, params: dict) -> list:
        """参数校验：默认按 PARAMS schema 逐项检查 + 拒绝未知键；子类可扩展。"""
        errors = []
        for key, spec in self.PARAMS.items():
            if key in params:
                err = spec.validate(params[key])
                if err:
                    errors.append(err)
        for key in params:
            if key not in self.PARAMS:
                errors.append(f"未知参数 {key!r}")
        return errors
