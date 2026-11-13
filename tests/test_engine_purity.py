"""E5（V0.7）：engine 包纯净性门禁。

`xccv/engine` 是流程引擎/模型训练推理的纯逻辑层——部署服务、无头场景
（fastapi 服务、剪枝子进程、批处理探针）都依赖它可在无 Qt 环境导入。
本测试把"0 处 PyQt/Qt 导入"从靠自觉变成回归门禁：

1. 源码扫描：engine 全部 .py 不得出现 PyQt5/PySide6/QtWidgets/QtCore/QtGui
   的 import 语句（白名单机制留给未来有理由的例外，新增先在此登记）；
2. 子进程导入复核：核心引擎子模块真实 import 后 sys.modules 不得含
   PyQt5（源码扫描防不了"间接拉起"——上游依赖链带 Qt 的回归）。

纯文本+子进程实现，不建 QApplication，offscreen/编译态均可跑。
"""

import re
import subprocess
import sys
import unittest
from pathlib import Path

_ENGINE_DIR = Path(__file__).resolve().parent.parent / "xccv" / "engine"

# 未来确有理由的例外在此登记相对 POSIX 路径（如 engine 可视化辅助模块）；
# 当前为空 = engine 全树必须零 Qt。
_PURITY_WHITELIST: set[str] = set()

# 命中即违规的导入语句（行首缩进允许：函数内延迟导入同样违规）
_BAD_IMPORT = re.compile(
    r"^\s*(?:from\s+(?:PyQt5|PySide6|QtWidgets|QtCore|QtGui)(?:\s|\.)"
    r"|import\s+(?:PyQt5|PySide6|QtWidgets|QtCore|QtGui)\b)")

# 子进程导入复核的核心子模块（覆盖 流程引擎/指标/检出流程/统一JSON/数据集）
_CORE_MODULES = [
    "xccv.engine.result_json",
    "xccv.engine.detection_flow",
    "xccv.engine.metrics.detection",
    "xccv.engine.flow.ir",
    "xccv.engine.flow.executor",
    "xccv.engine.dataset.prepare",
]


class EnginePurityTest(unittest.TestCase):
    """xccv/engine 全树禁止 Qt 导入（部署服务/无头场景复用的根基）。"""

    def test_no_qt_imports_in_engine_tree(self):
        offenders = []
        n_py = 0
        for path in sorted(_ENGINE_DIR.rglob("*.py")):
            n_py += 1
            rel = path.relative_to(_ENGINE_DIR).as_posix()
            if rel in _PURITY_WHITELIST:
                continue
            # utf-8-sig：容忍 BOM（Windows 工具写 UTF8 带 BOM 会 defeating
            # 行首锚定的正则——本测试自身就中招过，坑 P220）
            text = path.read_text(encoding="utf-8-sig")
            for i, line in enumerate(text.splitlines(), start=1):
                if _BAD_IMPORT.match(line):
                    offenders.append(f"{rel}:{i}: {line.strip()}")
        self.assertGreater(n_py, 50, "engine 树异常缩小，扫描可能失效")
        self.assertEqual(offenders, [],
                         "xccv/engine 出现 Qt 导入（部署服务/无头场景会炸）：\n"
                         + "\n".join(offenders))

    def test_core_modules_import_without_qt(self):
        code = (
            "import sys;"
            + ";".join(f"import {m}" for m in _CORE_MODULES)
            + ";bad=sorted(m for m in sys.modules if m.split('.')[0] in "
              "('PyQt5','PySide6'));"
              "print('BAD:'+','.join(bad) if bad else 'CLEAN')"
        )
        proc = subprocess.run([sys.executable, "-c", code],
                              capture_output=True, text=True, timeout=120)
        self.assertEqual(proc.returncode, 0,
                         f"引擎子模块导入失败：{proc.stderr[-2000:]}")
        tail = proc.stdout.strip().splitlines()[-1] if proc.stdout else ""
        self.assertTrue(tail.startswith("BAD:") or tail == "CLEAN",
                        f"探针输出异常：{proc.stdout[-500:]!r}")
        self.assertEqual(tail, "CLEAN",
                         f"引擎子模块间接拉起 Qt：{tail}")


if __name__ == "__main__":
    unittest.main()
