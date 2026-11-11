# 贡献指南

感谢关注 XCcv Community！

## 环境搭建

1. Windows 10/11（x64），Python 3.11；
2. 按 [README](README.md)「安装」章节创建虚拟环境并安装依赖；
3. 运行 `run_main.bat` 确认主界面可打开。

## 如何贡献

1. Fork 仓库并创建特性分支（`feat/xxx`、`fix/xxx`）；
2. 保持改动最小、聚焦单一问题；新增代码风格与所在模块保持一致
   （命名、注释密度、中文注释惯例）；
3. 涉及新依赖时：确认许可证与本项目（AGPL-3.0 + GPLv3 组件）兼容，
   并同步更新 `THIRD_PARTY_NOTICES.md`；
4. 修改 `labelme/` 内文件时，同步更新 `labelme/MODIFICATIONS.md`（GPLv3 §5a 要求）；
5. 提交 PR 并清晰描述变更内容与验证方式。

## 注意事项

- 本仓库不分发模型权重，请勿提交任何 `.pt/.pth/.onnx` 文件（.gitignore 已拦截）；
- 请勿提交真实生产图片、客户数据或任何敏感信息；测试素材使用合成数据；
- 提交前自查不引入密钥 / Token / 内网地址。

## 许可

提交的贡献将按 [AGPL-3.0](LICENSE) 授权随本仓库分发。
