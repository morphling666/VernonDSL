# 0.1.0 Alpha 与双 Showcase 收口计划

## 目标

将当前代码收口为可信的 `0.1.0` alpha 开发者预览，并把 Aurora 与
Mandelbulb 打磨成两项正式 showcase。

本轮不宣称稳定 GA。Linux CI、完整 v4 acceptance gates 与生产级资源
系统继续作为公开的已知限制。

## 1. 建立可发布的 alpha 元数据

- 在 `versions.toml` 将发布标识改为 PEP 440 兼容的 `0.1.0a1`。
- 在 `tools/generate_versions.py` 从同一版本真相源派生 CMake 可接受的
  数值 `PROJECT_VERSION`，不增加第二个手写版本。
- 添加 Apache-2.0 `LICENSE`。
- 添加 alpha changelog/release notes，明确支持矩阵及以下限制：
  - Metal 只有编译输出，没有 runtime。
  - 图形展示采用离屏渲染和主机读回，没有 swapchain。
  - 动态 TensorView layout、资源系统和部分后端覆盖仍有限制。
  - 环境相关的 GPU 测试可能 skip，发布报告需区分执行与跳过。
- 同步 `README.md`、`pyproject.toml` 与版本生成测试，统一
  `0.1.0 alpha / developer preview` 定位，并消除 frontend v3/v4
  状态叙述冲突。

## 2. 正式打磨两个视觉展示项目

正式 showcase：

- `examples/aurora_showcase.py`
- `examples/mandelbulb_showcase.py`

两者采用全屏 fragment 路径：Aurora 展示多 pass、纹理采样和自动资源
barrier，Mandelbulb 展示复杂控制流、数学 intrinsic 和 ray marching。
原 Dynamic Ocean 与 Terrain Erosion 保留为高级 compute/PBR 示例。

实施内容：

- 在 `examples/showcase_common.py` 集中处理确定性运行、输出目录创建、
  截图统计和机器可读结果。
- 为两项 demo 提供快速 smoke preset 与高质量 showoff preset，同时保持
  现有 CLI 参数兼容。
- 为参数错误、缺失 backend、缺失 asset、空截图及全黑截图提供清晰诊断。
- 继续通过共享路径处理 OpenGL Y 翻转及不同后端的坐标约定。
- 新增 `examples/README.md`，记录：
  - Vulkan 首选命令。
  - Windows DirectX/OpenGL 回退命令。
  - 交互与 headless 用法。
  - 预期输出和各 demo 展示的 VernonDSL 能力。

## 3. 增加可重复的 showcase 与 wheel 冒烟

- 新增轻量 showcase smoke runner：
  - 依次启动两个 demo 的小尺寸、少帧 headless preset。
  - 校验退出码、结构化统计、PNG 尺寸、alpha 和像素方差。
  - backend 不可用时输出明确 skip 原因；显式 required 模式下应失败。
- 扩展 `scripts/benchmark_baseline.py`，使 aurora 与 mandelbulb 均可独立测量，
  并复用同一结果契约。
- 增加 fresh-venv wheel 功能冒烟：
  - 安装新构建的 wheel。
  - 执行真实 CPU kernel/编译路径，而不只测试 import。
  - 核对 runtime/RHI CMake source delivery。
- 在 `python/tests` 覆盖 preset、结果解析和截图验证等不依赖 GPU 的逻辑。
- 将版本漂移检查和 fresh-wheel 功能冒烟接入
  `.github/workflows/windows-ci.yml`。
- GPU showcase 作为可用 backend 上的显式 smoke；无 GPU runner 不应被
  误判为产品失败。

## 4. 产出展示资产并完成 alpha 验收

- 在本机可用的 Vulkan backend 上运行两个高质量 headless preset。
- Vulkan 不可用时依次回退 DirectX、OpenGL。
- 生成真实渲染截图并用于 README gallery，不使用合成占位图。
- 执行以下验收：
  - Ruff。
  - 相关 Python 测试。
  - 版本生成漂移检查。
  - wheel build 与 `twine check`。
  - fresh-wheel 功能冒烟。
  - 两个 showcase 的 smoke 和高质量运行。
- 最终报告实际执行 backend、测试数量、截图路径、跳过项，以及仍阻止
  beta/GA 的事项。
- 不自动创建 tag、GitHub Release 或上传 PyPI；这些外部发布操作需要用户
  另行明确授权。

## 实施顺序

1. 设置 `0.1.0a1` 派生版本，并补 Apache-2.0、release notes 与一致的
   preview 文档。
2. 统一并打磨 Aurora/Mandelbulb presets、诊断、截图与结果契约。
3. 增加双 showcase、benchmark 和 fresh-wheel 功能冒烟，并接入适当测试
   与 CI。
4. 生成真实 gallery 截图，执行 alpha 发布验收矩阵并记录结果。

## 完成条件

- 所有发布版本均由 `versions.toml` 派生，无新增手写版本轴。
- 两个正式 demo 均能以 documented command 运行并产出非空、非全黑截图。
- wheel 在全新环境中通过真实功能测试，且能交付外部 CMake 使用所需的
  runtime/RHI 源码。
- Windows CI 覆盖版本漂移与 fresh-wheel 功能冒烟。
- README、示例文档和 release notes 对支持能力及限制的描述一致。
- 发布定位保持为 alpha，不暗示尚未满足的 beta/GA 条件。
