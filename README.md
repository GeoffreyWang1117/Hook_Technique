# LLM Hook Analysis Framework

一个可插拔的 LLM 挂钩分析框架，通过对推理和训练过程中的关键运行时节点进行动态 Hook，捕获推理轨迹与性能瓶颈。

## 🎯 项目目标

本框架专为小模型推理优化、结构裁剪与安全过滤研究而设计，提供：

- **动态Hook捕获**：Attention、KV Cache、激活值、Fisher 信息、梯度路径等运行时节点
- **多后端支持**：PyTorch forward/backward hook、CUDA kernel trace、TensorRT 日志注入
- **可视化分析**：推理轨迹可视化、性能瓶颈分析、可解释性报告
- **可插拔架构**：模块化设计，支持自定义Hook和分析器

## 🏗️ 架构设计

```
llm_hooks/
├── core/           # 核心Hook基础设施
├── pytorch/        # PyTorch Forward/Backward Hooks
├── attention/      # Attention 和 KV Cache 监控
├── activations/    # 激活值追踪
├── fisher/         # Fisher 信息矩阵
├── gradients/      # 梯度路径追踪
├── cuda/           # CUDA Kernel Trace
├── tensorrt/       # TensorRT 日志注入
├── visualization/  # 可视化工具
└── analysis/       # 可解释性分析
```

## ✨ 核心功能

### 1. PyTorch Hooks
- Forward Hook：捕获前向传播的中间激活
- Backward Hook：追踪反向传播的梯度流
- 自动Hook注册与管理

### 2. Attention 分析
- Multi-Head Attention 模式分析
- Attention Score 分布统计
- KV Cache 使用效率监控

### 3. 激活值监控
- 层级激活统计（均值、方差、稀疏度）
- 死神经元检测
- 激活值分布可视化

### 4. Fisher 信息
- 参数重要性评估
- 结构化剪枝支持
- 敏感度分析

### 5. 梯度路径追踪
- 梯度流可视化
- 梯度消失/爆炸检测
- 反向传播路径分析

### 6. CUDA Profiling
- Kernel 执行时间分析
- 内存带宽利用率
- SM 占用率统计

### 7. 性能分析
- 推理延迟分解
- 内存占用追踪
- 瓶颈识别与优化建议

## 🚀 快速开始

### 安装

```bash
pip install -r requirements.txt
pip install -e .
```

### 基础使用

```python
from llm_hooks import HookManager
from llm_hooks.pytorch import ForwardHook, BackwardHook
from llm_hooks.attention import AttentionMonitor

# 创建Hook管理器
manager = HookManager()

# 注册PyTorch Hooks
manager.register(ForwardHook(layer_name='transformer.layer.0'))
manager.register(BackwardHook(layer_name='transformer.layer.0'))

# 注册Attention监控
manager.register(AttentionMonitor(record_scores=True))

# 在模型上应用hooks
model = YourLLMModel()
manager.apply_to_model(model)

# 推理
output = model(input_ids)

# 获取分析结果
results = manager.get_results()
manager.visualize(results)
```

## 📊 可视化示例

框架提供多种可视化工具：

- **Attention热力图**：展示多头注意力模式
- **激活分布图**：统计各层激活值分布
- **梯度流图**：可视化反向传播路径
- **性能火焰图**：识别性能瓶颈
- **内存时间线**：追踪内存分配与释放

## 🔬 应用场景

### 1. 模型推理优化
- 识别推理瓶颈层
- 量化敏感度分析
- KV Cache优化策略

### 2. 结构化剪枝
- 基于Fisher信息的重要性排序
- 通道级剪枝决策
- 剪枝后精度评估

### 3. 安全过滤研究
- 异常激活检测
- 对抗样本追踪
- 模型行为分析

## 📦 依赖项

- Python >= 3.8
- PyTorch >= 2.0
- CUDA Toolkit >= 11.8 (可选)
- TensorRT >= 8.6 (可选)
- NumPy, Matplotlib, Seaborn

## 📖 文档

详细文档请参考 [docs/](docs/) 目录：

- [Hook API 文档](docs/api.md)
- [可视化指南](docs/visualization.md)
- [性能优化建议](docs/performance.md)
- [自定义Hook开发](docs/custom_hooks.md)

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

## 🔗 相关资源

- [PyTorch Hooks 官方文档](https://pytorch.org/docs/stable/notes/modules.html#module-hooks)
- [CUDA Profiling 工具](https://docs.nvidia.com/cuda/profiler-users-guide/)
- [TensorRT 开发指南](https://docs.nvidia.com/deeplearning/tensorrt/)
