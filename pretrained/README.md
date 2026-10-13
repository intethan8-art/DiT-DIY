# 发布权重

本目录分发小型 MNIST 推理权重。它们由本仓库代码从头训练，不包含下载的第三方预训练模型。

| 文件 | 用途 |
| --- | --- |
| `mnist-flow-ema.pt` | 条件生成 DiT 的 EMA 权重、完整训练/采样配置、选中轮次与来源哈希 |
| `mnist-classifier.pt` | 独立 MNIST CNN 的评估权重、训练拆分、参数与选中轮次 |

在仓库根目录执行：

```bash
python experiment.py --sample pretrained/mnist-flow-ema.pt --output outputs/pretrained_samples
python evaluate.py --checkpoint pretrained/mnist-flow-ema.pt --classifier pretrained/mnist-classifier.pt --output outputs/pretrained_evaluation
```

`mnist-flow-ema.pt` 只用于推理，不包含恢复训练需要的优化器和随机状态，不能用于 `--resume`。完整训练检查点保留在训练输出目录。

模型仅用于 28×28 单通道手写数字，条件标签为 0–9；不支持自然图像或文字提示。训练数据为 MNIST 官方训练集中的 55,000 张，另留出 5,000 张验证图像。所有参数与评估限制见 [实验报告](../docs/EXPERIMENT.md)，文件校验值见 [manifest.json](../docs/results/manifest.json)。
