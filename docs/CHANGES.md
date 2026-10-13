# 本次项目整理的改动范围

## 原有代码

本次未改动任何原有 Python 文件，包括模型、注意力层、VAE、损失函数、训练步骤和采样器。

开始任务前工作区已有两处用户修改，保留其原样：

- `train_vae.py`：VAE 训练轮数由 1 调整为 2。
- `latent_train.py`：训练轮数由 5 调整为 10；生成展示改为直接使用 decoder 均值，原先加噪声的两行改为注释。

这些修改不属于本次新增实验实现。

## 新增外围工具

- `experiment.py`：复用 `model.DIT`、`trainer.train_step`、`flow` 和 `sampler`，提供像素空间训练、EMA、验证、配置/权重保存、恢复与采样。预先将 MNIST 转成张量，等价归一化到 `[-1,1]`，减少 CPU 数据预处理开销。
- `evaluate.py`：独立的小型 CNN 评估器。生成器选型只使用验证 MSE，分类器测试集仅用于报告评估器的可靠程度。
- `configs/mnist_flow_cpu.json`：本机预算内的固定实验配置。
- `requirements.txt`、`docs/requirements-tested.txt`：基础依赖与实际训练环境记录。
- `README.md`：重写项目说明、运行命令、两种训练目标、复现与限制。
- `export_results.py`：核对检查点与评估结果的 SHA-256 后，导出小型推理权重、图表与日志。
- `tests/test_experiment.py`：验证中断后按 epoch 恢复与连续训练的权重完全一致，以及两种采样路径输出有限值且不扰动训练随机状态。
- `.gitignore`：为两份发布用小型权重增加精确例外；其他训练输出仍被忽略。

本次训练选择原有 Flow Matching 路径，与原始 `train.py` 默认的噪声预测路径不同；这是独立实验入口的配置选择，没有修改原始代码默认行为。
