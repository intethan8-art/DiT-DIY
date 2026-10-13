"""Export a finished bundled MNIST CPU run into GitHub project artifacts."""
import argparse
import hashlib
import json
import shutil
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from experiment import ROOT, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("outputs/mnist_flow_cpu"))
    parser.add_argument("--evaluation", type=Path, default=Path("outputs/mnist_flow_cpu/evaluation"))
    parser.add_argument("--classifier", type=Path, default=Path("outputs/evaluator/classifier.pt"))
    args = parser.parse_args()
    summary = json.loads((args.run / "summary.json").read_text())
    evaluation = json.loads((args.evaluation / "evaluation.json").read_text())
    checkpoint_path = args.run / "best.pt"
    checkpoint_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    if evaluation["checkpoint_sha256"] != checkpoint_hash:
        raise ValueError("Evaluation does not match the selected checkpoint")
    classifier_hash = hashlib.sha256(args.classifier.read_bytes()).hexdigest()
    if evaluation["classifier_sha256"] != classifier_hash:
        raise ValueError("Evaluation does not match the classifier")
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    release_config = json.loads((ROOT / "configs" / "mnist_flow_cpu.json").read_text())
    if state["config"] != release_config:
        raise ValueError("This release exporter requires configs/mnist_flow_cpu.json unchanged")
    if summary["selected_epoch"] != state["epoch"]:
        raise ValueError("Summary does not match the selected checkpoint")
    results = ROOT / "docs" / "results"
    pretrained = ROOT / "pretrained"
    results.mkdir(parents=True, exist_ok=True)
    pretrained.mkdir(parents=True, exist_ok=True)
    torch.save({"config": state["config"], "epoch": state["epoch"], "best_val": state["best_val"],
                "ema_state_dict": state["ema_state_dict"], "source_checkpoint_sha256": checkpoint_hash},
               pretrained / "mnist-flow-ema.pt")
    shutil.copy2(args.classifier, pretrained / "mnist-classifier.pt")
    for name in ("config.json", "summary.json", "metadata.json", "metrics.jsonl", "samples.png"):
        shutil.copy2(args.run / name, results / name)
    shutil.copy2(args.evaluation / "evaluation.json", results / "evaluation.json")
    for epoch in (1, 5, 10, 20, 40, 60):
        source = args.run / f"samples_epoch_{epoch:03d}.png"
        if source.exists():
            shutil.copy2(source, results / source.name)
    rows = [json.loads(line) for line in (args.run / "metrics.jsonl").read_text().splitlines()]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].plot([r["epoch"] for r in rows], [r["train/loss"] for r in rows], label="Train (raw)")
    axes[0].plot([r["epoch"] for r in rows], [r["val/ema_mse"] for r in rows], label="Validation (EMA)")
    axes[0].set(xlabel="Epoch", ylabel="Flow-matching MSE")
    axes[0].legend()
    axes[0].grid(alpha=0.2)
    axes[1].bar(range(10), evaluation["per_class_accuracy"], color="#3b82f6")
    axes[1].set(xlabel="Requested digit", ylabel="Classifier agreement", ylim=(0, 1), xticks=range(10))
    axes[1].grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(results / "training_and_evaluation.png", dpi=160)
    plt.close(fig)
    manifest = {"original_checkpoint_sha256": checkpoint_hash,
                "files": {str(p.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(pretrained.glob("*.pt"))},
                "dataset_raw_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                       for p in sorted((ROOT / "data" / "MNIST" / "raw").glob("*-ubyte"))}}
    write_json(results / "manifest.json", manifest)
    low, high = evaluation["conditional_accuracy_wilson_95"]
    class_rows = "\n".join(f"| {i} | {score:.1%} |" for i, score in enumerate(evaluation["per_class_accuracy"]))
    slowest = max(rows, key=lambda r: r["time/epoch_seconds"])
    timing_note = ""
    if slowest["time/epoch_seconds"] > 5 * statistics.median(r["time/epoch_seconds"] for r in rows):
        timing_note = (f"第 {slowest['epoch']} 轮记录了 {slowest['time/epoch_seconds']:.1f} 秒的异常耗时。"
                       "运行期间工具等待也出现长延迟，因此墙钟时间包含环境等待，不能视作纯计算耗时。")
    report = f"""# MNIST CPU 实验记录

## 设置

本次使用原有 `model.DIT`、`trainer.train_step`、`flow.make_flow_pair` 和 `sampler.sample_ode`，没有改动模型主体。实验入口新增了 EMA、cosine 学习率调度、固定验证集与检查点管理。

- 数据：MNIST 官方训练集以 seed=42 拆分为 55,000 张训练图和 5,000 张验证图；官方测试集不参与生成器训练或选择。
- 模型：168,784 参数，patch=4、width=64、depth=2、heads=4、MLP=256。
- 优化：Adam，batch=128，初始 lr=0.001，cosine 最低 lr=0.0001，EMA=0.995，标签丢弃概率=0.1。
- 采样：100 步 Euler ODE，CFG=2；展示 seed=2026；评估使用从 seed=9876 开始的分批独立噪声。
- 环境：CPU，4 个训练线程；版本与源文件哈希见 [metadata.json](results/metadata.json)。
- 选型：只按固定噪声验证集的最低 EMA MSE 选择，不按展示图挑选检查点或样本。

## 实测结果

| 项目 | 结果 |
| --- | --- |
| 完成轮数 | {summary['epochs_completed']} |
| 选中轮次 | {summary['selected_epoch']} |
| 停止原因 | `{summary['status']}` |
| 训练过程墙钟耗时（含等待） | {summary['elapsed_seconds'] / 60:.2f} 分钟 |
| 初始轮验证 MSE | {rows[0]['val/ema_mse']:.6f} |
| 最佳 EMA 验证 MSE | {summary['best_val_ema_mse']:.6f} |
| 独立分类器官方测试准确率 | {evaluation['classifier_test_accuracy']:.2%}（10,000 张） |
| 生成图类别一致率 | {evaluation['conditional_accuracy']:.2%}（{evaluation['sample_count']} 张） |
| 95% Wilson 区间 | {low:.2%}–{high:.2%} |

每类生成 {evaluation['samples_per_class']} 张，没有按分类器置信度或外观筛图。

{timing_note}

| 请求类别 | 类别一致率 |
| --- | --- |
{class_rows}

![训练与逐类评估](results/training_and_evaluation.png)

下图每行依次为 0–9，每类固定噪声生成 10 张，使用最佳 EMA 权重。

![最终生成结果](results/samples.png)

## 复现

```bash
python experiment.py --config configs/mnist_flow_cpu.json --output outputs/reproduce
python evaluate.py --checkpoint outputs/reproduce/best.pt --classifier pretrained/mnist-classifier.pt --output outputs/reproduce/evaluation
python experiment.py --sample pretrained/mnist-flow-ema.pt --output outputs/pretrained_samples
python -m unittest discover -s tests -v
```

发布权重为推理用 EMA 权重；完整可恢复检查点保留在本地 `outputs/mnist_flow_cpu/latest.pt` 和 `best.pt`。配置、逐轮指标、环境记录、数据与权重 SHA-256 位于 [results/](results/)。导出命令为 `python export_results.py`。

## 解读与限制

类别一致率衡量独立 CNN 是否把生成图识别为请求类别，不是 FID，也不能证明感知质量、风格多样性或无训练集记忆。应结合上方未经筛选的网格查看残余噪点和形状错误。

本次只训练一个生成器随机种子；Wilson 区间不包含不同训练种子的差异。分类器使用生成器相同训练/验证拆分训练 5 轮，按验证准确率选型，再在官方测试集报告准确率。没有为此实验训练或验证 latent DiT；本地已有 VAE 权重不属于本次生成质量结论。

原有代码与外围工具的改动范围见 [CHANGES.md](CHANGES.md)。
"""
    (ROOT / "docs" / "EXPERIMENT.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("Exported figures, logs, inference weights and checksums.")


if __name__ == "__main__":
    main()
