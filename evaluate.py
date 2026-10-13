"""Train an independent MNIST classifier and measure conditional sample agreement.

This is a diagnostic, not FID or a claim of perceptual quality/diversity.
The classifier uses the same train/validation split as the generative experiment.
"""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import MNIST

from experiment import ROOT, dataset_split, generate, write_json
from model import DIT


class DigitClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(32 * 7 * 7, 64), nn.ReLU(), nn.Linear(64, 10))

    def forward(self, x):
        return self.layers(x)


@torch.no_grad()
def accuracy(net, loader):
    net.eval()
    correct = sum((net(x).argmax(1) == y).sum().item() for x, y in loader)
    return correct / len(loader.dataset)


def wilson(correct, count):
    p, z = correct / count, 1.96
    denominator = 1 + z * z / count
    center = (p + z * z / (2 * count)) / denominator
    radius = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denominator
    return [center - radius, center + radius]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--config", default="configs/mnist_flow_cpu.json")
    parser.add_argument("--classifier", type=Path, default=Path("outputs/evaluator/classifier.pt"))
    parser.add_argument("--output", type=Path, default=Path("outputs/evaluation"))
    parser.add_argument("--train-classifier", action="store_true")
    parser.add_argument("--samples-per-class", type=int, default=100)
    parser.add_argument("--seed", type=int, default=9876)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(1234)
    args.output.mkdir(parents=True, exist_ok=True)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    model_checkpoint = None
    if args.checkpoint:
        model_checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        config = model_checkpoint["config"]
    classifier = DigitClassifier()
    split = {"seed": config["seed"], "validation_size": config["validation_size"]}
    if args.train_classifier:
        if args.classifier.exists():
            raise FileExistsError("Classifier already exists; omit --train-classifier to reuse it")
        train, val = dataset_split(config)
        loader = DataLoader(train, batch_size=256, shuffle=True)
        val_loader = DataLoader(val, batch_size=256)
        optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
        history, best = [], -1.0
        args.classifier.parent.mkdir(parents=True, exist_ok=True)
        for epoch in range(5):
            classifier.train()
            started = time.perf_counter()
            for x, y in loader:
                optimizer.zero_grad()
                loss = nn.functional.cross_entropy(classifier(x), y)
                loss.backward()
                optimizer.step()
            score = accuracy(classifier, val_loader)
            row = {"epoch": epoch + 1, "val_accuracy": score,
                   "seconds": time.perf_counter() - started}
            history.append(row)
            print(json.dumps(row), flush=True)
            if score > best:
                best = score
                torch.save({"state_dict": classifier.state_dict(), "split": split,
                            "seed": 1234, "epoch": epoch + 1, "val_accuracy": score,
                            "optimizer": "Adam", "learning_rate": 0.001, "batch_size": 256},
                           args.classifier)
        write_json(args.output / "classifier_training.json", history)
    saved = torch.load(args.classifier, map_location="cpu", weights_only=True)
    if saved["split"] != split:
        raise ValueError("Classifier and generator data splits differ")
    classifier.load_state_dict(saved["state_dict"])
    classifier.eval()
    test = MNIST(str(ROOT / "data"), train=False, download=True)
    test_loader = DataLoader(TensorDataset(test.data.float().unsqueeze(1).div(127.5).sub(1),
                                           test.targets), batch_size=256)
    test_accuracy = accuracy(classifier, test_loader)
    result = {"classifier_test_accuracy": test_accuracy, "classifier_test_count": len(test),
              "classifier_validation_accuracy": saved["val_accuracy"],
              "classifier_selected_epoch": saved["epoch"],
              "classifier_sha256": hashlib.sha256(args.classifier.read_bytes()).hexdigest(),
              "limitation": "Classifier agreement is not FID, perceptual quality, or diversity; one training seed."}
    if model_checkpoint:
        net = DIT(**config["model"])
        net.load_state_dict(model_checkpoint["ema_state_dict"])
        confusion = torch.zeros(10, 10, dtype=torch.long)
        all_samples, all_labels = [], []
        # Bounded generation batches keep CPU memory use modest.
        remaining, batch_index = args.samples_per_class, 0
        with torch.no_grad():
            while remaining:
                count = min(10, remaining)
                samples, labels = generate(net, config, torch.device("cpu"), count, args.seed + batch_index)
                predictions = classifier(samples.clamp(-1, 1)).argmax(1)
                confusion += torch.bincount(labels * 10 + predictions, minlength=100).reshape(10, 10)
                all_samples.append(samples)
                all_labels.append(labels)
                remaining -= count
                batch_index += 1
        correct, total = confusion.diag().sum().item(), confusion.sum().item()
        result.update({"checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                       "selected_epoch": model_checkpoint["epoch"], "sample_seed": args.seed,
                       "samples_per_class": args.samples_per_class, "sample_count": total,
                       "conditional_accuracy": correct / total,
                       "conditional_accuracy_wilson_95": wilson(correct, total),
                       "per_class_accuracy": (confusion.diag() / confusion.sum(1)).tolist(),
                       "confusion_matrix_true_rows_predicted_columns": confusion.tolist()})
        torch.save({"samples": torch.cat(all_samples), "labels": torch.cat(all_labels)},
                   args.output / "evaluation_samples.pt")
    write_json(args.output / "evaluation.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
