"""Reproducible MNIST experiments using the original DiT implementation unchanged."""
import argparse
import copy
import hashlib
import json
import math
import platform
import subprocess
import time
from pathlib import Path

import torch
import torchvision
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import MNIST
from torchvision.utils import save_image

from flow import make_flow_pair, make_score_pair
from model import DIT
from sampler import sample_ode, sample_sde
from trainer import train_step

ROOT = Path(__file__).resolve().parent


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def git_output(*args):
    result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def dataset_split(config):
    dataset = MNIST(str(ROOT / "data"), train=True, download=True)
    images = dataset.data.float().unsqueeze(1).div(127.5).sub(1)
    labels = dataset.targets
    order = torch.randperm(len(images), generator=torch.Generator().manual_seed(config["seed"]))
    n = config["validation_size"]
    val, train = order[:n], order[n:]
    return TensorDataset(images[train], labels[train]), TensorDataset(images[val], labels[val])


def generate(net, config, device, count=None, seed=None):
    # Keep evaluation from changing subsequent training RNG / data order.
    devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(config["sample_seed"] if seed is None else seed)
        k = config["samples_per_class"] if count is None else count
        labels = torch.arange(10, device=device).repeat_interleave(k)
        shape = config["model"]
        noise = torch.randn(len(labels), shape["C"], shape["H"], shape["W"], device=device)
        if config["objective"] == "flow":
            samples = sample_ode(net, noise, labels, config["sample_steps"], config["guidance"])
        else:
            samples = sample_sde(net, noise, labels, config["sample_steps"],
                                 lambda t: 0.5 * math.cos(math.pi * t / 2), config["guidance"])
        if not torch.isfinite(samples).all():
            raise RuntimeError("Non-finite generated samples")
        return samples.cpu(), labels.cpu()


@torch.no_grad()
def validate(net, loader, pair, config, device):
    devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(config["validation_seed"])
        net.eval()
        total = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            noisy, t, target = pair(x)
            total += (net(noisy, t, y) - target).square().mean().item() * len(x)
        return total / len(loader.dataset)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/mnist_flow_cpu.json")
    parser.add_argument("--output", default="outputs/mnist_flow_cpu")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--sample", type=Path, help="Generate from a checkpoint without training")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    checkpoint = None
    if args.resume or args.sample:
        checkpoint = torch.load(args.resume or args.sample, map_location="cpu", weights_only=True)
        if args.resume and checkpoint["config"] != config:
            raise ValueError("Resume requires the identical experiment config")
        config = checkpoint["config"]
    torch.set_num_threads(config["threads"])
    torch.manual_seed(config["seed"])
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    out = Path(args.output)
    if not args.sample and not args.resume and out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite an existing run: {out}")
    out.mkdir(parents=True, exist_ok=True)
    net = DIT(**config["model"]).to(device)
    ema = copy.deepcopy(net).eval().requires_grad_(False)
    if args.sample:
        ema.load_state_dict(checkpoint["ema_state_dict"])
        samples, _ = generate(ema, config, device)
        save_image((samples + 1).div(2).clamp(0, 1), out / "samples.png",
                   nrow=config["samples_per_class"])
        return
    write_json(out / "config.json", config)
    source_hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                     for name in ("experiment.py", "model.py", "layers.py", "flow.py", "sampler.py",
                                  "patch_utils.py", "trainer.py", "data.py")}
    metadata = {"git_commit": git_output("rev-parse", "HEAD"),
                "git_status": git_output("status", "--porcelain"),
                "python": platform.python_version(), "torch": str(torch.__version__),
                "torchvision": str(torchvision.__version__), "platform": platform.platform(),
                "device": str(device), "parameters": sum(p.numel() for p in net.parameters()),
                "source_sha256": source_hashes, "dataset": "MNIST official training split",
                "train_samples": 60000 - config["validation_size"],
                "validation_samples": config["validation_size"],
                "selection": "minimum fixed-noise validation MSE of EMA weights",
                "label_dropout": 0.1, "optimizer": "Adam", "resume_from": str(args.resume)}
    write_json(out / ("resume_metadata.json" if args.resume else "metadata.json"), metadata)
    train, val = dataset_split(config)
    loader = DataLoader(train, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val, batch_size=256)
    pair = {"flow": make_flow_pair, "score": make_score_pair}[config["objective"]]
    optimizer = torch.optim.Adam(net.parameters(), lr=config["learning_rate"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, config["epochs"], eta_min=config["learning_rate"] * config["minimum_lr_ratio"])
    start_epoch, best, elapsed_before = 0, float("inf"), 0.0
    if checkpoint:
        net.load_state_dict(checkpoint["model_state_dict"])
        ema.load_state_dict(checkpoint["ema_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        torch.set_rng_state(checkpoint["rng_state"])
        if device.type == "cuda" and checkpoint["cuda_rng_state"] is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
        start_epoch, best = checkpoint["epoch"], checkpoint["best_val"]
        elapsed_before = checkpoint["elapsed_seconds"]
    run = None
    try:
        import wandb
    except ImportError:
        print("W&B unavailable: config, metrics and checkpoints are recorded locally.", flush=True)
    else:
        run = wandb.init(project=config["wandb_project"], mode=config["wandb_mode"],
                         config=config, dir=str(out.resolve()))
    started = time.perf_counter()
    reason = "epochs_complete"
    completed = start_epoch
    try:
        for epoch in range(start_epoch, config["epochs"]):
            if time.perf_counter() - started + elapsed_before >= config["max_seconds"]:
                reason = "time_budget"
                break
            tick = time.perf_counter()
            net.train()
            total, seen = 0.0, 0
            for x, y in loader:
                loss = train_step(net, optimizer, x.to(device), y.to(device), pair)
                if not math.isfinite(loss):
                    raise RuntimeError("Non-finite training loss")
                with torch.no_grad():
                    for average, current in zip(ema.parameters(), net.parameters()):
                        average.lerp_(current, 1 - config["ema_decay"])
                total += loss * len(x)
                seen += len(x)
            val_loss = validate(ema, val_loader, pair, config, device)
            row = {"epoch": epoch + 1, "train/loss": total / seen, "val/ema_mse": val_loss,
                   "train/lr": optimizer.param_groups[0]["lr"],
                   "time/epoch_seconds": time.perf_counter() - tick,
                   "time/elapsed_seconds": elapsed_before + time.perf_counter() - started}
            scheduler.step()
            improved = val_loss < best
            best = min(best, val_loss)
            state = {"config": config, "epoch": epoch + 1, "best_val": best,
                     "model_state_dict": net.state_dict(), "ema_state_dict": ema.state_dict(),
                     "optimizer_state_dict": optimizer.state_dict(),
                     "scheduler_state_dict": scheduler.state_dict(), "rng_state": torch.get_rng_state(),
                     "cuda_rng_state": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                     "elapsed_seconds": row["time/elapsed_seconds"]}
            torch.save(state, out / "latest.tmp")
            (out / "latest.tmp").replace(out / "latest.pt")
            if improved:
                torch.save(state, out / "best.tmp")
                (out / "best.tmp").replace(out / "best.pt")
            with (out / "metrics.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            if run:
                run.log(row, step=epoch + 1)
            print(json.dumps(row), flush=True)
            completed = epoch + 1
            if completed == 1 or completed % config["preview_every"] == 0:
                samples, _ = generate(ema, config, device)
                save_image((samples + 1).div(2).clamp(0, 1), out / f"samples_epoch_{completed:03d}.png",
                           nrow=config["samples_per_class"])
        if not (out / "best.pt").exists():
            raise RuntimeError("No completed epoch; increase the training time budget")
        selected = torch.load(out / "best.pt", map_location=device, weights_only=True)
        ema.load_state_dict(selected["ema_state_dict"])
        samples, labels = generate(ema, config, device)
        save_image((samples + 1).div(2).clamp(0, 1), out / "samples.png",
                   nrow=config["samples_per_class"])
        torch.save({"samples": samples, "labels": labels}, out / "samples.pt")
        summary = {"status": reason, "epochs_completed": completed,
                   "selected_epoch": selected["epoch"], "best_val_ema_mse": selected["best_val"],
                   "elapsed_seconds": elapsed_before + time.perf_counter() - started}
        write_json(out / "summary.json", summary)
        print(json.dumps(summary), flush=True)
        if run:
            run.summary.update(summary)
            run.finish()
    except BaseException as exc:
        write_json(out / "failure.json", {"type": type(exc).__name__, "message": str(exc),
                                          "completed_epochs": completed})
        if run:
            run.finish(exit_code=1)
        raise


if __name__ == "__main__":
    main()
