"""Integration checks for checkpoint replay and the two existing sampler contracts."""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.data import TensorDataset

import experiment


class ExperimentTests(unittest.TestCase):
    def test_epoch_resume_replays_uninterrupted_weights(self):
        config = json.loads((experiment.ROOT / "configs/mnist_flow_cpu.json").read_text())
        config.update(epochs=2, batch_size=4, sample_steps=2, samples_per_class=1,
                      threads=1, max_seconds=600)
        generator = torch.Generator().manual_seed(7)
        data = TensorDataset(torch.randn(8, 1, 28, 28, generator=generator), torch.arange(8))
        original_validate = experiment.validate
        calls = 0

        def interrupt_second_epoch(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("Simulated interruption after first saved epoch")
            return original_validate(*args, **kwargs)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config))

            def run(output, resume=False):
                argv = ["experiment.py", "--config", str(config_path), "--output", str(output), "--device", "cpu"]
                if resume:
                    argv += ["--resume", str(output / "latest.pt")]
                with patch.object(sys, "argv", argv), patch.object(experiment, "dataset_split", return_value=(data, data)), \
                        patch.dict(sys.modules, {"wandb": None}), contextlib.redirect_stdout(io.StringIO()):
                    experiment.main()

            complete, interrupted = root / "complete", root / "interrupted"
            run(complete)
            with patch.object(experiment, "validate", side_effect=interrupt_second_epoch):
                with self.assertRaisesRegex(RuntimeError, "Simulated interruption"):
                    run(interrupted)
            self.assertTrue((interrupted / "failure.json").exists())
            run(interrupted, resume=True)
            a = torch.load(complete / "latest.pt", weights_only=True)
            b = torch.load(interrupted / "latest.pt", weights_only=True)
            self.assertEqual(a["epoch"], b["epoch"])
            for state in ("model_state_dict", "ema_state_dict"):
                for key in a[state]:
                    torch.testing.assert_close(a[state][key], b[state][key], rtol=0, atol=0)
            with self.assertRaises(FileExistsError):
                run(complete)

    def test_generation_preserves_rng_and_outputs_finite_images(self):
        config = json.loads((experiment.ROOT / "configs/mnist_flow_cpu.json").read_text())
        config.update(sample_steps=4, samples_per_class=1)
        torch.set_num_threads(1)
        net = experiment.DIT(**config["model"])
        for objective in ("flow", "score"):
            config["objective"] = objective
            rng = torch.get_rng_state().clone()
            samples, labels = experiment.generate(net, config, torch.device("cpu"))
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertEqual(tuple(samples.shape), (10, 1, 28, 28))
            self.assertTrue(torch.isfinite(samples).all())
            self.assertEqual(labels.tolist(), list(range(10)))


if __name__ == "__main__":
    unittest.main()
