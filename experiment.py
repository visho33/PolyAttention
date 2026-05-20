import csv
import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_device(name):
    if name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    if name == "mps" and not torch.backends.mps.is_available():
        return torch.device("cpu")
    return torch.device(name)

@dataclass
class ExperimentConfig:
    experiment_name: str = "experiment"
    run_id: str | None = None
    batch_size: int = 64
    val_batch_size: int = 64
    epochs: int = 100_000
    eval_every: int = 100
    lr: float = 1e-3
    seed: int = 0
    device: str = "cpu"
    out_root: str = "run"


class Experiment:

    def __init__(self, config, model, train_batch, val_batch, optimizer=None):
        self.config = config
        self.device = get_device(config.device)
        self.model = model.to(self.device)
        self.train_batch = train_batch
        self.val_batch = val_batch
        self.optimizer = optimizer or torch.optim.Adam(self.model.parameters(), lr=config.lr)

        self.run_dir = Path(config.out_root) / config.experiment_name
        if config.run_id is not None:
            self.run_dir = self.run_dir / config.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def accuracy(self, logits, targets):
        return (logits.argmax(dim=-1) == targets).float().mean().item()

    def train_step(self):
        self.model.train()
        x, y = self.train_batch.function_composition(self.config.batch_size)
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item(), self.accuracy(logits, y)

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        x, y = self.val_batch.function_composition(self.config.val_batch_size)
        logits = self.model(x)
        loss = F.cross_entropy(logits, y).item()
        return loss, self.accuracy(logits, y)

    def save_config(self):
        config = asdict(self.config)
        config["device_resolved"] = str(self.device)
        (self.run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    def run(self):
        self.save_config()
        metrics_path = self.run_dir / "metrics.csv"
        with metrics_path.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc"],
            )
            writer.writeheader()

            for epoch in range(self.config.epochs + 1):
                train_loss, train_acc = self.train_step()

                if epoch == 0 or epoch % self.config.eval_every == 0:
                    val_loss, val_acc = self.evaluate()
                    metrics = {
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "train_acc": train_acc,
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                    }
                    writer.writerow(metrics)
                    f.flush()
                    print(
                        f"epoch={epoch} train_loss={train_loss:.6f} "
                        f"train_acc={train_acc:.6f} val_loss={val_loss:.6f} "
                        f"val_acc={val_acc:.6f}",
                        flush=True,
                    )
