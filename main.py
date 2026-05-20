import argparse
from dataclasses import dataclass

from data import Batch
from experiment import Experiment, ExperimentConfig, get_device, set_seed
from model import FunctionCompositionTreeModel


@dataclass
class FunctionCompositionConfig(ExperimentConfig):
    experiment_name: str = "paper_function_composition_tree"
    t: int = 2
    n: int = 25
    d: int = 32
    heads: int = 4
    layers: int = 1
    d_ff: int = 128
    dropout: float = 0.0
    batch_size: int = 64
    val_batch_size: int = 64
    epochs: int = 20_000
    eval_every: int = 1_000
    lr: float = 1e-3
    seed: int = 0
    val_seed_offset: int = 10_000
    device: str = "cpu"
    out_root: str = "run"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name", default="paper_function_composition_tree")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--t", type=int, default=2)
    parser.add_argument("--n", type=int, default=25)
    parser.add_argument("--d", type=int, default=32)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--d-ff", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20_000)
    parser.add_argument("--eval-every", type=int, default=1_000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-seed-offset", type=int, default=10_000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-root", default="run")
    return parser.parse_args()


def main():
    args = parse_args()
    config = FunctionCompositionConfig(
        experiment_name=args.experiment_name,
        run_id=args.run_id,
        t=args.t,
        n=args.n,
        d=args.d,
        heads=args.heads,
        layers=args.layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        batch_size=args.batch_size,
        val_batch_size=args.val_batch_size,
        epochs=args.epochs,
        eval_every=args.eval_every,
        lr=args.lr,
        seed=args.seed,
        val_seed_offset=args.val_seed_offset,
        device=args.device,
        out_root=args.out_root,
    )

    device = get_device(config.device)
    set_seed(config.seed)
    model = FunctionCompositionTreeModel(
        t=config.t,
        n=config.n,
        d=config.d,
        heads=config.heads,
        layers=config.layers,
        d_ff=config.d_ff,
        dropout=config.dropout,
    )
    train_batch = Batch(config.t, config.n, seed=config.seed, device=device)
    val_batch = Batch(config.t, config.n, seed=config.seed + config.val_seed_offset, device=device)
    experiment = Experiment(config, model, train_batch, val_batch)

    print(
        f"run_dir={experiment.run_dir} t={config.t} n={config.n} d={config.d} "
        f"heads={config.heads} layers={config.layers} epochs={config.epochs} "
        f"device={experiment.device}",
        flush=True,
    )
    experiment.run()


if __name__ == "__main__":
    main()
