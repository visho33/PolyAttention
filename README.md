# PolyAttention

## Setup

```bash
source ~/venvs/poly-attention/bin/activate
```

## Smoke test

```bash
python main.py --experiment-name smoke --epochs 2 --eval-every 1 --device cpu
```

Outputs:

```text
run/smoke/config.json
run/smoke/metrics.csv
```

## Paper-style run

```bash
python main.py \
  --experiment-name paper_function_composition_tree \
  --epochs 20000 \
  --eval-every 1000 \
  --device cpu
```

Outputs:

```text
run/paper_function_composition_tree/config.json
run/paper_function_composition_tree/metrics.csv
```

Expected result:

```text
val_acc reaches 1.0 around 16k-20k epochs
```

## Keep multiple runs

Use `--run-id` to create a subfolder:

```bash
python main.py \
  --experiment-name paper_function_composition_tree \
  --run-id seed0 \
  --epochs 20000 \
  --eval-every 1000 \
  --device cpu
```

Outputs:

```text
run/paper_function_composition_tree/seed0/config.json
run/paper_function_composition_tree/seed0/metrics.csv
```

## Slurm

```bash
sbatch run.sh main.py --experiment-name paper_slurm --epochs 20000 --eval-every 1000 --device cuda
```
