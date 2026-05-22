# RoboSpatialBrain — RoboSpatial-Home Evaluation

This repository evaluates **RoboSpatialBrain** on the [RoboSpatial-Home](https://huggingface.co/datasets/chanhee-luke/RoboSpatial-Home) benchmark.

---

## Environment Setup

### Requirements

Python 3.10+ with the following packages:

```
torch>=2.10.0
transformers>=5.0.0
accelerate>=1.0.0
datasets>=2.0.0
qwen-vl-utils>=0.0.14
Pillow>=10.0.0
tqdm>=4.60.0
PyYAML>=6.0
numpy
```

### Conda environment (recommended)

Create and activate a dedicated environment:

```bash
conda create -n robospatial python=3.10 -y
conda activate robospatial
```

Install PyTorch with CUDA (adjust the CUDA version to match your driver):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

Install remaining dependencies:

```bash
pip install transformers>=5.0.0 accelerate datasets qwen-vl-utils Pillow tqdm PyYAML numpy
```

> **Note**: `transformers >= 5.0` is required for Qwen3/3.5 `enable_thinking` support.

### Flash Attention 2 (optional but recommended)

```bash
pip install flash-attn --no-build-isolation
```


---

## Downloading the Checkpoint

https://huggingface.co/lbx511/RoboSpatialBrain

Place the checkpoint under `./model`




---

## Running the Evaluation



```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
python -u main.py \
    robospatialBrain_$(date +%Y%m%d_%H%M%S) \
    --config config.yaml \
    2>&1 | tee /tmp/eval_robospatialBrain.log
```

Results are written to `./results/`.


---

## Output

Each run produces the following files under `results/`:

```
results/
├── context_<tag>_results.json
├── compatibility_<tag>_results.json
├── configuration_<tag>_results.json
└── aggregate_robospatial_home_<tag>.json
```

