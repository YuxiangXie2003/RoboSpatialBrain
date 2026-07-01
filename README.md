# RoboSpatialBrain — RoboSpatial-Home Evaluation

This repository evaluates **RoboSpatialBrain** on the [RoboSpatial-Home](https://huggingface.co/datasets/chanhee-luke/RoboSpatial-Home) benchmark.

---

## Team RoboSpatialBrain

Contributors:

- Yuxiang Xie, Harbin Institute of Technology, Shenzhen
- Qi Lv, Harbin Institute of Technology, Shenzhen
- Jianming Xing, Harbin Institute of Technology, Shenzhen
- Zijian Hong, Harbin Institute of Technology, Shenzhen
- Xiang Deng, Harbin Institute of Technology, Shenzhen; Ruoyu Technology
- Weili Guan, Harbin Institute of Technology, Shenzhen
- Liqiang Nie, Harbin Institute of Technology, Shenzhen

## Environment Setup

### Conda environment (recommended)

Create and activate a dedicated environment:

```bash
conda create -n robospatial python=3.10 -y
conda activate robospatial
```

Install all dependencies:

```bash
pip install -r requirements.txt
```

> **Note**: `requirements.txt` includes `--extra-index-url https://download.pytorch.org/whl/cu128` so that PyTorch and torchvision are pulled from the CUDA 12.8 wheel index automatically. If your CUDA driver version differs, replace `cu128` with the appropriate suffix (e.g. `cu118`, `cu121`) in `requirements.txt` before running the command above.

### Flash Attention 2 (optional but recommended)

Flash Attention cannot be installed via `requirements.txt` because it must be compiled against your local CUDA toolkit. Install it separately after the steps above:

```bash
pip install flash-attn --no-build-isolation
```

---

## Downloading the Model Weights

Download the checkpoint from HuggingFace: https://huggingface.co/lbx511/RoboSpatialBrain

The weights consist of three sub-models plus an inference scripts directory. After cloning this repo, manually create a `model/` directory and place the four downloaded directories inside it:

```bash
mkdir -p model
# then move/copy LM, VL-B, VL-F, scripts into model/
```

The final layout should look like this:

```
RoboSpatial-Eval-Submit/
├── config.yaml
├── main.py
├── models.py
├── evaluation.py
├── download_benchmark.py
├── upload.py
├── requirements.txt
├── README.md
├── ADDING_MODELS.md
├── model/              ← create this directory manually
│   ├── LM/
│   ├── VL-B/
│   └── scripts/
└── results/
```

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
