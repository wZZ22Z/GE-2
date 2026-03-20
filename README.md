GE^2

## GE2 Build Notes

The original setup pinned CUDA 11.3 + torch 1.12.1. On this machine, the verified working stack is:

- Python `3.10`
- PyTorch `2.9.0+cu128`
- CUDA toolkit `/usr/local/cuda-12.9`
- `g++ 11.4.0`
- CMake `4.1`

This combination builds `gege` successfully and imports correctly after local install.

## Recommended Environment Setup

```bash
source ~/miniconda3/etc/profile.d/conda.sh

conda create -n newgege310 python=3.10 -y
conda activate newgege310

# Avoid mixing packages from ~/.local into this conda env.
export PYTHONNOUSERSITE=1

python -m pip install --upgrade pip
python -m pip install --no-user --index-url https://download.pytorch.org/whl/cu128 torch==2.9.0+cu128
python -m pip install --no-user numpy pandas omegaconf psutil GPUtil importlib_metadata

export CUDA_HOME=/usr/local/cuda-12.9
export CUDACXX=$CUDA_HOME/bin/nvcc
export PATH=$CUDA_HOME/bin:$PATH

# Change if your GPU architecture is not sm80.
export TORCH_CUDA_ARCH_LIST=8.0
```

Sanity check:

```bash
g++ --version | head -n1
cmake --version | head -n1
PYTHONNOUSERSITE=1 python -c "import torch; print('torch', torch.__version__); print('torch cuda', torch.version.cuda); print('cuda available', torch.cuda.is_available())"
```

## Build

Run from `/home/zwang269/code/NewGE2/NewGE2/ge2/dandelion-dev`:

```bash
cd /home/zwang269/code/NewGE2/NewGE2/ge2/dandelion-dev
cmake -S . -B build-cu129 \
  -DCMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5

cmake --build build-cu129 --target gege -j 4
```

## Install Package

`pip-install` uses `--no-build-isolation --no-deps`.

```bash
cmake --build build-cu129 --target pip-install -j 2
```

## Run

Run from `/home/zwang269/code/NewGE2/NewGE2/ge2/dandelion-dev` after the package is installed.

```bash
export PATH=$HOME/.local/bin:/usr/local/cuda-12.9/bin:$PATH
```

### 1. Preprocess a Dataset

#### twitter dataset with 16 partitions

```bash
gege_preprocess --dataset twitter --output_dir datasets/twitter -ds 0.9 0.05 0.05 --num_partition 16
```

#### run on 4 GPU

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 GEGE_CSR_GATHER=0 GEGE_CSR_DEBUG=0 GEGE_STAGE_DEBUG=0 GEGE_UNIQUE_BACKEND=bitmap GEGE_PARTITION_BUFFER_PEER_RELAY=1 gege_train gege/configs/twitter_16p.yaml
```

```bash
gege_preprocess --dataset fb15k --output_dir datasets/fb15k -ds 0.9 0.05 0.05 --num_partition 1

CUDA_VISIBLE_DEVICES=0 GEGE_CSR_GATHER=0 GEGE_CSR_DEBUG=0 GEGE_STAGE_DEBUG=0 GEGE_UNIQUE_BACKEND=bitmap GEGE_PARTITION_BUFFER_PEER_RELAY=0 gege_train gege/configs/fb15k.yaml

```
#### Run on 2 GPUs

```bash
CUDA_VISIBLE_DEVICES=0,1 GEGE_CSR_GATHER=0 GEGE_CSR_DEBUG=0 GEGE_STAGE_DEBUG=0 GEGE_UNIQUE_BACKEND=bitmap GEGE_PARTITION_BUFFER_PEER_RELAY=1 gege_train gege/configs/livejournal_16p.yaml
```

### 2. Train

```bash
CUDA_VISIBLE_DEVICES=0 gege_train gege/configs/fb15k.yaml
```

Multi-GPU example:

```bash
CUDA_VISIBLE_DEVICES=0,1 gege_train gege/configs/twitter_16p.yaml
```

### 3. Evaluate

```bash
CUDA_VISIBLE_DEVICES=0 gege_eval gege/configs/fb15k.yaml
```

### 4. Quick Sanity Check

```bash
python -c "import gege; print('gege import ok')"
```

## Notes

- Config files under `gege/configs/` assume CUDA devices by default.
- If dataset paths differ from config defaults, update the YAML before train/eval.
- If GPU architecture is not `sm80`, change `TORCH_CUDA_ARCH_LIST` before build.

## Verified Working Build

```bash
export CUDA_HOME=/usr/local/cuda-12.9
export CUDACXX=$CUDA_HOME/bin/nvcc
export PATH=$CUDA_HOME/bin:$PATH
export TORCH_CUDA_ARCH_LIST=8.0

cmake -S . -B build-cu129-final \
  -DCMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5

cmake --build build-cu129-final --target gege -j 4
```
