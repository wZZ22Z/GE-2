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
# Hazel Cluster Guide

## Hazel (NC State) Cluster Adaptation Guide


### 0) Hazel quick checklist (登录节点)
```bash
module avail conda
module avail cuda
module avail gcc
module avail cmake
module list
which python
which nvcc
which cmake
which g++
nvidia-smi
```

### 1) Configure Conda per Hazel policy
```bash
module load conda
conda init bash
# 退出后重新登录/source ~/.bashrc
cat > ~/.condarc <<'EOF'
pkgs_dirs:
  - /share/$GROUP/$USER/conda/pkgs
channels:
  - conda-forge
EOF
```

### 2) Create per-project env in shared/app path
```bash
ENV_ROOT=/share/$GROUP/$USER/envs/ge2-py310
mkdir -p "$ENV_ROOT"
conda create --prefix "$ENV_ROOT" python=3.10 -y
conda activate "$ENV_ROOT"
export PYTHONNOUSERSITE=1
python -m pip install --upgrade pip
python -m pip install --no-user numpy pandas omegaconf psutil GPUtil importlib_metadata
```

### 3) Install PyTorch wheel according to Hazel CUDA module
先按 `module avail cuda` 和 `nvcc --version` 选对应 CUDA 版本 wheel（比如 cu117/cu121/cu126），不要硬编码旧服务器 cuda
```bash
module load cuda/12.6   # 具体版本请用 `module avail cuda` 结果
python -m pip install --no-user --index-url https://download.pytorch.org/whl/cu126 torch==2.9.0+cu126
```

### 4) Build GE2 on Hazel (login/compute node depends)
```bash
cd /home/zwang269/GE-2
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export CUDACXX=$CUDA_HOME/bin/nvcc
export PATH=$CUDA_HOME/bin:$PATH
export TORCH_CUDA_ARCH_LIST=8.0  # 或根据GPU架构调整

export Torch_DIR=$(python -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'share', 'cmake', 'Torch'))")

cmake -S . -B build-hazel -DCMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc -DTorch_DIR=$Torch_DIR -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build build-hazel --target gege -j 8
cmake --build build-hazel --target pip-install -j 2
```

### 5) Short import sanity check
```bash
PYTHONNOUSERSITE=1 python -c "import gege; print('gege import ok')"
```

### 6) Submit 4GPU training via LSF
用 `bsub` 提交，不直接在 login 上跑正式训练。
```bash
cat > run_ge2_4gpu.sh <<'EOF'
#!/bin/bash
#BSUB -q gpu
#BSUB -n 4
#BSUB -R "rusage[mem=8000]"  # 仅示例，请按 Hazel 要求
#BSUB -R "select[gpu_model0==A100]"  # 如果需要
#BSUB -gpu "num=4:mode=exclusive_process"
#BSUB -W 08:00
#BSUB -o ge2_4gpu.%J.out
#BSUB -e ge2_4gpu.%J.err

module load conda
module load cuda/12.6
module load gcc/11.4
module load cmake/4.1
conda activate /share/$GROUP/$USER/envs/ge2-py310
export PYTHONNOUSERSITE=1
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export CUDACXX=$CUDA_HOME/bin/nvcc
export PATH=$CUDA_HOME/bin:$PATH
export TORCH_CUDA_ARCH_LIST=8.0

cd /path/to/ge2/dandelion-dev
CUDA_VISIBLE_DEVICES=0,1,2,3 \
 GEGE_CSR_GATHER=0 GEGE_CSR_DEBUG=0 GEGE_STAGE_DEBUG=0 GEGE_UNIQUE_BACKEND=bitmap GEGE_PARTITION_BUFFER_PEER_RELAY=1 \
 gege_train gege/configs/twitter_16p.yaml
EOF

bsub < run_ge2_4gpu.sh
```

### 7) 常见要点
- 先用 `nvidia-smi` 和 `python -c "import torch; print(torch.cuda.is_available())"` 验证环境。
- 如果模块版本变化，先更新 `CUDA_HOME`/`CUDACXX`/`TORCH_CUDA_ARCH_LIST`。
- 如果你要复现 4GPU 训练，`gege/configs/twitter_16p.yaml` 与分区数据路径必须一致。

### 8) 简化执行顺序（快速版）
1) 登录节点：`module load conda` + 建 conda env + pip install
2) 登录节点：`module avail cuda` 确认 CUDA 版本
3) 计算节点（LSF 作业）：执行 build + 训练

> ⚠️ 重点：不要把旧机器的绝对路径（`/usr/local/cuda-12.9`）直接照搬到 Hazel；用 `module show cuda/...` 查真实路径。

---

## Hazel 4GPU Recommended Quickstart

1. 用 `module avail` 查可用版本。
2. 创建 `conda` env 在 `/share` 或 `/usr/local/usrapps/<group>`。
3. `module load` + 安装 torch wheel 对应 CUDA。
4. 本地 `cmake` 构建 `build-hazel`。
5. 写 `bsub` 4GPU 作业脚本跑 `gege_train`。

