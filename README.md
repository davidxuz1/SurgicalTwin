# Surgical Twin Project

## Environment Setup Guide

This guide provides comprehensive instructions for setting up the environment for the Surgical Twin project, including installation of required dependencies and models.

```bash
git clone 
cd SurgicalTwin

### Conda Environment Setup

Create and activate a new Conda environment:

```bash
conda create -n surgical_twin python=3.11
conda activate surgical_twin
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```

### Required Manual Installations

#### 1. SAM2 Installation

```bash
cd third_party
git clone https://github.com/facebookresearch/segment-anything-2.git
cd segment-anything-2/
pip install -e .
python setup.py build_ext --inplace
```

#### 2. Monst3r Setup

```bash
cd third_party
git clone --recursive https://github.com/junyi42/monst3r
cd monst3r
cd data
bash download_ckpt.sh
```

**Note:** Check if the file `Tartan-C-T-TSKH-spring540x960-M.pth` exists in `monst3r/third_party/RAFT/models`.
If it does not exist, download it manually from this [drive link](https://drive.google.com/file/d/1a0C5FTdhjM4rKrfXiGhec7eq2YM141lu/view) and place it in the `monst3r/third_party/RAFT/models` folder.

#### 3. SurgicalGaussian Installation

```bash
git clone https://github.com/xwx0924/SurgicalGaussian
```

Substitute submodules:
Replace `submodules/diff-gaussian-rasterization` with the following repository:
https://github.com/JonathonLuiten/diff-gaussian-rasterization-w-depth

Then:

```bash
cd SurgicalGaussian
pip install -e submodules/simple-knn
cd submodules/diff-gaussian-rasterization-w-depth
python setup.py install
```

Install PyTorch3D:

```bash
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d
pip install -e .
```

Install tiny-cuda-nn:

```bash
git clone https://github.com/nvlabs/tiny-cuda-nn
cd tiny-cuda-nn
git submodule update --init --recursive
cmake . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build --config RelWithDebInfo -j
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

#### 4. YOLO Setup

To train your own YOLO model, you need:
- A `data.yaml` file
- The following folder structure:
  - `train/images`
  - `valid/images`
  - `test/images`

**Dataset Example (Yolov11):**
https://universe.roboflow.com/laparoscopic-yolo/laparoscopy/dataset/14