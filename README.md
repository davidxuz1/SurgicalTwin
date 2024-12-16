# Surgical Twin Project

## Environment Setup Guide

This guide provides detailed instructions for setting up, using, and understanding the features of the Surgical Twin project, including installation of required dependencies and models.

```bash
git clone https://github.com/davidxuz1/SurgicalTwin.git
cd SurgicalTwin
```

### 🗂️ Folder Setup

Within the `SurgicalTwin` directory, set up the following folder structure:

1. **Create `data` folder with `input` subfolder**:
   - `data`: This folder will store intermediate and processed data.
   - `data/input`: Place the `video.mp4` file to be processed here.

   ```bash
   mkdir -p data/input
   ```

2. **Create `models` folder**:
   - Within `models`, create a `pretrained` folder to store all pre-trained models.
   - Inside `pretrained`, create the following subfolders for different models:
     - `depth_model`
     - `pose_model`
     - `SAM2_model`
     - `STTN_inpainting_model`
     - `Surgical_Tools_Detection_Yolov11_Model`

   ```bash
   mkdir -p models/pretrained/{depth_model,pose_model,SAM2_model,STTN_inpainting_model,Surgical_Tools_Detection_Yolov11_Model}
   ```

3. **Create `third_party` folder**:
   - This folder will store external repositories and third-party dependencies.

   ```bash
   mkdir third_party
   ```

### Conda Environment Setup

Create and activate a new Conda environment:

```bash
conda create -n surgical_twin python=3.11
conda activate surgical_twin
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```

Install the required Python dependencies:

```bash
cd SurgicalTwin
pip install -r requirements.txt
```


### 🔧 Required Manual Installations

#### 1. YOLO Pre-trained Model

If you want to use the pre-trained YOLO model, download it from [this link](https://drive.google.com/file/d/1uPcsP2ISl4gAHR5xPXO-CnaZdc7OXwds/view?usp=sharing) and place it to `SurgicalTwin/models/pretrained/Surgical_Tools_Detection_Yolov11_Model`. If you prefer to train your own model, refer to Section 4 in **Tools** for YOLO Training.

#### 2. SAM2 Installation

```bash
cd third_party
git clone https://github.com/facebookresearch/segment-anything-2.git
cd segment-anything-2/
pip install -e .
python setup.py build_ext --inplace
```

**Download SAM2 Model**  
Download `sam2_hiera_tiny.pt` from [this link](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt) and place it in `SurgicalTwin/models/pretrained/SAM2_model/sam2_hiera_tiny.pt`.

#### 3. Monst3r Setup

```bash
cd third_party
git clone --recursive https://github.com/junyi42/monst3r
cd monst3r
cd data
bash download_ckpt.sh
```

**Download Monst3r Model**  
Download the model from [this link](https://drive.google.com/file/d/1Z1jO_JmfZj0z3bgMvCwqfUhyZ1bIbc9E/view) and place it in `SurgicalTwin/models/pretrained/pose_model/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt.pth`.

#### 4. Inpainting Setup

```bash
cd third_party
git clone https://github.com/researchmm/STTN
```

**Download STTN Model**  
Download `sttn.pth` from [this link](https://drive.google.com/file/d/1ZAMV8547wmZylKRt5qR_tC5VlosXD4Wv/view) and place it in `SurgicalTwin/models/pretrained/STTN_inpainting_model/sttn.pth`.

#### 5. Depth Setup

```bash
cd third_party
git clone https://github.com/DepthAnything/Depth-Anything-V2
```

**Download Depth Model**  
Download the model from [this link](https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true) and place it in `SurgicalTwin/models/pretrained/depth_model/depth_anything_v2_vitl.pth`.

#### 6. SurgicalGaussian Installation

```bash
cd third_party
git clone https://github.com/xwx0924/SurgicalGaussian
```

Then:

```bash
cd third_party
cd SurgicalGaussian/submodules
git clone https://github.com/camenduru/simple-knn 
cd simple-knn
pip install .
cd ..
rm -rf depth-diff-gaussian-rasterization
git clone https://github.com/JonathonLuiten/diff-gaussian-rasterization-w-depth 
cd diff-gaussian-rasterization-w-depth
pip install .

```

Install PyTorch3D:

```bash
cd third_party
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d
pip install -e .
```

Install tiny-cuda-nn:

```bash
cd third_party
git clone https://github.com/nvlabs/tiny-cuda-nn
cd tiny-cuda-nn
git submodule update --init --recursive
cmake . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build --config RelWithDebInfo -j
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

## 🚀 How to Use

To use the Surgical Twin Pipeline, navigate to the `src` directory and run the `main.py` script with the appropriate arguments:

```bash
cd src
python3 main.py
```

### Available Arguments

The script supports the following arguments:

- **Input Video**:
  - `--input_video`: Path to the input video file.  
    Default: `data/input/video.mp4`.

- **Stages**:
  - `--stages`: Select the stages to run.  
    Options: `all`, `detection`, `dehaze`, `segmentation`, `inpainting`, `depth`, `pose`, `gaussian`, `render`.  
    Default: `all`.

- **Detection Stage**:
  - `--model_path_detection`: Path to the YOLO model weights.  
    Default: `models/pretrained/Surgical_Tools_Detection_Yolov11_Model/surgical_tools_detection_model.pt`.
  - `--threshold_detection`: Detection confidence threshold (0-1).  
    Default: `0.6`.
  - `--dilation_factor_detection`: Bounding box dilation factor (>1).  
    Default: `1.2`.
  - `--fixed_bbox_watermark`: Fixed bounding box coordinates (x_min, y_min, x_max, y_max) where the video watermark is.

- **Segmentation Stage**:
  - `--batch_size_segmentation`: Number of frames to process in each batch.  
    Default: `300`.
  - `--dilatation_factor_segmentation`: Mask dilation factor.  
    Default: `10.0`.
  - `--mask_segmentation`: Choose 1 to save binary masks or 2 to skip saving.  
    Default: `2`.

- **Depth Stage**:
  - `--encoder_depth`: Encoder type for depth estimation.  
    Options: `vits`, `vitb`, `vitl`, `vitg`.  
    Default: `vitl`.

- **Pose Stage**:
  - `--image_size_pose`: Image size for pose estimation.  
    Options: `224`, `512`.  
    Default: `224`.
  - `--num_frames_pose`: Maximum number of frames to process.  
    Default: `300`.

- **Gaussian Stage**:
  - `--config_gaussian`: Path to the 3D Gaussian Splatting configuration file.  
    Default: `video_gaussian.py`.

### Outputs

1. **Intermediate Results**:
   - Stored in the `data/intermediate` folder, which contains subfolders for each stage:
     - `dehaze`
     - `depth`
     - `detection`
     - `inpainting`
     - `pose`
     - `segmentation`
     - `gaussian` (corresponding to the input needed for the Gaussian process)
     
2. **Logs**:
   - Execution times for each stage are logged in the `data/logs` folder.

3. **Final Output**:
   - The final digital twin generated using Dynamic 3D Gaussian Splatting is saved in the `output` folder.
   - For the `render` stage, the output is exported as a `video.mp4` file in `data/output/train/interpolate_40000/renders`.

### Example Usage

To process a video using all stages with default settings:

```bash
python3 main.py --input_video data/input/video.mp4 --stages all
```

To run only the segmentation stage:

```bash
python3 main.py --input_video data/input/video.mp4 --stages segmentation
```

#### Recommendation

We recommend processing the video stage by stage in the following order: `detection`, `dehaze`, `segmentation`, `inpainting`, `depth`, `pose`, `gaussian`, `render`. This approach allows you to review each output in the corresponding intermediate results folder before proceeding to the next stage, enabling parameter tuning and ensuring optimal results without waiting for all stages to complete.


### 🔨 Tools

Several utility tools are available in the `tools` directory to assist with various stages of the pipeline. Below are the tools and their usage:

#### 1. **Bounding Box Selection**
- **Script**: `tools/get_bounding_box_from_video.py`
- **Description**: Selects the bounding box of a watermark manually from the input video.
- **Output**: Creates a file `fixed_bbox_watermark.txt` in `data/tools_output/`, containing the bounding box coordinates.
- **Usage**:
  ```bash
  python3 tools/get_bounding_box_from_video.py
  ```
  - Input: `data/input/video.mp4`
  - Output: `data/tools_output/fixed_bbox_watermark.txt`
  - Use the generated bounding box with the `--fixed_bbox_watermark` argument in `main.py`.

#### 2. **Video Cutter**
- **Script**: `tools/video_cut.py`
- **Description**: Cuts a video between specified time intervals or extracts a single frame at a specific time. To specify the time intervals, use the arguments `--start` and `--end` for cutting, or `--frame` for extracting a frame. Specify the mode using the `--i` argument: `1` for cutting a video and `2` for extracting a frame.
- **Usage**:
  ```bash
  python3 tools/video_cut.py --input <path_to_video> --i <1_or_2> --start <start_time_in_minutes> --end <end_time_in_minutes> --frame <frame_time_in_minutes>
  ```
  - Input: A video file located at the specified path.
  - Output: A cut video (`video_cut.mp4`) if `--i 1`, or a single extracted frame (`frame_minuto_<time>.png`) if `--i 2`.


#### 3. **Pose Visualizer**
- **Script**: `tools/pose_visualizer/pose_visualizer.py`
- **Description**: Visualizes the interactive 4D results from the pose stage.
- **Output**: Interactive visualization of results from `data/intermediate/pose/pose_output`.
- **Usage**:
  ```bash
  python3 tools/pose_visualizer/pose_visualizer.py
  ```
  - Input: `data/intermediate/pose/pose_output`.

#### 4. **YOLO Training**
- **Training Script**: `tools/train_yolo/yolov11_train.py`
- **Description**: Trains the YOLOv11 model for surgical tool detection.
- **Setup**:
  1. Place your dataset with the following structure in `data/yolo_dataset`:
     ```
     data/yolo_dataset/
     ├── data.yaml
     ├── train/
     │   ├── images/
     │   └── labels/
     ├── valid/
     │   ├── images/
     │   └── labels/
     └── test/
         ├── images/
         └── labels/
     ```
  2. Example dataset: [Laparoscopic Yolo Dataset](https://universe.roboflow.com/laparoscopic-yolo/laparoscopy/dataset/14).
- **Usage**:
  ```bash
  python3 tools/train_yolo/yolov11_train.py
  ```
  - Output: Trained YOLO model (`best.pt`).
  - After obtaining the `best.pt` model, rename it to `surgical_tools_detection_model.pt` and move it to `models/pretrained/Surgical_Tools_Detection_Yolov11_Model`.


#### 5. **YOLO Testing**
- **Testing Script**: `tools/train_yolo/yolov11_test.py`
- **Description**: Tests the YOLO model on the input video and exports a video with bounding boxes.
- **Setup**:
  1. Place the trained YOLO model (`best.pt`) in `tools/train_yolo`.
  2. Run the script:
     ```bash
     python3 tools/train_yolo/yolov11_test.py
     ```
  - Input: `data/input/video.mp4` and `tools/train_yolo/best.pt`
  - Output: Processed video with bounding boxes in `data/tools_output/yolo_output`.


## 💡 Acknowledgments

We would like to express our gratitude to the developers of the following projects ([Haze Removal](https://github.com/trishababu/desmoking-algorithm-using-dcp-and-cnn), [ultralytics](https://github.com/ultralytics/ultralytics), [monst3r](https://github.com/Junyi42/monst3r), [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2/tree/main), [SurgicalGaussian](https://github.com/xwx0924/SurgicalGaussian), [STTN](https://github.com/researchmm/STTN), [SAM2](https://github.com/facebookresearch/sam2)), as some of our source code is borrowed from them. 

Their incredible work has been instrumental in making this project possible. 🙏
