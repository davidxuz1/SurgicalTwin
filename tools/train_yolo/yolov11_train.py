import torch
from ultralytics import YOLO
import logging
from datetime import datetime
import optuna
import os
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Define paths
DATA_YAML = os.path.join(PROJECT_ROOT, "data/yolo_dataset/data.yaml")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data/tools_output/yolo_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Start timing
start_time = time.time()

# Liberar memoria CUDA
torch.cuda.empty_cache()

# Check CUDA availability
cuda_available = torch.cuda.is_available()
logger.info(f"CUDA available: {cuda_available}")
logger.info(f"Training data location: {os.path.join(PROJECT_ROOT, 'data/yolo_dataset')}")
logger.info(f"Training outputs will be saved to: {OUTPUT_DIR}")

# Load the pre-trained YOLOv11 model
model = YOLO("yolo11n.pt")

def objective(trial):
    # Reducir el rango de epochs y batch size para evitar problemas de memoria
    epochs = trial.suggest_int('epochs', 50, 200)
    imgsz = trial.suggest_categorical('imgsz', [480, 640])
    batch_size = trial.suggest_int('batch_size', 4, 32)
    
    # Data augmentation parameters
    hsv_h = trial.suggest_float('hsv_h', 0.0, 0.1)
    hsv_s = trial.suggest_float('hsv_s', 0.0, 0.9)
    hsv_v = trial.suggest_float('hsv_v', 0.0, 0.9)
    degrees = trial.suggest_int('degrees', 0, 45)
    translate = trial.suggest_float('translate', 0.0, 0.5)
    scale = trial.suggest_float('scale', 0.1, 1.0)
    shear = trial.suggest_int('shear', 0, 10)
    perspective = trial.suggest_float('perspective', 0.0, 0.001)
    flipud = trial.suggest_float('flipud', 0.0, 1.0)
    fliplr = trial.suggest_float('fliplr', 0.0, 1.0)
    mosaic = trial.suggest_float('mosaic', 0.0, 1.0)
    mixup = trial.suggest_float('mixup', 0.0, 1.0)
    
    # Configurar variables de entorno para gestión de memoria
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'
    
    try:
        train_results = model.train(
            data=DATA_YAML,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            device=0 if cuda_available else 'cpu',
            classes=[2, 3, 6],
            project=OUTPUT_DIR,
            name=f"trial_{trial.number}",
            augment=True,
            hsv_h=hsv_h,
            hsv_s=hsv_s,
            hsv_v=hsv_v,
            degrees=degrees,
            translate=translate,
            scale=scale,
            shear=shear,
            perspective=perspective,
            flipud=flipud,
            fliplr=fliplr,
            mosaic=mosaic,
            mixup=mixup,
            cache=False
        )
        return train_results.results_dict['metrics/mAP50-95(B)']
    except RuntimeError as e:
        if "out of memory" in str(e):
            torch.cuda.empty_cache()
            return float('-inf')
        raise e

# Start optimization timing
optimization_start_time = time.time()

# Perform hyperparameter optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)

optimization_duration = time.time() - optimization_start_time
logger.info(f"Hyperparameter optimization completed in {optimization_duration:.2f} seconds")

best_params = study.best_params
logger.info(f"Best hyperparameters: {best_params}")

# Start final training timing
final_training_start_time = time.time()

# Train the model with the best parameters
try:
    final_model_path = os.path.join(OUTPUT_DIR, "final_training")
    train_results = model.train(
        data=DATA_YAML,
        epochs=best_params['epochs'],
        imgsz=best_params['imgsz'],
        batch=best_params['batch_size'],
        device=0 if cuda_available else 'cpu',
        project=OUTPUT_DIR,
        name="final_training",
        classes=[2, 3, 6],
        augment=True,
        hsv_h=best_params['hsv_h'],
        hsv_s=best_params['hsv_s'],
        hsv_v=best_params['hsv_v'],
        degrees=best_params['degrees'],
        translate=best_params['translate'],
        scale=best_params['scale'],
        shear=best_params['shear'],
        perspective=best_params['perspective'],
        flipud=best_params['flipud'],
        fliplr=best_params['fliplr'],
        mosaic=best_params['mosaic'],
        mixup=best_params['mixup']
    )
    logger.info(f"Training completed successfully. Results saved in: {final_model_path}")
except Exception as e:
    logger.error(f"An error occurred during training: {str(e)}")

# Evaluate the model's performance on the validation set
metrics = model.val()
logger.info(f"Validation metrics: {metrics}")

# Export the model to .pt format
try:
    model_version = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_path = os.path.join(OUTPUT_DIR, f"best_model_{model_version}.pt")
    model.save(export_path)
    logger.info(f"Model exported to: {export_path}")
except Exception as e:
    logger.error(f"Error exporting model: {str(e)}")

# Calculate and log total duration
total_duration = time.time() - start_time
hours = int(total_duration // 3600)
minutes = int((total_duration % 3600) // 60)
seconds = int(total_duration % 60)

logger.info(f"\nTotal execution time: {hours:02d}:{minutes:02d}:{seconds:02d}")
logger.info(f"- Optimization phase: {optimization_duration:.2f} seconds")
logger.info(f"- Final training phase: {time.time() - final_training_start_time:.2f} seconds")