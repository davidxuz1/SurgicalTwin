import os
import sys
import argparse
import torch
import cv2
import shutil
import numpy as np
from pathlib import Path
import subprocess
from stages.detection.surgical_tools_detection_stage import SurgicalToolsDetectionStage
from stages.dehaze.dehaze_video import DehazeStage
from stages.segmentation.segmentation_stage import SegmentationStage
from stages.inpainting.inpainting_stage import InpaintingStage
from stages.depth.depth_stage import DepthStage
from stages.pose.pose_stage import PoseStage
import time
from datetime import datetime

# Get absolute project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    parser = argparse.ArgumentParser(description="Surgical Twin Pipeline")
    parser.add_argument("--input_video", type=str,
                      default=os.path.join(PROJECT_ROOT, "data/input/video.mp4"),
                      help="Path to input video")
    
    # Stages argument
    parser.add_argument("--stages", type=str, default="all",
                      help="Stages to run: all | segmentation | dehaze | detection | depth | inpainting | pose | gaussian | render")

    # Detection stage
    parser.add_argument("--model_path_detection", type=str,
                      default=os.path.join(PROJECT_ROOT, "models/pretrained/Surgical_Tools_Detection_Yolov11_Model/surgical_tools_detection_model.pt"),
                      help="Path to YOLO model weights")
    parser.add_argument("--threshold_detection", type=float,
                      default=0.6,
                      help="Detection confidence threshold (0-1)")
    parser.add_argument("--dilation_factor_detection", type=float,
                      default=1.2,
                      help="Bounding box dilation factor (>1) for detection stage")
    parser.add_argument("--fixed_bbox_watermark", type=int, nargs=4,
                      help="Fixed bounding box coordinates (x_min y_min x_max y_max) for video watermark")
    
    # Segmentation stage
    parser.add_argument("--batch_size_segmentation", type=int, default=300,
                      help="Number of frames to process in each batch for segmentation stage")
    parser.add_argument("--dilatation_factor_segmentation", type=float, default=10.0,
                      help="Factor for mask dilatation for segmentation stage")
    parser.add_argument("--mask_segmentation", type=int, default=2,
                      help="1 to save binary masks, 2 to skip mask saving")
    
    # Depth stage
    parser.add_argument("--encoder_depth", type=str, default='vitl',
                    choices=['vits', 'vitb', 'vitl', 'vitg'],
                    help="Encoder type for depth estimation")
    
    # Pose stage
    parser.add_argument("--image_size_pose", type=int, default=224, 
                    help="Image size for pose estimation [224 or 512]")
    parser.add_argument('--num_frames_pose', type=int, default=300, 
                        help='Maximum number of frames for video processing in pose stage')


    # Gaussian stage
    parser.add_argument("--config_gaussian", type=str, default='video_gaussian.py', 
                    help="3D Gaussian Splatting config file")
    
    args = parser.parse_args()
    
    # Validate arguments

    if not os.path.exists(args.input_video):
        sys.exit(f"Error: Input video not found at {args.input_video}")
    
    valid_stages = [
        "all",  # all stages
        "segmentation",
        "dehaze",
        "detection",
        "inpainting",
        "depth",
        "pose",
        "gaussian", 
        "render"
    ]

    
    if args.stages not in valid_stages:
        sys.exit(f"Error: Invalid --stages value. Must be one of: {', '.join(valid_stages)}")

    # Detection stage validations
    if args.stages in ["all", "detection"]:
        if not os.path.exists(args.model_path_detection):
            sys.exit(f"Error: Model weights for detection stage not found at {args.model_path_detection}")
        if not 0 <= args.threshold_detection <= 1:
            sys.exit(f"Error: Threshold detection stage must be between 0 and 1")
        if args.dilation_factor_detection <= 1:
            sys.exit(f"Error: Dilation factor for detection stage must be greater than 1")
        if args.fixed_bbox_watermark is not None and len(args.fixed_bbox_watermark) != 4:
            sys.exit(f"Error: Fixed bounding box of video watermark must have 4 coordinates")

    # Segmentation stage validations
    if args.stages in ["all", "segmentation"]:
        if args.dilatation_factor_segmentation <= 1:
            sys.exit(f"Error: Dilation factor for segmentation stage must be greater than 1")
        if args.mask_segmentation not in [1, 2]:
            sys.exit(f"Error: Mask saving option in segmentation stage must be 1 or 2")
        if args.batch_size_segmentation <= 0:
            sys.exit(f"Error: Batch size for segmentation stage must be greater than 0")

    # Depth stage validations
    if args.stages in ["all", "depth"]:
        if args.encoder_depth not in ['vits', 'vitb', 'vitl', 'vitg']:
            sys.exit(f"Error: Encoder type must be one of ['vits', 'vitb', 'vitl', 'vitg']")


    # Pose stage validations
    if args.stages in ["all", "pose"]:
        if args.image_size_pose not in [224, 512]:
            sys.exit(f"Error: Image size for pose stage must be 224 or 512")
        if args.num_frames_pose <= 0:
            sys.exit(f"Error: Number of frames for pose stage must be greater than 0")
    
    # Gaussian stage validations
    if args.stages in ["all", "gaussian"]:
        if not args.config_gaussian.endswith('.py'):
            sys.exit(f"Error: Config file for gaussian stage must be a .py file")

    return args


def log_time(file_path, stage_name, time_taken):
    with open(file_path, 'a') as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"{timestamp} - {stage_name}: {time_taken:.2f} seconds\n")

def main():
    args = parse_args()
    total_start_time = time.time()

    # Create a log file for processing times
    log_dir = os.path.join(PROJECT_ROOT, "data/logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "processing_times.txt")

    # Fixed paths
    detection_output_dir = os.path.join(PROJECT_ROOT, "data/intermediate/detection")
    dehaze_output_dir = os.path.join(PROJECT_ROOT, "data/intermediate/dehaze")
    segmentation_output_dir = os.path.join(PROJECT_ROOT, "data/intermediate/segmentation")
    inpainting_output_dir = os.path.join(PROJECT_ROOT, "data/intermediate/inpainting")
    depth_output_dir = os.path.join(PROJECT_ROOT, "data/intermediate/depth")
    pose_output_dir = os.path.join(PROJECT_ROOT, "data/intermediate/pose")
    gaussian_output_dir = os.path.join(PROJECT_ROOT, "data/output")
    os.makedirs(detection_output_dir, exist_ok=True)
    os.makedirs(dehaze_output_dir, exist_ok=True)
    os.makedirs(segmentation_output_dir, exist_ok=True)
    os.makedirs(inpainting_output_dir, exist_ok=True)
    os.makedirs(depth_output_dir, exist_ok=True)
    os.makedirs(pose_output_dir, exist_ok=True)
    os.makedirs(gaussian_output_dir, exist_ok=True)

    # Output path for detection stage
    detected_tools_video = os.path.join(detection_output_dir, "surgical_tools_detected.mp4")
    tools_bbox_file = os.path.join(detection_output_dir, "surgical_tools_bbox.txt")

    # Output path for dehaze stage
    dehazed_video = os.path.join(dehaze_output_dir, "dehazed_video.mp4")

    # Output path for segmentation stage
    segmented_video = os.path.join(segmentation_output_dir, "segmented_video.mp4")
    segmented_masks = os.path.join(segmentation_output_dir, "segmented_video_masks.npy")

    # Output path for inpainting stage
    inpainted_video = os.path.join(inpainting_output_dir, "inpainted_video.mp4")

    # Output path for depth stage
    depth_frames_dir = os.path.join(depth_output_dir, "depth_frames")

    # Output path for pose stage
    poses_bounds = os.path.join(pose_output_dir, "poses_bounds.npy")
    pose_output_dir = os.path.join(pose_output_dir, "pose_output")


    ####################################################################################
    ###########################SurgicalToolsDetectionStage##############################
    ####################################################################################
    if args.stages in ["all", "detection"]:
        print("\n" + "="*80)
        print("Starting Surgical Tools Detection Stage")
        print(f"Input video: {args.input_video}")
        print("="*80 + "\n")

        detection_start_time = time.time()

        try:
            # Initialize detection stage
            detection_stage = SurgicalToolsDetectionStage(
                model_path=args.model_path_detection,
                threshold=args.threshold_detection,
                dilation_factor=args.dilation_factor_detection,
                fixed_bbox_watermark=args.fixed_bbox_watermark
            )
            
            # Process video
            video_output, txt_output = detection_stage.process(
                args.input_video,
                detected_tools_video,
                tools_bbox_file
            )
            
            torch.cuda.empty_cache()
            print(f"Detection video saved at: {video_output}")
            print(f"Coordinates file saved at: {txt_output}")

            detection_time = time.time() - detection_start_time
            log_time(log_file, "Tools Detection", detection_time)

            print("\n" + "="*80)
            print("Detection Process completed successfully")
            print(f"Processing time: {detection_time:.2f} seconds")
            print(f"Detection video saved at: {video_output}")
            print(f"Coordinates file saved at: {txt_output}")
            print("="*80 + "\n")

        except Exception as e:
            sys.exit(f"Error during detection: {str(e)}")


    ####################################################################################
    #################################DehazeStage########################################
    ####################################################################################
    if args.stages in ["all", "dehaze"]:
        print("\n" + "="*80)
        print("Starting Dehaze stage")
        print(f"Input video: {args.input_video}")
        print("="*80 + "\n")

        dehaze_start_time = time.time()
        
        try:
            # Initialize and run dehaze stage
            dehaze_stage = DehazeStage()
            dehazed_video_path = dehaze_stage.process(args.input_video, dehazed_video)
            
            torch.cuda.empty_cache()
            dehaze_time = time.time() - dehaze_start_time
            log_time(log_file, "Dehaze", dehaze_time)
            
            print("\n" + "="*80)
            print("Dehaze process completed successfully")
            print(f"Processing time: {dehaze_time:.2f} seconds")
            print(f"Processed video saved at: {dehazed_video_path}")
            print("="*80 + "\n")
            
        except Exception as e:
            sys.exit(f"Error during dehaze processing: {str(e)}")

    ####################################################################################
    #################################SegmentationStage##################################
    ####################################################################################
    if args.stages in ["all", "segmentation"]:
        print("\n" + "="*80)
        print("Starting Segmentation stage")
        print(f"Input video: {dehazed_video}")
        print("="*80 + "\n")

        segmentation_start_time = time.time()

        try:
            # Initialize and run segmentation stage
            sam2_model_path = os.path.join(PROJECT_ROOT, "models/pretrained/SAM2_model/sam2_hiera_tiny.pt")
            if not os.path.exists(sam2_model_path):
                sys.exit(f"Error: SAM2 model not found at {sam2_model_path}")

            segmentation_stage = SegmentationStage(
                model_path=sam2_model_path,
                batch_size=args.batch_size_segmentation,
                dilatation_factor=args.dilatation_factor_segmentation,
                save_masks=(args.mask_segmentation == 1)
            )
            
            segmented_video_path = segmentation_stage.process(
                dehazed_video,
                tools_bbox_file,
                segmented_video
            )
            
            torch.cuda.empty_cache()
            segmentation_time = time.time() - segmentation_start_time
            log_time(log_file, "Segmentation", segmentation_time)
            
            print("\n" + "="*80)
            print("Segmentation process completed successfully")
            print(f"Processing time: {segmentation_time:.2f} seconds")
            print(f"Segmented video saved at: {segmented_video_path}")
            print("="*80 + "\n")
            
        except Exception as e:
            sys.exit(f"Error during segmentation processing: {str(e)}")

    ####################################################################################
    #################################InpaintingStage####################################
    ####################################################################################

    if args.stages in ["all", "inpainting"]:
        print("\n" + "="*80)
        print("Starting Inpainting stage")
        print(f"Input video: {dehazed_video} and segmentation masks: {segmented_masks}")
        print("="*80 + "\n")

        inpainting_start_time = time.time()
        
        try:
            # Initialize and run inpainting stage
            sttn_model_path = os.path.join(PROJECT_ROOT, "models/pretrained/STTN_inpainting_model/sttn.pth")
            if not os.path.exists(sttn_model_path):
                sys.exit(f"Error: STTN model not found at {sttn_model_path}")
                
            inpainting_stage = InpaintingStage(sttn_model_path)
            inpainted_video_path = inpainting_stage.process(
                dehazed_video,
                segmented_masks,
                inpainted_video
            )
            
            torch.cuda.empty_cache()
            inpainting_time = time.time() - inpainting_start_time
            log_time(log_file, "Inpainting", inpainting_time)
            
            print("\n" + "="*80)
            print("Inpainting process completed successfully")
            print(f"Processing time: {inpainting_time:.2f} seconds")
            print(f"Inpainted video saved at: {inpainted_video_path}")
            print("="*80 + "\n")
            
        except Exception as e:
            sys.exit(f"Error during inpainting processing: {str(e)}")


    ####################################################################################
    #################################DepthStage#########################################
    ####################################################################################
    if args.stages in ["all", "depth"]:
        print("\n" + "="*80)
        print("Starting Depth stage")
        print(f"Input video: {inpainted_video}")
        print("="*80 + "\n")

        # Verify Depth-Anything model existence
        depth_model_path = os.path.join(PROJECT_ROOT, "models/pretrained/depth_model", f"depth_anything_v2_{args.encoder_depth}.pth")
        if not os.path.exists(depth_model_path):
            sys.exit(f"Error: Depth-Anything model not found at {depth_model_path}")

        depth_start_time = time.time()
        
        try:
            # Initialize and run depth stage
            depth_stage = DepthStage(
                model_path=depth_model_path,
                model_type=args.encoder_depth
            )
            depth_frames_path = depth_stage.process(inpainted_video, depth_frames_dir)
            
            torch.cuda.empty_cache()
            depth_time = time.time() - depth_start_time
            log_time(log_file, "Depth Estimation", depth_time)
            
            print("\n" + "="*80)
            print("Depth process completed successfully")
            print(f"Processing time: {depth_time:.2f} seconds")
            print(f"Depth frames saved at: {depth_frames_path}")
            print("="*80 + "\n")
            
        except Exception as e:
            sys.exit(f"Error during depth processing: {str(e)}")


    ####################################################################################
    #################################PoseStage##########################################
    ####################################################################################
    if args.stages in ["all", "pose"]:
        print("\n" + "="*80)
        print("Starting Pose stage")
        print(f"Input video directory: {inpainted_video}")
        print("="*80 + "\n")

        # Initialize and run segmentation stage
        pose_model_path = os.path.join(PROJECT_ROOT, "models/pretrained/pose_model/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt.pth")
        if not os.path.exists(pose_model_path):
            sys.exit(f"Error: Pose model not found at {pose_model_path}")
            
        pose_start_time = time.time()
        
        try:
            # Initialize and run pose stage
            pose_stage = PoseStage(
            model_path=pose_model_path,
            image_size=args.image_size_pose,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            output_dir=pose_output_dir,
            use_gt_mask=False,
            fps=0,
            num_frames=args.num_frames_pose
            )
            poses_bounds_path = pose_stage.process(
                input_video= inpainted_video,
                output_path=poses_bounds,
                output_dir=pose_output_dir
            )

            torch.cuda.empty_cache()
            pose_time = time.time() - pose_start_time
            log_time(log_file, "Pose Estimation", pose_time)
            
            print("\n" + "="*80)
            print("Pose process completed successfully")
            print(f"Processing time: {pose_time:.2f} seconds")
            print(f"Poses bounds saved at: {poses_bounds_path}")
            print("="*80 + "\n")
            
        except Exception as e:
            sys.exit(f"Error during pose processing: {str(e)}")
                
    ####################################################################################### 
    #################################GaussianStage#########################################
    #######################################################################################

    def process_frame_to_square(img, target_size):

        h, w = img.shape[:2]
        ratio = min(target_size / w, target_size / h)
        new_w = int(w * ratio)
        new_h = int(h * ratio)
        
        resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Check if input is grayscale or RGB
        channels = 1 if len(img.shape) == 2 else img.shape[2]
        square_img = np.zeros((target_size, target_size, channels), dtype=img.dtype)
        
        x_offset = (target_size - new_w) // 2
        y_offset = (target_size - new_h) // 2
        
        if channels == 1:
            # Handle grayscale images
            if len(resized_img.shape) == 2:
                resized_img = resized_img[..., np.newaxis]
            square_img[y_offset:y_offset+new_h, x_offset:x_offset+new_w, 0] = resized_img[..., 0]
        else:
            # Handle RGB images
            square_img[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_img
        
        return square_img

    if args.stages in ["all", "gaussian"]:
        # Define paths
        gaussian_input_dir = os.path.join(PROJECT_ROOT, "data/intermediate/gaussian")
        depth_input_dir = os.path.join(gaussian_input_dir, "depth")
        images_input_dir = os.path.join(gaussian_input_dir, "images")
        poses_bounds_path = os.path.join(gaussian_input_dir, "poses_bounds.npy")

        # Function to remove existing directories and files
        def remove_existing_dirs_and_file():
            if os.path.exists(depth_input_dir):
                shutil.rmtree(depth_input_dir)
                print(f"Deleted folder: {depth_input_dir}")
            if os.path.exists(images_input_dir):
                shutil.rmtree(images_input_dir)
                print(f"Deleted folder: {images_input_dir}")
            if os.path.exists(poses_bounds_path):
                os.remove(poses_bounds_path)
                print(f"Deleted file: {poses_bounds_path}")

        # Remove existing directories and files
        remove_existing_dirs_and_file()

        # Create necessary directories
        os.makedirs(gaussian_input_dir, exist_ok=True)
        os.makedirs(depth_input_dir, exist_ok=True)
        os.makedirs(images_input_dir, exist_ok=True)
            
        print("\n" + "="*80)
        print("Starting 3D Dynamic Gaussian Splatting stage")
        print(f"Input directory: {gaussian_input_dir}")
        print("="*80 + "\n")
            
        gaussian_start_time = time.time()
        
        try:
            # Process video frames
            cap = cv2.VideoCapture(inpainted_video)
            if not cap.isOpened():
                raise Exception(f"Could not open video: {inpainted_video}")

            frame_count = 0
            target_size = args.image_size_pose
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                square_frame = process_frame_to_square(frame, target_size)
                frame_path = os.path.join(images_input_dir, f"frame_{frame_count:06d}.png")
                cv2.imwrite(frame_path, square_frame, [cv2.IMWRITE_PNG_COMPRESSION, 9])
                frame_count += 1
                
            cap.release()
            
            if frame_count == 0:
                raise Exception("No frames could be extracted from the video")

            # Process depth frames
            depth_files = sorted(os.listdir(depth_frames_dir))
            if not depth_files:
                raise Exception(f"No depth files found in: {depth_frames_dir}")

            for idx, depth_file in enumerate(depth_files):
                depth_path = os.path.join(depth_frames_dir, depth_file)
                depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
                if depth_img is None:
                    raise Exception(f"Could not read depth file: {depth_path}")

                processed_depth = process_frame_to_square(depth_img, target_size)
                output_depth_path = os.path.join(depth_input_dir, f"depth_{idx:06d}.png")
                cv2.imwrite(output_depth_path, processed_depth, [cv2.IMWRITE_PNG_COMPRESSION, 9])

            # Copy pose bounds
            if not os.path.exists(poses_bounds):
                raise Exception(f"Poses bounds file not found: {poses_bounds}")
 
            shutil.copy2(poses_bounds, os.path.join(gaussian_input_dir, "poses_bounds.npy"))
            
            # Configure and execute Gaussian stage
            arguments_gaussian = os.path.join(PROJECT_ROOT, f"config/gaussian_stage/{args.config_gaussian}")
            if not os.path.exists(arguments_gaussian):
                raise Exception(f"Configuration file not found: {arguments_gaussian}")

            # Get absolute path to SurgicalGaussian directory
            gaussian_dir = os.path.join(PROJECT_ROOT, "third_party/SurgicalGaussian")
            
            # Save current working directory
            original_dir = os.getcwd()
            
            # Change to SurgicalGaussian directory
            os.chdir(gaussian_dir)

            # command = [
            #     "python",
            #     "train.py",
            #     "-s", "data/ownvideo",
            #     "-m", "output/ownvideo",
            #     "--config", "arguments/endonerf/ownvideo.py"
            # ]
            
            # Prepare relative paths
            relative_input_dir = os.path.relpath(gaussian_input_dir, gaussian_dir)
            relative_output_dir = os.path.relpath(gaussian_output_dir, gaussian_dir)
            relative_config = os.path.relpath(arguments_gaussian, gaussian_dir)

            command = [
                "python",
                "train.py",
                "-s", relative_input_dir,
                "-m", relative_output_dir,
                "--config", relative_config
            ]

            process = subprocess.run(command, check=True)

            
            # Restore original working directory
            os.chdir(original_dir)
            
            if process.returncode == 0:
                print("\nGaussian Splatting training completed successfully")
                torch.cuda.empty_cache()
            else:
                raise Exception("Error during Gaussian Splatting training")

            gaussian_time = time.time() - gaussian_start_time
            log_time(log_file, "Gaussian Splatting", gaussian_time)
            
            print("\n" + "="*80)
            print("Gaussian Splatting process completed successfully")
            print(f"Processing time: {gaussian_time:.2f} seconds")
            print("="*80 + "\n")

        except Exception as e:
            # Ensure original directory is restored in case of error
            os.chdir(original_dir)
            sys.exit(f"Error during Gaussian Splatting processing: {str(e)}")

    #######################################################################################
    ################################## Render Stage #######################################
    #######################################################################################

    if args.stages in ["all", "render"]:
        print("\n" + "="*80)
        print("Starting Render stage")
        print("="*80 + "\n")
        
        render_start_time = time.time()
        
        # Change to SurgicalGaussian directory
        gaussian_dir = os.path.join(PROJECT_ROOT, "third_party/SurgicalGaussian")
        original_dir = os.getcwd()
        os.chdir(gaussian_dir)

        relative_output_dir = os.path.relpath(gaussian_output_dir, gaussian_dir)
        render_command = [
            "python",
            "render.py",
            "--model_path", relative_output_dir,
            "--mode", "time",
            "--iteration", "40000"
        ]

        try:
            process = subprocess.run(render_command, check=True)
            os.chdir(original_dir)

            if process.returncode == 0:
                print("\nRender process completed successfully")    
            else:
                raise Exception("Error during render")
            
            torch.cuda.empty_cache()
            render_time = time.time() - render_start_time
            log_time(log_file, "Render", render_time)

            print("\n" + "="*80)
            print("Render process completed successfully")
            print(f"Processing time: {render_time:.2f} seconds")
            print("="*80 + "\n")

        except Exception as e:
            os.chdir(original_dir)
            sys.exit(f"Error during Render processing: {str(e)}")


    #######################################################################################


    # Total processing time
    total_time = time.time() - total_start_time
    log_time(log_file, "Total Time", total_time)
    
    print("\n" + "="*80)
    print("PROCESSING SUMMARY")
    print(f"Total execution time: {total_time:.2f} seconds")
    print(f"Processing times log saved at: {log_file}")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()














