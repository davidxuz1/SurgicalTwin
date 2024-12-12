import os
import shutil
import tempfile
import numpy as np
import torch
import cv2
from sam2.build_sam import build_sam2_video_predictor

class SegmentationStage:
    def __init__(self, model_path, batch_size=50, dilatation_factor=1.0, save_masks=False):
        self.model_path = model_path
        self.batch_size = batch_size
        self.dilatation_factor = dilatation_factor
        self.save_masks = save_masks
        self.frame_boxes = None

    def parse_bbox_line(self, line):
        boxes = []
        parts = line.split(": ")
        if len(parts) < 2 or not parts[1].strip():
            return boxes
        bbox_parts = parts[1].strip()
        bbox_strings = bbox_parts.split("] [")
        for bbox_str in bbox_strings:
            bbox_str = bbox_str.replace("[", "").replace("]", "")
            if bbox_str:
                coords = list(map(int, bbox_str.split(",")))
                boxes.append(coords)
        return boxes

    def read_bbox_file(self, file_path):
        frame_boxes = {}
        with open(file_path, 'r') as f:
            for line in f:
                frame_num = int(line.split("_")[1].split(":")[0])
                boxes = self.parse_bbox_line(line)
                if boxes:
                    frame_boxes[frame_num] = boxes
        return frame_boxes

    def dilate_mask(self, mask, dilatation_factor):
        kernel_size = int(round(dilatation_factor * 2))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        return cv2.dilate(mask, kernel, iterations=1)

    def calculate_iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        return intersection / float(area1 + area2 - intersection)


    def check_forward_consistency(self, frame_boxes, current_frame, lookforward=3):
        """
        Verifica la consistencia de los boxes en los siguientes frames
        """
        boxes_sequence = []
        if current_frame in frame_boxes:
            boxes_sequence.append(frame_boxes[current_frame])
        else:
            return None

        # Verificar los siguientes frames
        for frame in range(current_frame + 1, current_frame + lookforward):
            if frame not in frame_boxes:
                return None
            boxes_sequence.append(frame_boxes[frame])

        if len(boxes_sequence) < lookforward:
            return None

        # Verificar consistencia usando IoU
        consistent_boxes = []
        for box in boxes_sequence[0]:
            is_consistent = True
            for frame_boxes in boxes_sequence[1:]:
                found_match = False
                for current_box in frame_boxes:
                    if self.calculate_iou(box, current_box) > 0.7:
                        found_match = True
                        break
                if not found_match:
                    is_consistent = False
                    break
            if is_consistent:
                consistent_boxes.append(box)

        return consistent_boxes if consistent_boxes else None


    def process(self, video_input, bbox_file, output_path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        frame_boxes = self.read_bbox_file(bbox_file)
        if not frame_boxes:
            raise ValueError("No valid bounding boxes found in the file")

        print("Initializing SAM2 model...")
        predictor = build_sam2_video_predictor(
            "sam2_hiera_t.yaml",
            self.model_path,
            device=device
        )

        all_binary_masks = []
        if self.save_masks:
            masks_dir = os.path.join(os.path.dirname(output_path), 'binary_masks')
            os.makedirs(masks_dir, exist_ok=True)

        with tempfile.TemporaryDirectory() as temp_dir:
            original_frames_dir = os.path.join(temp_dir, 'original_frames')
            batch_frames_dir = os.path.join(temp_dir, 'batch_frames')
            os.makedirs(original_frames_dir, exist_ok=True)
            os.makedirs(batch_frames_dir, exist_ok=True)

            print(f"Extracting frames from video: {video_input}")
            os.system(f"ffmpeg -i {video_input} -q:v 2 -start_number 0 {original_frames_dir}/%05d.jpg")

            frame_names = sorted([f for f in os.listdir(original_frames_dir) if f.endswith('.jpg')])
            total_frames = len(frame_names)
            print(f"Total frames to process: {total_frames}")

            first_frame = cv2.imread(os.path.join(original_frames_dir, frame_names[0]))
            frame_height, frame_width, _ = first_frame.shape
            print(f"Frame dimensions: {frame_width}x{frame_height}")

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, 30, (frame_width, frame_height))

            try:
                batch_start = 0
                while batch_start < total_frames:
                    batch_frame_num = batch_start + 1

                    # Verificar consistencia hacia adelante
                    forward_consistent_boxes = self.check_forward_consistency(frame_boxes, batch_frame_num)
                    
                    if forward_consistent_boxes:
                        initial_boxes = forward_consistent_boxes
                        batch_end = min(batch_start + self.batch_size, total_frames)
                        print(f"Found consistent boxes in frames {batch_frame_num}-{batch_frame_num+2}: {initial_boxes}")
                        print(f"\nProcessing batch {batch_start}-{batch_end}")
                    else:
                        # Si no hay boxes consistentes, intentar con el siguiente frame
                        batch_start += 1
                        print(f"No consistent boxes found for frames {batch_frame_num}-{batch_frame_num+2}, moving to next frame")
                        continue

                    batch_frames = frame_names[batch_start:batch_end]
                    for frame_name in os.listdir(batch_frames_dir):
                        os.remove(os.path.join(batch_frames_dir, frame_name))
                    
                    for frame_name in batch_frames:
                        shutil.copy(
                            os.path.join(original_frames_dir, frame_name),
                            os.path.join(batch_frames_dir, frame_name)
                        )

                    try:
                        inference_state = predictor.init_state(video_path=batch_frames_dir)

                        for obj_id, box in enumerate(initial_boxes, start=1):
                            _, _, _ = predictor.add_new_points_or_box(
                                inference_state=inference_state,
                                frame_idx=0,
                                obj_id=obj_id,
                                box=np.array(box, dtype=np.float32)
                            )

                        print("Propagating segmentation...")
                        video_segments = {}
                        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
                            video_segments[out_frame_idx] = {
                                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                                for i, out_obj_id in enumerate(out_obj_ids)
                            }

                        for rel_frame_idx, frame_name in enumerate(batch_frames):
                            frame = cv2.imread(os.path.join(original_frames_dir, frame_name))
                            if frame is None:
                                print(f"Warning: Failed to read frame {frame_name}")
                                continue

                            combined_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)

                            if rel_frame_idx in video_segments:
                                for out_obj_id, mask in video_segments[rel_frame_idx].items():
                                    if mask is None or mask.size == 0:
                                        continue

                                    mask_uint8 = (mask * 255).astype(np.uint8)
                                    if mask_uint8.ndim == 3 and mask_uint8.shape[0] == 1:
                                        mask_uint8 = np.squeeze(mask_uint8, axis=0)

                                    if mask_uint8.shape[0] == 0 or mask_uint8.shape[1] == 0:
                                        continue

                                    try:
                                        resized_mask = cv2.resize(mask_uint8, (frame_width, frame_height), interpolation=cv2.INTER_NEAREST)
                                        dilated_mask = self.dilate_mask(resized_mask, self.dilatation_factor)
                                        combined_mask = np.logical_or(combined_mask, dilated_mask > 0)

                                        mask_overlay = cv2.merge([dilated_mask, dilated_mask, dilated_mask])
                                        mask_color = np.array([0, 255, 0], dtype=np.uint8) if out_obj_id % 2 == 0 else np.array([255, 0, 0], dtype=np.uint8)
                                        mask_overlay = (mask_overlay > 0) * mask_color
                                        frame = cv2.addWeighted(frame, 0.7, mask_overlay, 0.3, 0)

                                    except cv2.error as e:
                                        print(f"Error processing mask for frame {frame_name}: {e}")
                                        continue

                            if self.save_masks:
                                mask_filename = f"mask_{batch_start + rel_frame_idx:05d}.png"
                                mask_path = os.path.join(masks_dir, mask_filename)
                                binary_mask = (combined_mask * 255).astype(np.uint8)
                                cv2.imwrite(mask_path, binary_mask, [cv2.IMWRITE_PNG_COMPRESSION, 9])

                            all_binary_masks.append(combined_mask.astype(np.bool_))

                            if rel_frame_idx == 0 and batch_frame_num in frame_boxes:
                                for box in frame_boxes[batch_frame_num]:
                                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 255), 2)

                            out.write(frame)

                    except RuntimeError as e:
                        print(f"Error processing batch {batch_start}-{batch_end}: {e}")

                    batch_start = batch_end

                out.release()
                masks_output_path = output_path.rsplit('.', 1)[0] + '_masks.npy'
                np.save(masks_output_path, np.array(all_binary_masks))
                print(f"Segmented video saved to {output_path}")
                print(f"Binary masks saved to {masks_output_path}")

            except Exception as e:
                print(f"An error occurred during video processing: {e}")
                raise

        return output_path
