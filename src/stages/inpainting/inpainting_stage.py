import torch
import numpy as np
import cv2
import sys
import imageio
import imageio.v2 as iio
import importlib
import os
from pathlib import Path
from typing import List
from PIL import Image
from torchvision import transforms

# Obtener la ruta absoluta al directorio STTN
STTN_PATH = str(Path(__file__).resolve().parent.parent.parent.parent / "third_party" / "STTN")
sys.path.insert(0, STTN_PATH)
from core.utils import Stack, ToTorchFormatTensor

# Consolidated transforms
_to_tensors = transforms.Compose([
    Stack(),
    ToTorchFormatTensor()
])

class VideoInpainter:
    def __init__(self, vi_ckpt):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.inpainter = self._build_sttn_model(vi_ckpt)
        self.w, self.h = 432, 240
        self.neighbor_stride = 5
        self.ref_length = 10

    def _build_sttn_model(self, ckpt_p, model_type="sttn"):
        sys.path.insert(0, STTN_PATH)
        net = importlib.import_module(f'model.{model_type}')
        model = net.InpaintGenerator().to(self.device)
        data = torch.load(ckpt_p, map_location=self.device)
        model.load_state_dict(data['netG'])
        model.eval()
        return model


    def _get_ref_index(self, neighbor_ids, length):
        ref_index = []
        for i in range(0, length, self.ref_length):
            if not i in neighbor_ids:
                ref_index.append(i)
        return ref_index

    @torch.no_grad()
    def inpaint_video(self, frames: List[Image.Image], masks: List[Image.Image]) -> List[Image.Image]:
        video_length = len(frames)
        
        # Prepare features and masks
        feats = [frame.resize((self.w, self.h)) for frame in frames]
        feats = _to_tensors(feats).unsqueeze(0) * 2 - 1
        _masks = [mask.resize((self.w, self.h), Image.NEAREST) for mask in masks]
        _masks = _to_tensors(_masks).unsqueeze(0)

        feats, _masks = feats.to(self.device), _masks.to(self.device)
        comp_frames = [None] * video_length

        # Process features
        feats = (feats * (1 - _masks).float()).view(video_length, 3, self.h, self.w)
        feats = self.inpainter.encoder(feats)
        _, c, feat_h, feat_w = feats.size()
        feats = feats.view(1, video_length, c, feat_h, feat_w)

        # Complete holes using spatial-temporal transformers
        for f in range(0, video_length, self.neighbor_stride):
            neighbor_ids = list(range(max(0, f - self.neighbor_stride),
                                    min(video_length, f + self.neighbor_stride + 1)))
            ref_ids = self._get_ref_index(neighbor_ids, video_length)

            pred_feat = self.inpainter.infer(feats[0, neighbor_ids + ref_ids, :, :, :],
                                           _masks[0, neighbor_ids + ref_ids, :, :, :])
            pred_img = self.inpainter.decoder(pred_feat[:len(neighbor_ids), :, :, :])
            pred_img = ((torch.tanh(pred_img) + 1) / 2).permute(0, 2, 3, 1) * 255

            for i, idx in enumerate(neighbor_ids):
                b_mask = (_masks.squeeze()[idx].unsqueeze(-1) != 0).int()
                frame = torch.from_numpy(np.array(frames[idx].resize((self.w, self.h)))).to(self.device)
                img = pred_img[i] * b_mask + frame * (1 - b_mask)
                img = img.cpu().numpy()
                comp_frames[idx] = img if comp_frames[idx] is None else comp_frames[idx] * 0.5 + img * 0.5

        # Post-process frames
        ori_w, ori_h = frames[0].size
        for idx in range(video_length):
            frame = np.array(frames[idx])
            b_mask = np.uint8(np.array(masks[idx])[..., np.newaxis] != 0)
            comp_frame = np.uint8(comp_frames[idx])
            comp_frame = Image.fromarray(comp_frame).resize((ori_w, ori_h))
            comp_frame = np.array(comp_frame)
            comp_frame = comp_frame * b_mask + frame * (1 - b_mask)
            comp_frames[idx] = Image.fromarray(np.uint8(comp_frame))

        return comp_frames

class InpaintingStage:
    def __init__(self, vi_ckpt):
        self.inpainter = VideoInpainter(vi_ckpt)

    def process(self, input_video, input_mask, output_path):
        # Load video and mask
        video = imageio.get_reader(input_video)
        fps = video.get_meta_data()['fps']
        frames = [Image.fromarray(frame) for frame in video]
        
        mask = np.load(input_mask)
        masks = [Image.fromarray(np.uint8(m * 255)) for m in mask]

        if mask.shape[:2] != (len(frames), np.array(frames[0]).shape[0]):
            raise ValueError("Mask dimensions must match video dimensions")

        # Process video
        with torch.no_grad():
            inpainted_frames = self.inpainter.inpaint_video(frames, masks)

        # Save result
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        imageio.mimsave(str(output_path), inpainted_frames, fps=fps)
        return output_path

# import torch
# import numpy as np
# import cv2
# import argparse
# import sys
# import imageio
# import imageio.v2 as iio
# import importlib
# import os
# from pathlib import Path
# from typing import List
# from PIL import Image
# from torchvision import transforms

# sys.path.insert(0, str(Path(__file__).resolve().parent / "sttn"))
# from core.utils import Stack, ToTorchFormatTensor

# # Consolidated transforms
# _to_tensors = transforms.Compose([
#     Stack(),
#     ToTorchFormatTensor()
# ])

# class VideoInpainter:
#     def __init__(self, args):
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         self.inpainter = self._build_sttn_model(args.vi_ckpt)
#         self.w, self.h = 432, 240
#         self.neighbor_stride = 5
#         self.ref_length = 10

#     def _build_sttn_model(self, ckpt_p, model_type="sttn"):
#         net = importlib.import_module(f'model.{model_type}')
#         model = net.InpaintGenerator().to(self.device)
#         data = torch.load(ckpt_p, map_location=self.device)
#         model.load_state_dict(data['netG'])
#         model.eval()
#         return model

#     def _get_ref_index(self, neighbor_ids, length):
#         ref_index = []
#         for i in range(0, length, self.ref_length):
#             if not i in neighbor_ids:
#                 ref_index.append(i)
#         return ref_index

#     @torch.no_grad()
#     def inpaint_video(self, frames: List[Image.Image], masks: List[Image.Image]) -> List[Image.Image]:
#         video_length = len(frames)
        
#         # Prepare features and masks
#         feats = [frame.resize((self.w, self.h)) for frame in frames]
#         feats = _to_tensors(feats).unsqueeze(0) * 2 - 1
#         _masks = [mask.resize((self.w, self.h), Image.NEAREST) for mask in masks]
#         _masks = _to_tensors(_masks).unsqueeze(0)

#         feats, _masks = feats.to(self.device), _masks.to(self.device)
#         comp_frames = [None] * video_length

#         # Process features
#         feats = (feats * (1 - _masks).float()).view(video_length, 3, self.h, self.w)
#         feats = self.inpainter.encoder(feats)
#         _, c, feat_h, feat_w = feats.size()
#         feats = feats.view(1, video_length, c, feat_h, feat_w)

#         # Complete holes using spatial-temporal transformers
#         for f in range(0, video_length, self.neighbor_stride):
#             neighbor_ids = list(range(max(0, f - self.neighbor_stride),
#                                     min(video_length, f + self.neighbor_stride + 1)))
#             ref_ids = self._get_ref_index(neighbor_ids, video_length)

#             pred_feat = self.inpainter.infer(feats[0, neighbor_ids + ref_ids, :, :, :],
#                                            _masks[0, neighbor_ids + ref_ids, :, :, :])
#             pred_img = self.inpainter.decoder(pred_feat[:len(neighbor_ids), :, :, :])
#             pred_img = ((torch.tanh(pred_img) + 1) / 2).permute(0, 2, 3, 1) * 255

#             for i, idx in enumerate(neighbor_ids):
#                 b_mask = (_masks.squeeze()[idx].unsqueeze(-1) != 0).int()
#                 frame = torch.from_numpy(np.array(frames[idx].resize((self.w, self.h)))).to(self.device)
#                 img = pred_img[i] * b_mask + frame * (1 - b_mask)
#                 img = img.cpu().numpy()
#                 comp_frames[idx] = img if comp_frames[idx] is None else comp_frames[idx] * 0.5 + img * 0.5

#         # Post-process frames
#         ori_w, ori_h = frames[0].size
#         for idx in range(video_length):
#             frame = np.array(frames[idx])
#             b_mask = np.uint8(np.array(masks[idx])[..., np.newaxis] != 0)
#             comp_frame = np.uint8(comp_frames[idx])
#             comp_frame = Image.fromarray(comp_frame).resize((ori_w, ori_h))
#             comp_frame = np.array(comp_frame)
#             comp_frame = comp_frame * b_mask + frame * (1 - b_mask)
#             comp_frames[idx] = Image.fromarray(np.uint8(comp_frame))

#         return comp_frames

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--input_video", type=str, required=True)
#     parser.add_argument("--input_mask", type=str, required=True)
#     parser.add_argument("--output_dir", type=str, required=True)
#     parser.add_argument("--vi_ckpt", type=str, required=True)
#     parser.add_argument("--fps", type=int, default=25)
#     args = parser.parse_args()

#     # Load video and mask
#     video = imageio.get_reader(args.input_video)
#     fps = video.get_meta_data()['fps']
#     frames = [Image.fromarray(frame) for frame in video]
    
#     mask = np.load(args.input_mask)
#     masks = [Image.fromarray(np.uint8(m * 255)) for m in mask]

#     if mask.shape[:2] != (len(frames), np.array(frames[0]).shape[0]):
#         raise ValueError("Mask dimensions must match video dimensions")

#     # Process video
#     inpainter = VideoInpainter(args)
#     with torch.no_grad():
#         inpainted_frames = inpainter.inpaint_video(frames, masks)

#     # Save result
#     output_dir = Path(args.output_dir)
#     output_dir.mkdir(parents=True, exist_ok=True)
#     output_path = output_dir / "inpainted_video.mp4"
    
#     imageio.mimsave(str(output_path), inpainted_frames, fps=fps)
#     print(f"Inpainted video saved to: {output_path}")

# if __name__ == "__main__":
#     main()
