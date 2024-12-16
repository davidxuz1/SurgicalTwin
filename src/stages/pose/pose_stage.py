import os
import sys
import torch
import numpy as np
from pathlib import Path
import cv2
import shutil
import copy
import matplotlib.pyplot as plt

# Añadir MonST3R al path, si es necesario
MONST3R_PATH = str(Path(__file__).resolve().parent.parent.parent.parent / "third_party" / "monst3r")
if MONST3R_PATH not in sys.path:
    sys.path.insert(0, MONST3R_PATH)

from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images, enlarge_seg_masks
from dust3r.utils.device import to_numpy
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.utils.viz_demo import convert_scene_output_to_glb

torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"
torch.cuda.set_per_process_memory_fraction(0.85)
torch.backends.cuda.matmul.allow_tf32 = True


class PoseStage:
    def __init__(self,
                 model_path: str,
                 image_size: int = 512,
                 device: str = 'cuda',
                 output_dir: str = './demo_tmp',
                 use_gt_mask: bool = False,
                 fps: int = 0,
                 num_frames: int = 200):
        """
        Clase para la etapa de pose 3D utilizando MonST3R.
        """
        if not os.path.exists(model_path):
            sys.exit(f"Error: Pose model not found at {model_path}")

        self.model_path = model_path
        self.image_size = image_size
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.output_dir = output_dir
        self.use_gt_mask = use_gt_mask
        self.fps = fps
        self.num_frames = num_frames

        # Cargar el modelo
        self.model = AsymmetricCroCo3DStereo.from_pretrained(self.model_path).to(self.device)
        self.model.eval()

        os.makedirs(self.output_dir, exist_ok=True)

    def process(self,
                input_video: str,
                output_path: str,
                output_dir: str,
                schedule='linear',
                niter=300,
                min_conf_thr=1.1,
                as_pointcloud=True,
                mask_sky=False,
                clean_depth=True,
                transparent_cams=False,
                cam_size=0.05,
                show_cam=True,
                scenegraph_type='swinstride',
                winsize=5,
                refid=0,
                new_model_weights=None,
                temporal_smoothing_weight=0.01,
                translation_weight='1.0',
                shared_focal=True,
                flow_loss_weight=0.00,
                flow_loss_start_iter=0.1,
                flow_loss_threshold=25,
                use_gt_mask=None):
        """
        Procesa un video de entrada para obtener la reconstrucción 3D y generar poses_bounds.npy.
        """
        target_size = self.image_size
        redimensioned_frames = []

        # Convertir video a frames redimensionados a 512x512
        cap = cv2.VideoCapture(input_video)
        frame_count = 0
        frames_dir = os.path.join(output_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)

        while True:
            ret, img = cap.read()
            if not ret:
                break
            h, w = img.shape[:2]
            ratio = min(target_size / w, target_size / h)
            new_w = int(w * ratio)
            new_h = int(h * ratio)
            resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            square_img = np.zeros((target_size, target_size, 3), dtype=img.dtype)
            x_offset = (target_size - new_w) // 2
            y_offset = (target_size - new_h) // 2
            square_img[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_img
            frame_filename = os.path.join(frames_dir, f"frame_{frame_count:06d}.png")
            cv2.imwrite(frame_filename, square_img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
            redimensioned_frames.append(frame_filename)
            frame_count += 1

        cap.release()

        input_files = redimensioned_frames
        use_gt_mask = use_gt_mask if use_gt_mask is not None else self.use_gt_mask
        new_model_weights = new_model_weights or self.model_path

        # Guardar resultados en output_dir directamente
        save_folder = output_dir
        os.makedirs(save_folder, exist_ok=True)

        # Cargar imágenes
        imgs = load_images(input_files, size=self.image_size, verbose=True,
                           dynamic_mask_root=None, fps=self.fps, num_frames=self.num_frames)

        # Si solo hay 1 frame, duplicarlo
        if len(imgs) == 1:
            imgs.append(copy.deepcopy(imgs[0]))
            imgs[1]['idx'] = 1

        # Ajustar scenegraph_type
        if scenegraph_type in ["swin", "swinstride", "swin2stride"]:
            scenegraph_type = f"{scenegraph_type}-{winsize}-noncyclic"
        elif scenegraph_type == "oneref":
            scenegraph_type = f"{scenegraph_type}-{refid}"

        # Crear pares de imágenes
        pairs = make_pairs(imgs, scene_graph=scenegraph_type, prefilter=None, symmetrize=True)

        output = inference(pairs, self.model, self.device, batch_size=1, verbose=True)
        torch.cuda.empty_cache()

        # Alineamiento global
        if len(imgs) > 2:
            mode = GlobalAlignerMode.PointCloudOptimizer
            scene = global_aligner(output, device=self.device, mode=mode, verbose=True,
                                   shared_focal=shared_focal,
                                   temporal_smoothing_weight=temporal_smoothing_weight,
                                   translation_weight=float(translation_weight),
                                   flow_loss_weight=flow_loss_weight,
                                   flow_loss_start_epoch=flow_loss_start_iter,
                                   flow_loss_thre=flow_loss_threshold,
                                   use_self_mask=not use_gt_mask,
                                   num_total_iter=niter,
                                   empty_cache=len(imgs) > 72)
        else:
            mode = GlobalAlignerMode.PairViewer
            scene = global_aligner(output, device=self.device, mode=mode, verbose=True)

        if mode == GlobalAlignerMode.PointCloudOptimizer:
            scene.compute_global_alignment(init='mst', niter=niter, schedule=schedule, lr=0.01)

        # Guardar modelo 3D
        self.get_3D_model_from_scene(save_folder, scene, min_conf_thr, as_pointcloud,
                                     mask_sky, clean_depth, transparent_cams, cam_size, show_cam)

        # Guardar resultados
        scene.save_tum_poses(f'{save_folder}/pred_traj.txt')
        scene.save_intrinsics(f'{save_folder}/pred_intrinsics.txt')
        scene.save_depth_maps(save_folder)
        scene.save_dynamic_masks(save_folder)
        scene.save_conf_maps(save_folder)
        scene.save_init_conf_maps(save_folder)
        scene.save_rgb_imgs(save_folder)
        enlarge_seg_masks(save_folder, kernel_size=5 if use_gt_mask else 3)

        # Guardar poses_bounds.npy
        poses_bounds_path = self.save_poses_bounds(scene, save_folder)

        # Mover a output_path si es necesario
        if os.path.abspath(poses_bounds_path) != os.path.abspath(output_path):
            shutil.move(poses_bounds_path, output_path)
            poses_bounds_path = output_path

        # Borrar frames temporales
        shutil.rmtree(frames_dir)

        print(f"Processing completed. Output saved in {save_folder}")
        return poses_bounds_path

    def get_3D_model_from_scene(self, outdir, scene, min_conf_thr=3, as_pointcloud=False,
                                mask_sky=False, clean_depth=False, transparent_cams=False,
                                cam_size=0.05, show_cam=True, save_name=None, thr_for_init_conf=True):
        if scene is None:
            return None
        if clean_depth:
            scene = scene.clean_pointcloud()
        if mask_sky:
            scene = scene.mask_sky()

        rgbimg = scene.imgs
        focals = scene.get_focals().cpu()
        cams2world = scene.get_im_poses().cpu()
        pts3d = to_numpy(scene.get_pts3d(raw_pts=True))
        scene.min_conf_thr = min_conf_thr
        scene.thr_for_init_conf = thr_for_init_conf
        msk = to_numpy(scene.get_masks())
        cmap = plt.get_cmap('viridis')
        cam_color = [(255*c[0], 255*c[1], 255*c[2]) for c in [cmap(i / len(rgbimg))[:3] for i in range(len(rgbimg))]]

        return convert_scene_output_to_glb(outdir, rgbimg, pts3d, msk, focals, cams2world,
                                           as_pointcloud=as_pointcloud, transparent_cams=transparent_cams,
                                           cam_size=cam_size, show_cam=show_cam, silent=True,
                                           save_name=save_name, cam_color=cam_color)

    def save_poses_bounds(self, scene, save_folder):
        # Obtener poses de cámara (camera-to-world)
        poses = scene.get_im_poses().detach().cpu().numpy()  # shape: (N, 4, 4)

        # Obtener intrínsecos
        K = scene.get_intrinsics().detach().cpu().numpy()  # shape: (N, 3, 3)

        # Obtener profundidades
        depths = scene.get_depthmaps()
        near_fars = []
        for depth in depths:
            near = depth[depth > 0].min().detach().item()
            far = depth.max().detach().item()
            near_fars.append([near, far])

        # Crear formato LLFF (Nx17)
        llff_poses = []
        for i in range(len(poses)):
            # Extraer rotación y traslación (3x4)
            R = poses[i, :3, :3]  # Rotación 3x3
            t = poses[i, :3, 3:]  # Traslación 3x1

            # Obtener altura, ancho y focal length
            height = K[i, 0, 0]  # Asumiendo que K[0,0] es la focal length
            width = K[i, 1, 1]
            focal = K[i, 0, 0]  # Asumiendo focal length igual para x e y

            # Crear matriz de pose LLFF (3x5)
            pose_mat = np.concatenate([
                np.concatenate([R, t], axis=1),  # [R|t] (3x4)
                np.array([[height, width, focal]]).T  # [h,w,f] (3x1)
            ], axis=1)

            # Aplanar y concatenar con near/far
            llff_pose = np.concatenate([
                pose_mat.reshape(-1),  # 15 elementos
                near_fars[i]           # 2 elementos
            ])
            llff_poses.append(llff_pose)

        # Apilar todas las poses
        llff_poses = np.stack(llff_poses)

        print(f"\nLLFF poses shape: {llff_poses.shape}")  # Debería ser (N, 17)

        # Guardar
        np.save(f'{save_folder}/poses_bounds.npy', llff_poses)
        print(f"\nPoses and bounds saved to {save_folder}/poses_bounds.npy")
        return f'{save_folder}/poses_bounds.npy'
    