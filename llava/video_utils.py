#modified from  https://github.com/LaVi-Lab/Video-3D-LLM/tree/main

import os
import json
import torch
import pickle
import cv2
import numpy as np
from PIL import Image
from transformers.image_utils import to_numpy_array
import json
from tqdm import tqdm
import random
import copy
from pathlib import Path
def convert_from_uvd(u, v, d, intr, pose):
    fx = intr[0, 0]
    fy = intr[1, 1]
    cx = intr[0, 2]
    cy = intr[1, 2]
    depth_scale = 1000
    
    z = d / depth_scale
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    world = (pose @ np.array([x, y, z, 1]))
    return world[:3] / world[3]
    
def load_matrix_from_txt(path, shape=(4, 4)):
    with open(path) as f:
        txt = f.readlines()
    txt = ''.join(txt).replace('\n', ' ')
    matrix = [float(v) for v in txt.split()]
    return np.array(matrix).reshape(shape)


def unproject(intrinsics, poses, depths):
    """
        intrinsics: (V, 4, 4)
        poses: (V, 4, 4)
        depths: (V, H, W)
    """
    V, H, W = depths.shape
    y = torch.arange(0, H).to(depths.device)
    x = torch.arange(0, W).to(depths.device)
    y, x = torch.meshgrid(y, x)

    x = x.unsqueeze(0).repeat(V, 1, 1).view(V, H*W)     # (V, H*W)
    y = y.unsqueeze(0).repeat(V, 1, 1).view(V, H*W)     # (V, H*W)

    fx = intrinsics[:, 0, 0].unsqueeze(-1).repeat(1, H*W)
    fy = intrinsics[:, 1, 1].unsqueeze(-1).repeat(1, H*W)
    cx = intrinsics[:, 0, 2].unsqueeze(-1).repeat(1, H*W)
    cy = intrinsics[:, 1, 2].unsqueeze(-1).repeat(1, H*W)

    z = depths.view(V, H*W) / 1000       # (V, H*W)
    x = (x - cx) * z / fx
    y = (y - cy) * z / fy
    cam_coords = torch.stack([
        x, y, z, torch.ones_like(x)
    ], -1)      # (V, H*W, 4)

    world_coords = (poses @ cam_coords.permute(0, 2, 1)).permute(0, 2, 1)       # (V, H*W, 4)
    world_coords = world_coords[..., :3] / world_coords[..., 3].unsqueeze(-1)   # (V, H*W, 3)
    world_coords = world_coords.view(V, H, W, 3)

    return world_coords


class VideoProcessor:
    def __init__(
        self, 
        video_folder="/data/", 
        annotation_dir="/data/embodiedscan/",
        voxel_size=None,
        min_xyz_range=None,
        max_xyz_range=None,
        frame_sampling_strategy='uniform',
        val_box_type='pred',
        dataset_list = 'scannet',
        bbox_type = None,
    ):
        self.video_folder = video_folder
        self.voxel_size = voxel_size
        self.min_xyz_range = torch.tensor(min_xyz_range) if min_xyz_range is not None else None
        self.max_xyz_range = torch.tensor(max_xyz_range) if max_xyz_range is not None else None
        self.frame_sampling_strategy = frame_sampling_strategy
        self.scene = {}
        self.bbox_type = bbox_type
        print('============frame sampling strategy: {}============='.format(self.frame_sampling_strategy))

        for split in ["train", "val", "test"]:
            if 'scannet' in dataset_list:
                with open(os.path.join(annotation_dir, f"embodiedscan_infos_{split}.pkl"), "rb") as f:
                    data = pickle.load(f)["data_list"]
                    for item in data:
                        if item["sample_idx"].startswith("scannet"):
                            self.scene[item["sample_idx"]] = item

        self.scan2obj = {}

        if 'scannet' in dataset_list:
            for split in ['train', 'val']:
                if self.bbox_type is not None:
                    box_type = self.bbox_type
                else:
                    box_type = "gt" if split == "train" else val_box_type
                filename = os.path.join("/data", "metadata", f"scannet_{split}_{box_type}_box.json")
                with open(filename) as f:
                    data = json.load(f)
                    self.scan2obj.update(data)

        if 'mc' in self.frame_sampling_strategy:
            sampling_file = "/data/metadata/scannet_select_frames.json"
            self.mc_sampling_files = {}
            with open(sampling_file) as f:
                data = json.load(f)
                for dd in data:
                    self.mc_sampling_files[dd['video_id']] = dd

            with open('/data/metadata/pcd_discrete_0.1.pkl', 'rb') as f:
                pc_data = pickle.load(f)
            self.pc_min = {}
            self.pc_max = {}
            for scene_id in pc_data:
                pc_points = pc_data[scene_id]
                min_xyz = [1000, 1000, 1000]
                max_xyz = [-1000, -1000, -1000]
                for data in pc_points:
                    min_xyz = [min(v1, v2) for v1, v2 in zip(min_xyz, data)]
                    max_xyz = [max(v1, v2) for v1, v2 in zip(max_xyz, data)]
                self.pc_min[scene_id] = torch.Tensor(min_xyz) / 10
                self.pc_max[scene_id] = torch.Tensor(max_xyz) / 10


    def sample_frame_files_mc(self, video_id: str, frames_upbound: int = 32, do_shift=False):
        mc_files = self.mc_sampling_files[video_id]
        frame_files = mc_files['frame_files'][:frames_upbound]
        voxel_nums = mc_files['voxel_nums'][:frames_upbound]

        ratio = 1.0
        if 'ratio95' in self.frame_sampling_strategy:
            ratio = 0.95
        elif 'ratio90' in self.frame_sampling_strategy:
            ratio = 0.9

        if ratio != 1.0:
            num_all_voxels = mc_files['num_all_voxels']
            out = []
            cc = 0
            for frame_file, voxel_num in zip(frame_files, voxel_nums):
                out.append(frame_file)
                cc += voxel_num
                if cc >= num_all_voxels * ratio:
                    break
            frame_files = out

        frame_files.sort(key=lambda file: int(file.split('/')[-1].split('.')[0]))

        return frame_files  


    def sample_frame_files(
        self,
        video_id: str,
        force_sample: bool = False,
        frames_upbound: int = 0,
    ):
        meta_info = self.scene[video_id]

        if meta_info["images"][0]["img_path"].startswith('scannet'):
            frame_files = [os.path.join(self.video_folder, img["img_path"].replace("scannet/","")) for img in meta_info["images"]]
            frame_depth_files = [os.path.join(self.video_folder, img["depth_path"]) for img in meta_info["images"]]
        
        if force_sample:
            num_frames_to_sample = frames_upbound
        else:
            num_frames_to_sample = 10

        avg_fps = 3
        total_frames = len(frame_files)
        sampled_indices = np.linspace(0, total_frames - 1, num_frames_to_sample, dtype=int)

        return [frame_files[i] for i in sampled_indices], [frame_depth_files[i] for i in sampled_indices]

    def calculate_world_coords(
        self,
        video_id: str, 
        frame_files,
        do_normalize=False,
    ):
        meta_info = self.scene[video_id]
        scene_id = video_id.split('/')[-1]

        axis_align_matrix = torch.from_numpy(np.array(meta_info['axis_align_matrix']))
        depth_intrinsic = torch.from_numpy(np.array(meta_info["depth_cam2img"]))

        depths = []
        poses = []
 
        # Read and store the sampled frames
        for frame_path in frame_files:
            depth_path = frame_path.replace(".jpg", ".png")
            with Image.open(depth_path) as depth_img:
                depth = np.array(depth_img).astype(np.int32)
                depths.append(torch.from_numpy(depth))

            # pose
            pose_file = frame_path.replace("jpg", "txt")
            pose = np.loadtxt(pose_file)
            poses.append(torch.from_numpy(pose))


        depths = torch.stack(depths)   # (V, H, W)
        poses = torch.stack([axis_align_matrix @ pose for pose in poses])     # (V, 4, 4)
        depth_intrinsic = depth_intrinsic.unsqueeze(0).repeat(len(frame_files), 1, 1)
        
        world_coords = unproject(depth_intrinsic.float(), poses.float(), depths.float())    # (V, H, W, 3)

        if do_normalize:
            world_coords = torch.maximum(world_coords, self.pc_min[scene_id].to(world_coords.device))
            world_coords = torch.minimum(world_coords, self.pc_max[scene_id].to(world_coords.device))
        
        return {
            "world_coords": world_coords,
        }
    def resize_K(self,K, scale_x, scale_y):
        K_resized = K  #.clone()
        K_resized[0, 0] *= scale_x  # fx
        K_resized[0, 2] *= scale_x  # cx
        K_resized[1, 1] *= scale_y  # fy
        K_resized[1, 2] *= scale_y  # cy
        return K_resized

    def crop_K(self,K, left, top):
        K_cropped = K  #.clone()
        K_cropped[0, 2] -= left  # cx
        K_cropped[1, 2] -= top   # cy
        return K_cropped
    

    def generate_patch_uv_tensor(self,H, W, patch_size, B=1, device='cpu'):
        n_h = H // patch_size
        n_w = W // patch_size

        u_grid, v_grid = torch.meshgrid(torch.arange(n_w), torch.arange(n_h), indexing='xy')  # (n_w, n_h)
        uv = torch.stack([u_grid, v_grid], dim=-1).reshape(-1, 2).to(device)  # (N, 2)

        uv_batch = uv.unsqueeze(0).repeat(B, 1, 1)  # (B, N, 2)
        return uv_batch

    def compute_ray_from_patch_batch(self,patch_uv_tensor, depth_maps, K, poses, patch_size=16):
        B, N, _ = patch_uv_tensor.shape
        H, W = depth_maps.shape[1:]
        u_center = patch_uv_tensor[:, :, 0] * patch_size + patch_size // 2  # (B, N)
        v_center = patch_uv_tensor[:, :, 1] * patch_size + patch_size // 2  # (B, N)
        u_center = u_center.clamp(0, W - 1)
        v_center = v_center.clamp(0, H - 1)
        d = depth_maps[torch.arange(B)[:, None], v_center, u_center]  # (B, N)
        ones = torch.ones_like(u_center)  # (B, N, 3)
        pixel_coords = torch.stack([u_center, v_center, ones], dim=-1).float()  # (B, N, 3)
        K_inv = torch.inverse(K).to(dtype = pixel_coords.dtype )  # (B, 3, 3)
        poses = poses.to(dtype = pixel_coords.dtype )
        K_inv = K_inv.unsqueeze(0).repeat(B, 1, 1)  
        cam_coords = torch.bmm(K_inv[:,:3,:3], pixel_coords.transpose(1, 2)).transpose(1, 2) * d.unsqueeze(-1)
        cam_coords_h = torch.cat([cam_coords, torch.ones(B, N, 1, device=cam_coords.device)], dim=-1)  # (B, N, 4)
        p_world_h = torch.bmm(cam_coords_h, poses.transpose(1, 2))  # (B, N, 4)
        p_world = p_world_h[:, :, :3] / (p_world_h[:, :, 3:4] + 1e-6)  # (B, N, 3)
        o_world = poses[:, :3, 3].unsqueeze(1).expand(-1, N, -1)  # (B, N, 3)
        d_vec = p_world - o_world  # (B, N, 3)
        d_unit = d_vec / (d_vec.norm(dim=-1, keepdim=True) + 1e-6)
        return o_world, d_unit, p_world

    def preprocess(
        self,
        video_id: str, 
        image_processor,
        force_sample: bool = False,
        frames_upbound: int = 0,
        strategy: str = "center_crop",
        patch_size: int = 14,
    ):

        if 'mc' in self.frame_sampling_strategy:
            frame_files = self.sample_frame_files_mc(
                video_id,
                frames_upbound=frames_upbound,
                do_shift=('shift' in self.frame_sampling_strategy),
            )
        else:
            frame_files ,frame_depth_files = self.sample_frame_files(
                video_id,
                force_sample=force_sample,
                frames_upbound=frames_upbound,
            )

        video_dict = self.calculate_world_coords(
            video_id,
            frame_files,
            do_normalize=('norm' in self.frame_sampling_strategy),
        )
        world_coords = video_dict["world_coords"]
        V, H, W, _ = world_coords.shape
        world_coords_flat = world_coords.reshape(-1, 3)
        x_min, x_max = world_coords_flat[:, 0].min().item(), world_coords_flat[:, 0].max().item()
        y_min, y_max = world_coords_flat[:, 1].min().item(), world_coords_flat[:, 1].max().item()
        z_min, z_max = world_coords_flat[:, 2].min().item(), world_coords_flat[:, 2].max().item()
        boundry = torch.tensor([x_min, x_max, y_min, y_max, z_min, z_max])

        images = []
        for frame_file in frame_files:
            with Image.open(frame_file) as img:
                frame = img.convert("RGB")
                images.append(frame)

        meta_info = self.scene[video_id]
        axis_align_matrix = torch.from_numpy(np.array(meta_info['axis_align_matrix']))
        depth_intrinsics = torch.from_numpy(np.array(meta_info["depth_cam2img"]))
        depth_images = []
        camera_poses = []
        for frame_path in frame_files:
            
            # depth image
            depth_path = frame_path.replace(".jpg", ".png")

            with Image.open(depth_path) as depth:
                depth.load()
                depth_images.append(depth)
            pose_file = frame_path.replace("jpg", "txt")
            pose = np.loadtxt(pose_file)
            camera_poses.append(torch.from_numpy(pose))
        
        camera_poses = torch.stack([axis_align_matrix @ pose for pose in camera_poses])     # (V, 4, 4)

        crop_size = image_processor.crop_size["width"]
        if strategy == "resize":
            images = [frame.resize((crop_size, crop_size)) for frame in images]
            resized_coords = [cv2.resize(coords.numpy(), (384, 384), interpolation=cv2.INTER_NEAREST) for coords in world_coords] 
        elif strategy == "center_crop":
            new_height = crop_size
            new_width = int(W * (crop_size / H))  # 512
            images = [frame.resize((new_width, new_height)) for frame in images]
            resized_coords = [cv2.resize(coords.numpy(), (new_width, new_height), interpolation=cv2.INTER_NEAREST) for coords in world_coords]
            # Calculate the position and perform the center crop
            left = (new_width - crop_size) // 2
            right = left + crop_size
            top = (new_height - crop_size) // 2
            bottom = top + crop_size
            images = [frame.crop((left, top, right, bottom)) for frame in images]
            resized_coords = [coords[top:bottom, left:right, :] for coords in resized_coords]

            depth_images = [frame.resize((new_width, new_height), resample=Image.NEAREST) for frame in depth_images]
            depth_images = [torch.from_numpy(np.array(frame.crop((left, top, right-6, bottom-6))).astype(np.int32)) for frame in depth_images]

            scale_x = 0.8 #new_width/ori_width
            scale_y = 0.8 #new_height/ori_height
            depth_intrinsics = self.resize_K(depth_intrinsics, scale_x, scale_y)
            depth_intrinsics =  torch.tensor(self.crop_K(depth_intrinsics, left, top)) 

        depth_images = torch.stack(depth_images)   # (V, H, W)
        B, H, W = depth_images.shape
        patch_size = 27
        patch_uv_tensor = self.generate_patch_uv_tensor(H, W, patch_size, B, device='cpu')
        o_world, d_unit, p_world = self.compute_ray_from_patch_batch(patch_uv_tensor, depth_images, depth_intrinsics, camera_poses, patch_size)
        dir_patch = d_unit   
        return {
            "images": images,
            "world_coords": torch.from_numpy(np.stack(resized_coords)),
            "video_size": len(images),
            "boundry": boundry,
            "objects": torch.tensor(self.scan2obj[video_id]),
            "depths" :depth_images,
            "camera_pose" :camera_poses,
            "camera_intrinsics" :depth_intrinsics,
            "dir_patch":dir_patch,
        }


    def process_3d_video(
        self,
        video_id: str, 
        image_processor,
        force_sample: bool = False,
        frames_upbound: int = 0,
        strategy: str = "center_crop",
        patch_size: int = 14,
    ):
        video_dict = self.preprocess(
            video_id,
            image_processor,
            force_sample,
            frames_upbound,
            strategy,
            patch_size,
        )
        video_dict["images"] = image_processor.preprocess(video_dict["images"], return_tensors="pt")["pixel_values"]
        return video_dict

    
    def discrete_point(self, xyz):
        xyz = torch.tensor(xyz)
        if self.min_xyz_range is not None:
            xyz = torch.maximum(xyz, self.min_xyz_range.to(xyz.device))
        if self.max_xyz_range is not None:
            xyz = torch.minimum(xyz, self.max_xyz_range.to(xyz.device))
        if self.min_xyz_range is not None:
            xyz = (xyz - self.min_xyz_range.to(xyz.device)) 
            
        xyz = xyz / self.voxel_size
        return xyz.round().int().tolist()
    

def merge_video_dict(video_dict_list):
    new_video_dict = {}
    new_video_dict['box_input'] = []
    for k in video_dict_list[0]:
        if k in ["world_coords", 'images', 'objects','dir_patch']:
            new_video_dict[k] = torch.stack([video_dict[k] for video_dict in video_dict_list])
        elif k in ['box_input']:
            for video_dict in video_dict_list:
                if video_dict[k] is not None:
                    new_video_dict['box_input'].append(video_dict[k])

    new_video_dict['box_input'] = torch.Tensor(new_video_dict['box_input'])
    return new_video_dict
