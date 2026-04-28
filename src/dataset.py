import torch
from torch.utils.data import Dataset
import numpy as np
import os
import pandas as pd
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import open3d as o3d

class MultimodalDataset(Dataset):
    def __init__(self, mode='train', data_root='./Data/Multi_Modal', split_root='./Data/splits', scenario_name="scenario32"):
        self.data_dir = data_root
        
        # 读取拆分好的 CSV
        csv_path = os.path.join(split_root, f"{scenario_name}_{mode}.csv")
        self.df = pd.read_csv(csv_path)
        
        self.img_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self._init_gps_normalization()

    def _init_gps_normalization(self):
        all_dx, all_dy = [], []
        for idx in range(len(self.df)):
            bs_lat, bs_lon = self._read_gps_raw(self.df.iloc[idx]['unit1_loc'])
            ue_lat, ue_lon = self._read_gps_raw(self.df.iloc[idx]['unit2_loc_1'])
            if bs_lat != 0.0:
                dx = (ue_lon - bs_lon) * 111320 * np.cos(np.radians(bs_lat))
                dy = (ue_lat - bs_lat) * 111320
                all_dx.append(dx)
                all_dy.append(dy)
        if all_dx:
            self.min_dx, self.max_dx = np.min(all_dx), np.max(all_dx)
            self.min_dy, self.max_dy = np.min(all_dy), np.max(all_dy)
        else:
            self.min_dx, self.max_dx, self.min_dy, self.max_dy = 0, 1, 0, 1

    def _read_gps_raw(self, rel_path):
        try:
            with open(os.path.join(self.data_dir, rel_path), 'r') as f:
                lines = f.readlines()
                return float(lines[0].strip()), float(lines[1].strip())
        except: return 0.0, 0.0

    def _calc_gps_bemamba_eq1(self, bs_lat, bs_lon, ue_lat, ue_lon):
        dx = (ue_lon - bs_lon) * 111320 * np.cos(np.radians(bs_lat))
        dy = (ue_lat - bs_lat) * 111320
        dx_norm = (dx - self.min_dx) / (self.max_dx - self.min_dx + 1e-8)
        dy_norm = (dy - self.min_dy) / (self.max_dy - self.min_dy + 1e-8)
        dist = np.sqrt(dx_norm**2 + dy_norm**2)
        angle = np.arctan2(dy_norm, dx_norm)
        return [dist, angle]

    def _read_power(self, rel_path):
        try:
            with open(os.path.join(self.data_dir, rel_path), 'r') as f:
                return [float(x) for x in f.read().split()]
        except: return [0.0] * 64

    def _ply_to_base_bev(self, rel_path, grid_size=256):
        bev = np.zeros((grid_size, grid_size), dtype=np.float32)
        try:
            full_path = os.path.join(self.data_dir, rel_path)
            if not os.path.exists(full_path): return bev
            pcd = o3d.io.read_point_cloud(full_path)
            points = np.asarray(pcd.points)
            if len(points) > 0:
                x, y = points[:, 0], points[:, 1]
                x_idx = np.clip(((x - x.min()) / (x.max() - x.min() + 1e-5)) * (grid_size-1), 0, grid_size-1).astype(int)
                y_idx = np.clip(((y - y.min()) / (y.max() - y.min() + 1e-5)) * (grid_size-1), 0, grid_size-1).astype(int)
                bev[x_idx, y_idx] = 1.0
        except: pass
        return bev

    def _generate_virtual_points(self, current_bev, prev_bev):
        diff = current_bev - prev_bev
        moving_points = np.where(diff > 0)
        if len(moving_points[0]) > 0:
            offset_x = np.random.randint(-1, 2, size=len(moving_points[0]))
            offset_y = np.random.randint(-1, 2, size=len(moving_points[1]))
            virtual_x = np.clip(moving_points[0] + offset_x, 0, 255)
            virtual_y = np.clip(moving_points[1] + offset_y, 0, 255)
            current_bev[virtual_x, virtual_y] = 1.0
        return current_bev

    def _resize_tensor(self, t, size=(256, 256)):
        return F.interpolate(t.unsqueeze(0), size=size, mode='bilinear', align_corners=False).squeeze(0)

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        imgs, radars, lidars = [], [], []
        prev_bev = None
        
        for t in range(1, 6):
            # 1. 图像
            img_path = os.path.join(self.data_dir, row[f'unit1_rgb_{t}'])
            imgs.append(self.img_transform(Image.open(img_path).convert('RGB')))
            
            # 2. 雷达
            ang_p = os.path.join(self.data_dir, row[f'unit1_radar_{t}'].replace('/radar_data/', '/radar_data_ang/'))
            vel_p = os.path.join(self.data_dir, row[f'unit1_radar_{t}'].replace('/radar_data/', '/radar_data_vel/'))
            try:
                ang_arr, vel_arr = np.nan_to_num(np.load(ang_p)), np.nan_to_num(np.load(vel_p))
                radar_tensor = torch.tensor(np.stack([ang_arr, vel_arr], axis=0), dtype=torch.float32)
            except: radar_tensor = torch.zeros((2, 256, 256))
            radars.append(self._resize_tensor(radar_tensor))
            
            # 3. LiDAR (保留虚拟点生成)
            ply_p = row[f'unit1_lidar_{t}']
            current_bev = self._ply_to_base_bev(ply_p)
            if prev_bev is not None:
                current_bev = self._generate_virtual_points(current_bev, prev_bev)
            prev_bev = current_bev.copy()
            lidars.append(self._resize_tensor(torch.tensor(current_bev).unsqueeze(0)))
            
        # 4. GPS
        bs_lat, bs_lon = self._read_gps_raw(row['unit1_loc'])
        u1_lat, u1_lon = self._read_gps_raw(row['unit2_loc_1'])
        u2_lat, u2_lon = self._read_gps_raw(row['unit2_loc_2'])
        gps_start = self._calc_gps_bemamba_eq1(bs_lat, bs_lon, u1_lat, u1_lon)
        gps_end   = self._calc_gps_bemamba_eq1(bs_lat, bs_lon, u2_lat, u2_lon)
        gps = torch.tensor([gps_start, gps_end], dtype=torch.float32)
        
        # 5. Label & Power Vector
        target = torch.tensor(int(row['unit1_beam']) - 1, dtype=torch.long)
        power_vec = torch.tensor(self._read_power(row['unit1_pwr_60ghz']), dtype=torch.float32)
        
        return torch.stack(imgs), torch.stack(radars), torch.stack(lidars), gps, target, power_vec