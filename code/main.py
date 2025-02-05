import os
import random
import math
import json
import logging
from pathlib import Path
from tqdm import tqdm
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import MeanAbsoluteError, MeanSquaredError  
from sklearn.model_selection import train_test_split
import albumentations as A
import kornia as K
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
import zipfile


# Additional imports for point cloud processing and tensor batching.
# (Ensure that these functions are installed and available in your environment.)
from torch_geometric.nn import PointNetConv, fps, radius
from torch_geometric.utils import to_dense_batch
from torch_scatter import scatter_mean, scatter_std
from pytorch3d.ops import box3d_overlap


# Get the directory where the notebook is located
notebook_dir = os.getcwd()  

# Path to the zip file
zip_file_path = os.path.join(notebook_dir, "sample_dataset.zip")

# Extract the contents into the same directory as the notebook
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(notebook_dir)

# Logging Setup
log_file_path = os.path.join(os.getcwd(), 'logging.txt')
logging.basicConfig(
    filename=log_file_path,
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True
)

# Global Paths & Configuration
BASE_PATH = os.path.join(os.getcwd(), 'dataset')
LOG_DIR = os.path.join(os.getcwd(), 'logs')

class Config:
    """
    Global configuration settings.
    """
    TRAIN_SPLIT = 0.7
    VAL_SPLIT = 0.15
    TEST_SPLIT = 0.15
    RGB_SIZE = (128, 128)
    LIDAR_POINTS = 1024
    LIDAR_ROT_RANGE = (-15, 15)  # degrees
    LIDAR_SCALE_RANGE = (0.9, 1.1)
    BATCH_SIZE = 4
    NUM_WORKERS = 2
    SEED = 42
    NUM_EPOCHS = 50
    DEBUG_MODE = False  # Set to True to enable single-sample visualization


# Logging & Visualization Setup
class DebugVisualizer:
    """
    Provides simple logging and visualization utilities.
    All outputs are saved to LOG_DIR.
    """
    def __init__(self):
        self.log_dir = LOG_DIR
        self.image_dir = os.path.join(self.log_dir, 'images')
        self.pointcloud_dir = os.path.join(self.log_dir, 'pointclouds')
        self.distribution_dir = os.path.join(self.log_dir, 'distributions')

        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.pointcloud_dir, exist_ok=True)

    def denormalize(self, tensor):
        """Convert normalized image tensor to displayable format. IMAGENET"""
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        tensor = tensor * std + mean
        return tensor

    def log_image(self, image, bbox, title):
        """
        Save an RGB image to file with an optional bounding box.
        """
        if isinstance(image, torch.Tensor):
            image = image.cpu().detach()
            if image.dim() == 4 and image.size(0) == 1:
                image = image.squeeze(0)
            # Denormalize the image to visualize
            image = self.denormalize(image)
            image = image.numpy().transpose(1, 2, 0)

        image = np.clip(image, 0, 1)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(image)
        if bbox is not None:
            rect = plt.Rectangle((bbox[0], bbox[1]),
                                 bbox[2] - bbox[0],
                                 bbox[3] - bbox[1],
                                 linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
        plt.title(title)
        image_path = os.path.join(self.image_dir, f"{title}.png")
        plt.savefig(image_path)
        plt.close()

    def log_pointcloud(self, pc, title='pointcloud', include_plotlyjs='cdn'):
        """
        Save a 3D point cloud visualization as an HTML file.
        """
        fig = go.Figure(data=[go.Scatter3d(
            x=pc[:, 0],
            y=pc[:, 1],
            z=pc[:, 2],
            mode='markers',
            marker=dict(size=2, color=pc[:, 2])
        )])
        pointcloud_path = os.path.join(self.pointcloud_dir, f"{title}.html")
        fig.show()
        fig.write_html(pointcloud_path, include_plotlyjs=include_plotlyjs)


# Data Augmentation
class Augmentor:
    """
    Performs data augmentation for RGB and LiDAR data.
    """
    def __init__(self, split):
        self.split = split
        self.rgb_transform = self._get_rgb_transforms()
        self.lidar_transform = self._get_lidar_transforms()

    def _get_rgb_transforms(self):
        transforms = [
            A.Resize(*Config.RGB_SIZE),
            A.Normalize(mean=[0.485, 0.456, 0.406],  # normalization with imagenet mean/std
                        std=[0.229, 0.224, 0.225]),
            A.pytorch.ToTensorV2()
        ]
        if self.split == 'train':
            transforms.insert(1, A.HorizontalFlip(p=0.5))
            transforms.insert(1, A.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.7
            ))
        return A.Compose(transforms)

    def _get_lidar_transforms(self):
        if self.split != 'train':
            return None
        return nn.Sequential(
            K.augmentation.RandomRotation3D(degrees=Config.LIDAR_ROT_RANGE, p=0.7),
            K.augmentation.RandomAffine3D(degrees=Config.LIDAR_ROT_RANGE,
                                            scale=Config.LIDAR_SCALE_RANGE, p=0.5)
        )


# Dataset & Preprocessing
class FrustumDataset(Dataset):
    """
    Loads and preprocesses scene data for 3D bounding box prediction.
    Each scene includes an RGB image, point cloud, mask, and 3D bbox.
    """
    def __init__(self, scene_paths, split='train'):
        self.scene_paths = scene_paths
        self.split = split
        self.augmentor = Augmentor(split)
        self.instances = []
        for path in tqdm(scene_paths, desc="Processing Scenes"):
            # Check if required files exist for the scene.
            if all([(path / 'rgb.jpg').exists(),
                    (path / 'pc.npy').exists(),
                    (path / 'mask.npy').exists()]):
                scene_data = self._load_scene(path)
                for instance in scene_data['instances']:
                    self.instances.append({
                        'scene_id': scene_data['scene_id'],
                        'instance': instance
                    })
            else:
                logging.warning(f"Invalid scene: {path.name}")

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        instance_data = self.instances[idx]
        # If debug mode is enabled, visualize a single random instance.
        if Config.DEBUG_MODE and random.random() < 0.01:
            self._debug_visualization(instance_data)
        return instance_data

    def _load_scene(self, path):
        try:
            rgb = cv2.cvtColor(cv2.imread(str(path / 'rgb.jpg')), cv2.COLOR_BGR2RGB)
            pc = np.load(path / 'pc.npy')
            masks = np.load(path / 'mask.npy')
            bboxes = np.load(path / 'bbox3d.npy') if (path / 'bbox3d.npy').exists() else None

            instances = []
            for i in range(masks.shape[0]):
                instance = self._process_instance(rgb, pc, masks[i], bboxes[i] if bboxes is not None else None)
                if instance is not None:
                    instances.append(instance)

            return {
                'scene_id': path.name,
                'instances': instances,
                'num_instances': len(instances)
            }
        except Exception as e:
            logging.error(f"Error loading scene {path.name}: {str(e)}")
            return {'scene_id': path.name, 'instances': [], 'num_instances': 0}

    def _process_instance(self, rgb, pc, mask, bbox):
        try:
            if np.isnan(rgb).any() or np.isnan(pc).any() or np.isnan(mask).any():
                logging.warning("NaN detected in input data!")
                return None
            if np.isinf(rgb).any() or np.isinf(pc).any() or np.isinf(mask).any():
                logging.warning("Inf detected in input data!")
                return None

            if np.sum(mask) < 1:
                logging.warning("Empty mask detected in input data!")
                return None

            bbox_2d = self._get_bbox_from_mask(mask)
            if bbox_2d is None:
                return None

            xmin, ymin, xmax, ymax = bbox_2d
            cropped_rgb = rgb[ymin:ymax, xmin:xmax]
            transformed = self.augmentor.rgb_transform(image=cropped_rgb)
            rgb_crop = transformed['image']

            if len(rgb_crop.shape) != 3:
                logging.error(f"Inconsistent RGB shape: {rgb_crop.shape}")
                return None

            pc_tensor, centroid = self._process_lidar(pc, mask)
            bbox_data = self._process_bbox(bbox) if bbox is not None else None

            return {
                'rgb': rgb_crop,
                'pc': pc_tensor,
                'centroid': centroid,
                'bbox': bbox_data
            }
        except Exception as e:
            logging.warning(f"Instance processing failed: {str(e)}")
            return None

    def _process_lidar(self, pc, mask):
        """
        Extract LiDAR points from the point cloud using the mask and ensure a fixed number of points.
        """
        pc_reshaped = np.transpose(pc, (1, 2, 0))
        if mask.shape != pc_reshaped.shape[:2]:
            raise ValueError(f"Mask shape {mask.shape} does not match LiDAR dimensions {pc_reshaped.shape[:2]}")
        processed_points = pc_reshaped[mask.astype(bool)]
        if processed_points.shape[0] == 0:
            raise ValueError("No LiDAR points extracted from mask!")
        centroid = np.mean(processed_points, axis=0)
        pc_tensor = torch.tensor(processed_points, dtype=torch.float32)
        #reshape to meet kornia transforms
        if self.augmentor.lidar_transform:
            augmented = self.augmentor.lidar_transform(pc_tensor.unsqueeze(0))
            while augmented.dim() > 2 and augmented.size(0) == 1:
                augmented = augmented.squeeze(0)
            if augmented.dim() != 2 or augmented.size(1) != 3:
                raise ValueError(f"After augmentation, expected shape (N,3) but got {augmented.shape}")
            pc_tensor = augmented
        if pc_tensor.shape[0] < Config.LIDAR_POINTS:
            padding = torch.zeros(Config.LIDAR_POINTS - pc_tensor.shape[0], 3)
            pc_tensor = torch.cat([pc_tensor, padding], dim=0)
        else:
            idx = torch.randperm(pc_tensor.shape[0])[:Config.LIDAR_POINTS]
            pc_tensor = pc_tensor[idx]
        return pc_tensor, torch.tensor(centroid, dtype=torch.float32)

    def _get_bbox_from_mask(self, mask):
        """
        Extract a 2D bounding box from a binary mask.
        """
        try:
            mask_uint8 = mask.astype(np.uint8)
            rows = np.any(mask_uint8, axis=1)
            cols = np.any(mask_uint8, axis=0)
            ymin, ymax = np.where(rows)[0][[0, -1]]
            xmin, xmax = np.where(cols)[0][[0, -1]]
            if (xmax - xmin) < 2 or (ymax - ymin) < 2:
                return None
            return (xmin, ymin, xmax, ymax)
        except Exception as e:
            logging.error(f"BBox extraction failed: {str(e)}")
            return None

    def _process_bbox(self, bbox, num_bins=2):
        """
        Process a raw 3D bounding box (8,3) into target parameters:
          - center, dimensions, orientation class and residual.
        """
        if bbox.shape != (8, 3):
            raise ValueError(f"Expected bbox shape (8,3), got {bbox.shape}")
        center = np.mean(bbox, axis=0)
        dimensions = np.max(bbox, axis=0) - np.min(bbox, axis=0)
        delta = bbox[1] - bbox[0]
        yaw = np.arctan2(delta[1], delta[0])
        # Discretizing the continuous yaw angle into one of several bins.
        bin_size = 2 * np.pi / num_bins
        yaw_shifted = yaw + np.pi
        bin_idx = int(np.floor(yaw_shifted / bin_size))
        bin_idx = min(bin_idx, num_bins - 1)
        bin_center = bin_idx * bin_size + bin_size / 2.0 - np.pi
        residual = yaw - bin_center
        return {
            'center': torch.tensor(center, dtype=torch.float32),
            'dims': torch.tensor(dimensions, dtype=torch.float32),
            'orient_cls': torch.tensor(bin_idx, dtype=torch.long),
            'orient_reg': torch.tensor(residual, dtype=torch.float32)
        }

    def _debug_visualization(self, instance_data):
        """
        Visualize a single preprocessed sample:
          - Log the augmented RGB image with its 2D bounding box overlay.
          - Log the masked 3D point cloud.
        This is only invoked if Config.DEBUG_MODE is True.
        """
        try:
            instance = instance_data['instance']
            scene_id = instance_data['scene_id']
            # Augmented RGB image
            rgb_tensor = instance['rgb']
            rgb_np = rgb_tensor.cpu().numpy().transpose(1, 2, 0)
            bbox_2d = instance.get('bbox_2d', None)
            debugger.log_image(rgb_np, bbox_2d, f"{scene_id}_augmented_rgb_debug")
            # Masked 3D point cloud visualization
            pc = instance['pc'].cpu().numpy()
            debugger.log_pointcloud(pc, f"{scene_id}_pc_debug")
        except Exception as e:
            logging.error(f"Visualization failed: {str(e)}")

# Data Loader and Collation
def collate_fn(batch):
    """
    Custom collate function to merge individual instances into a batch.
    """
    collated = {
        'scene_ids': [], # for later scene level instance grouping at post-processing
        'rgb': [],
        'pc': [],
        'centroid': [], # to scale back pc points
        'bbox': {'center': [], 'dims': [], 'orient_cls': [], 'orient_reg': []}
    }
    for data in batch:
        instance = data['instance']
        collated['scene_ids'].append(data['scene_id'])
        collated['rgb'].append(instance['rgb'])
        collated['pc'].append(instance['pc'])
        collated['centroid'].append(instance['centroid'])
        bbox = instance.get('bbox', None)
        if bbox is None:
            continue
        collated['bbox']['center'].append(bbox['center'])
        collated['bbox']['dims'].append(bbox['dims'])
        collated['bbox']['orient_cls'].append(bbox['orient_cls'])
        collated['bbox']['orient_reg'].append(bbox['orient_reg'])
    collated['rgb'] = torch.stack(collated['rgb'])
    collated['pc'] = torch.stack(collated['pc'])
    collated['bbox']['center'] = torch.stack(collated['bbox']['center'])
    collated['bbox']['dims'] = torch.stack(collated['bbox']['dims'])
    collated['bbox']['orient_cls'] = torch.stack(collated['bbox']['orient_cls'])
    collated['bbox']['orient_reg'] = torch.stack(collated['bbox']['orient_reg'])
    return collated

def get_loaders():
    """
    Create and return DataLoaders for training, validation, and testing.
    """
    dataset_path = BASE_PATH
    all_scenes = list(Path(dataset_path).glob('*'))
    logging.info(f"Total scenes detected: {len(all_scenes)}")
    train_scenes, test_val_scenes = train_test_split(
        all_scenes, test_size=1-Config.TRAIN_SPLIT, random_state=Config.SEED
    )
    val_scenes, test_scenes = train_test_split(
        test_val_scenes, test_size=0.5, random_state=Config.SEED
    )
    #Dataset creation
    train_dataset = FrustumDataset(train_scenes, 'train')
    val_dataset = FrustumDataset(val_scenes, 'val')
    test_dataset = FrustumDataset(test_scenes, 'test')
    #Dataset loaders
    loaders = {
        'train': DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            collate_fn=collate_fn,
            pin_memory=True
        ),
        'val': DataLoader(
            val_dataset,
            batch_size=Config.BATCH_SIZE,
            num_workers=Config.NUM_WORKERS,
            collate_fn=collate_fn,
            pin_memory=True
        ),
        'test': DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            num_workers=Config.NUM_WORKERS,
            collate_fn=collate_fn,
            pin_memory=True
        )
    }
    return loaders

# Model Components
class SimplePointNetPP(nn.Module):
    """
    Simplified PointNet++ module for processing LiDAR point clouds.
    ## refernce link: https://pytorch-geometric.readthedocs.io/en/2.5.1/tutorial/point_cloud.html
    imported modules: PointNetConv,fps,radius
    """
    def __init__(self, in_channels=3, out_channels=512):
        super().__init__()
        self.conv1 = PointNetConv(
            nn.Sequential(
                nn.Linear(3 + 3, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU()
            ),
            add_self_loops=False
        )
        self.conv2 = PointNetConv(
            nn.Sequential(
                nn.Linear(128 + 3, 256),  
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256, 512),  
                nn.ReLU()
            ),
            add_self_loops=False
        )
    # Normalizing the lidar points batch-wise (instead during pre-processing)
    def normalize_pos(self, pos, batch):
        batch = batch.to(pos.device)
        batch = batch - batch.min()
        batch_size = batch.max().item() + 1
        mean = scatter_mean(pos, batch, dim=0, dim_size=batch_size).to(pos.device)
        std = scatter_std(pos, batch, dim=0, dim_size=batch_size).to(pos.device) + 1e-6
        pos_norm = (pos - mean[batch]) / std[batch]
        return pos_norm, batch

    def forward(self, pos, batch):
        pos, batch = self.normalize_pos(pos, batch)
        # stage-1: sampling & grouping, pointnet
        idx1 = fps(pos, batch, ratio=0.5)
        pos1 = pos[idx1]
        batch1 = batch[idx1]
        edge_index1 = radius(pos1, pos1, r=0.1, batch_x=batch1, batch_y=batch1, max_num_neighbors=64) #hyperparam: max_num_neighbors, radii
        x1 = self.conv1(x=pos1, pos=pos1, edge_index=edge_index1)
        # stage-2: sampling & grouping, pointnet
        idx2 = fps(pos1, batch1, ratio=0.5)
        pos2 = pos1[idx2]
        batch2 = batch1[idx2]
        edge_index2 = radius(pos2, pos2, r=0.25, batch_x=batch2, batch_y=batch2, max_num_neighbors=64)
        x2_input = x1[idx2]
        x2 = self.conv2(x=x2_input, pos=pos2, edge_index=edge_index2)
        return x2, batch2

class ResNetBackbone(nn.Module):
    """
    ResNet-18 based backbone for image feature extraction.
    """
    def __init__(self, freeze=True):
        super().__init__()
        base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.stem = nn.Sequential(
            base_model.conv1, base_model.bn1, base_model.relu, base_model.maxpool
        )
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4
        #Transfer learning
        if freeze:
            for param in self.stem.parameters():
                param.requires_grad = False
            for param in self.layer1.parameters():
                param.requires_grad = False
            for param in self.layer2.parameters(): 
                param.requires_grad = False
            for param in self.layer3.parameters(): 
                param.requires_grad = False
            for param in self.layer4.parameters(): 
                param.requires_grad = False
            

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.layer4(x)

class CrossAttentionFusion(nn.Module):
    """
    Fuses LiDAR and image features using cross-attention.
    """
    def __init__(self, lidar_dim=512, image_dim=512):
        super().__init__()
        self.image_proj = nn.Linear(image_dim, lidar_dim)
        self.attention = nn.MultiheadAttention(lidar_dim, 4, batch_first=True)

    def forward(self, lidar_feats, image_feats, mask=None):
        image_feats = self.image_proj(image_feats)
        attn_out, _ = self.attention(
            query=lidar_feats,
            key=image_feats,
            value=image_feats,
            key_padding_mask=mask
        )
        return attn_out

# Complete model archietcture
    
class MultiModalBBoxPredictor(nn.Module):
    """
    End-to-end model that fuses image and LiDAR features to predict 3D bounding boxes.
    """
    def __init__(self, num_bins=2):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_bins = num_bins
        self.pointnet = SimplePointNetPP()
        self.resnet = ResNetBackbone(freeze=True)
        self.img_norm = nn.LayerNorm(512)
        self.point_norm = nn.LayerNorm(512)
        self.fusion = CrossAttentionFusion(lidar_dim=512, image_dim=512)
        #prediction heads
        self.head_center = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, 3)
        )
        self.head_dims = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, 3),
            nn.Softplus()
        )
        self.head_orient_cls = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, self.num_bins)
        )
        self.head_orient_reg = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, self.num_bins)
        )
        self.scale_factor_dims = nn.Parameter(torch.tensor(1.0))
        self.global_pool_proj = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
        )


    def forward(self, rgb, lidar):
        rgb = rgb.to(self.device)
        lidar = lidar.to(self.device)
        # Image pathway
        img_feats = self.resnet(rgb)
        b, c, h, w = img_feats.shape
        img_feats = img_feats.view(b, c, -1).permute(0, 2, 1)
        img_feats = self.img_norm(img_feats)
        # LiDAR pathway
        batch_size = rgb.shape[0]
        num_points = lidar.shape[1]
        pos = lidar.reshape(-1, 3).to(self.device)
        # Batch processing
        # Create a batch index tensor that assigns each point to its corresponding batch
        # This is done by repeating each batch index 'num_points' times
        batch = torch.arange(batch_size, device=self.device).repeat_interleave(num_points)

        # Pass the positional data and batch indices through the PointNet module
        # 'lidar_feats' contains the extracted features for each point
        # 'lidar_batch' contains the batch indices corresponding to each feature
        lidar_feats, lidar_batch = self.pointnet(pos, batch)

        # Convert the sparse batch of point features into a dense tensor and reshaping with corresponding Batch dimensions
        # 'point_feats_dense' has the shape [batch_size, max_num_nodes, feature_dim]
        # 'mask' is a boolean tensor indicating the presence of valid nodes
        point_feats_dense, mask = to_dense_batch(lidar_feats, lidar_batch, max_num_nodes=256, fill_value=0.0)
        point_feats_dense = self.point_norm(point_feats_dense)
        # Fusion via cross-attention
        fused_dense = self.fusion(point_feats_dense, img_feats)
        # Global pooling - max+avg pooling to preserve sparse features
        fused_max = fused_dense.max(dim=1)[0]
        fused_avg = fused_dense.mean(dim=1)
        fused_global = self.global_pool_proj(torch.cat([fused_max, fused_avg], dim=1))
        # Prediction heads
        pred_center = self.head_center(fused_global)
        pred_dims_raw = self.head_dims(fused_global)
        pred_dims = pred_dims_raw * self.scale_factor_dims #predicted dimensions are too low relative to the ground truth,so defined a learnable dim scaling factor
        pred_orient_cls = self.head_orient_cls(fused_global)
        pred_orient_reg = self.head_orient_reg(fused_global)
        return {
            'center': pred_center,
            'dims': pred_dims,
            'orient_cls': pred_orient_cls,
            'orient_reg': pred_orient_reg
        }
    
# Training Infrastructure
class BBoxTrainer:
    """
    Trainer for the 3D bounding box prediction model.
    Handles training, validation, and testing loops.
    """
    def __init__(self, model, train_loader, val_loader, test_loader, patience = 4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.writer = SummaryWriter()
        self.optimizer = self.configure_optimizer()
        # Using ReduceLROnPlateau scheduler to dynamically reduce the LR when validation loss plateaus.
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=2, verbose=True)
        self.loss_weights = {'center': 1.0, 'dims': 1.0, 'yaw': 2.0}
        self.patience = patience  # Early stopping patience
        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0

    def configure_optimizer(self):
        params = [
            {'params': self.model.resnet.layer3.parameters(), 'lr': 1e-4},
            {'params': self.model.resnet.layer4.parameters(), 'lr': 1e-4},
            {'params': self.model.pointnet.parameters(), 'lr': 5e-4},
            {'params': self.model.fusion.parameters(), 'lr': 5e-4},
            {'params': self.model.head_center.parameters(), 'lr': 5e-4},
            {'params': self.model.head_dims.parameters(), 'lr': 5e-4},
            {'params': self.model.head_orient_cls.parameters(), 'lr': 1e-4},
            {'params': self.model.head_orient_reg.parameters(), 'lr': 1e-4},
        ]
        return torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-3)

    #orientation loss compute
    def yaw_loss_multibin(self, pred_cls, pred_reg, gt_cls, gt_reg):
        loss_cls = F.cross_entropy(pred_cls, gt_cls)
        pred_residual = pred_reg.gather(1, gt_cls.unsqueeze(1)).squeeze(1)
        loss_reg = F.smooth_l1_loss(pred_residual, gt_reg)
        return loss_cls + loss_reg
    
    #total loss compute
    def compute_loss(self, preds, targets):
        center_loss = F.smooth_l1_loss(preds['center'], targets['center'])
        dims_loss = F.smooth_l1_loss(preds['dims'], targets['dims'])
        loss_yaw = self.yaw_loss_multibin(preds['orient_cls'], preds['orient_reg'], targets['orient_cls'], targets['orient_reg'])
        total_loss = (self.loss_weights['center'] * center_loss +
                      self.loss_weights['dims'] * dims_loss +
                      self.loss_weights['yaw'] * loss_yaw)
        return total_loss

    #training loop
    def train_epoch(self):
        self.model.train()
        avg_loss = 0
        for batch in self.train_loader:
            rgb = batch['rgb'].to(self.device, non_blocking=True)
            pc = batch['pc'].to(self.device, non_blocking=True)
            targets = {
                'center': batch['bbox']['center'].to(self.device),
                'dims': batch['bbox']['dims'].to(self.device),
                'orient_cls': batch['bbox']['orient_cls'].to(self.device),
                'orient_reg': batch['bbox']['orient_reg'].to(self.device)
            }
            # Reset gradients to avoid accumulation from previous steps
            self.optimizer.zero_grad()
            # Forward pass: Predict bounding boxes using the model
            preds = self.model(rgb, pc)
             # Compute the loss by comparing predictions with ground truth labels
            loss = self.compute_loss(preds, targets)
            # Backpropagation: Compute gradients of the loss w.r.t. model parameters
            loss.backward()
             # Gradient Clipping: Prevents exploding gradients by regulating their norm
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            # Update model parameters using the optimizer
            self.optimizer.step()
            
            avg_loss += loss.item()
        return avg_loss / len(self.train_loader)

    #validation loop
    def validate(self, epoch):
        metrics = BBoxMetrics(stage='val', num_bins=self.model.num_bins)
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in self.val_loader:
                rgb = batch['rgb'].to(self.device)
                pc = batch['pc'].to(self.device)
                targets = {
                    'center': batch['bbox']['center'].to(self.device),
                    'dims': batch['bbox']['dims'].to(self.device),
                    'orient_cls': batch['bbox']['orient_cls'].to(self.device),
                    'orient_reg': batch['bbox']['orient_reg'].to(self.device)
                }
                preds = self.model(rgb, pc)
                loss = self.compute_loss(preds, targets)
                val_loss += loss.item()
                metrics.update(preds, targets, batch_size=rgb.size(0))
        avg_val_loss = val_loss / len(self.val_loader)
        metric_values = metrics.get_metrics()
        self.log_metrics(metric_values, epoch, 'val')
        # Log validation loss and metrics
        logging.info(f"Validation Metrics: {metric_values}")
        return avg_val_loss

    def train(self, num_epochs):
        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_epoch()
            val_loss = self.validate(epoch)
            # Update the LR scheduler based on validation loss.
            self.scheduler.step(val_loss)
            logging.info(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            # Early stopping check.
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_no_improve = 0
                torch.save(self.model.state_dict(), "best_model.pth")
                logging.info(f"New best model saved at epoch {epoch} with Val Loss: {val_loss:.4f}")
            else:
                self.epochs_no_improve += 1
                logging.info(f"No improvement for {self.epochs_no_improve} epochs.")
                if self.epochs_no_improve >= self.patience:
                    logging.info("Early stopping triggered.")
                    break
    # evaluation loop
    def evaluate_metrics(self, loader, stage):
        """
        Evaluate detailed metrics on the given DataLoader using BBoxMetrics.
        """
        metrics = BBoxMetrics(stage='test', num_bins=self.model.num_bins)
        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                rgb = batch['rgb'].to(self.device)
                pc = batch['pc'].to(self.device)
                targets = {
                    'center': batch['bbox']['center'].to(self.device),
                    'dims': batch['bbox']['dims'].to(self.device),
                    'orient_cls': batch['bbox']['orient_cls'].to(self.device),
                    'orient_reg': batch['bbox']['orient_reg'].to(self.device)
                }
                preds = self.model(rgb, pc)
                metrics.update(preds, targets, batch_size=rgb.size(0))
                loss = self.compute_loss(preds, targets)
                test_loss += loss.item()
        avg_test_loss = test_loss / len(self.test_loader)
        metric_values = metrics.get_metrics()
        self.log_metrics(metric_values, -1, 'test')
        logging.info(f"{stage} Metrics: {metric_values}")
        logging.info(f"{stage} Test Loss: {avg_test_loss:.4f}")

        return metric_values, avg_test_loss

    def log_metrics(self, metrics, epoch, stage):
        """Log metrics to TensorBoard"""
        for k, v in metrics.items():
            self.writer.add_scalar(f'{stage}/{k}', v, int(epoch) if isinstance(epoch, int) else 0)


# BBoxMetrics for Detailed Evaluation
class BBoxMetrics:
    """
    Metrics for 3D bounding boxes, adapted for multi-bin orientation.
    """
    def __init__(self, stage='val', num_bins=2):
        self.stage = stage
        self.num_bins = num_bins
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mae_center = MeanAbsoluteError().to(self.device)
        self.mse_center = MeanSquaredError().to(self.device)
        self.mae_dims = MeanAbsoluteError().to(self.device)
        self.mse_dims = MeanSquaredError().to(self.device)
        self.angle_error = MeanAbsoluteError().to(self.device)
        self.iou_3d = MeanAbsoluteError().to(self.device)  

    def update(self, preds, targets, batch_size):
        # Center metrics
        self.mae_center(preds['center'], targets['center'])
        self.mse_center(preds['center'], targets['center'])
        # Dimension metrics
        self.mae_dims(preds['dims'], targets['dims'])
        self.mse_dims(preds['dims'], targets['dims'])
        # Orientation error
        num_bins = self.num_bins
        bin_size = 2 * math.pi / num_bins
        pred_yaw_angles = multibin_to_yaw(preds['orient_cls'], preds['orient_reg'], num_bins)
        gt_yaw_angles = targets['orient_reg'] + (targets['orient_cls'].float() * bin_size + bin_size / 2.0 - math.pi)
        # Compute angular difference 
        angle_diff = torch.remainder(pred_yaw_angles - gt_yaw_angles + math.pi, 2 * math.pi) - math.pi
        angle_diff_deg = torch.abs(torch.rad2deg(angle_diff))
        self.angle_error(angle_diff_deg, torch.zeros_like(angle_diff_deg))
        
        # Convert yaw angles to (sin, cos) vectors
        pred_yaw_vec = torch.stack([torch.sin(pred_yaw_angles), torch.cos(pred_yaw_angles)], dim=1)
        gt_yaw_vec = torch.stack([torch.sin(gt_yaw_angles), torch.cos(gt_yaw_angles)], dim=1)
        
        # 3D IoU (via box3d_overlap from pytorch 3d)
        pred_boxes = convert_to_box3d(preds['center'], preds['dims'], pred_yaw_vec)
        gt_boxes = convert_to_box3d(targets['center'], targets['dims'], gt_yaw_vec)
        try:
            _, ious = box3d_overlap(pred_boxes, gt_boxes)
            valid_ious = ious[~torch.isnan(ious)]
            if len(valid_ious) > 0:
                self.iou_3d(valid_ious.mean(), torch.tensor(0.0))
            else:
                logging.warning(f"[WARNING {self.stage}] All IoUs are NaN!")
        except Exception as e:
            logging.error(f"[ERROR {self.stage}] IoU calculation failed: {str(e)}")

    def get_metrics(self):
        return {
            'center_mae': self.mae_center.compute().item(),
            'center_mse': self.mse_center.compute().item(),
            'dims_mae': self.mae_dims.compute().item(),
            'dims_mse': self.mse_dims.compute().item(),
            'angle_error': self.angle_error.compute().item(),
            'iou_3d': self.iou_3d.compute().item()
        }

    def reset(self):
        self.mae_center.reset()
        self.mse_center.reset()
        self.mae_dims.reset()
        self.mse_dims.reset()
        self.angle_error.reset()
        self.iou_3d.reset()


# Box Conversion Helpers
def multibin_to_yaw(pred_cls, pred_reg, num_bins):
    """
    Convert multi-bin predictions into a continuous yaw angle(in radians).
    """
    bin_idx = torch.argmax(pred_cls, dim=1)
    bin_size = 2 * math.pi / num_bins
    bin_center = bin_idx.float() * bin_size + (bin_size / 2.0) - math.pi
    pred_residual = pred_reg.gather(1, bin_idx.unsqueeze(1)).squeeze(1)
    pred_angle = bin_center + pred_residual
    return pred_angle

def convert_to_box3d(centers, dims, orientations):
    """
    Convert center, dimensions, and orientation (sinθ, cosθ) to 8-corner 3D boxes.
    """
    dims = torch.clamp(dims, min=0.05)
    batch_size = centers.shape[0]
    device = centers.device
    angles = torch.atan2(orientations[:, 0], orientations[:, 1])
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    rot_z = torch.zeros((batch_size, 3, 3), device=device)
    rot_z[:, 0, 0] = cos
    rot_z[:, 0, 1] = -sin
    rot_z[:, 1, 0] = sin
    rot_z[:, 1, 1] = cos
    rot_z[:, 2, 2] = 1
    lwh = dims / 2
    corners = torch.tensor([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]
    ], device=device).float()
    corners = corners[None] * lwh[:, None]
    corners = torch.bmm(corners, rot_z)
    boxes = corners + centers[:, None]
    return boxes

# main script

if __name__ == "__main__":
    # for visualization/logging
    debugger = DebugVisualizer()
    # Data preprocessing & loading
    loaders = get_loaders()
    # Model creation
    model = MultiModalBBoxPredictor()

    #print model architecture
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dummy_rgb = torch.randn(4, 3, 128, 128).to(device)
    dummy_lidar = torch.randn(4, 512, 3).to(device)
    model_summary = summary(model, input_data=[dummy_rgb, dummy_lidar])
    print(model_summary)
    logging.info(model_summary)
    #initialize model trainer
    trainer = BBoxTrainer(model, loaders['train'], loaders['val'], loaders['test'], patience=50) 

    # Model training   
    Config.NUM_EPOCHS
    trainer.train(Config.NUM_EPOCHS)

    # Evaluate detailed metrics on the test set.
    test_metrics, test_loss = trainer.evaluate_metrics(loaders['test'], stage='test')
    # Save metrics
    with open("test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)
    # Ensure logs are flushed before the script exits
    logging.shutdown()
    # Save final model
    torch.save(model.state_dict(), "final_model.pth")
    logging.info("Training complete. Models and metrics saved.")