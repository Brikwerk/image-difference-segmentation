from os import listdir
from os.path import join
import random

import torch
from torchvision.transforms import functional as TF
from torchvision import transforms as T
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class MaskDataset(Dataset):
    def __init__(self, img_root, mask_root, img_names=None,
                 mask_names=None, train=True, height=800,
                 width=533):
        self.img_root = img_root
        self.mask_root = mask_root
        self.train = train
        self.height = height
        self.width = width

        # If names for images or masks are specified,
        # use the specified names instead of all names
        # in the image/mask root.
        if img_names is None:
            self.imgs = sorted(listdir(img_root))
        else:
            self.imgs = img_names
        if mask_names is None:
            self.masks = sorted(listdir(mask_root))
        else:
            self.masks = mask_names
        # Check that we have the same number of masks
        # and images.
        assert len(self.imgs) == len(self.masks)

    def transform(self, img, masks):
        resize = T.Resize((self.height, self.width))
        img = resize(img)
        masks = resize(masks)

        # Random horizontal flip
        if random.random() > 0.5 and self.train:
            img = TF.hflip(img)
            masks = TF.hflip(masks)

        # Random vertical flip
        if random.random() > 0.5 and self.train:
            img = TF.vflip(img)
            masks = TF.vflip(masks)

        img = TF.to_tensor(img)
        # masks = TF.to_tensor(masks)

        return img, masks

    def __getitem__(self, idx):
        # Load the image and mask.
        # The mask is loaded as a simple 2D array, not an image
        img = Image.open(join(self.img_root, self.imgs[idx])).convert("RGB")
        mask = np.array(Image.open(join(self.mask_root, self.masks[idx])))

        # Identify the unique mask IDs, excluding the background (ID = 0)
        mask_ids = np.unique(mask)[1:]

        # Generate binary masks based on each unique mask
        masks = mask == mask_ids[:, None, None]

        # Apply the same transforms to image and masks
        masks = torch.tensor(masks, dtype=torch.uint8)
        img, masks = self.transform(img, masks)

        # Get the bounding boxes for each mask
        bboxes = []
        for i in range(len(masks)):
            coords = np.where(masks[i])
            x_min = np.min(coords[1])
            x_max = np.max(coords[1])
            if x_min == x_max:  # Ensure single pixel bboxes are counted
                x_max += 1
            y_min = np.min(coords[0])
            y_max = np.max(coords[0])
            if y_min == y_max:
                y_max += 1
            bboxes.append([x_min, y_min, x_max, y_max])

        # Create COCO-type segmentation data tensors
        labels = torch.tensor([1] * len(masks), dtype=torch.int64)
        bboxes = torch.tensor(bboxes, dtype=torch.float32)
        area = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
        is_crowd = torch.tensor([0] * len(masks), dtype=torch.int64)

        seg_data = {
            "labels": labels,
            "boxes": bboxes,
            "masks": masks,
            "image_id": torch.tensor([idx]),
            "area": area,
            "iscrowd": is_crowd
        }

        return img, seg_data

    def __len__(self):
        return len(self.imgs)
