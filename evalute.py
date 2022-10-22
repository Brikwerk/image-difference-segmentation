import argparse
from os import listdir

import torch
from torch.utils.data.dataloader import DataLoader

from src.data.mask_dataset import MaskDataset
from src.utils import mask_dataset_collate
from src.models.mask_rcnn import get_mask_rcnn
from src.evaluate import evaluate


parser = argparse.ArgumentParser()
parser.add_argument('--weights_path', type=str, required=True,
                    help="""Path to model weights.""")
parser.add_argument('--images_path', required=True, type=str,
                    help="""Path to a folder containing input images.""")
parser.add_argument('--masks_path', required=True, type=str,
                    help="""Path to a folder containing input masks.""")
parser.add_argument('--height', default=800, type=int,
                    help="""Image height for testing.""")
parser.add_argument('--width', default=533, type=int,
                    help="""Image width for testing.""")
parser.add_argument('--device', default="cpu", type=str,
                    help="""The device to use when running the model.""")
parser.add_argument('--batch_size', default=16, type=int,
                    help="""Batch size to pass to the DiffNet model.""")
parser.add_argument('--num_workers', default=8, type=int,
                    help="""Number of threads for loading data.""")
args = parser.parse_args()


if __name__ == "__main__":
    # General config
    IMAGES_PATH = args.images_path
    MASKS_PATH = args.masks_path
    WEIGHTS_PATH = args.weights_path
    NUM_WORKERS = args.num_workers
    HEIGHT = args.height
    WIDTH = args.width

    # Data parameters
    DEVICE = args.device
    BATCH_SIZE = args.batch_size

    # Get the names of all available images/masks
    images_all = sorted(listdir(IMAGES_PATH))
    masks_all = sorted(listdir(MASKS_PATH))
    assert images_all == masks_all

    # Create the datasets
    test_dataset = MaskDataset(
        IMAGES_PATH, MASKS_PATH,
        img_names=images_all,
        mask_names=masks_all,
        train=False
    )

    # Create the dataloaders
    test_loader = DataLoader(
        test_dataset,
        BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=mask_dataset_collate
    )

    # Model setup
    model = get_mask_rcnn()
    model.load_state_dict(torch.load(WEIGHTS_PATH))
    model.to(DEVICE)

    # Evaluate the model and print the results
    coco_evaluator = evaluate(model, test_loader, DEVICE)
