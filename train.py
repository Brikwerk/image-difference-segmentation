import argparse
from os import listdir
from os.path import join
import datetime

import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
import numpy as np

from src.data.mask_dataset import MaskDataset
from src.utils import mask_dataset_collate
from src.models.mask_rcnn import get_mask_rcnn
from src.evaluate import evaluate


parser = argparse.ArgumentParser()
parser.add_argument('--weights_path', type=str, default="./weights",
                    help="""Output path to the weights for the
                            segmentation model being trained.""")
parser.add_argument('--images_path', required=True, type=str,
                    help="""Path to a folder containing input images.""")
parser.add_argument('--masks_path', required=True, type=str,
                    help="""Path to a folder containing input masks.""")
parser.add_argument('--height', default=800, type=int,
                    help="""Image height for training and testing.""")
parser.add_argument('--width', default=533, type=int,
                    help="""Image width for training and testing.""")
parser.add_argument('--epochs', default=100, type=int,
                    help="""Number of training epochs.""")
parser.add_argument('--warmup_epochs', default=10, type=int,
                    help="""Number of warmup epochs.""")
parser.add_argument('--lr', default=4e-3, type=float,
                    help="""Training learning rate.""")
parser.add_argument('--device', default="cpu", type=str,
                    help="""The device to use when running the model.""")
parser.add_argument('--batch_size', default=16, type=int,
                    help="""Batch size to pass to the DiffNet model.""")
parser.add_argument('--num_workers', default=8, type=int,
                    help="""Number of threads for loading data.""")
parser.add_argument('--use_half_precision', default=False, type=bool,
                    help="""Whether to train using half precision.""")
parser.add_argument('--use_pretrained', default=False, type=bool,
                    help="""Whether to use a pretrained Mask RCNN model.""")
args = parser.parse_args()


def train_for_one_epoch(model, dataloader, optimizer, scheduler, device, epoch,
                        scaler=None, writer=None):
    model.train()
    losses = []
    for i, batch in enumerate(tqdm(dataloader)):
        images, seg_data = batch
        images = list(image.to(device) for image in images)
        seg_data = [{k: v.to(device) for k, v in t.items()} for t in seg_data]

        # Process the images and get the total loss
        if scaler is not None:
            with torch.cuda.amp.autocast():
                output = model(images, seg_data)
                loss = sum(loss for loss in output.values())
        else:
            output = model(images, seg_data)
            loss = sum(loss for loss in output.values())

        # Back prop the loss
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # Record loss
        losses.append(loss.item())
        if writer is not None:
            writer.add_scalar('train/loss', loss.item(),
                              (epoch * len(dataloader)) + i)

    return losses


if __name__ == "__main__":
    # General config
    IMAGES_PATH = args.images_path
    MASKS_PATH = args.masks_path
    WEIGHTS_PATH = args.weights_path
    NUM_WORKERS = args.num_workers
    HEIGHT = args.height
    WIDTH = args.width
    USE_HALF_PRECISION = args.use_half_precision

    # Model parameters
    EPOCHS = args.epochs
    WARMUP_EPOCHS = args.warmup_epochs
    LR = args.lr
    DEVICE = args.device
    BATCH_SIZE = args.batch_size
    DATETIME_STR = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    USE_PRETRAINED = args.use_pretrained

    # Get the names of all available images/masks
    images_all = sorted(listdir(IMAGES_PATH))
    masks_all = sorted(listdir(MASKS_PATH))
    assert images_all == masks_all

    # Get a random split of names for training/testing
    all_indices = torch.randperm(len(images_all))
    split_len = int(len(all_indices) * 0.1)
    train_indices = all_indices[split_len:]
    test_indices = all_indices[:split_len]

    images_train = [images_all[i] for i in train_indices]
    images_test = [images_all[i] for i in test_indices]
    masks_train = [masks_all[i] for i in train_indices]
    masks_test = [masks_all[i] for i in test_indices]

    # Create the datasets
    train_dataset = MaskDataset(
        IMAGES_PATH, MASKS_PATH,
        img_names=images_train,
        mask_names=masks_train,
        train=True
    )
    test_dataset = MaskDataset(
        IMAGES_PATH, MASKS_PATH,
        img_names=images_test,
        mask_names=masks_test,
        train=False
    )

    # Create the dataloaders
    train_loader = DataLoader(
        train_dataset,
        BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=mask_dataset_collate
    )
    test_loader = DataLoader(
        test_dataset,
        BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=mask_dataset_collate
    )

    # Model setup
    model = get_mask_rcnn(pretrained=USE_PRETRAINED)
    model.to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR)
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer, warmup_epochs=WARMUP_EPOCHS,
        max_epochs=EPOCHS)

    if USE_HALF_PRECISION:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    # Logging
    writer = SummaryWriter()

    # Training loop
    for epoch in range(EPOCHS):
        # Train
        losses = train_for_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=DEVICE,
            epoch=epoch,
            scaler=scaler,
            writer=writer
        )
        print(f'Epoch {epoch}: Loss {np.mean(losses)}')

        coco_evaluator = evaluate(model, test_loader, DEVICE)
        ap_score = coco_evaluator.coco_eval['segm'].stats[0]  # 0.5 - 0.95 AP
        writer.add_scalar('test/AP_0.5_0.95',
                          scheduler.get_last_lr()[0], epoch)

        # Update the learning rate scheduler
        scheduler.step()
        writer.add_scalar('learning_rate', scheduler.get_last_lr()[0], epoch)

        # Save the model
        torch.save(model.state_dict(),
                   join(WEIGHTS_PATH,
                   f"mask_rcnn_fpn_{DATETIME_STR}.pth"))
