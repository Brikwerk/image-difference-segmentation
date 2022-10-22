import argparse
from os.path import basename, splitext, join
from PIL import Image

import torch
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms as T
from tqdm import tqdm

from src.models.mask_rcnn import get_mask_rcnn
from src.data.basic_dataset import BasicDataset
from src.utils import combine_masks, get_palette, make_missing_dirs


parser = argparse.ArgumentParser()
parser.add_argument('--weights_path', required=True, type=str,
                    help="""Path to the weights for a segmentation model.""")
parser.add_argument('--input_path', required=True, type=str,
                    help="""Path to a folder containing input images.""")
parser.add_argument('--output_path', required=True, type=str,
                    help="""Path to the location where masks will be stored.
                            This location does not necessarily need to
                            exist.""")
parser.add_argument('--device', default="cpu", type=str,
                    help="""The device to use when running the model.""")
parser.add_argument('--batch_size', default=16, type=int,
                    help="""Batch size to pass to the DiffNet model.""")
parser.add_argument('--num_workers', default=8, type=int,
                    help="""Number of threads for loading data.""")
parser.add_argument('--mask_thresh', default=0.2, type=float,
                    help="""Threshold for generating binary masks.""")
args = parser.parse_args()


if __name__ == "__main__":
    # Create a binary mask classification model
    model = get_mask_rcnn(num_classes=2)
    model.eval()
    # Load DiffNet weights
    model.load_state_dict(torch.load(args.weights_path))
    model.to(args.device)

    transform = T.Compose([
        T.Grayscale(),
        T.ToTensor()
    ])
    dataset = BasicDataset(args.input_path, transform=transform)
    dataloader = DataLoader(dataset,
                            args.batch_size,
                            num_workers=args.num_workers)

    # Make output location, if it doesn't exist
    make_missing_dirs(args.output_path)

    palette = get_palette(num_colours=255)
    for batch in tqdm(dataloader):
        imgs, paths = batch
        imgs = imgs.to(args.device)
        with torch.no_grad():
            output = model(imgs)

        for i in range(len(imgs)):
            masks = output[i]['masks'].detach().cpu().squeeze(1).numpy()
            name = splitext(basename(paths[i]))[0]

            combined_masks = combine_masks(masks, thresh=args.mask_thresh)

            # print(np.max(combined_masks))
            img = Image.fromarray(combined_masks).convert('P')
            img.putpalette(palette)
            img.save(join(args.output_path, name + ".png"), format='PNG')
