import argparse
import os

from PIL import Image
import numpy as np
from tqdm import tqdm

from src.utils import get_image_paths


parser = argparse.ArgumentParser()
parser.add_argument('--images_path', required=True, type=str,
                    help="""Path to a folder containing input images.""")
parser.add_argument('--diffs_path', required=True, type=str,
                    help="""Path to a folder containing input images.""")
parser.add_argument('--masks_path', required=True, type=str,
                    help="""Path to a folder containing input masks.""")
args = parser.parse_args()


if __name__ == "__main__":
    images = sorted(get_image_paths(args.images_path))
    masks = sorted(get_image_paths(args.masks_path))
    diffs = sorted(get_image_paths(args.diffs_path))
    assert len(images) == len(masks) == len(diffs)

    for i in tqdm(range(len(masks))):
        mask_path = masks[i]
        image_path = images[i]
        diff_path = diffs[i]

        # Check basenames match
        mask_bn = os.path.basename(mask_path)
        image_bn = os.path.basename(image_path)
        diff_bn = os.path.basename(diff_path)
        assert mask_bn == image_bn == diff_bn

        mask = np.array(Image.open(mask_path))
        if mask.sum() < 10:
            print(mask_path)
            pass
            # Remove data that has no mask data
            # os.remove(mask_path)
            # os.remove(image_path)
            # os.remove(diff_path)
