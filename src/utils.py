from os import listdir, makedirs
from os.path import join, isdir, isfile
import re

import numpy as np
import cv2 as cv
from PIL import Image
from skimage import feature
import torch
from skimage.morphology import label


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def getfiles(path):
    files = [f for f in listdir(path) if isfile(join(path, f))]
    files.sort(key=natural_keys)  # Sorted in human order
    return files


def getfolders(path):
    folders = [f for f in listdir(path) if isdir(join(path, f))]
    folders.sort(key=natural_keys)  # Sorted in human order
    return folders


def get_image_paths(path):
    paths = getfiles(path)
    imgs = []
    for img in paths:
        if (img.endswith('.jpg') or img.endswith('.png')
                or img.endswith('.jpeg')):
            imgs.append(join(path, img))
    return imgs


def warp_to_match(img1, img2, min_match_count=10):
    """
    img1 is warped to match img2 through homographic warping based upon
    keypoints found through SIFT. img1 and img2 are expected to be
    numpy image arrays.

    False is returned on failure if less than 10 matching keypoints are found
    A Pillow is returned of the warped image on success
    """
    MIN_MATCH_COUNT = min_match_count

    # Initiate SIFT detector
    sift = cv.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32(
            [kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32(
            [kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    else:
        return False

    im_out = cv.warpPerspective(img1, M, (img2.shape[1], img2.shape[0]))

    return Image.fromarray(im_out)


def mass_img_open(path):
    """
    Copy image into the memory and close the file handler.
    This function prevents a bug with Pillow when thousands of
    images are loaded in quick succession.
    """
    temp = Image.open(path)
    img = temp.copy()
    temp.close()
    return img


def load_all_imgs(img_list):
    """
    Loads all images as Pillow Images and appends them into a
    Python list.
    """
    imgs = []
    for img_path in img_list:
        imgs.append(mass_img_open(img_path))
    return imgs


def create_img_diff(img1, img2, resize_width=533, resize_height=800):
    """
    Creates an image differential based on 2 Pillow images.
    The differential is constructed from Canny edges.
    Differences in img1 compared to img2 are returned.
    """
    img1_resize = np.array(img1.convert('L').resize(
        (resize_width, resize_height)))
    img1_canny = feature.canny(img1_resize, sigma=1)

    img2_resize = np.array(img2.convert('L').resize(
        (resize_width, resize_height)))
    img2_canny = feature.canny(img2_resize, sigma=1)

    diff = img2_canny < img1_canny

    return Image.fromarray(diff)


def make_missing_dirs(path):
    if not isdir(path):
        makedirs(path)


def combine_masks(masks, thresh=0.2):
    masks_shape = masks.shape
    mask_combined = np.zeros((masks_shape[1], masks_shape[2]))
    if len(masks) > 0:
        for i, mask in enumerate(masks):
            # Threshold mask to binary result
            mask[mask >= thresh] = 1
            mask[mask < thresh] = 0
            mask = mask.astype(bool)

            # Slot the mask values into the combined
            # mask with a unique value
            mask_combined[mask] = 1

    mask_combined = label(mask_combined)

    return mask_combined.astype(np.uint8)


def get_palette(num_colours=8):
    if num_colours > 255:
        raise ValueError("Palette size is too large")

    # Generate an initial seed to generate palette values with
    palette_seed = torch.tensor([2 ** 25 - 1, 2 ** 20 - 1, 2 ** 22 - 1])
    # Create palette colours in an RGB range
    # Only 255 truly unique palette colours are possible
    # using this method.
    palette = (torch.arange(0, num_colours).unsqueeze(-1) * palette_seed) % 255
    palette = palette.numpy().astype("uint8")

    return palette


def mask_dataset_collate(batch):
    return tuple(zip(*batch))
