"""
Finds matching images between two folders of images.
Matches are warped such that the difference between
them is as small as possible.

This works best for two folders full of images that
might contain extras or one folder has images that
are cropped/transformed slightly.
"""


from os.path import join, isdir
import argparse

from skimage.metrics import structural_similarity as ssim
import numpy as np
from tqdm import tqdm

from src.utils import get_image_paths, load_all_imgs, warp_to_match
from src.utils import create_img_diff, make_missing_dirs


parser = argparse.ArgumentParser()
parser.add_argument('--a_root', required=True, type=str,
                    help="""Path to the root folder containing images
                            for matching.""")
parser.add_argument('--b_root', required=True, type=str,
                    help="""Path to the root folder containing alternate
                            images that roughly match the A folder.""")
parser.add_argument('--dst_root', required=True, type=str,
                    help="""Path to the output folder. This folder does
                            not necessarily need to exist.""")
parser.add_argument('--ssim_thresh',
                    required=False, default=0.3, type=float,
                    help="""SSIM threshold for initially finding matching
                            pages in the A and B folders. Pages that don't
                            find a match with at least the threshold are
                            not processed in further steps of this script.
                            SSIM values should range from 0.0 - 1.0""")
parser.add_argument('--match_thresh',
                    required=False, default=1000, type=int,
                    help="""The threshold for how many matching keypoints
                            are needed before performing a homographical
                            transformation to match A and B images. Good
                            values range from 2000 - 6000. A poor match
                            ranges from 1000 - 2000.""")
parser.add_argument('--final_ssim_thresh',
                    required=False, default=0.65, type=float,
                    help="""SSIM threshold for performing the final
                            filtering of pages. Image should match
                            fairly well at this stage and, thus,
                            should exhibit a high SSIM (typically
                            greater than 0.5)""")
parser.add_argument('--difference_thresh',
                    required=False, default=15000, type=int,
                    help="""Threshold for how many white pixels are
                            in a difference image. For images 800 x 533,
                            15,000 pixels is typically a good upper
                            limit.""")
parser.add_argument('--output_height',
                    required=False, default=800, type=int,
                    help="""Height of the output difference image.""")
parser.add_argument('--output_width',
                    required=False, default=533, type=int,
                    help="""Height of the output difference image.""")
args = parser.parse_args()


if __name__ == "__main__":
    # Get paths to source images
    a_root = args.a_root
    b_root = args.b_root

    # Check image folders exist
    if not isdir(a_root):
        print("No valid path at", a_root)
        exit(1)
    if not isdir(b_root):
        print("No valid path at", b_root)
        exit(1)

    # Get image names
    a_img_paths = get_image_paths(a_root)
    b_img_paths = get_image_paths(b_root)

    # Load images
    a_imgs = load_all_imgs(a_img_paths)
    b_imgs = load_all_imgs(b_img_paths)

    # Match images based on ssim score
    ssim_a = []
    ssim_b = []
    for a_img in tqdm(a_imgs):
        # Check each a image against each b image for a match
        best_score = 0
        best_score_idx = 0
        for idx, b_img in enumerate(b_imgs):
            score = ssim(np.array(a_img.convert('L').resize(
                            (args.output_width, args.output_height))),
                         np.array(b_img.convert('L').resize(
                            (args.output_width, args.output_height))))
            if score > best_score:
                best_score = score
                best_score_idx = idx
        # Threshold images on similarity
        if best_score > args.ssim_thresh:
            # Store a img + b match in respective arrays
            ssim_a.append(a_img)
            ssim_b.append(b_imgs[best_score_idx])

    if len(ssim_a) == 0:
        print(f"No SSIM matches found for all {len(a_imgs)} A image(s)")
        print("Try changing the SSIM threshold")
        exit(1)
    else:
        print(f"Found SSIM matches for {len(ssim_a)} " +
              f"out of {len(a_imgs)} A image(s)")

    # Warp images to match with SIFT/homography
    # Threshold images based upon the number of detected keypoints
    sift_a = []
    sift_b = []
    for i in tqdm(range(len(ssim_a))):
        a_img = ssim_a[i]
        b_img = ssim_b[i]
        # Solid matches range from 2000 - 6000
        # Poor matches range from 1000 - 2000
        match_threshold = args.match_thresh
        try:
            b_img_warped = warp_to_match(
                np.array(b_img.convert('L')),
                np.array(a_img.convert('L')),
                min_match_count=match_threshold
            )
        except Exception as e:
            print("Error when processing", str(e))
            continue

        # Store a img and warped b img on good match
        if b_img_warped:
            sift_a.append(a_img)
            sift_b.append(b_img_warped)
        # Otherwise, store a img and unwarped b img
        else:
            sift_a.append(a_img)
            sift_b.append(b_img)

    if len(sift_a) == 0:
        print(f"No SIFT matches found for all {len(ssim_a)} A image(s)")
        print("Try changing the match threshold")
        exit(1)
    else:
        print(f"Found SIFT matches for {len(sift_a)} " +
              f"out of {len(ssim_a)} A image(s)")

    # Create image diffs and threshold based on the number of white pixels in
    # the image difference. Poor matches will contain an order of magnitude
    # more white pixels than good matches.
    # Images are additionally filtered for another round of SSIM score checks.
    # At this stage, A/B images should match fairly well (>0.65)
    a_diffs = []
    a_thresh_imgs = []
    b_diffs = []
    b_thresh_imgs = []
    for i in tqdm(range(len(sift_a))):
        a_img = sift_a[i]
        b_img = sift_b[i]
        score = ssim(np.array(a_img.convert('L').resize(
                        (args.output_width, args.output_height))),
                     np.array(b_img.convert('L').resize(
                        (args.output_width, args.output_height))))

        a_diff = create_img_diff(a_img, b_img)
        b_diff = create_img_diff(b_img, a_img)

        diff_threshold = args.difference_thresh
        temp_diff = np.array(a_diff)
        if (temp_diff[temp_diff == 1].sum() < diff_threshold
                and score > args.final_ssim_thresh):
            a_diffs.append(a_diff)
            b_diffs.append(b_diff)
            a_thresh_imgs.append(a_img)
            b_thresh_imgs.append(b_img)
    sift_a = a_thresh_imgs
    sift_b = b_thresh_imgs

    if len(sift_a) == 0:
        print(f"No difference matches found for all {len(sift_a)} A image(s)")
        print("Try changing the difference threshold or final SSIM threshold")
        exit(1)
    else:
        print(f"Found difference matches for {len(sift_a)} " +
              f"out of {len(ssim_a)} A image(s)")

    # Save all images to the destination, if matched images exist
    if len(a_diffs) > 0 and len(sift_a) > 0:
        # Make/get paths to destination dirs
        a_dst = join(args.dst_root, "a_images")
        make_missing_dirs(a_dst)
        b_dst = join(args.dst_root, "b_images")
        make_missing_dirs(b_dst)
        a_diff_dst = join(a_dst, "Diff")
        make_missing_dirs(a_diff_dst)
        b_diff_dst = join(b_dst, "Diff")
        make_missing_dirs(b_diff_dst)

        print("Saving images...")

        for i in tqdm(range(len(sift_a))):
            sift_a[i].save(join(a_dst, f"{i:0>5}.png"))
            a_diffs[i].save(join(a_diff_dst, f"{i:0>5}.png"))
            sift_b[i].save(join(b_dst, f"{i:0>5}.png"))
            b_diffs[i].save(join(b_diff_dst, f"{i:0>5}.png"))
    else:
        print("No pages matched")
