from torch.utils.data.dataset import Dataset
from torchvision.transforms.functional import to_tensor
from PIL import Image

from ..utils import get_image_paths


class BasicDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # Get all image paths from root
        self.img_paths = get_image_paths(root_dir)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path)

        if self.transform is not None:
            img = self.transform(img)
        else:
            img = to_tensor(img)

        return img, img_path
