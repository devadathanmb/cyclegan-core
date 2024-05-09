import glob
import os

import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


def convert_to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image


class ImageDataset(Dataset):
    def __init__(self, img_path, transforms_=None, unaligned=False, mode="test"):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        if not os.path.exists(img_path):
            raise Exception(f"{img_path} is not a valid directory.")

        self.files = sorted(glob.glob(img_path))
        print(self.files)

    def __getitem__(self, index):
        image = Image.open(self.files[index % len(self.files)])

        if image.mode != "RGB":
            image = convert_to_rgb(image)

        item = self.transform(image)

        return {
            "img": item,
        }

    def __len__(self):
        return len(self.files)
