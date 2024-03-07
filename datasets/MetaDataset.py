from __future__ import print_function, division
from torch.utils.data import Dataset
import json
import numpy as np
from PIL import Image

mode_key = {
    'train': 'TRAIN',
    'val': 'VALID',
    'test': 'TEST'
}

dir_map = {
    'ilsvrc_2012': 'ILSVRC2012_img_train',
    'aircraft': 'fgvc-aircraft-2013b/data/images',
    'cu_birds': 'CUB_200_2011/images',
    'dtd': 'dtd/images',
    'fungi': 'fungi/images',
    'vgg_flower': 'vgg_flower/',
    'quickdraw': 'quickdraw'
}


def load_image(img):
    """Load image img.

    Args:
      img: a 1D numpy array of shape [side**2]

    Returns:
      a PIL Image
    """
    # We make the assumption that the images are square.
    side = int(np.sqrt(img.shape[0]))
    # To load an array as a PIL.Image we must first reshape it to 2D.
    img = Image.fromarray(img.reshape((side, side)))
    img = img.convert('RGB')
    return img


class MetaDataset(Dataset):
    """MetaDataset dataset."""

    def __init__(self, data_path, label_path, mode='train', transform=None, data_type="meta_dataset"):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        root_dir = f'{data_path}/data/'
        processed_file_path = f'{data_path}/processed_data'
        self.base_path = label_path

        self.root_dir = root_dir
        self.processed_file_path = processed_file_path
        self.transform = transform
        self.mode = mode_key[mode]

        if data_type == 'imagenet':
            classes = ['ilsvrc_2012']
            all_paths_file = self.base_path + f"/ilsvrc_2012_paths_{mode}.json"
            all_labels_file = self.base_path + f"/ilsvrc_2012_labels_{mode}.json"
        else:
            classes = ['ilsvrc_2012', 'omniglot', 'aircraft', 'cu_birds', 'dtd', 'quickdraw', 'fungi', 'vgg_flower']
            all_paths_file = self.base_path + f"/all_paths_{mode}.json"
            all_labels_file = self.base_path + f"/all_labels_{mode}.json"

        with open(all_paths_file, 'r') as f:
            self.paths = json.load(f)
        with open(all_labels_file, 'r') as f:
            self.labels = json.load(f)

        self.count = len(self.paths)
        self.num_class = len(np.unique(self.labels))

    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        img_name = self.paths[idx]
        label = self.labels[idx]
        image = Image.open(img_name).convert('RGB')

        if self.transform:
            image = self.transform(image)
        return image, label
