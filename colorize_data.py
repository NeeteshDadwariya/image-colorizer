import glob
from typing import Tuple

import PIL.Image
import numpy as np
import torch
import torchvision.transforms as T
from skimage import io
from skimage.color import rgb2lab, rgb2gray
from torch.utils.data import Dataset

import Transforms
from utils import image_to_tensor, tensor_to_image


class ColorizeData(Dataset):
    def __init__(self, root, split='train'):
        self.input_shape_transform = Transforms.input_shape_transform

        # Data Augmentation for training
        if split == 'train':
            self.input_shape_transform = T.Compose([self.input_shape_transform, T.RandomHorizontalFlip()])

        self.root = root
        file_list = glob.glob(self.root + "*")
        print(file_list)
        self.paths = []
        for class_path in file_list:
            for img_path in glob.glob(class_path + "/*.jpg"):
                self.paths.append(img_path)
        print(self.paths)

    def __len__(self) -> int:
        # return Length of dataset
        return len(self.paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Return the input tensor and output tensor for training
        path = self.paths[index]

        img_rgb = PIL.Image.open(path).convert('RGB')
        img_original = self.input_shape_transform(img_rgb)
        img_original = tensor_to_image(img_original)
        img_lab = rgb2lab(img_original)
        img_ab = img_lab[:, :, 1:3]
        img_ab = torch.from_numpy(img_ab.transpose((2, 0, 1))).float()
        gray_image = rgb2gray(img_original)
        gray_image = torch.from_numpy(gray_image).unsqueeze(0).float()

        return gray_image, img_ab
