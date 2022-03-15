from typing import Tuple


import torchvision.transforms as T
import torch
import numpy as np
import os
import glob
import torchvision.transforms.functional as TF

from skimage import io
from skimage.color import rgb2lab, rgb2gray
from torch.utils.data import Dataset
from utils import image_to_tensor, tensor_to_image

class ColorizeData(Dataset):
    def __init__(self, root, split='train'):
        self.input_shape_transform = T.Compose([T.ToTensor(),
                                                T.Resize(size=(256, 256)),
                                                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        # Data Augmentation for training
        if split == 'train':
            self.input_shape_transform = T.Compose([self.input_shape_transform, T.RandomHorizontalFlip()])

        #To convert images to grayscale
        self.gray_transform = T.Compose([self.input_shape_transform, T.Grayscale])

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
        img_rgb = io.imread(path)
        shaped_tensor = self.input_shape_transform(img_rgb)
        reshaped_image = tensor_to_image(shaped_tensor)

        img_lab = rgb2lab(reshaped_image)
        img_ab = img_lab[:, :, 1:3]

        gray_image = rgb2gray(reshaped_image)
        return image_to_tensor(gray_image), image_to_tensor(img_ab)
