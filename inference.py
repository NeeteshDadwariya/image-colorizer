import PIL
import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from skimage import io
from skimage.color import lab2rgb, rgb2lab, rgb2gray
from torch.autograd import Variable

import Transforms
from Transforms import input_shape_transform
from basic_model import ColorNet
from train import best_model_path
from utils import tensor_to_image

use_gpu = torch.cuda.is_available()

class Inference:

    def predict(self, gray_image_path):
        # Loading image as tensor
        img_rgb = PIL.Image.open(gray_image_path).convert('RGB')
        img_original = Transforms.input_shape_transform(img_rgb)
        img_original = tensor_to_image(img_original)
        gray_image = rgb2gray(img_original).reshape((1, 256, 256))
        gray_image = torch.from_numpy(gray_image).unsqueeze(0).float()

        # Rebuilding model from params
        model = ColorNet()
        model.load_state_dict(torch.load(best_model_path))
        model.eval()

        # Use GPU if available
        with torch.no_grad():
            input_gray_variable = Variable(gray_image).cuda() if use_gpu else Variable(gray_image)

            # Run forward pass
            output_ab = model(input_gray_variable)

            output_ab = output_ab.cpu().squeeze(0)
            gray_image = gray_image.cpu().squeeze(0)

            color_image = torch.cat((gray_image, output_ab), 0)
            color_image = color_image.numpy()
            color_image = color_image.transpose((1, 2, 0))
            color_image[:, :, 0:1] = color_image[:, :, 0:1] * 100
            color_image = lab2rgb(color_image.astype(np.float64))

            plt.imsave(arr=color_image, fname='./data/inference_output.jpeg')

        return color_image
