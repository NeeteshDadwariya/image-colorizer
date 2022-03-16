import numpy as np
import torch
from PIL.Image import Image
from skimage import io
from skimage.color import lab2rgb
from torch.autograd import Variable

from Transforms import input_shape_transform
from basic_model import ColorNet
from train import best_model_path

use_gpu = torch.cuda.is_available()

class Inference:

    def predict(self, gray_image_path):
        # Loading image as tensor
        gray_img = io.imread(gray_image_path)
        gray_tensor = input_shape_transform(gray_img)

        # Rebuilding model from params
        model = ColorNet()
        model.load_state_dict(torch.load(best_model_path))
        model.eval()

        input_gray_variable, input_ab_variable = None, None
        # Use GPU if available
        with torch.no_grad():
            input_gray_variable = Variable(gray_tensor).cuda() if use_gpu else Variable(gray_tensor)

        # Run forward pass
        output_ab = model(input_gray_variable)

        output_ab = output_ab.cpu()
        gray_tensor = gray_tensor.cpu()

        color_image = torch.cat((gray_tensor, output_ab), 0).numpy()
        color_image = color_image.transpose((1, 2, 0))
        color_image[:, :, 0:1] = color_image[:, :, 0:1] * 100
        color_image[:, :, 1:3] = color_image[:, :, 1:3] * 255 - 128
        color_image = lab2rgb(color_image.astype(np.float64))

        pil_image = Image.fromarray(color_image.astype('uint8'), 'RGB')
        pil_image.show()

        return color_image
