import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
from skimage.color import lab2rgb

plt.switch_backend('agg')

import torch


def save_checkpoint(state, is_best_so_far, filename, best_model_path):
    '''Saves checkpoint, and replace the old best model if the current model is better'''
    torch.save(state, filename)
    if is_best_so_far:
        torch.save(state['state_dict'], best_model_path)

class AverageMeter(object):
    '''An easy way to compute and store both average and current values'''
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def visualize_image(grayscale_input, ab_input=None, show_image=False, save_path=None, save_name=None):
    '''Show or save image given grayscale (and ab color) inputs. Input save_path in the form {'grayscale': '/path/', 'colorized': '/path/'}'''
    plt.clf() # clear matplotlib plot
    ab_input = ab_input.cpu()
    grayscale_input = grayscale_input.cpu()
    if ab_input is None:
        grayscale_input = grayscale_input.squeeze().numpy()
        if save_path is not None and save_name is not None:
            plt.imsave(grayscale_input, '{}.{}'.format(save_path['grayscale'], save_name) , cmap='gray')
        if show_image:
            plt.imshow(grayscale_input, cmap='gray')
            plt.show()
    else:
        color_image = torch.cat((grayscale_input, ab_input), 0).numpy()
        color_image = color_image.transpose((1, 2, 0))
        color_image[:, :, 0:1] = color_image[:, :, 0:1] * 100
        color_image[:, :, 1:3] = color_image[:, :, 1:3] * 255 - 128
        color_image = lab2rgb(color_image.astype(np.float64))
        grayscale_input = grayscale_input.squeeze().numpy()
        if save_path is not None and save_name is not None:
            plt.imsave(arr=grayscale_input, fname='{}{}'.format(save_path['grayscale'],
                                                                 save_name.replace('%', 'gray')), cmap='gray')
            plt.imsave(arr=color_image, fname='{}{}'.format(save_path['colorized'],
                                                            save_name.replace('%', 'color')))
        if show_image:
            f, axarr = plt.subplots(1, 2)
            axarr[0].imshow(grayscale_input, cmap='gray')
            axarr[1].imshow(color_image)
            plt.show()

def image_to_tensor(image):
    if len(image.shape) == 2:
        return torch.from_numpy(image).unsqueeze(0).float()
    return torch.from_numpy(image.transpose((2, 0, 1)))

def tensor_to_image(tensor):
    return np.asarray(tensor.permute(1, 2, 0))

def create_and_get_path(prefix, name, remove=False):
    dir = os.path.join(prefix, name)
    if os.path.exists(dir) and remove:
        shutil.rmtree(dir)
    os.makedirs(dir, exist_ok=True)
    return dir
