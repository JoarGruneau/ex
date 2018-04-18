# tf_unet is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# tf_unet is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with tf_unet.  If not, see <http://www.gnu.org/licenses/>.


'''
Created on Aug 10, 2016

author: jakeret
'''
from __future__ import print_function, division, absolute_import, unicode_literals
import numpy as np
from PIL import Image
import cv2
import operator

def plot_prediction(x_test, y_test, prediction, save=False):
    import matplotlib
    import matplotlib.pyplot as plt
    
    test_size = x_test.shape[0]
    fig, ax = plt.subplots(test_size, 3, figsize=(12,12), sharey=True, sharex=True)
    
    x_test = crop_to_shape(x_test, prediction.shape)
    y_test = crop_to_shape(y_test, prediction.shape)
    
    ax = np.atleast_2d(ax)
    for i in range(test_size):
        cax = ax[i, 0].imshow(x_test[i])
        plt.colorbar(cax, ax=ax[i,0])
        cax = ax[i, 1].imshow(y_test[i, ..., 1])
        plt.colorbar(cax, ax=ax[i,1])
        pred = prediction[i, ..., 1]
        pred -= np.amin(pred)
        pred /= np.amax(pred)
        cax = ax[i, 2].imshow(pred)
        plt.colorbar(cax, ax=ax[i,2])
        if i==0:
            ax[i, 0].set_title("x")
            ax[i, 1].set_title("y")
            ax[i, 2].set_title("pred")
    fig.tight_layout()
    
    if save:
        fig.savefig(save)
    else:
        fig.show()
        plt.show()

def to_rgb(img):
    """
    Converts the given array into a RGB image. If the number of channels is not
    3 the array is tiled such that it has 3 channels. Finally, the values are
    rescaled to [0,255) 
    
    :param img: the array to convert [nx, ny, channels]
    
    :returns img: the rgb image [nx, ny, 3]
    """
    img = np.atleast_3d(img)
    channels = img.shape[2]
    if channels < 3:
        img = np.tile(img, 3)
    
    img[np.isnan(img)] = 0
    # img -= np.amin(img)
    # img /= np.amax(img)
    img *= 255
    return img

def rgb_multi_prediction(img):
    mapping = {0: [255, 255, 255], 1: [0, 0, 255], 2: [0, 255, 255], 3: [0, 255, 0], 4: [255, 255, 0], 5: [255, 0, 0]}
    shape=img.shape
    out= np.zeros((shape[1], shape[2], 3))
    arg_max = np.argmax(img, axis=3)
    arg_max = arg_max.reshape((img.shape[1], img.shape[2]))
    print (arg_max.shape)
    for x in range(shape[1]):
        for y in range(shape[2]):
            out[x,y,:]=mapping[arg_max[x,y]]
    return out

def crop_to_shape(data, shape):
    """
    Crops the array to the given image shape by removing the border (expects a tensor of shape [batches, nx, ny, channels].
    
    :param data: the array to crop
    :param shape: the target shape
    """
    offset0 = (data.shape[1] - shape[1])//2
    offset1 = (data.shape[2] - shape[2])//2
    return data[:, offset0:(-offset0), offset1:(-offset1)]

def combine_img_prediction(data, gt, pred, crop=False):
    """
    Combines the data, grouth thruth and the prediction into one rgb image
    
    :param data: the data tensor
    :param gt: the ground thruth tensor
    :param pred: the prediction tensor
    
    :returns img: the concatenated rgb image 
    """
    ny = pred.shape[2]
    ch = data.shape[3]
    if crop:
        img = np.concatenate((to_rgb(crop_to_shape(data, pred.shape).reshape(-1, ny, ch)),
                          to_rgb(crop_to_shape(gt[..., 1], pred.shape).reshape(-1, ny, 1)), 
                          to_rgb(pred[..., 1].reshape(-1, ny, 1))), axis=1)
    else:
        img = np.concatenate((to_rgb(data.reshape(-1, ny, ch)),
                          to_rgb(gt[..., 1].reshape(-1, ny, 1)),
                          to_rgb(pred[..., 1].reshape(-1, ny, 1))), axis=1)
    return img

def save_image(img, path):
    """
    Writes the image to disk
    
    :param img: the rgb image to save
    :param path: the target path
    """
    Image.fromarray(img.round().astype(np.uint8)).save(path, 'JPEG', dpi=[300,300], quality=90)


def compute_occurrences(n_labels, component_image):
    occurrences = [0] * (n_labels - 1)

    for x in range(component_image.shape[0]):
        for y in range(component_image.shape[1]):
            if component_image[x, y] != 0:
                occurrences[component_image[x, y] - 1] += 1
    return occurrences


def filter_image(image, filter_size):
    n_labels, component_image = cv2.connectedComponents(image.astype(dtype=np.uint8), 4)
    occurrences = compute_occurrences(n_labels, component_image)
    remove_lables = [True if occurrence < filter_size else False for occurrence in occurrences]

    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            if component_image[x, y] != 0 and remove_lables[component_image[x, y] - 1]:
                image[x, y] = 0
    return image


def combine(ground_truth, prediction):
    blue_channel = np.zeros(prediction.shape, dtype=np.float32)
    red_channel = np.abs(prediction - ground_truth)
    return np.stack([red_channel, ground_truth, blue_channel], axis=2)


def calculate_f1_score(ground_truth, prediction):
    n_found = 0
    n_ground_labels, ground_component_image = cv2.connectedComponents(ground_truth.astype(dtype=np.uint8), 4)
    n_pred_labels, pred_component_image = cv2.connectedComponents(prediction.astype(dtype=np.uint8), 4)
    ground_occurrences = compute_occurrences(n_ground_labels, ground_component_image)
    pred_occurrences = compute_occurrences(n_pred_labels, pred_component_image)
    label_pixels = [[] for _ in range(n_ground_labels - 1)]
    print(len(label_pixels))
    print(n_ground_labels)
    print(np.amax(ground_component_image))
    for x in range(ground_truth.shape[0]):
        for y in range(ground_truth.shape[1]):
            if ground_component_image[x, y] != 0:
                label_pixels[ground_component_image[x, y] - 1].append([x, y])

    for label in range(n_ground_labels - 1):
        intersections = {}
        for x, y in label_pixels[label]:
            corresponding_label = pred_component_image[x, y]
            if corresponding_label != 0:
                if not intersections.get(corresponding_label):
                    intersections[corresponding_label] = 1
                else:
                    intersections[corresponding_label] += 1
        if intersections:
            pred_label, intersect = max(intersections.items(), key=operator.itemgetter(1))
            if float(intersect) / (ground_occurrences[label - 1] + pred_occurrences[pred_label - 1] - intersect) >= 0.5:
                n_found += 1
    tp = n_found
    fp = n_pred_labels - 1 - n_found
    fn = n_ground_labels - 1 - n_found
    with np.errstate(divide='ignore', invalid='ignore'):
        precision = np.divide(n_found, n_pred_labels - 1)
        recall = np.divide(n_found, n_ground_labels - 1)
        f1_score = np.divide(2 * precision * recall, precision + recall)

    return [n_ground_labels - 1, tp, fp, fn, precision, recall, f1_score]

