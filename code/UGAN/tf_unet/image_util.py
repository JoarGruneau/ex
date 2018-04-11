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
author: jakeret
'''
from __future__ import print_function, division, absolute_import, unicode_literals
from tf_unet import util
import cv2
import glob
import numpy as np
import os
import pickle
import shutil
from PIL import Image
from random import randint
mapping={0:[255, 255, 255], 1:[0, 0, 255], 2:[0, 255, 255], 3:[0, 255, 0], 4:[255, 255, 0], 5:[255, 0, 0]}
inv_mapping={str(v): k for k, v in mapping.items()}


class BaseDataProvider(object):
    """
    Abstract base class for DataProvider implementation. Subclasses have to
    overwrite the `_next_data` method that load the next data and label array.
    This implementation automatically clips the data with the given min/max and
    normalizes the values to (0,1]. To change this behavoir the `_process_data`
    method can be overwritten. To enable some post processing such as data
    augmentation the `_post_process` method can be overwritten.

    :param a_min: (optional) min value used for clipping
    :param a_max: (optional) max value used for clipping

    """
    
    # channels = 1
    # n_class = 2




    def __init__(self, channels, n_class, border_size=0, a_min=None, a_max=None):
        self.channels = channels
        self.n_class = n_class
        self.border_size = border_size
        self.a_min = a_min if a_min is not None else -np.inf
        self.a_max = a_max if a_min is not None else np.inf

    def _load_data_and_label(self):
        data, label = self._next_data()
            
        train_data = self._process_data(data)
        labels = self._process_labels(label)
        
        train_data, labels = self._post_process(train_data, labels)
        # nx = train_data.shape[0]+2*self.border_size
        # ny = train_data.shape[1]+2*self.border_size
        # border_train = np.zeros((nx, ny, self.channels))
        # border_label = np.zeros((nx, ny, self.n_class))
        # border_train[self.border_size:-self.border_size, self.border_size:-self.border_size, :] = train_data
        # border_label[self.border_size:-self.border_size, self.border_size:-self.border_size, :] = labels
        nx = train_data.shape[0]
        ny = train_data.shape[1]
        # print (type(train_data))
        # train_data = cv2.copyMakeBorder(train_data, top=self.border_size, bottom=self.border_size, left=self.border_size, right=self.border_size,
        #                    borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
        # train_data = cv2.copyMakeBorder(train_data, top=self.border_size, bottom=self.border_size,
        #                                 left=self.border_size, right=self.border_size,
        #                                 borderType=cv2.BORDER_CONSTANT, value=[0]*self.channels)
        # labels = cv2.copyMakeBorder(labels, top=self.border_size, bottom=self.border_size, left=self.border_size,
        #                             right=self.border_size,
        #                             borderType=cv2.BORDER_CONSTANT, value=[0]*self.n_class)

        return train_data.reshape(1, nx, ny, self.channels), \
               labels.reshape(1, label.shape[0], label.shape[1], self.n_class)
    
    def _process_labels(self, label):
        nx = label.shape[0]
        ny = label.shape[1]
        labels = np.zeros((nx, ny, self.n_class), dtype=np.float32)
        if self.n_class == 1:
            label = label/255
            labels[..., 0] = label
        elif self.n_class == 2:
            label = label/255
            labels[..., 1] = label
            labels[..., 0] = 1.0 - label
            # print("conversion done")
            # print(np.amax(labels))
            # print(np.amin(labels))
        else:
            for x in range(nx):
                for y in range(ny):
                    pixel = list(label[x,y,:])
                    labels[x,y,inv_mapping[str(pixel)]]=1.0
        return labels

    def _process_weights(selfself, weights):
        return weights/255
    
    def _process_data(self, data):
        # normalization
        data = np.clip(np.fabs(data), self.a_min, self.a_max)
        data -= np.amin(data)
        data /= np.amax(data)
        return data
    
    def _post_process(self, data, labels):
        """
        Post processing hook that can be used for data augmentation
        
        :param data: the data array
        :param labels: the label array
        """
        return data, labels
    
    def __call__(self, n):
        train_data, labels = self._load_data_and_label()
        nx = train_data.shape[1]
        ny = labels.shape[2]

        X = np.zeros((n, train_data.shape[1], train_data.shape[2], self.channels))
        Y = np.zeros((n, labels.shape[1], labels.shape[2], self.n_class))

        X[0] = train_data
        Y[0] = labels


        for i in range(1, n):
            train_data, labels = self._load_data_and_label()
            X[i] = train_data
            Y[i] = labels
    
        return X, Y
    
class SimpleDataProvider(BaseDataProvider):
    """
    A simple data provider for numpy arrays. 
    Assumes that the data and label are numpy array with the dimensions
    data `[n, X, Y, channels]`, label `[n, X, Y, classes]`. Where
    `n` is the number of images, `X`, `Y` the size of the image.

    :param data: data numpy array. Shape=[n, X, Y, channels]
    :param label: label numpy array. Shape=[n, X, Y, classes]
    :param a_min: (optional) min value used for clipping
    :param a_max: (optional) max value used for clipping
    :param channels: (optional) number of channels, default=1
    :param n_class: (optional) number of classes, default=2
    
    """
    
    def __init__(self, data, label, a_min=None, a_max=None, channels=1, n_class = 2):
        super(SimpleDataProvider, self).__init__(a_min, a_max)
        self.data = data
        self.label = label
        self.file_count = data.shape[0]
        self.n_class = n_class
        self.channels = channels

    def _next_data(self):
        idx = np.random.choice(self.file_count)
        return self.data[idx], self.label[idx]


class ImageDataProvider(BaseDataProvider):
    """
    Generic data provider for images, supports gray scale and colored images.
    Assumes that the data images and label images are stored in the same folder
    and that the labels have a different file suffix 
    e.g. 'train/fish_1.tif' and 'train/fish_1_mask.tif'

    Usage:
    data_provider = ImageDataProvider("..fishes/train/*.tif")
        
    :param search_path: a glob search pattern to find all data and label images
    :param a_min: (optional) min value used for clipping
    :param a_max: (optional) max value used for clipping
    :param data_suffix: suffix pattern for the data images. Default '.tif'
    :param mask_suffix: suffix pattern for the label images. Default '_mask.tif'
    :param shuffle_data: if the order of the loaded file path should be randomized. Default 'True'
    :param channels: (optional) number of channels, default=1
    :param n_class: (optional) number of classes, default=2
    
    """
    
    def __init__(self, image_path, label_path=None, patch_size=1000, border_size=0,
                 a_min=None, a_max=None, data_suffix=".tif", mask_suffix='_mask.tif', weight_suffix='_weight.tif',
                 shuffle_data=True, channels=1, n_class = 2, load_saved=False):
        super(ImageDataProvider, self).__init__(channels, n_class, border_size, a_min, a_max)
        self.data_suffix = data_suffix
        self.mask_suffix = mask_suffix
        self.weight_suffix = weight_suffix
        self.file_idx = -1
        self.patch_size=patch_size
        self.load_saved = load_saved

        self.shuffle_data = shuffle_data
        
        self.data_files = self._find_data_files(image_path)
        self.label_path = label_path
        
        if self.shuffle_data:
            np.random.shuffle(self.data_files)
        
        # assert len(self.data_files) > 0, "No training files"
        # print("Number of files used: %s" % len(self.data_files))
        img = self._load_file(self.data_files[0], type=-1, add_borders=True, dtype=np.float32)
        # # print(np.amax(img))
        # # print (np.amin(img))
        # self.channels = 1 if len(img.shape) == 2 else img.shape[-1]
        self.size = img.shape[0]

    def agument(self, image, mirror, rotate, swap_channels):
        if mirror == 1:
            image = np.fliplr(image)
        elif mirror == 2:
            image = np.flipud(image)

        if rotate > 0:
            image = np.rot90(image, rotate)
        if swap_channels:
            image =  cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image

    def get_patches(self, batch_size=1, get_coordinates=False):
        self._cylce_file()
        image_name = self.data_files[self.file_idx]
        if self.load_saved:
            patches = pickle.load(open(image_name,'rb'))
        else:
            base_name = os.path.join(self.label_path, os.path.basename(image_name))
            image = self._load_file(image_name, type=-1, add_borders=True, dtype=np.float32)
            label = self._load_file(base_name.replace(self.data_suffix, self.mask_suffix),
                                    type=0, add_borders=False, dtype=np.uint8)
            weights = self._load_file(base_name.replace(self.data_suffix, self.weight_suffix),
                                    type=0, add_borders=False, dtype=np.uint8)

            if self.shuffle_data:
                mirror = randint(0, 2)
                rotate = randint(0, 3)
                swap_channel = randint(0, 1)

                image = self.agument(image, mirror, rotate, swap_channel)
                label = self.agument(label, mirror, rotate, 0)
                weights = self.agument(weights, mirror, rotate, 0)

            patches = self._get_patches(image,label, weights, get_coordinates)
        return patches

    def save_patches(self, save_path):
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        os.makedirs(os.path.dirname(save_path))
        for image_name in self.data_files:
            save_name = os.path.basename(image_name.replace(self.data_suffix, '.pkl'))
            print (save_name)
            label_name = os.path.join(self.label_path, os.path.basename(image_name)
                    .replace(self.data_suffix, self.mask_suffix))
            patches = self._get_patches(
                self._load_file(image_name, type=-1, add_borders=True, dtype=np.float32),
                self._load_file(label_name, type=0, add_borders=False, dtype=np.uint8), get_coordinates=True)
            pickle.dump(patches, open(os.path.join(save_path, save_name), 'wb'))
        print ('all patches saved')

    def get_border_size(self):
        return  self.border_size

    def get_patch_size(self):
        return  self.patch_size

    def get_input_size(self):
        return self.size - 2 * self.border_size

    def _find_data_files(self, search_path):
        if self.load_saved:
            return glob.glob(search_path+'*.pkl')
        else:
            all_files = glob.glob(search_path)
            return [name for name in all_files if self.data_suffix in name and not self.mask_suffix in name]
    
    
    def _load_file(self, path, type=-1, add_borders=False, dtype=np.float32):
        img = cv2.imread(path, type)
        if type == -1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if add_borders:
            shape=list(img.shape)
            shape[0] += 2*self.border_size
            shape[1] += 2*self.border_size
            border_img = np.zeros(shape)
            border_img[self.border_size:-self.border_size, self.border_size:-self.border_size, ...] = img.copy()

            left = np.fliplr(img[:, 0:self.border_size].copy())
            right = np.fliplr(img[:, -self.border_size:].copy())
            border_img[self.border_size:-self.border_size, :self.border_size]=left
            border_img[self.border_size:-self.border_size, -self.border_size:]=right

            up = np.flipud(border_img[self.border_size:2*self.border_size, :].copy())
            down = np.flipud(border_img[-2*self.border_size:-self.border_size, :].copy())

            border_img[0:self.border_size, :] = up
            border_img[-self.border_size:, :] = down
            img=border_img
            # img.astype(dtype=np.uint8)
            # img_p = Image.fromarray(img, 'RGB')
            # img_p.save('my.png')
            # util.save_image(img, 'test')

        img.astype(dtype=dtype)
        return img

    def _cylce_file(self):
        self.file_idx += 1
        if self.file_idx >= len(self.data_files):
            self.file_idx = 0 
            if self.shuffle_data:
                np.random.shuffle(self.data_files)

    def _get_patches(self, img, label, weights, get_coordinates=False):
        patches = []
        for x in range(self.border_size, self.size-self.border_size, self.patch_size):
            for y in range(self.border_size, self.size-self.border_size, self.patch_size):

                patch_img=img[x-self.border_size:x + self.patch_size + self.border_size,
                                y-self.border_size:y + self.patch_size + self.border_size, ...]
                patch_label=label[x-self.border_size:x + self.patch_size - self.border_size,
                                     y-self.border_size:y + self.patch_size -self.border_size, ...]
                patch_weights= weights[x-self.border_size:x + self.patch_size - self.border_size,
                                     y-self.border_size:y + self.patch_size -self.border_size, ...]

                patch_img = self._process_data(patch_img)
                patch_label = self._process_labels(patch_label)
                patch_weights =self._process_weights(weights)
                patch_img = patch_img.reshape(1, patch_img.shape[0], patch_img.shape[1], self.channels)
                patch_label = patch_label.reshape(1, patch_label.shape[0], patch_label.shape[1], self.n_class)
                patch_weights = patch_weights.reshape(1, patch_weights.shape[0], patch_weights.shape[1])
                if get_coordinates:
                    patches.append([patch_img, patch_label, patch_weights, [x-self.border_size, y-self.border_size]])
                else:
                    patches.append([patch_img, patch_label, patch_weights])
        return patches

