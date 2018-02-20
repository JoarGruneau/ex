import tensorflow as tf
import numpy as np
import os
import glob
from PIL import Image
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
base_dir = os.path.dirname(os.path.abspath(__file__))


class InputPipeline():
    # creates an put pipleine from data stored in file
    def __init__(self, x_path, y_path, x_dim, y_dim):
        self.x_path = x_path
        self.y_path = y_path
        self.x_dim = x_dim
        self.y_dim = y_dim

    def _get_paths(self):
        x_names = sorted(glob.glob(os.path.join(self.x_path, '*.png')))[1:100]
        x_identifiers = [os.path.basename(x_name.split('_')[0]) for x_name in x_names]
        y_names = [sorted(glob.glob(os.path.join(self.y_path, x_id + '*.png'))) for x_id in x_identifiers]
        return x_names, y_names

    def _parse(self, x_name, y_name):
        image_string = tf.read_file(x_name)
        image_decoded = tf.image.decode_png(image_string, channels=3)
        image_decoded.set_shape(self.x_dim)
        masks = []

        for mask_name in y_name:
            image_string = tf.read_file(mask_name)
            decoded_mask = tf.image.decode_png(image_string, channels=1)
            decoded_mask.set_shape(self.y_dim)
            masks.append(decoded_mask)
            # image_decoded.set_shape([None, None])
            # masks.append(tf.image.resize_images(image_decoded, y_dims))

        return image_decoded, tf.stack(masks, axis=3)

    def get_batches(self):
        x_names, y_names = self._get_paths()
        dataset = tf.data.Dataset.from_tensor_slices((x_names, y_names))
        dataset = dataset.map(
            lambda x_name, y_name: tuple(tf.py_func(
                self._parse, [x_name, y_name], [tf.uint8, tf.uint8])))

        dataset = dataset.shuffle(buffer_size=10000)
        dataset = dataset.batch(32)
        dataset = dataset.repeat(5)
        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()


if __name__ == "__main__":
    image_dir = os.path.join(base_dir, 'images')
    mask_dir = os.path.join(base_dir, 'ground_truth_masks')
    pipe = InputPipeline(image_dir, mask_dir, (1024, 1024, 3), (1024, 1024, 1))
    x, y = pipe.get_batches()
    print(x.shape)