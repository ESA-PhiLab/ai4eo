import gzip
import random

import numpy as np
import pandas as pd
from keras.utils import Sequence
from skimage.filters import gaussian
from skimage.io import imread
from skimage.transform import resize, rotate
from sklearn.preprocessing import label_binarize


__all__ = ['InputTargetSequence']


class InputTargetSequence(Sequence):
    def __init__(
            self, inputs, targets, classes, batch_size=None, image_size=None, normalize=False, augmentation=True,
            zoom=False, brightness_factors=None, input_postprocess=None, input_channels=None, target_postprocess=None,
            target_type=None, blur=False, sunset=False, shuffle=True
    ):
        """
        Args:
            target_type: Can be *standard* (default), *one-hot* or *image*.

        """

        self.inputs = inputs.values if isinstance(inputs, pd.Series) else inputs
        self.targets = targets.values if isinstance(targets, pd.Series) else targets

        self.batch_size = batch_size
        if self.batch_size is None:
            self.batch_size = 32

        self.shuffle = shuffle
        self._current_index = 0

        self.image_size = image_size
        self.normalize = normalize
        self.augmentation = augmentation
        self.zoom = zoom
        self.input_postprocess = input_postprocess or self._dummy
        self.input_channels = input_channels
        self.target_postprocess = target_postprocess or self._dummy
        self.blur = blur
        self.sunset = sunset

        self.classes = classes
        self.target_type = target_type
        if self.target_type is None:
            self.target_type = 'one-hot'
        self.brightness_factors = brightness_factors

        counts = dict(zip(*np.unique(self.targets, return_counts=True)))
        self.class_weights = {
            i: max(counts.values()) / counts[classes[i]]
            for i in range(len(classes))
        }

    def __len__(self):
        return len(self.inputs) // self.batch_size

    def __getitem__(self, idx=None):
        return self.get_samples(self.batch_size)

    @staticmethod
    def _dummy(x):
        return x

    @staticmethod
    def read(filename, image_size=None, normalize=False):
        if filename.endswith('.npy'):
            image = np.load(filename, allow_pickle=False)
        elif filename.endswith('.npy.gz'):
            with gzip.open(filename, 'r') as f:
                image = np.load(f, allow_pickle=False)
        else:
            image = imread(filename)

        if normalize:
            image = image / 255

        if image_size is not None:
            image = resize(image, image_size, anti_aliasing=True, preserve_range=True)

        return image

    def get_samples(self, n):

        if self.shuffle:
            sample_ids = random.sample(range(len(self.inputs)), n)
        else:
            sample_ids = [
                idx if idx < len(self.inputs) else idx % len(self.inputs)
                for idx in range(self._current_index + 1, self._current_index + n + 1)
            ]
            self._current_index = sample_ids[-1]

        samples = [self.get(idx) for idx in sample_ids]

        inputs, targets = zip(*samples)
        return np.array(inputs), np.array(targets)

    def get(self, idx):
        # Trailing underscore due to python's builtin 'input' function
        input_ = self.read(self.inputs[idx], self.image_size, self.normalize)

        target = self.targets[idx]
        if self.target_type == 'image':
            target = self.read(target, self.image_size)
        elif self.target_type == 'one-hot':
            target = label_binarize([target], classes=self.classes)[0]

        if self.augmentation:
            # Create a random transformation
            transformation = {
                "rotation": random.randint(-180, 179),
                "fliplr": random.sample([True, False], 1),
            }

            input_ = self.transform(input_, transformation)
            if self.target_type == 'image':
                # We do not want to change the values of the target mask
                target = self.transform(target, transformation, target=True)

        return self.input_postprocess(input_), self.target_postprocess(target)

    def transform(self, image, transformation=None, target=False):
        """Apply transformations to the image

        Args:
            image:
            transformation:
            target: If true, no transformations involving value changes are performed
                (brightness, blur, etc.)
        """
        if transformation is None:
            transformation = {}

        # Try to keep the original data type of the image. It will be
        # probably changed during the transformations
        dtype = image.dtype
        image = image.astype(float)

        # Rotate the image
        if transformation.get("rotation", None) is not None:
            image = rotate(
                image, transformation['rotation'], mode='symmetric',
                preserve_range=True
            )

        if not target:
            # Blur the image
            if self.blur:
                sigma = random.uniform(0, float(self.blur))
                image = gaussian(image, sigma=sigma, multichannel=True)

            # Add a sunset filter sometimes
            if self.sunset and random.random() < 0.25:
                image[:, :, 0] *= 1.1
                image[:, :, 2] *= 0.9

                image = np.clip(image, 0, 255)

            # Change brightness:
            if self.brightness_factors is not None:
                image *= random.uniform(*self.brightness_factors)

                image = np.clip(image, 0, 255)

        # Zoom and shift a little bit?
        # if self.zoom:
        #    factor = 1.1
        #    original_size = image.shape
        #    x_size, y_size = int(image.shape[1]*factor), int(image.shape[0]*factor)

        #    image = resize(
        #        image, (x_size, y_size),
        #        anti_aliasing=True, preserve_range=True
        #    )

        #    x_shift = random.randint(0, image.shape[1] - original_size[1])
        #    y_shift = random.randint(0, image.shape[0] - original_size[0])
        #    x0 = x_shift
        #    y0 = y_shift
        #    x1 = original_size[1] + x_shift
        #    y1 = original_size[0] + y_shift

        #    image = image[y0:y1, x0:x1, :]

        # Flip the image
        if transformation.get("fliplr", False):
            image = np.fliplr(image)

        return image.astype(dtype)