"""Provide preprocessing tools like patches extractor and an advanced data
generator for keras.
"""


from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from random import Random

from tensorflow.keras.utils import Sequence
import numpy as np
import pandas as pd
import rasterio as rio
from rasterio.windows import Window
from shapely.geometry import box as Box
from skimage.io import imread
from sklearn.preprocessing import label_binarize
from typhon.files import FileSet

def extract_patches(
    image, polygons, shape, stride=None, threshold=0.5
):
    """Extract patches from a geocoded image

    Args:
        image: Filename of geocoded image or rasterio.Dataset
        polygons: GeoDataFrame with labelled polygons.
        shape: Desired shape of the extracted patches. Is also used to
            calculate the number of extracted patches.
        stride:
        threshold: How many points of the extracted rectangle should be covered
            by the labelled polygon? Must be a float be 0 and 1.
    Yields:
        A tuple of two elements: the index of the polygon and the corresponding
        patch.
    """
    if stride is None:
        stride = shape

    image_bounds = Box(*image.bounds)
    BoundingBox = namedtuple(
        'BoundingBox',
        ['col_start', 'col_end', 'row_start', 'row_end', 'height', 'width']
    )

    # We want to extract as many rectangles from the labelled polygons as possible.
    # We are working with two coordinate systems: the index system (row, column) to
    # extract the pixel values from the image matrix and the geo-referenced
    # coordinates (x,y) to find the patches corresponding to the labelled polygons.
    for idx, polygon in polygons.iterrows():
        if not polygon.geometry.is_valid:
            print('Polygon {} is not valid!'.format(idx))
            continue

        if not polygon.geometry.intersects(image_bounds):
            continue

        # Extract the bounding box of the polygon, so we can divide it easily
        # into sub-rectangles:
        row1, col1 = image.index(polygon.geometry.bounds[0], polygon.geometry.bounds[1])
        row2, col2 = image.index(polygon.geometry.bounds[2], polygon.geometry.bounds[3])
        polygon_bbox = BoundingBox(
            col_start=min([col1, col2]), col_end=max([col1, col2]),
            row_start=min([row1, row2]), row_end=max([row1, row2]),
            height=abs(row2-row1), width=abs(col2-col1)
        )

        for column in range((polygon_bbox.width // stride[0]) + 1):
            for row in range((polygon_bbox.height // stride[1]) + 1):
                # Transform the patch bounding box indices to coordinates to
                # calculate the percentage that is covered by the labelled
                # polygon:
                x1, y1 = image.xy(
                    polygon_bbox.row_start + row*stride[1],
                    polygon_bbox.col_start + column*stride[0]
                )
                x2, y2 = image.xy(
                    polygon_bbox.row_start + row*stride[1] + shape[1],
                    polygon_bbox.col_start + column*stride[0] + shape[1]
                )
                patch_bounds = Box(x1, y1, x2, y2)

                # We check first whether the threshold condition is fullfilled
                # and then we extract the patch from the image:
                overlapping_area = \
                    polygon.geometry.intersection(patch_bounds).area
                if overlapping_area / patch_bounds.area <  threshold:
                    # The labelled polygon covers less than $threshold$ of the
                    # whole patch, i.e. we reject it:
                    continue

                # rasterio returns data in CxHxW format, so we have to transpose
                # it:
                patch = image.read(
                    window=Window(
                        polygon_bbox.col_start + column*stride[0],
                        polygon_bbox.row_start + row*stride[1],
                        shape[1], shape[0]
                    )
                ).T

                # The fourth channel shows the alpha (transparency) value. We do not
                # allow patch with transparent pixels:
                if not patch.size or (patch.shape[-1] == 4 and 0 in patch[..., 3]):
                    continue

                yield idx, patch


class ImageLoader:
    def __init__(
            self, images, labels=None, augmentator=None, reader=None,
            batch_size=None, balance=False, label_encoding='one-hot',
            shuffle=True, random_seed=42, max_workers=None, classes=None,
            preprocess_input=None,
        ):
        """Create an ImageLoader

        Args:
            images: Must be either an iterable of image filenames, a path to a
                directory (e.g. /path/to/images/*.tif) or a path containing the
                placeholder *{label}* (e.g. /path/to/{label}/*.tif to match
                /path/to/car/001.tif). In the latter case, you do not have to
                set the parameter *labels*.
            labels: This must be given or *images* must contain a placeholder
                with *{label}* if you want to balance this dataset. Must be
                an iterable of labels with the same length as *images*.
            yield_labels: Yield labels if *True*.
                Default: True if *labels* are set otherwise False.
            reader: Function to read the images. If None, images will be read
                by scikit-image.imread function.
                Default None.
            shuffle: Shuffle the dataset once before yielding. Default: True.
            random_seed: Number to initialize a random state. Default: 42.
            augmentator: Use your favourite augmentator object. Actively
                supported are keras, imgaug and Albumentations image
                augmentators. Can be also set to a function that will be called
                on each image before yielding it to the model. Default: None
            classes: Classes which will be encoded in this dataset.
            batch_size: Size of one batch. Default: 32.
            balance: Can be either:
                * *True*: the minority classes are going to be oversampled so
                    they have the same number as the majority class. If this is
                    used, *labels* must be given.
                * *iterable*: An iterable with the weights for each sample. The
                    sum of all weights should be 1.
                Default: False.
            balance_batch: If *True*, all classes appear with equal numbers in
                each batch. Works only if the number of classes is equal or
                lower than the batch size. Defult: False.
            label_encoding: Can be either:
                * *False*: No encoding.
                * *one-hot*: 1D numpy array of binary labels
                *
                ...
                Default: *one-hot*.

        Examples:

            from ai4eo.preprocessing import ImageLoader
            from keras.preprocessing.image import ImageDataGenerator


            keras_augmentator = ImageDataGenerator(
                featurewise_center=True,
                featurewise_std_normalization=True,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True
            )

            data = ImageLoader(
                '/path/to/images/{label}/*.tif', augmentator=keras_augmentator,
            )

            # Create keras model
            model = ...

            model.fit_generator(data, ...)
        """

        if isinstance(images, str):
            # Let's try to find all images in the given path
            files = FileSet(images).to_dataframe()
            images = files.index.values

            if "label" in files.columns:
                labels = files['label'].values

        if labels is not None and len(labels) != len(images):
            raise ValueError("images and labels must have the same length!")

        self.images = np.array(images)
        self.labels = None if labels is None else np.array(labels)

        self.classes = classes
        if self.classes is None and self.labels is not None:
            self.classes = np.unique(self.labels)

        if self.classes is not None:
            self.class_indices = {
                index: label for index, label in enumerate(self.classes)
            }
        else:
            self.class_indices = None

        if label_encoding == 'one-hot':
            self.labels = label_binarize(self.labels, classes=self.classes)

        self.reader = reader
        self.augmentator = augmentator
        self.augmentator_type = None
        if callable(getattr(self.augmentator, "random_transform", None)):
            self.augmentator_type = 'keras'
        elif callable(getattr(self.augmentator, "augment_batches", None)):
            self.augmentator_type = 'imgaug'
        # elif callable(getattr(self.augmentator, "augment_batches", None)):
        #     self.augmentator_type = 'imgaug'
        self.batch_size = batch_size or 32
        self.preprocess_input = preprocess_input

        # To make the experiments reproducible:
        self.random_state = np.random.RandomState(random_seed)
        self.random_seed = random_seed

        self.max_workers = max_workers

        self._indices = list(range(len(self.images)))
        if shuffle:
            self.random_state.shuffle(self._indices)

        if not balance:
            self._weights = None
        # Check explicitly for True because iterables could also return True in
        # a boolean context
        elif balance is True:
            # We want to oversample the minority classes, i.e. the set the
            # weights accordingly (the lower the amount of samples per class,
            # the higher the weight for them).
            if self.labels is None:
                raise ValueError('Cannot balance samples by myself without'
                                 'having any labels! Please set *labels*!')
            unique_labels, counts = np.unique(labels, return_counts=True)
            label_counts = pd.Series(counts, index=unique_labels)
            self._weights = \
                1 / label_counts.loc[labels].values / len(label_counts)
        else:
            self._weights = balance

    def __len__(self):
        return len(self.images) // self.batch_size

    def __getitem__(self, idx):
        if self._weights is None:
            sample_ids = \
                self._indices[idx*self.batch_size:(idx+1)*self.batch_size]
            return self.get_samples(sample_ids)
        else:
            # CAVE-AT: It could be that during one epoch some samples are never
            # seen.
            return self.get_random_samples(self.batch_size)

    def get_random_ids(self, n):
        weights = \
            None if self._weights is None else self._weights[self._indices]
        return self.random_state.choice(
            self._indices, size=n, replace=False, p=weights
        )

    def get_random_samples(self, n):
        sample_ids = self.get_random_ids(n)
        return self.get_samples(sample_ids)

    def get_samples(self, sample_ids):
        """Get a batch of samples

        Returns:
            A batch of images as numpy.array (BxHxWxC or BxCxHxW) and - if
            *yield_labels* is True - a numpy array with the encoded labels.
        """
        filenames = self.images[sample_ids]

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            # imgaug is much faster applied one complete batch
            if self.augmentator_type == 'imgaug':
                batch = list(pool.map(self.read, filenames))
                batch = self.augmentator.augment_images(batch)
            else:
                batch = list(pool.map(self.read_and_augment, filenames))

        if self.preprocess_input is not None:
            batch = [self.preprocess_input(img) for img in batch]

        if self.labels is None:
            return np.array(batch)
        else:
            return np.array(batch), self.labels[sample_ids]

    def read_and_augment(self, filename):
        return self.augment(self.read(filename))

    def augment(self, image):
        if self.augmentator is None:
            return image
        if self.augmentator_type == 'keras':
            return self.augmentator.random_transform(image)
        else:
            return self.augmentator(image)

    @staticmethod
    def read(filename):
        try:
            if filename.endswith('.npy'):
                return np.load(filename, allow_pickle=False)
            elif filename.endswith('.npy.gz'):
                with gzip.open(filename, 'r') as f:
                    return np.load(f, allow_pickle=False)
            else:
                return imread(filename)
        except Exception as error:
            print(f"ERROR: Could not read {filename}!")
            raise error


    def reset(self):
        self.random_state = np.random.RandomState(self.random_seed)
