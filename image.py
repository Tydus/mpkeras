"""Fairly basic set of tools for real-time data augmentation on image data.
Can easily be extended to include new transformations,
new preprocessing methods, etc...
"""
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import re
from scipy import linalg
import scipy.ndimage as ndi
from six.moves import range
import os
import threading
import warnings
import json
import random
import sys
from operator import itemgetter

# from .. import backend as K
import tensorflow.keras.backend as K


try:
    from PIL import Image as pil_image
except ImportError:
    pil_image = None


def random_rotation(x, rg, row_axis=1, col_axis=2, channel_axis=0,
                    fill_mode='nearest', cval=0.):
    """Performs a random rotation of a Numpy image tensor.

    # Arguments
        x: Input tensor. Must be 3D.
        rg: Rotation range, in degrees.
        row_axis: Index of axis for rows in the input tensor.
        col_axis: Index of axis for columns in the input tensor.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.

    # Returns
        Rotated Numpy image tensor.
    """
    theta = np.pi / 180 * np.random.uniform(-rg, rg)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])

    h, w = x.shape[row_axis], x.shape[col_axis]
    transform_matrix = transform_matrix_offset_center(rotation_matrix, h, w)
    x = apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x


def random_shift(x, wrg, hrg, row_axis=1, col_axis=2, channel_axis=0,
                 fill_mode='nearest', cval=0.):
    """Performs a random spatial shift of a Numpy image tensor.

    # Arguments
        x: Input tensor. Must be 3D.
        wrg: Width shift range, as a float fraction of the width.
        hrg: Height shift range, as a float fraction of the height.
        row_axis: Index of axis for rows in the input tensor.
        col_axis: Index of axis for columns in the input tensor.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.

    # Returns
        Shifted Numpy image tensor.
    """
    h, w = x.shape[row_axis], x.shape[col_axis]
    tx = np.random.uniform(-hrg, hrg) * h
    ty = np.random.uniform(-wrg, wrg) * w
    translation_matrix = np.array([[1, 0, tx],
                                   [0, 1, ty],
                                   [0, 0, 1]])

    transform_matrix = translation_matrix  # no need to do offset
    x = apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x


def random_shear(x, intensity, row_axis=1, col_axis=2, channel_axis=0,
                 fill_mode='nearest', cval=0.):
    """Performs a random spatial shear of a Numpy image tensor.

    # Arguments
        x: Input tensor. Must be 3D.
        intensity: Transformation intensity.
        row_axis: Index of axis for rows in the input tensor.
        col_axis: Index of axis for columns in the input tensor.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.

    # Returns
        Sheared Numpy image tensor.
    """
    shear = np.random.uniform(-intensity, intensity)
    shear_matrix = np.array([[1, -np.sin(shear), 0],
                             [0, np.cos(shear), 0],
                             [0, 0, 1]])

    h, w = x.shape[row_axis], x.shape[col_axis]
    transform_matrix = transform_matrix_offset_center(shear_matrix, h, w)
    x = apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x


def random_zoom(x, zoom_range, row_axis=1, col_axis=2, channel_axis=0,
                fill_mode='nearest', cval=0.):
    """Performs a random spatial zoom of a Numpy image tensor.

    # Arguments
        x: Input tensor. Must be 3D.
        zoom_range: Tuple of floats; zoom range for width and height.
        row_axis: Index of axis for rows in the input tensor.
        col_axis: Index of axis for columns in the input tensor.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.

    # Returns
        Zoomed Numpy image tensor.

    # Raises
        ValueError: if `zoom_range` isn't a tuple.
    """
    if len(zoom_range) != 2:
        raise ValueError('zoom_range should be a tuple or list of two floats. '
                         'Received arg: ', zoom_range)

    if zoom_range[0] == 1 and zoom_range[1] == 1:
        zx, zy = 1, 1
    else:
        zx, zy = np.random.uniform(zoom_range[0], zoom_range[1], 2)
    zoom_matrix = np.array([[zx, 0, 0],
                            [0, zy, 0],
                            [0, 0, 1]])

    h, w = x.shape[row_axis], x.shape[col_axis]
    transform_matrix = transform_matrix_offset_center(zoom_matrix, h, w)
    x = apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x


def random_channel_shift(x, intensity, channel_axis=0):
    x = np.rollaxis(x, channel_axis, 0)
    min_x, max_x = np.min(x), np.max(x)
    channel_images = [np.clip(x_channel + np.random.uniform(-intensity, intensity), min_x, max_x)
                      for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_axis + 1)
    return x


def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix


def apply_transform(x, transform_matrix, channel_axis=0, fill_mode='nearest', cval=0.):
    x = np.rollaxis(x, channel_axis, 0)
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]
    channel_images = [ndi.interpolation.affine_transform(x_channel, final_affine_matrix,
                                                         final_offset, order=0, mode=fill_mode, cval=cval) for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_axis + 1)
    return x


def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x

def standardize(x,
                preprocessing_function=None,
                rescale=None,
                channel_axis=None,
                samplewise_center=False,
                featurewise_center=False,
                samplewise_std_normalization=False,
                featurewise_std_normalization=False,
                mean=None,
                std=None,
                zca_whitening=False,
                principal_components=None,
                rng=None):
    if preprocessing_function:
        x = preprocessing_function(x)
    if rescale:
        x *= rescale
    # x is a single image, so it doesn't have image number at index 0
    img_channel_axis = channel_axis - 1
    if samplewise_center:
        x -= np.mean(x, axis=img_channel_axis, keepdims=True)
    if samplewise_std_normalization:
        x /= (np.std(x, axis=img_channel_axis, keepdims=True) + 1e-7)

    if featurewise_center:
        if mean is not None:
            x -= mean
        else:
            warnings.warn('This ImageDataGenerator specifies '
                          '`featurewise_center`, but it hasn\'t'
                          'been fit on any training data. Fit it '
                          'first by calling `.fit(numpy_data)`.')
    if featurewise_std_normalization:
        if std is not None:
            x /= (std + 1e-7)
        else:
            warnings.warn('This ImageDataGenerator specifies '
                          '`featurewise_std_normalization`, but it hasn\'t'
                          'been fit on any training data. Fit it '
                          'first by calling `.fit(numpy_data)`.')
    if zca_whitening:
        if principal_components is not None:
            flatx = np.reshape(x, (x.size))
            whitex = np.dot(flatx, principal_components)
            x = np.reshape(whitex, (x.shape[0], x.shape[1], x.shape[2]))
        else:
            warnings.warn('This ImageDataGenerator specifies '
                          '`zca_whitening`, but it hasn\'t'
                          'been fit on any training data. Fit it '
                          'first by calling `.fit(numpy_data)`.')
    return x

def random_transform(x,
                     row_axis=None,
                     col_axis=None,
                     channel_axis=None,
                     rotation_range=0.,
                     height_shift_range=0.,
                     width_shift_range=0.,
                     shear_range=0.,
                     zoom_range=0.,
                     fill_mode='nearest',
                     cval=0.,
                     channel_shift_range=0.,
                     horizontal_flip=False,
                     vertical_flip=False,
                     rng=None):

    supplied_rngs = True
    if rng is None:
        supplied_rngs = False
        rng = np.random

    # x is a single image, so it doesn't have image number at index 0
    img_row_axis = row_axis - 1
    img_col_axis = col_axis - 1
    img_channel_axis = channel_axis - 1

    # use composition of homographies
    # to generate final transform that needs to be applied
    if rotation_range:
        theta = np.pi / 180 * rng.uniform(-rotation_range, rotation_range)
    else:
        theta = 0
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])
    if height_shift_range:
        tx = rng.uniform(-height_shift_range, height_shift_range) * x.shape[img_row_axis]
    else:
        tx = 0

    if width_shift_range:
        ty = rng.uniform(-width_shift_range, width_shift_range) * x.shape[img_col_axis]
    else:
        ty = 0

    translation_matrix = np.array([[1, 0, tx],
                                   [0, 1, ty],
                                   [0, 0, 1]])
    if shear_range:
        shear = rng.uniform(-shear_range, shear_range)
    else:
        shear = 0
    shear_matrix = np.array([[1, -np.sin(shear), 0],
                             [0, np.cos(shear), 0],
                             [0, 0, 1]])

    if zoom_range[0] == 1 and zoom_range[1] == 1:
        zx, zy = 1, 1
    else:
        zx, zy = rng.uniform(zoom_range[0], zoom_range[1], 2)
    zoom_matrix = np.array([[zx, 0, 0],
                            [0, zy, 0],
                            [0, 0, 1]])

    transform_matrix = np.dot(np.dot(np.dot(rotation_matrix,
                                            translation_matrix),
                                     shear_matrix),
                              zoom_matrix)

    h, w = x.shape[img_row_axis], x.shape[img_col_axis]
    transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
    x = apply_transform(x, transform_matrix, img_channel_axis,
                        fill_mode=fill_mode, cval=cval)
    if channel_shift_range != 0:
        x = random_channel_shift(x,
                                 channel_shift_range,
                                 img_channel_axis)

    get_random = None
    if supplied_rngs:
        get_random = rng.rand
    else:
        get_random = np.random.random

    if horizontal_flip:
        if get_random() < 0.5:
            x = flip_axis(x, img_col_axis)

    if vertical_flip:
        if get_random() < 0.5:
            x = flip_axis(x, img_row_axis)

    return x

def array_to_img(x, dim_ordering='default', scale=True):
    """Converts a 3D Numpy array to a PIL Image instance.

    # Arguments
        x: Input Numpy array.
        dim_ordering: Image data format.
        scale: Whether to rescale image values
            to be within [0, 255].

    # Returns
        A PIL Image instance.

    # Raises
        ImportError: if PIL is not available.
        ValueError: if invalid `x` or `dim_ordering` is passed.
    """
    if pil_image is None:
        raise ImportError('Could not import PIL.Image. '
                          'The use of `array_to_img` requires PIL.')
    x = np.asarray(x)
    if x.ndim != 3:
        raise ValueError('Expected image array to have rank 3 (single image). '
                         'Got array with shape:', x.shape)

    if dim_ordering == 'default':
        dim_ordering = K.image_dim_ordering()
    if dim_ordering not in {'th', 'tf'}:
        raise ValueError('Invalid dim_ordering:', dim_ordering)

    # Original Numpy array x has format (height, width, channel)
    # or (channel, height, width)
    # but target PIL image has format (width, height, channel)
    if dim_ordering == 'th':
        x = x.transpose(1, 2, 0)
    if scale:
        x = x + max(-np.min(x), 0)
        x_max = np.max(x)
        if x_max != 0:
            x /= x_max
        x *= 255
    if x.shape[2] == 3:
        # RGB
        return pil_image.fromarray(x.astype('uint8'), 'RGB')
    elif x.shape[2] == 1:
        # grayscale
        return pil_image.fromarray(x[:, :, 0].astype('uint8'), 'L')
    else:
        raise ValueError('Unsupported channel number: ', x.shape[2])


def img_to_array(img, dim_ordering='default'):
    """Converts a PIL Image instance to a Numpy array.

    # Arguments
        img: PIL Image instance.
        dim_ordering: Image data format.

    # Returns
        A 3D Numpy array (float32).

    # Raises
        ValueError: if invalid `img` or `dim_ordering` is passed.
    """
    if dim_ordering == 'default':
        dim_ordering = K.image_dim_ordering()
    if dim_ordering not in {'th', 'tf'}:
        raise ValueError('Unknown dim_ordering: ', dim_ordering)
    # Numpy array x has format (height, width, channel)
    # or (channel, height, width)
    # but original PIL image has format (width, height, channel)
    x = np.asarray(img, dtype='float32')
    if len(x.shape) == 3:
        if dim_ordering == 'th':
            x = x.transpose(2, 0, 1)
    elif len(x.shape) == 2:
        if dim_ordering == 'th':
            x = x.reshape((1, x.shape[0], x.shape[1]))
        else:
            x = x.reshape((x.shape[0], x.shape[1], 1))
    else:
        raise ValueError('Unsupported image shape: ', x.shape)
    return x

def load_img(path, grayscale=False, target_size=None):
    """Loads an image into PIL format.

    # Arguments
        path: Path to image file
        grayscale: Boolean, whether to load the image as grayscale.
        target_size: Either `None` (default to original size)
            or tuple of ints `(img_height, img_width)`.

    # Returns
        A PIL Image instance.

    # Raises
        ImportError: if PIL is not available.
    """
    if pil_image is None:
        raise ImportError('Could not import PIL.Image. '
                          'The use of `array_to_img` requires PIL.')
    img = pil_image.open(path)
    if grayscale:
        img = img.convert('L')
    else:  # Ensure 3 channel even when loaded image is grayscale
        img = img.convert('RGB')

    if target_size:
        x, y = img.size
        size = max(x, y)
        new_im = pil_image.new('RGB', (size, size), (0, 0, 0))
        new_im.paste(img, (int((size - x) / 2), int((size - y) / 2)))
        img.close()
        img = new_im.resize((target_size[1], target_size[0]), pil_image.ANTIALIAS)
    return img

def list_pictures(directory, ext='jpg|jpeg|bmp|png'):
    return [os.path.join(root, f)
            for root, _, files in os.walk(directory) for f in files
            if re.match('([\w]+\.(?:' + ext + '))', f)]


class ImageDataGenerator(object):
    """Generate minibatches of image data with real-time data augmentation.

    # Arguments
        featurewise_center: set input mean to 0 over the dataset.
        samplewise_center: set each sample mean to 0.
        featurewise_std_normalization: divide inputs by std of the dataset.
        samplewise_std_normalization: divide each input by its std.
        zca_whitening: apply ZCA whitening.
        rotation_range: degrees (0 to 180).
        width_shift_range: fraction of total width.
        height_shift_range: fraction of total height.
        shear_range: shear intensity (shear angle in radians).
        zoom_range: amount of zoom. if scalar z, zoom will be randomly picked
            in the range [1-z, 1+z]. A sequence of two can be passed instead
            to select this range.
        channel_shift_range: shift range for each channels.
        fill_mode: points outside the boundaries are filled according to the
            given mode ('constant', 'nearest', 'reflect' or 'wrap'). Default
            is 'nearest'.
        cval: value used for points outside the boundaries when fill_mode is
            'constant'. Default is 0.
        horizontal_flip: whether to randomly flip images horizontally.
        vertical_flip: whether to randomly flip images vertically.
        rescale: rescaling factor. If None or 0, no rescaling is applied,
            otherwise we multiply the data by the value provided
            (before applying any other transformation).
        preprocessing_function: function that will be implied on each input.
            The function will run before any other modification on it.
            The function should take one argument:
            one image (Numpy tensor with rank 3),
            and should output a Numpy tensor with the same shape.
        dim_ordering: 'th' or 'tf'. In 'th' mode, the channels dimension
            (the depth) is at index 1, in 'tf' mode it is at index 3.
            It defaults to the `image_dim_ordering` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "tf".
        pool: an open multiprocessing.Pool that will be used to
            process multiple images in parallel. If left off or set to
            None, then the default serial processing with a single
            process will be used.
    """

    def __init__(self,
                 featurewise_center=False,
                 samplewise_center=False,
                 featurewise_std_normalization=False,
                 samplewise_std_normalization=False,
                 zca_whitening=False,
                 rotation_range=0.,
                 width_shift_range=0.,
                 height_shift_range=0.,
                 shear_range=0.,
                 zoom_range=0.,
                 channel_shift_range=0.,
                 fill_mode='nearest',
                 cval=0.,
                 horizontal_flip=False,
                 vertical_flip=False,
                 rescale=None,
                 preprocessing_function=None,
                 dim_ordering='default',
                 pool=None):
        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        self.featurewise_center = featurewise_center
        self.samplewise_center = samplewise_center
        self.featurewise_std_normalization = featurewise_std_normalization
        self.samplewise_std_normalization = samplewise_std_normalization
        self.zca_whitening = zca_whitening
        self.rotation_range = rotation_range
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range
        self.shear_range = shear_range
        self.zoom_range = zoom_range
        self.channel_shift_range = channel_shift_range
        self.fill_mode = fill_mode
        self.cval = cval
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.rescale = rescale
        self.preprocessing_function = preprocessing_function
        self.pool = pool

        if dim_ordering not in {'tf', 'th'}:
            raise ValueError('dim_ordering should be "tf" (channel after row and '
                             'column) or "th" (channel before row and column). '
                             'Received arg: ', dim_ordering)
        self.dim_ordering = dim_ordering
        if dim_ordering == 'th':
            self.channel_axis = 1
            self.row_axis = 2
            self.col_axis = 3
        if dim_ordering == 'tf':
            self.channel_axis = 3
            self.row_axis = 1
            self.col_axis = 2

        self.mean = None
        self.std = None
        self.principal_components = None

        if np.isscalar(zoom_range):
            self.zoom_range = [1 - zoom_range, 1 + zoom_range]
        elif len(zoom_range) == 2:
            self.zoom_range = [zoom_range[0], zoom_range[1]]
        else:
            raise ValueError('zoom_range should be a float or '
                             'a tuple or list of two floats. '
                             'Received arg: ', zoom_range)

    def flow(self, X, y=None, batch_size=32, shuffle=True, seed=None,
             save_to_dir=None, save_prefix='', save_format='jpeg'):
        return NumpyArrayIterator(
            X, y, self,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            dim_ordering=self.dim_ordering,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format,
            pool=self.pool)

    def flow_from_directory(self, directory,
                            target_size=(256, 256), color_mode='rgb',
                            classes=None, class_mode='categorical',
                            batch_size=32, shuffle=True, seed=None,
                            save_to_dir=None,
                            save_prefix='',
                            save_format='jpeg',
                            follow_links=False):
        return DirectoryIterator(
            directory, self,
            target_size=target_size, color_mode=color_mode,
            classes=classes, class_mode=class_mode,
            dim_ordering=self.dim_ordering,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format,
            follow_links=follow_links,
            pool=self.pool)


    def flow_from_directory_with_hard_samples(self, directory,
                            target_size=(256, 256), color_mode='rgb',
                            classes=None, class_mode='categorical',
                            batch_size=32, shuffle=True, seed=None,
                            save_to_dir=None,
                            save_prefix='',
                            save_format='jpeg',
                            follow_links=False,
                            hard_samples=None,
                            hard_sample_rate=0.05):
        self.hard_samples=hard_samples
        self.hard_sample_rate = hard_sample_rate
        return DirectoryIterator_with_hard_samples(
            directory, self,
            target_size=target_size, color_mode=color_mode,
            classes=classes, class_mode=class_mode,
            dim_ordering=self.dim_ordering,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format,
            follow_links=follow_links,
            pool=self.pool,
            hard_samples=self.hard_samples,
            hard_sample_rate=self.hard_sample_rate
            
            )    
    
    
    def flow_from_metaFunc(self, metaFunc,
                            target_size=(256, 256), color_mode='rgb',
                            classes=None, class_mode='categorical',
                            batch_size=32, shuffle=True, seed=None,
                            save_to_dir=None,
                            save_prefix='',
                            save_format='jpeg',
                            image_in_memory = False,
                            follow_links=False):
        return MetadataIterator(
            metaFunc, self,
            target_size=target_size, color_mode=color_mode,
            classes=classes, class_mode=class_mode,
            dim_ordering=self.dim_ordering,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format,
            follow_links=follow_links,
            image_in_memory = image_in_memory,
            pool=self.pool)
    
    def flow_from_metaseq( self, metaSeq, 
                            target_size=(256, 256), color_mode='rgb',
                            classes=None, class_mode='categorical',
                            batch_size=32, shuffle=True, seed=None,
                            save_to_dir=None,
                            save_prefix='',
                            save_format='jpeg',
                            follow_links=False, 
                            raise_exception = False,
                            image_in_memory = False):
        """ Returning a sequence for training/evaluation
        """
        return MetadataSeqIterator(
            metaSeq, self,
            target_size=target_size, color_mode=color_mode,
            classes=classes, class_mode=class_mode,
            dim_ordering=self.dim_ordering,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format,
            follow_links=follow_links,
            raise_exception = raise_exception,
            image_in_memory = image_in_memory,
            pool=self.pool)
    
    def flow_from_metaseq_siamese( self, metaSeq, 
                            target_size=(256, 256), color_mode='rgb',
                            classes=None, # class_mode='binary',
                            steps=None,
                            batch_size=32, shuffle=True, seed=None,
                            save_to_dir=None,
                            save_prefix='',
                            save_format='jpeg',
                            follow_links=False):
        """ Returning a sequence for training/evaluation
        """
        return MetadataSeqSiameseIterator(
            metaSeq, self,
            target_size=target_size, color_mode=color_mode,
            classes=classes, # class_mode=class_mode,
            dim_ordering=self.dim_ordering,
            steps=steps,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format,
            follow_links=follow_links,
            pool=self.pool)


    def pipeline(self):
        """A pipeline of functions to apply in order to an image.
        """
        return [
            (random_transform, dict(
                row_axis=self.row_axis,
                col_axis=self.col_axis,
                channel_axis=self.channel_axis,
                rotation_range=self.rotation_range,
                height_shift_range=self.height_shift_range,
                width_shift_range=self.width_shift_range,
                shear_range=self.shear_range,
                zoom_range=self.zoom_range,
                fill_mode=self.fill_mode,
                cval=self.cval,
                channel_shift_range=self.channel_shift_range,
                horizontal_flip=self.horizontal_flip,
                vertical_flip=self.vertical_flip)
            ),

            (standardize, dict(
                preprocessing_function=self.preprocessing_function,
                rescale=self.rescale,
                channel_axis=self.channel_axis,
                samplewise_center=self.samplewise_center,
                samplewise_std_normalization=self.samplewise_std_normalization,
                featurewise_center=self.featurewise_center,
                mean=self.mean,
                featurewise_std_normalization=self.featurewise_std_normalization,
                std=self.std,
                zca_whitening=self.zca_whitening,
                principal_components=self.principal_components)
            )
        ]

    def standardize(self, x):
        return standardize(x,
            preprocessing_function=self.preprocessing_function,
            rescale=self.rescale,
            channel_axis=self.channel_axis,
            samplewise_center=self.samplewise_center,
            samplewise_std_normalization=self.samplewise_std_normalization,
            featurewise_center=self.featurewise_center,
            mean=self.mean,
            featurewise_std_normalization=self.featurewise_std_normalization,
            std=self.std,
            zca_whitening=self.zca_whitening,
            principal_components=self.principal_components)

    def random_transform(self, x):
        return random_transform(x,
            row_axis=self.row_axis,
            col_axis=self.col_axis,
            channel_axis=self.channel_axis,
            rotation_range=self.rotation_range,
            height_shift_range=self.height_shift_range,
            width_shift_range=self.width_shift_range,
            shear_range=self.shear_range,
            zoom_range=self.zoom_range,
            fill_mode=self.fill_mode,
            cval=self.cval,
            channel_shift_range=self.channel_shift_range,
            horizontal_flip=self.horizontal_flip,
            vertical_flip=self.vertical_flip)

    def fit(self, x,
            augment=False,
            rounds=1,
            seed=None):
        """Required for featurewise_center, featurewise_std_normalization
        and zca_whitening.

        # Arguments
            x: Numpy array, the data to fit on. Should have rank 4.
                In case of grayscale data,
                the channels axis should have value 1, and in case
                of RGB data, it should have value 3.
            augment: Whether to fit on randomly augmented samples
            rounds: If `augment`,
                how many augmentation passes to do over the data
            seed: random seed.

        # Raises
            ValueError: in case of invalid input `x`.
        """
        x = np.asarray(x)
        if x.ndim != 4:
            raise ValueError('Input to `.fit()` should have rank 4. '
                             'Got array with shape: ' + str(x.shape))
        if x.shape[self.channel_axis] not in {1, 3, 4}:
            raise ValueError(
                'Expected input to be images (as Numpy array) '
                'following the dimension ordering convention "' + self.dim_ordering + '" '
                '(channels on axis ' + str(self.channel_axis) + '), i.e. expected '
                'either 1, 3 or 4 channels on axis ' + str(self.channel_axis) + '. '
                'However, it was passed an array with shape ' + str(x.shape) +
                ' (' + str(x.shape[self.channel_axis]) + ' channels).')

        if seed is not None:
            np.random.seed(seed)

        x = np.copy(x)
        if augment:
            ax = np.zeros(tuple([rounds * x.shape[0]] + list(x.shape)[1:]))
            for r in range(rounds):
                for i in range(x.shape[0]):
                    ax[i + r * x.shape[0]] = self.random_transform(x[i])
            x = ax

        if self.featurewise_center:
            self.mean = np.mean(x, axis=(0, self.row_axis, self.col_axis))
            broadcast_shape = [1, 1, 1]
            broadcast_shape[self.channel_axis - 1] = x.shape[self.channel_axis]
            self.mean = np.reshape(self.mean, broadcast_shape)
            x -= self.mean

        if self.featurewise_std_normalization:
            self.std = np.std(x, axis=(0, self.row_axis, self.col_axis))
            broadcast_shape = [1, 1, 1]
            broadcast_shape[self.channel_axis - 1] = x.shape[self.channel_axis]
            self.std = np.reshape(self.std, broadcast_shape)
            x /= (self.std + K.epsilon())

        if self.zca_whitening:
            flat_x = np.reshape(x, (x.shape[0], x.shape[1] * x.shape[2] * x.shape[3]))
            sigma = np.dot(flat_x.T, flat_x) / flat_x.shape[0]
            u, s, _ = linalg.svd(sigma)
            self.principal_components = np.dot(np.dot(u, np.diag(1. / np.sqrt(s + 10e-7))), u.T)


class Iterator(object):

    def __init__(self, n, batch_size, shuffle, seed):
        self.n = n
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batch_index = 0
        self.total_batches_seen = 0
        self.lock = threading.Lock()
        self.index_generator = self._flow_index(n, batch_size, shuffle, seed)

        # create multiple random number generators to be used separately in
        # each process when using a multiprocessing.Pool
        if seed:
            self.rngs = [np.random.RandomState(seed + i) for i in range(batch_size)]
        else:
            self.rngs = [np.random.RandomState(i) for i in range(batch_size)]

    def reset(self):
        self.batch_index = 0

    def _flow_index(self, n, batch_size=32, shuffle=False, seed=None):
        # ensure self.batch_index is 0
        self.reset()
        while 1:
            if seed is not None:
                np.random.seed(seed + self.total_batches_seen)
            if self.batch_index == 0:
                index_array = np.arange(n)
                if shuffle:
                    index_array = np.random.permutation(n)

            current_index = (self.batch_index * batch_size) % n
            if n >= current_index + batch_size:
                current_batch_size = batch_size
                self.batch_index += 1
            else:
                current_batch_size = n - current_index
                self.batch_index = 0
            self.total_batches_seen += 1
            yield (index_array[current_index: current_index + current_batch_size],
                   current_index, current_batch_size)

    def __iter__(self):
        # needed if we want to do something like:
        # for x, y in data_gen.flow(...):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)


def process_image_pipeline(tup):
    """ Worker function for NumpyArrayIterator multiprocessing.Pool
    """
    (pipeline, x, rng) = tup
    x = x.astype('float32')
    for (func, kwargs) in pipeline:
        x = func(x, rng=rng, **kwargs)
    return x

def process_image_pipeline_dir(tup):
    """ Worker function for DirectoryIterator multiprocessing.Pool
    """
    (pipeline, fname, directory, grayscale,
    target_size, dim_ordering, rng) = tup
    img = load_img(os.path.join(directory, fname),
                   grayscale=grayscale,
                   target_size=target_size)
    x = img_to_array(img, dim_ordering=dim_ordering)
    for (func, kwargs) in pipeline:
        x = func(x, rng=rng, **kwargs)
    return x

def process_image_pipeline_dir_cache(tup):
    """ Worker function for DirectoryIterator multiprocessing.Pool
    """
    (pipeline, fname, directory, grayscale,
    target_size, dim_ordering, rng,image_cache) = tup
    if image_cache is not None:
        img = image_cache
    else:
        img = load_img(os.path.join(directory, fname),
                       grayscale=grayscale,
                       target_size=target_size)
    x = img_to_array(img, dim_ordering=dim_ordering)
    for (func, kwargs) in pipeline:
        x = func(x, rng=rng, **kwargs)
    return (x,fname,img)


class NumpyArrayIterator(Iterator):

    def __init__(self, x, y, image_data_generator,
                 batch_size=32, shuffle=False, seed=None,
                 dim_ordering='default',
                 save_to_dir=None, save_prefix='', save_format='jpeg',
                 pool=None):
        if y is not None and len(x) != len(y):
            raise ValueError('X (images tensor) and y (labels) '
                             'should have the same length. '
                             'Found: X.shape = %s, y.shape = %s' %
                             (np.asarray(x).shape, np.asarray(y).shape))
        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        self.x = np.asarray(x)
        if self.x.ndim != 4:
            raise ValueError('Input data in `NumpyArrayIterator` '
                             'should have rank 4. You passed an array '
                             'with shape', self.x.shape)
        channels_axis = 3 if dim_ordering == 'tf' else 1
        if self.x.shape[channels_axis] not in {1, 3, 4}:
            raise ValueError('NumpyArrayIterator is set to use the '
                             'dimension ordering convention "' + dim_ordering + '" '
                             '(channels on axis ' + str(channels_axis) + '), i.e. expected '
                             'either 1, 3 or 4 channels on axis ' + str(channels_axis) + '. '
                             'However, it was passed an array with shape ' + str(self.x.shape) +
                             ' (' + str(self.x.shape[channels_axis]) + ' channels).')
        if y is not None:
            self.y = np.asarray(y)
        else:
            self.y = None
        self.image_data_generator = image_data_generator
        self.dim_ordering = dim_ordering
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        self.pool = pool

        super(NumpyArrayIterator, self).__init__(x.shape[0], batch_size, shuffle, seed)

    def next(self):
        # for python 2.x.
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch
        # see http://anandology.com/blog/using-iterators-and-generators/
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel

        batch_x = None

        if self.pool:
            pipeline = self.image_data_generator.pipeline()
            result = self.pool.map(process_image_pipeline, (
                (pipeline, self.x[j], self.rngs[i%self.batch_size])
                for i, j in enumerate(index_array)))
            batch_x = np.array(result)
        else:
            # TODO: also utilize image_data_generator.pipeline()?
            batch_x = np.zeros(tuple([current_batch_size] + list(self.x.shape)[1:]))
            for i, j in enumerate(index_array):
                x = self.x[j]
                x = self.image_data_generator.random_transform(x.astype('float32'))
                x = self.image_data_generator.standardize(x)
                batch_x[i] = x

        if self.save_to_dir:
            for i in range(current_batch_size):
                img = array_to_img(batch_x[i], self.dim_ordering, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                  index=current_index + i,
                                                                  hash=np.random.randint(1e4),
                                                                  format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        if self.y is None:
            return batch_x
        batch_y = self.y[index_array]
        return batch_x, batch_y


class DirectoryIterator(Iterator):

    def __init__(self, directory, image_data_generator,
                 target_size=(256, 256), color_mode='rgb',
                 dim_ordering='default',
                 classes=None, class_mode='categorical',
                 batch_size=32, shuffle=True, seed=None,
                 save_to_dir=None, save_prefix='', save_format='jpeg',
                 follow_links=False, pool=None):
        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        self.directory = directory
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        if color_mode not in {'rgb', 'grayscale'}:
            raise ValueError('Invalid color mode:', color_mode,
                             '; expected "rgb" or "grayscale".')
        self.color_mode = color_mode
        self.dim_ordering = dim_ordering
        if self.color_mode == 'rgb':
            if self.dim_ordering == 'tf':
                self.image_shape = self.target_size + (3,)
            else:
                self.image_shape = (3,) + self.target_size
        else:
            if self.dim_ordering == 'tf':
                self.image_shape = self.target_size + (1,)
            else:
                self.image_shape = (1,) + self.target_size
        self.classes = classes
        if class_mode not in {'categorical', 'binary', 'sparse', None}:
            raise ValueError('Invalid class_mode:', class_mode,
                             '; expected one of "categorical", '
                             '"binary", "sparse", or None.')
        self.class_mode = class_mode
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        self.pool = pool

        white_list_formats = {'png', 'jpg', 'jpeg', 'bmp'}

        # first, count the number of samples and classes
        self.nb_sample = 0

        if not classes:
            classes = []
            for subdir in sorted(os.listdir(directory)):
                if os.path.isdir(os.path.join(directory, subdir)):
                    classes.append(subdir)
        self.nb_class = len(classes)
        self.class_indices = dict(zip(classes, range(len(classes))))

        def _recursive_list(subpath):
            return sorted(os.walk(subpath, followlinks=follow_links), key=lambda tpl: tpl[0])

        for subdir in classes:
            subpath = os.path.join(directory, subdir)
            for root, _, files in _recursive_list(subpath):
                for fname in files:
                    is_valid = False
                    for extension in white_list_formats:
                        if fname.lower().endswith('.' + extension):
                            is_valid = True
                            break
                    if is_valid:
                        self.nb_sample += 1
        print('Found %d images belonging to %d classes.' % (self.nb_sample, self.nb_class))

        # second, build an index of the images in the different class subfolders
        self.filenames = []
        self.classes = np.zeros((self.nb_sample,), dtype='int32')
        i = 0
        for subdir in classes:
            subpath = os.path.join(directory, subdir)
            for root, _, files in _recursive_list(subpath):
                for fname in files:
                    is_valid = False
                    for extension in white_list_formats:
                        if fname.lower().endswith('.' + extension):
                            is_valid = True
                            break
                    if is_valid:
                        self.classes[i] = self.class_indices[subdir]
                        i += 1
                        # add filename relative to directory
                        absolute_path = os.path.join(root, fname)
                        self.filenames.append(os.path.relpath(absolute_path, directory))
        super(DirectoryIterator, self).__init__(self.nb_sample, batch_size, shuffle, seed)

    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel

        batch_x = None
        grayscale = self.color_mode == 'grayscale'

        if self.pool:
            pipeline = self.image_data_generator.pipeline()
            result = self.pool.map(process_image_pipeline_dir, ((pipeline,
                self.filenames[j],
                self.directory,
                grayscale,
                self.target_size,
                self.dim_ordering,
                self.rngs[i%self.batch_size]) for i, j in enumerate(index_array)))
            batch_x = np.array(result)
        else:
            # TODO: also utilize image_data_generator.pipeline()?
            batch_x = np.zeros((current_batch_size,) + self.image_shape)
            # build batch of image data
            for i, j in enumerate(index_array):
                fname = self.filenames[j]
                img = load_img(os.path.join(self.directory, fname),
                               grayscale=grayscale,
                               target_size=self.target_size)
                x = img_to_array(img, dim_ordering=self.dim_ordering)
                x = self.image_data_generator.random_transform(x)
                x = self.image_data_generator.standardize(x)
                batch_x[i] = x
        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            for i in range(current_batch_size):
                img = array_to_img(batch_x[i], self.dim_ordering, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                  index=current_index + i,
                                                                  hash=np.random.randint(1e4),
                                                                  format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        # build batch of labels
        if self.class_mode == 'sparse':
            batch_y = self.classes[index_array]
        elif self.class_mode == 'binary':
            batch_y = self.classes[index_array].astype('float32')
        elif self.class_mode == 'categorical':
            batch_y = np.zeros((len(batch_x), self.nb_class), dtype='float32')
            for i, label in enumerate(self.classes[index_array]):
                batch_y[i, label] = 1.
        else:
            return batch_x
        return batch_x, batch_y


class DirectoryIterator_with_hard_samples(Iterator):

    def __init__(self, directory, image_data_generator,
                 target_size=(256, 256), color_mode='rgb',
                 dim_ordering='default',
                 classes=None, class_mode='categorical',
                 batch_size=32, shuffle=True, seed=None,
                 save_to_dir=None, save_prefix='', save_format='jpeg',
                 follow_links=False, pool=None, hard_samples=None, hard_sample_rate=0):
        
        self.hard_samples = hard_samples
        self.hard_sample_rate = hard_sample_rate
        
        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        self.directory = directory
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        if color_mode not in {'rgb', 'grayscale'}:
            raise ValueError('Invalid color mode:', color_mode,
                             '; expected "rgb" or "grayscale".')
        self.color_mode = color_mode
        self.dim_ordering = dim_ordering
        if self.color_mode == 'rgb':
            if self.dim_ordering == 'tf':
                self.image_shape = self.target_size + (3,)
            else:
                self.image_shape = (3,) + self.target_size
        else:
            if self.dim_ordering == 'tf':
                self.image_shape = self.target_size + (1,)
            else:
                self.image_shape = (1,) + self.target_size
        self.classes = classes
        if class_mode not in {'categorical', 'binary', 'sparse', None}:
            raise ValueError('Invalid class_mode:', class_mode,
                             '; expected one of "categorical", '
                             '"binary", "sparse", or None.')
        self.class_mode = class_mode
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        self.pool = pool

        white_list_formats = {'png', 'jpg', 'jpeg', 'bmp'}

        # first, count the number of samples and classes
        self.nb_sample = 0

        if not classes:
            classes = []
            for subdir in sorted(os.listdir(directory)):
                if os.path.isdir(os.path.join(directory, subdir)):
                    classes.append(subdir)
        self.nb_class = len(classes)
        self.class_indices = dict(zip(classes, range(len(classes))))

        def _recursive_list(subpath):
            return sorted(os.walk(subpath, followlinks=follow_links), key=lambda tpl: tpl[0])

        for subdir in classes:
            subpath = os.path.join(directory, subdir)
            for root, _, files in _recursive_list(subpath):
                for fname in files:
                    is_valid = False
                    for extension in white_list_formats:
                        if fname.lower().endswith('.' + extension):
                            is_valid = True
                            break
                    if is_valid:
                        self.nb_sample += 1
        print('Found %d images belonging to %d classes.' % (self.nb_sample, self.nb_class))

        # second, build an index of the images in the different class subfolders
        self.filenames = []
        self.classes = np.zeros((self.nb_sample,), dtype='int32')
        i = 0
        for subdir in classes:
            subpath = os.path.join(directory, subdir)
            for root, _, files in _recursive_list(subpath):
                for fname in files:
                    is_valid = False
                    for extension in white_list_formats:
                        if fname.lower().endswith('.' + extension):
                            is_valid = True
                            break
                    if is_valid:
                        self.classes[i] = self.class_indices[subdir]
                        i += 1
                        # add filename relative to directory
                        absolute_path = os.path.join(root, fname)
                        self.filenames.append(os.path.relpath(absolute_path, directory))
        super(DirectoryIterator_with_hard_samples, self).__init__(self.nb_sample, batch_size, shuffle, seed)

    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel

        
        hard_samples_index = None
        if self.hard_samples is not None and self.hard_sample_rate > 0:
            try:
                hard_samples = np.array(self.hard_samples["hard_samples_index"])
                hard_sample_num = min(len(hard_samples), int(current_batch_size * self.hard_sample_rate))
                if hard_sample_num > 0 :
                    selected_hard_samples = hard_samples[np.random.permutation(len(hard_samples))[:hard_sample_num]]
                    for i in range(hard_sample_num):
                        if selected_hard_samples[i] not in index_array:
                            index_array[i] = selected_hard_samples[i]
            except Exception as e:
                print(str(e))
                pass 

        self.current_batch_filenames = [self.filenames[j] for j in index_array]
        self.current_index_array = index_array            
        
            
        batch_x = None
        grayscale = self.color_mode == 'grayscale'

        if self.pool:
            pipeline = self.image_data_generator.pipeline()
            result = self.pool.map(process_image_pipeline_dir, ((pipeline,
                self.filenames[j],
                self.directory,
                grayscale,
                self.target_size,
                self.dim_ordering,
                self.rngs[i%self.batch_size]) for i, j in enumerate(index_array)))
            batch_x = np.array(result)
        else:
            # TODO: also utilize image_data_generator.pipeline()?
            batch_x = np.zeros((current_batch_size,) + self.image_shape)
            # build batch of image data
            for i, j in enumerate(index_array):
                fname = self.filenames[j]
                img = load_img(os.path.join(self.directory, fname),
                               grayscale=grayscale,
                               target_size=self.target_size)
                x = img_to_array(img, dim_ordering=self.dim_ordering)
                x = self.image_data_generator.random_transform(x)
                x = self.image_data_generator.standardize(x)
                batch_x[i] = x
        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            for i in range(current_batch_size):
                img = array_to_img(batch_x[i], self.dim_ordering, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                  index=current_index + i,
                                                                  hash=np.random.randint(1e4),
                                                                  format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        # build batch of labels
        if self.class_mode == 'sparse':
            batch_y = self.classes[index_array]
        elif self.class_mode == 'binary':
            batch_y = self.classes[index_array].astype('float32')
        elif self.class_mode == 'categorical':
            batch_y = np.zeros((len(batch_x), self.nb_class), dtype='float32')
            for i, label in enumerate(self.classes[index_array]):
                batch_y[i, label] = 1.
        else:
            return batch_x
        
        self.current_data = batch_x
        self.current_label = batch_y
        return batch_x, batch_y

    def get_current_batch_meta(self):
        return self.current_index_array,self.current_data,self.current_label

class MetadataIterator(Iterator):

    def __init__(self, metadataFunc, image_data_generator,
                 target_size=(256, 256), color_mode='rgb',
                 dim_ordering='default',
                 classes=None, class_mode='categorical',
                 batch_size=32, shuffle=True, seed=None,
                 save_to_dir=None, save_prefix='', save_format='jpeg',
                 image_in_memory = False,
                 follow_links=False, pool=None):
        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        self.metadataFunc = metadataFunc
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        if color_mode not in {'rgb', 'grayscale'}:
            raise ValueError('Invalid color mode:', color_mode,
                             '; expected "rgb" or "grayscale".')
        self.color_mode = color_mode
        self.dim_ordering = dim_ordering
        if self.color_mode == 'rgb':
            if self.dim_ordering == 'tf':
                self.image_shape = self.target_size + (3,)
            else:
                self.image_shape = (3,) + self.target_size
        else:
            if self.dim_ordering == 'tf':
                self.image_shape = self.target_size + (1,)
            else:
                self.image_shape = (1,) + self.target_size
        self.classes = classes
        if class_mode not in {'categorical', 'binary', 'sparse', None}:
            raise ValueError('Invalid class_mode:', class_mode,
                             '; expected one of "categorical", '
                             '"binary", "sparse", or None.')
        self.class_mode = class_mode
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        self.pool = pool

        white_list_formats = {'png', 'jpg', 'jpeg', 'bmp'}

        # first, count the number of samples and classes
        self.nb_sample = 0

        if not classes:
            classes = []
            for subdir in sorted(os.listdir(directory)):
                if os.path.isdir(os.path.join(directory, subdir)):
                    classes.append(subdir)

        filenames, labels, classnames = metadataFunc()

        nb_sample = len(filenames)
        self.nb_class = len(classnames)
        self.class_indices = dict(zip(classnames, range(len(classnames))))
        self.nb_sample = nb_sample
        self.filenames = filenames

        if self.class_mode == 'sparse' or self.class_mode == 'binary' or self.class_mode == 'categorical':
            self.classes = np.asarray(labels, dtype='int32')
        else:
            self.classes = labels

        self.directory = "/"
        self.firstPrint = True # Disable printing for debugging. 

        self.image_in_memory = image_in_memory
        self.image_cache = {}        
        
        super(MetadataIterator, self).__init__(self.nb_sample, batch_size, shuffle, seed)

    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
            
        # The transformation of images is not under thread lock
        # so it can be done in parallel
            if self.image_in_memory:
                for filename in filenames:
                    if filename not in self.image_cache:
                        self.image_cache[filename] = None      
                        
        batch_x = None
        grayscale = self.color_mode == 'grayscale'
        
        if not self.firstPrint:
            import traceback
            print( "index_array shape ==== %s" % index_array.shape)
            print( "index_array[0] ====== %s" % index_array[0])
            for line in traceback.format_stack():
                print(line.strip())
            
            self.firstPrint = True

        if self.pool:
            bDone = False 
            while not bDone:
                try:
                    pipeline = self.image_data_generator.pipeline()
                    
                    if self.image_in_memory:
                        tmp_results = self.pool.map(process_image_pipeline_dir_cache, ((pipeline,
                            filenames[i],
                            self.directory,
                            grayscale,
                            self.target_size,
                            self.dim_ordering,
                            self.rngs[i%self.batch_size],
                            self.image_cache[filenames[i]]) for i in range(nsize) ))
                        result = [x for x,filename,img in tmp_results]
                        for x,filename,img in tmp_results:
                            if self.image_cache[filename] is None:
                                self.image_cache[filename] = img          
                    else:                    
                        result = self.pool.map(process_image_pipeline_dir, ((pipeline,
                            self.filenames[j],
                            self.directory,
                            grayscale,
                            self.target_size,
                            self.dim_ordering,
                            self.rngs[i%self.batch_size]) for i, j in enumerate(index_array)))
                    bDone = True
                except:
                    # Error happens in the last block 
                    print( "Skipped batch %d because of exception (file read error?), index === %s " % (current_index, index_array) )
                    with self.lock:
                        index_array, current_index, current_batch_size = next(self.index_generator)
            batch_x = np.array(result)
        else:
            # TODO: also utilize image_data_generator.pipeline()?
            batch_x = np.zeros((current_batch_size,) + self.image_shape)
            # build batch of image data
            for i, j in enumerate(index_array):
                fname = self.filenames[j]
                img = load_img(os.path.join(self.directory, fname),
                               grayscale=grayscale,
                               target_size=self.target_size)
                x = img_to_array(img, dim_ordering=self.dim_ordering)
                x = self.image_data_generator.random_transform(x)
                x = self.image_data_generator.standardize(x)
                batch_x[i] = x
        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            for i in range(current_batch_size):
                img = array_to_img(batch_x[i], self.dim_ordering, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                  index=current_index + i,
                                                                  hash=np.random.randint(1e4),
                                                                  format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        # build batch of labels
        if self.class_mode == 'sparse':
            batch_y = self.classes[index_array]
        elif self.class_mode == 'binary':
            batch_y = self.classes[index_array].astype('float32')
        elif self.class_mode == 'categorical':
            batch_y = np.zeros((len(batch_x), self.nb_class), dtype='float32')
            for i, label in enumerate(self.classes[index_array]):
                batch_y[i, label] = 1.
        else:
            return batch_x
        return batch_x, batch_y
    
class MetadataSeqIterator(Iterator):
    def __init__(self, metadataSeqFunc, image_data_generator,
                 target_size=(256, 256), color_mode='rgb',
                 dim_ordering='default',
                 classes=None, class_mode='categorical',
                 steps = None, 
                 batch_size=32, shuffle=True, seed=None,
                 save_to_dir=None, save_prefix='', save_format='jpeg',
                 follow_links=False, 
                 raise_exception = False, 
                 image_in_memory = False,
                 pool=None):
        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        self.metadataSeq = metadataSeqFunc()
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        if color_mode not in {'rgb', 'grayscale'}:
            raise ValueError('Invalid color mode:', color_mode,
                             '; expected "rgb" or "grayscale".')
        self.color_mode = color_mode
        self.dim_ordering = dim_ordering
        if self.color_mode == 'rgb':
            if self.dim_ordering == 'tf':
                self.image_shape = self.target_size + (3,)
            else:
                self.image_shape = (3,) + self.target_size
        else:
            if self.dim_ordering == 'tf':
                self.image_shape = self.target_size + (1,)
            else:
                self.image_shape = (1,) + self.target_size
        self.classes = classes
        if class_mode not in {'categorical', 'binary', 'sparse', None}:
            raise ValueError('Invalid class_mode:', class_mode,
                             '; expected one of "categorical", '
                             '"binary", "sparse", or None.')
        self.class_mode = class_mode
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        self.pool = pool

        white_list_formats = {'png', 'jpg', 'jpeg', 'bmp'}

        # first, count the number of samples and classes
        
        nb_sample = 0
        cnames = None
        labels = None
        for ( filenames, uselabels, classnames ) in metadataSeqFunc():
            nb_sample += len( filenames ) 
            cnames = classnames
            labels = uselabels
            
            
        self.nb_class = len(cnames)
        self.class_indices = dict(zip(cnames, range(len(cnames))))
        self.nb_sample = nb_sample

        if self.class_mode == 'sparse' or self.class_mode == 'binary' or self.class_mode == 'categorical':
            self.classes = np.asarray(labels, dtype='int32')
        else:
            self.classes = labels

        self.directory = "/"
        self.firstPrint = True # Disable printing for debugging. 
        self.raise_exception = raise_exception
        
        self.image_in_memory = image_in_memory
        self.image_cache = {}
        
        super(MetadataSeqIterator, self).__init__(self.nb_sample, batch_size, shuffle, seed)
    
    def reset(self):
        with self.lock:
            self.metadataSeq.reset()
        super(MetadataSeqIterator, self).reset()
   
    def __len__(self):
        return len(self.metadataSeq)
        
    def next(self):
        with self.lock:
            try:
                filenames, labels, classnames = next(self.metadataSeq)
            except StopIteration:
                if self.raise_exception:
                    raise
                filenames, labels, classnames = next(self.metadataSeq)
            if self.image_in_memory:
                for filename in filenames:
                    if filename not in self.image_cache:
                        self.image_cache[filename] = None                
        # The transformation of images is not under thread lock
        # so it can be done in parallel

        batch_x = None
        grayscale = self.color_mode == 'grayscale'
        
        if not self.firstPrint:
            import traceback
            print( "Labels shape ==== %s" % labels.shape)           
            self.firstPrint = True

        if self.pool:
            bDone = False 
            while not bDone:
               
                bDone = True                
                try:
                    nsize = len( filenames ) 
                    pipeline = self.image_data_generator.pipeline()
                    if self.image_in_memory:
                        tmp_results = self.pool.map(process_image_pipeline_dir_cache, ((pipeline,
                            filenames[i],
                            self.directory,
                            grayscale,
                            self.target_size,
                            self.dim_ordering,
                            self.rngs[i%self.batch_size],
                            self.image_cache[filenames[i]]) for i in range(nsize) ))
                        result = [x for x,filename,img in tmp_results]
                        for x,filename,img in tmp_results:
                            if self.image_cache[filename] is None:
                                self.image_cache[filename] = img          
                    else:
                        result = self.pool.map(process_image_pipeline_dir, ((pipeline,
                            filenames[i],
                            self.directory,
                            grayscale,
                            self.target_size,
                            self.dim_ordering,
                            self.rngs[i%self.batch_size]) for i in range(nsize) ))
                        
                    
                    bDone = True
                except:
                    # Error happens in the last block 
                    print( "Skipped batch because of exception (file read error?), filenames === %s " % (filenames) )
                    with self.lock:
                        try:
                            filenames, labels, classnames = next(self.metadataSeq)
                        except StopIteration:
                            if self.raise_exception:
                                raise
                            filenames, labels, classnames = next(self.metadataSeq)
            batch_x = np.array(result)
        else:
            while not bDone:
                try:
                    # TODO: also utilize image_data_generator.pipeline()?
                    batch_x = np.zeros((current_batch_size,) + self.image_shape)
                    # build batch of image data
                    nsize = len( filenames ) 
                    for i in enumerate(index_array):
                        fname = filenames[i]
                        img = load_img(os.path.join(self.directory, fname),
                                       grayscale=grayscale,
                                       target_size=self.target_size)
                        x = img_to_array(img, dim_ordering=self.dim_ordering)
                        x = self.image_data_generator.random_transform(x)
                        x = self.image_data_generator.standardize(x)
                        batch_x[i] = x
                except:
                    # Error happens in the last block 
                    print( "Skipped batch because of exception (file read error?), filenames === %s " % (filenames) )
                    with self.lock:
                        try:
                            filenames, labels, classnames = next(self.metadataSeq)
                        except StopIteration:
                            if self.raise_exception:
                                rais
                            filenames, labels, classnames = next(self.metadataSeq)

                
        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            for i in range(nsize):
                img = array_to_img(batch_x[i], self.dim_ordering, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                  index=current_index + i,
                                                                  hash=np.random.randint(1e4),
                                                                  format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        # build batch of labels
        if self.class_mode == 'sparse':
            batch_y = np.zeros( (nsize, self.nb_class), dtype='float32' )
            batch_y[ np.arrange( nsize), labels ] = 1
        elif self.class_mode == 'binary':
            batch_y = np.array(labels).astype('float32')
        elif self.class_mode == 'categorical':
            batch_y = np.zeros((nsize, self.nb_class), dtype='float32')
            batch_y[ np.arange( nsize), labels ] = 1
            # for i, label in enumerate(labels):
            #    batch_y[i, label] = 1.
        else:
            return batch_x
        return batch_x, batch_y

# Siamese should be binary 
# metadataSeqFunc should return [(filename1, filename2)], label
class MetadataSeqSiameseIterator(Iterator):
    def __init__(self, metadataSeqFunc, image_data_generator,
                 target_size=(256, 256), color_mode='rgb',
                 dim_ordering='default',
                 classes=None, # class_mode='binary',
                 steps = None, 
                 batch_size=32, shuffle=True, seed=None,
                 save_to_dir=None, save_prefix='', save_format='jpeg',
                 follow_links=False, pool=None):
        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        self.metadataSeq = metadataSeqFunc()
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        if color_mode not in {'rgb', 'grayscale'}:
            raise ValueError('Invalid color mode:', color_mode,
                             '; expected "rgb" or "grayscale".')
        self.color_mode = color_mode
        self.dim_ordering = dim_ordering
        if self.color_mode == 'rgb':
            if self.dim_ordering == 'tf':
                self.image_shape = self.target_size + (3,)
            else:
                self.image_shape = (3,) + self.target_size
        else:
            if self.dim_ordering == 'tf':
                self.image_shape = self.target_size + (1,)
            else:
                self.image_shape = (1,) + self.target_size
        self.classes = classes
        #if class_mode not in {'binary'}: # {'categorical', 'binary', 'sparse', None}:
        #    raise ValueError('Invalid class_mode:', class_mode,
        #                     '; expected one of "categorical", '
        #                     '"binary", "sparse", or None.')
        self.class_mode = 'binary' # class_mode
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        self.pool = pool

        white_list_formats = {'png', 'jpg', 'jpeg', 'bmp'}
        
        if steps is None:
            # first, count the number of samples and classes
            nb_sample = 0
            labels = None
            for ( filenames, uselabels ) in metadataSeqFunc():
                nb_sample += len( filenames ) 
                labels = uselabels
        else:
            nb_sample = step * batch_size
            labels = [0]
           
        self.nb_class = 1 # len(cnames)
        self.class_indices = {'similarity': 0 } # dict(zip(cnames, range(len(cnames))))
        self.nb_sample = nb_sample

        if self.class_mode == 'sparse' or self.class_mode == 'binary' or self.class_mode == 'categorical':
            self.classes = np.asarray(labels, dtype='int32')
        else:
            self.classes = labels

        self.directory = "/"
        self.firstPrint = True # Disable printing for debugging. 
        super(MetadataSeqSiameseIterator, self).__init__(self.nb_sample, batch_size, shuffle, seed)
    
    def reset(self):
        with self.lock:
            self.metadataSeq.reset()
        super(MetadataSeqSiameseIterator, self).reset()

        
    def next(self):
        with self.lock:
            try:
                filenames, labels = next(self.metadataSeq)
            except StopIteration:
                filenames, labels = next(self.metadataSeq)
        # The transformation of images is not under thread lock
        # so it can be done in parallel

        batch_x = None
        grayscale = self.color_mode == 'grayscale'
        
        if not self.firstPrint:
            import traceback
            print( "Labels shape ==== %s" % labels.shape)           
            self.firstPrint = True

        if self.pool:
            bDone = False 
            while not bDone:
                try:
                    nsize = len( filenames ) 
                    pipeline = self.image_data_generator.pipeline()
                    result0 = self.pool.map(process_image_pipeline_dir, ((pipeline,
                        filenames[i][0],
                        self.directory,
                        grayscale,
                        self.target_size,
                        self.dim_ordering,
                        self.rngs[i%self.batch_size]) for i in range(nsize) ))
                    result1 = self.pool.map(process_image_pipeline_dir, ((pipeline,
                        filenames[i][1],
                        self.directory,
                        grayscale,
                        self.target_size,
                        self.dim_ordering,
                        self.rngs[i%self.batch_size]) for i in range(nsize) ))
                    bDone = True
                    # print( "Successful to read in the batch " )
                except:
                    # Error happens in the last block 
                    print( "Skipped batch because of exception (file read error?), filanames === %s " % (filenames) )
                    with self.lock:
                        try:
                            filenames, labels = next(self.metadataSeq)
                        except StopIteration:
                            filenames, labels = next(self.metadataSeq)
            batch_x = [np.array(result0), np.array( result1) ]
            # print( "Batch size === %d" % (batch_x[0].shape[0]) )
            # pairs = []
            #for i in range( nsize ):
            #    pairs += [[result0[i], result1[i]]]
            # batch_x = {"0":result0, "1":result1}
            #x = batch_x
            #if x is None or len(x) == 0:
            #            # Handle data tensors support when no input given
            #            # step-size = 1 for data tensors
            #            batch_size = 1
            #elif isinstance(x, list):
            #            batch_size = x[0].shape[0]
            #elif isinstance(x, dict):
            #            batch_size = list(x.values())[0].shape[0]
            #else:
            #            batch_size = x.shape[0]
            #print( "Batch size === %d " % batch_size )
        else:
            while not bDone:
                try:
                    # TODO: also utilize image_data_generator.pipeline()?
                    pairs0 = []
                    pairs1 = []
                    # build batch of image data
                    nsize = len( filenames ) 
                    for i in enumerate(index_array):
                        fname = filenames[i][0]
                        img = load_img(os.path.join(self.directory, fname),
                                       grayscale=grayscale,
                                       target_size=self.target_size)
                        x = img_to_array(img, dim_ordering=self.dim_ordering)
                        x = self.image_data_generator.random_transform(x)
                        x0 = self.image_data_generator.standardize(x)
                        fname = filenames[i][1]
                        img = load_img(os.path.join(self.directory, fname),
                                       grayscale=grayscale,
                                       target_size=self.target_size)
                        x = img_to_array(img, dim_ordering=self.dim_ordering)
                        x = self.image_data_generator.random_transform(x)
                        x1 = self.image_data_generator.standardize(x)
                        pairs0 += [x0]
                        pairs1 += [x1]
                    bDone = True
                except:
                    # Error happens in the last block 
                    print( "Skipped batch because of exception (file read error?), filenames === %s " % (filenames) )
                    with self.lock:
                        try:
                            filenames, labels = next(self.metadataSeq) 
                        except StopIteration:
                            filenames, labels = next(self.metadataSeq)
            batch_x = [ np.array( pairs0 ), np.array( pairs1 ) ]

        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            for i in range(nsize):
                img0 = array_to_img(batch_x[i][0], self.dim_ordering, scale=True)
                img1 = array_to_img(batch_x[i][1], self.dim_ordering, scale=True)
                label = labels[i]
                hashinfo = np.random.randint(1e4)
                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                  index=current_index + i,
                                                                  hash=hashinfo,
                                                                  format=self.save_format)
                img0.save(os.path.join(self.save_to_dir, fname))
                flag = ['neg', 'pos'][label]
                fname1 = '{prefix}_{index}_{hash}_{flag}.{format}'.format(prefix=self.save_prefix,
                                                                  index=current_index + i,
                                                                  hash=hashinfo,
                                                                  flag=flag,
                                                                  format=self.save_format)
                img1.save(os.path.join(self.save_to_dir, fname1))

        # build batch of labels
        if self.class_mode == 'binary':
            batch_y = np.array( labels ) 
        else:
            return batch_x
        return batch_x, batch_y

def make_closure(content):
    # This is the outer enclosing function
    def invoke():
    # This is the nested function
        return content
    return invoke     
############################################################
#  Dataset
############################################################
class MetadataSeq():
    def __init__(self, classnames, filelist, metadata, mapping, batch_size, verbose = True, root_dir = None ):
        self.verbose = verbose
        if root_dir is None:
            self.root_dir = "./" 
        else:
            self.root_dir = root_dir
        self.classnames = classnames
        self.filelist = filelist
        self.index = 0
        self.batch_size = batch_size
        self.metadata = metadata
        self.mapping = mapping 
    def reset():
        self.index = 0
    def __iter__(self):
        return self  
    def __len__(self):
        return len(self.filelist) // self.batch_size 
    def __next__(self):
        filenames = []
        labels = []
        nsize = len( self.filelist )
        if self.index + self.batch_size > nsize:
            self.index = 0
            raise StopIteration
        for i in range( self.batch_size):
            cur = self.filelist[ (i+self.index)%nsize ]
            filenames.append( os.path.join( self.root_dir, cur[0] ) )
            labels.append( cur[1] )
        self.index = self.index + self.batch_size
        return filenames, labels, self.classnames
    @property
    def classes(self):
        np.asarray(list( map( lambda x: x[1], self.filelist )), dtype = 'int' )
    @property
    def steps(self):
        nsize = len( self.filelist )
        return nsize // self.batch_size
    @property
    def histogram(self):
        hist = np.zeros( (len(self.classnames)) )
        for key, value in self.metadata.items():
            hist[self.mapping[key]] = len(value)
        return hist
    
class MulticropSeq():
    def __init__(self, classnames, filelist, metadata, mapping, batch_size, ncrop, verbose = True, root_dir = None ):
        self.verbose = verbose
        if root_dir is None:
            self.root_dir = "./" 
        else:
            self.root_dir = root_dir
        self.classnames = classnames
        self.filelist = filelist
        self.index = 0
        self.batch_size = batch_size
        self.ncrop = ncrop
        self.metadata = metadata
        self.mapping = mapping 
    def reset():
        self.index = 0
    def __iter__(self):
        return self        
    def __next__(self):
        filenames = []
        labels = []
        nsize = len( self.filelist )
        if ( self.index + 1 )* self.batch_size > nsize * self.ncrop:
            self.index = 0
            raise StopIteration
        idx = self.index * self.batch_size
        for i in range( self.batch_size):
            cur = self.filelist[ (i+idx) // self.ncrop ]
            filenames.append( os.path.join( self.root_dir, cur[0] ) )
            labels.append( cur[1] )
        self.index += 1
        return filenames, labels, self.classnames
    @property
    def classes(self):
        return np.asarray(list( map( lambda x: x[1], self.filelist )), dtype = 'int' )
    @property
    def steps(self):
        nsize = len( self.filelist )
        return nsize * self.ncrop // self.batch_size
    @property
    def histogram(self):
        hist = np.zeros( (len(self.classnames)) )
        for key, value in self.metadata.items():
            hist[self.mapping[key]] = len(value)
        return hist

# Siamese sequence: return a pair of image (positive, negative). 
class MetadataSeqSiamese():
    def __init__(self, classnames, filelist, metadata, mapping, batch_size, verbose = True, root_dir = None, seed = 0, equiv = None ):
        self.verbose = verbose
        if root_dir is None:
            self.root_dir = "./" 
        else:
            self.root_dir = root_dir 
        self.classnames = classnames
        self.filelist = filelist
        self.index = 0
        self.batch_size = batch_size
        self.metadata = metadata
        self.mapping = mapping 
        self.seed = seed 
        self.rng = random.Random(self.seed)
        self.equiv = equiv
    def reset():
        self.index = 0
        self.rng = random.Random(self.seed)
    def __iter__(self):
        return self        
    def __next__(self):
        filenames = []
        labels = []
        nsize = len( self.filelist )
        # print ("Iteration with index === %d" % (self.index ) )
        if self.index + self.batch_size > nsize * 2:
            # print( "Done iteration with index === %d, batch === %d, files === %d" % ( self.index, self.batch_size, nsize ) )
            self.index = 0
            raise StopIteration
        for i in range( self.batch_size):
            idx = (i+self.index) // 2
            label = (i+self.index) % 2
            cur = self.filelist[ idx % nsize ] # Base example
            fname0 = os.path.join( self.root_dir, cur[0] )
            cl = cur[1]
            classname = self.classnames[cl]
            if label == 1:
                if classname in self.metadata:
                    clsize = len( self.metadata[classname] )
                    # print( "Size of %s is %d" % (classname, clsize) )
                    # Choose an example from the same dataset
                    if clsize > 1:
                        # Find a positive example in class:
                        bFind = False
                        while not bFind:
                            fidx = self.rng.randint( 0, len( self.metadata[classname] ) - 1 ) # at least one more number 
                            fname1 = os.path.join( self.root_dir, classname, self.metadata[classname][fidx] )
                            bFind = ( fname0 != fname1 ) # We have at least two members, so there should be at least one other file in class
                else:
                    label = 0 # No positive example, choose a negative example. 
            if label == 0:
                # Generate negative example
                bFind = False
                while not bFind:
                    nclasses = len( self.classnames)
                    cidx = self.rng.randint( 0, nclasses - 1 )
                    bNegative = cidx != cl
                    if self.equiv and bNegative:
                        # Evaluate on equivalent class 
                        classname1 = self.classnames[cidx]
                        if classname in self.equiv:
                            if classname1 in self.equiv[classname]:
                                # The two classes are equivalent
                                bNegative = False
                                print( "Search again, classes %s and %s are equivalent ... " % (classname, classname1) )
                    if bNegative:
                        # find a negative example
                        classname = self.classnames[cidx]
                        if classname in self.metadata:
                            clsize = len( self.metadata[classname] )
                            # print ( "Pick negative example from %s with size %d" % ( classname, clsize) )
                            if clsize > 0:
                                fidx = self.rng.randint( 0, len( self.metadata[classname] ) - 1 ) # at least one more number 
                                fname1 = os.path.join( self.root_dir, classname, self.metadata[classname][fidx] )
                                bFind = ( fname0 != fname1 ) 
            # print( "Pair === %s, %s" %(fname0, fname1) )
            filenames.append( ( fname0, fname1) )
            labels.append( label )
        self.index = self.index + self.batch_size
        return filenames, labels
    @property
    def steps(self):
        nsize = len( self.filelist )
        return ( nsize * 2 ) // self.batch_size
    @property
    def histogram(self):
        hist = np.zeros( (len(self.classnames)) )
        for key, value in self.metadata.items():
            hist[self.mapping[key]] = len(value)
        return hist
    

############################################################
#  Dataset
############################################################
    
class DatasetSubdirectory():
    def __init__(self, root_dir, metadata_file, data_dir, equiv_file = None, verbose = True, seed = 0, splits = {"train": 80, "val": 20 } ):
        super().__init__() 
        self.metadata = { }
        self.metadata_nontrain = { }
        self.classnames = [ ]
        self.mapping = { }
        self.fileinfo = {}
        self.verbose = verbose
        self.root_dir = root_dir
        self.data_dir = os.path.join( root_dir, data_dir ) 
        self.ready = False
        self.limits = None
        self.metadata = {}
        self.list = {}
        self.seed = seed
        self.metadata_file = os.path.join( self.root_dir, metadata_file)
        if not os.path.isfile( self.metadata_file):
            self.prepare_metadata()
        self.equiv_file = equiv_file
    
    def prepare_metadata(self):
        classmapping = {}
    
        numdir = 0 
        metadata = {}
        for root, dirs, files in os.walk( self.data_dir ):
            if len( files ) > 0:
                basename = os.path.basename( root )
                metadata[basename] = files
                classmapping[basename] = numdir
                numdir += 1
                if numdir % 1000 == 0:
                    print( "Proccess %d directories ... " % numdir )

        info = metadata
        with open( self.metadata_file, "w") as outfile:
            json.dump( info, outfile )
        
    # this should be called to initialize all necessary data structure 
    # Train threshold: at least this number of samples in training. 
    # Mapping: a dictionary that maps a class name (subdirectory) to a category
    # classes: a list of classes that intrepret classname -> pos
    def prepare( self, seed = 0, splits = {"train": 80, "val": 20 }, train_threshold = 5, classes = None, 
                mapping = None, class_index = None ): 
        with open( self.metadata_file, "r") as fp:
            metadata = json.load( fp )
        lst = []
        cnt = 0
        bComputeMapping = False
        if class_index is not None:
            # class_index is in the form of imagenet_utils.get_imagenet_class_index()
            # it is a dictionary with entry '465': ['n02916936', 'bulletproof_vest'],
            tuples = sorted( class_index.items(), key = itemgetter(1) )
            lastname = ""
            bOrdered = True
            self.classnames = []
            self.mapping = {}
            for key, keyinfo in tuples:
                if int( key ) != len( self.classnames ):
                    print ("Out of order: %s %s" % (key, keyinfo) )
                self.classnames.append( keyinfo[0] )
                if keyinfo[0] < lastname: 
                    print( "Out of order: %s: %s (last: %s)" % (key, keyinfo, lastname ) )
                    bOrdered = False
                lastname = keyinfo[0]
                for name in keyinfo:
                    self.mapping[name] = int( key ) 
                
            if not bOrdered:
                print( "Caution: class_index is not ordered" )
            else:
                print( "class_index is properly ordered" )
        else:
            if classes is None and mapping is None:
                classes = sorted( map( lambda x : x[0], metadata.items() ))
                # print (classes)

            if not (mapping is None): 
                self.mapping = mapping
                mx = 0 
                for key, value in mapping.items():
                    mx = max( mx, value )
                self.classnames = "0" * (mx + 1 )
                for key, value in mapping.items():
                    self.classnames[value] = key
            else:
                if not (classes is None):
                    for idx, classname in enumerate(classes):
                        self.mapping[classname] = idx
                    self.classnames = classes
        if splits is None:
            splits = {}
            
        # print(len(self.metadata))
        for classname, filesinfo in metadata.items():
            cl = self.mapping[classname]
            for file in filesinfo:
                lst.append( (file, cl) ) 
            cnt = cnt + 1
            
        random.Random(seed).shuffle(lst)
        total = 0
        for key, value in splits.items():
            total = total + value
            self.metadata[key] = {}
        # Slot item to "train", "val", etc.. 
        start = 0
        cumul = 0
        for key, value in splits.items():
            cumul = cumul + value
            end = ( cumul * len( lst ) + (total//2)) // total 
            #  print( "Data %s of size %d, with start = %d, end = %d" % (key, len(lst ), start, end) )
            self.list[key] = []
            for tup in lst[start:end]:
                fname = tup[0]
                cl = tup[1]
                classname = self.classnames[cl]
                if not ( classname in self.metadata[key] ):
                    self.metadata[key][classname] = []
                self.metadata[key][classname].append( fname ) 
            start = end
            
        # Identify if any item has very low number of class in training (not trainable ). 
        if train_threshold > 0 and "train" in self.metadata and "val" in self.metadata:
            move_class = []
            for classname, filelists in self.metadata["train"].items():
                if len( filelists ) < train_threshold:
                    move_class.append( classname )
            for classname in move_class:
                if not (classname in self.metadata["val"]):
                    self.metadata["val"][classname] = self.metadata["train"][classname]
                else:
                    self.metadata["val"][classname] += self.metadata["train"][classname]
                self.metadata["train"].pop(classname)
        # Form list. move proper list from train to val if of lower count
        start = 0
        cumul = 0
        for key, value in splits.items():
            cumul = cumul + value
            end = ( cumul * len( lst ) + (total//2)) // total 
            #  print( "Data %s of size %d, with start = %d, end = %d" % (key, len(lst ), start, end) )
            for tup in lst[start:end]:
                fname = tup[0]
                cl = tup[1]
                classname = self.classnames[cl]
                if classname in self.metadata[key]:
                    self.list[key].append( ( os.path.join(classname, fname),cl))
                else:
                    # Move train to val
                    self.list["val"].append( ( os.path.join(classname, fname),cl))
            start = end
            
            
        self.list["all"] = []
        self.metadata["all"] = {}
        for tup in lst:
            fname = tup[0]
            cl = tup[1]
            try:
                classname = self.classnames[cl]
                self.list["all"].append( (os.path.join( classname, fname), cl ) )
                if not ( classname in self.metadata["all"] ):
                    self.metadata["all"][classname] = []
                self.metadata["all"][classname].append( fname ) 
            except:
                print( "Entry not properly formatted fname, cl == %s, %s" % (fname, cl) )
            
        for key, value in self.list.items():
            print( "%s Data %s has %d items" % (self.data_dir, key, len(self.list[key]) ) )

    def metadata_seq( self, subset=None, batch_size=32 ):
        if subset is None:
            subset = "all"
        assert subset in self.list
        return make_closure( MetadataSeq( self.classnames, self.list[subset], self.metadata[subset], self.mapping, batch_size, verbose = self.verbose, root_dir=self.data_dir ) )
    
    def metadata_multicrop_seq( self, subset, batch_size, ncrop ):
        assert subset in self.list
        return make_closure( MulticropSeq( self.classnames, self.list[subset], self.metadata[subset], self.mapping, batch_size, ncrop, verbose = self.verbose, root_dir=self.data_dir ) )
    
    def metadata_seq_siamese( self, subset, batch_size ):
        equiv = None
        if self.equiv_file:
            filename = os.path.join( self.root_dir, self.equiv_file)
            with open( filename, "r") as fp:
                equiv = json.load( fp )
        if subset == "train":
            return make_closure( MetadataSeqSiamese( self.classnames, self.list[subset], self.metadata[subset], self.mapping, batch_size, verbose = self.verbose, root_dir=self.data_dir, equiv = equiv ) )
        else:
            return make_closure( MetadataSeqSiamese( self.classnames, self.list[subset], { **self.metadata[subset], **self.metadata["train"] }, self.mapping, batch_size, verbose = self.verbose, root_dir=self.data_dir, seed = self.seed, equiv = equiv ) )

# Helper function for image classification        
class ClassificationResults():
    def __init__(self, pred, truth, num_classes):
        self.pred = pred
        self.truth = truth
        self.num_classes = num_classes
        self.truepositive = np.array([0]*num_classes)
        self.truenegative = np.array([0]*num_classes)
        self.falsepositive = np.array([0]*num_classes)
        ff = np.array([0]*num_classes)
        for i in range(len(truth)):
            j = truth[i]
            if j == pred[i]:
                # Actual class is j and it is predicted as j:
                self.truepositive[j] += 1
            else:
                # Prediction result is different 
                self.truenegative[j] += 1
                k = pred[i]
                self.falsepositive[k] += 1
    @property
    def recall(self):
        return self.truepositive / ( self.truepositive + self.truenegative )
    @property
    def precision(self):
        return self.truepositive / ( self.truepositive + self.falsepositive )
    @property
    def f1(self):
        return 2/( (1/self.recall) + 1/ (self.precision) )
        
import itertools
import matplotlib.pyplot as plt
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def process_ncrop_result( result, ncrop):
    nx, ny = result.shape
    nsize = nx // ncrop
    ret = np.zeros( (nsize, ny) ) 
    for i in range( nsize):
        ret[i,:] = np.mean( result[i*ncrop:(i+1)*ncrop, :], axis=0 )
        # if i == 0:
        #     print( "Result %d X %d Average %s to %s" % ( nx, ny, result[i*ncrop:(i+1)*ncrop, :], ret[i,:] ) )
    return ret

class ImageReader:
    def __init__(self, rootdir, dim_ordering='default', **kwargs ):
        self.rootdir = rootdir
        self.kwargs = kwargs
        self.dim_ordering = dim_ordering
    def read( self, file ):
        image = load_img( os.path.join(self.rootdir, file), **self.kwargs )
        img = img_to_array( image, dim_ordering = self.dim_ordering )
        # print (img.shape )
        return img

def read_in_images( rootdir, pattern=".*.jpg", pool = None, dim_ordering='default', **kwargs ):
    ex = re.compile( pattern )
    filelist = []
    for file in os.listdir(rootdir):
        if ex.match( file ):
            filelist.append(file)
    filelist.sort()
    if pool:
        imgReader = ImageReader( rootdir, dim_ordering, **kwargs ) 
        images = pool.map( imgReader.read, filelist )
    else:
        images = []
        for f in filelist:
            image = load_img( os.path.join(rootdir, f), **kwargs)
            img = img_to_array( image, dim_ordering = dim_ordering )
            # print (img.shape )
            images.append( img )
    return filelist, np.array( images )

def print_layers( model, first = None, last = None ):
    nlayers = len( model.layers )
    idx = 0
    for layer in model.layers:
        bPrint = True
        if first or last:
            bPrint = False
            if first and idx < first:
                bPrint = True
            if last and idx >= nlayers - last:
                bPrint = True
        if bPrint:
            print ( "Layer %d ==== %s" % (idx, layer.name ) )
        idx += 1
        
class Tee(object):
    def __init__(self, name):
        print( "Tee started, output buffered to %s" %name )
        self.file = open(name, "w", 1)
        self.stdout = sys.stdout
        sys.stdout = self
    def __del__(self):
        print( "Tee stopped" )
        sys.stdout = self.stdout
        self.file.close()
    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
    def flush(self):
        self.file.flush()
        
