# -*- coding: utf-8 -*-
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
import tensorflow as tf


class MaxPoolingWithArgmax2D(Layer):
    def __init__(self, pool_size=(2, 2), strides=(2, 2), padding='same', **kwargs):
        super(MaxPoolingWithArgmax2D, self).__init__(**kwargs)
        self.padding = padding
        self.pool_size = pool_size
        self.strides = strides

    def call(self, inputs, **kwargs):
        padding = self.padding
        pool_size = self.pool_size
        strides = self.strides
        if K.backend() == 'tensorflow':
            ksize = [1, pool_size[0], pool_size[1], 1]
            padding = padding.upper()
            strides = [1, strides[0], strides[1], 1]
            output, argmax = tf.nn.max_pool_with_argmax(inputs, ksize=ksize, strides=strides, padding=padding)
        else:
            errmsg = '{} backend is not supported for layer {}'.format(K.backend(), type(self).__name__)
            raise NotImplementedError(errmsg)
        argmax = K.cast(argmax, K.floatx())
        return [output, argmax]

    def compute_output_shape(self, input_shape):
        ratio = (1, 2, 2, 1)
        output_shape = [dim // ratio[idx] if dim is not None else None for idx, dim in enumerate(input_shape)]
        output_shape = tuple(output_shape)
        return [output_shape, output_shape]

    def compute_mask(self, inputs, mask=None):
        return 2 * [None]


class MaxUnpooling2D(Layer):
    def __init__(self, size=(2, 2), **kwargs):
        super(MaxUnpooling2D, self).__init__(**kwargs)
        self.size = size

    def call(self, inputs, output_shape=None):
        updates, mask = inputs[0], inputs[1]
        with tf.variable_scope(self.name):
            mask = K.cast(mask, 'int32')
            input_shape = tf.shape(updates, out_type='int32')
            #  calculation new shape
            if output_shape is None:
                output_shape = (input_shape[0], input_shape[1] * self.size[0], input_shape[2] * self.size[1], input_shape[3])
            self.output_shape1 = output_shape

            # calculation indices for batch, height, width and feature maps
            one_like_mask = K.ones_like(mask, dtype='int32')
            batch_shape = K.concatenate([[input_shape[0]], [1], [1], [1]], axis=0)
            batch_range = K.reshape(tf.range(output_shape[0], dtype='int32'), shape=batch_shape)
            b = one_like_mask * batch_range
            y = mask // (output_shape[2] * output_shape[3])
            x = (mask // output_shape[3]) % output_shape[2]
            feature_range = tf.range(output_shape[3], dtype='int32')
            f = one_like_mask * feature_range

            # transpose indices & reshape update values to one dimension
            updates_size = tf.size(updates)
            indices = K.transpose(K.reshape(K.stack([b, y, x, f]), [4, updates_size]))
            values = K.reshape(updates, [updates_size])
            ret = tf.scatter_nd(indices, values, output_shape)
            ret.set_shape([updates.shape[0], updates.shape[1] * self.size[0], updates.shape[2] * self.size[1], updates.shape[3]])
            return ret

    def compute_output_shape(self, input_shape):
        mask_shape = input_shape[1]
        return mask_shape[0], mask_shape[1] * self.size[0], mask_shape[2] * self.size[1], mask_shape[3]


class CombineMotionWithImg(Layer):
    '''
    Combines upsampled velocity vector with a grid of pixel positions scaled to
    match the size of image input.
    '''
    def __init__(self, **kwargs):
        super(CombineMotionWithImg, self).__init__(**kwargs)

    def call(self, inputs, output_shape=None):
        with tf.variable_scope(self.name):
            # Create a 2D array of x and y coordinates of each pixel, with the centre
            # of the image being (0,0)
            input_shape = tf.shape(inputs)
            input_shape_fl = tf.cast(input_shape, tf.float32)


            width = input_shape_fl[-3]
            height = input_shape_fl[-2]
            xs = tf.range(width, dtype=tf.float32)
            xs = tf.floor(tf.subtract(xs, tf.divide(width, tf.constant(2, dtype=tf.float32))))
            xs = tf.divide(xs, width)
            ys = tf.range(height, dtype=tf.float32)
            ys = tf.floor(tf.subtract(ys, tf.divide(height, tf.constant(2, dtype=tf.float32))))
            ys = tf.divide(ys, height)
            xs, ys = tf.meshgrid(ys, xs)
            xs = tf.expand_dims(xs, axis=-1)
            ys = tf.expand_dims(ys, axis=-1)
            xy_grid = tf.concat([xs, ys], axis=-1)
        
            xy_grid = tf.expand_dims(xy_grid, axis=0)
            batch_size = input_shape[0:1]
            batch_size = tf.concat([batch_size, tf.ones((3,), dtype=tf.int32)], axis=0)
            xy_grid = tf.tile(xy_grid, batch_size)
            
            pos_vel_pre_input = tf.concat([inputs, xy_grid], axis=-1)

            return pos_vel_pre_input

    def compute_output_shape(self, input_shape):
        return input_shape[-3], input_shape[-2], input_shape[-1]
