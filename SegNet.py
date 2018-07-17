# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Reshape, Permute
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Multiply, Concatenate
# from tensorflow.keras.utils import np_utils
import tensorflow.keras.backend as K
from tensorflow.python import debug as tf_debug

from Mylayers import MaxPoolingWithArgmax2D, MaxUnpooling2D
from generator import data_gen

import os
import numpy as np
import argparse
import json
import pandas as pd
from PIL import Image
import pickle
import pdb


def CreateSegNet(input_shape, kernel=3, pool_size=(2, 2), output_mode="softmax"):
    # encoder
    K.set_image_data_format('channels_last')

    inputs = Input(shape=(input_shape[1], input_shape[0], input_shape[2]))

    layers = [(2, 64), (2, 128), (3, 256), (3, 512), (3, 512)]
    layers = [(1, 64), (1, 64)]
    end_dec, masks = SegNetEncoderDecoderGenerator(inputs, layers=layers,
                                kernel=3, pool_size=(2, 2),
                                shave_off_decoder_end=0)


    # conv_24 = Convolution2D(64, (kernel, kernel), padding="same")(end_dec)
    # conv_24 = BatchNormalization()(conv_24)
    # conv_24 = Activation("relu")(conv_24)

    # print(masks[0].shape)
    # unpool_5 = MaxUnpooling2D(pool_size)([conv_24, masks[0]])

    # conv_25 = Convolution2D(64, (kernel, kernel), padding="same")(unpool_5)
    # conv_25 = BatchNormalization()(conv_25)
    # conv_25 = Activation("relu")(conv_25)

    # conv_26 = Convolution2D(1, (1, 1), padding="valid")(end_dec)
    # conv_26 = BatchNormalization()(conv_26)
    # conv_26 = Reshape((input_shape[0] * input_shape[1], 1), input_shape=(input_shape[0], input_shape[1], 1))(conv_26)

    outputs = Activation(output_mode)(conv_26)

    segnet = Model(inputs=inputs, outputs=outputs, name="OFNet")

    return segnet

def SegNetEncoderDecoderGenerator(inputs, layers, kernel=3, pool_size=(2, 2), shave_off_decoder_end=0):
    '''
    Creates an convolutional symmetric encoder-decoder architecture similar to
    that of a SegNet. The exact structure is defined by the 'layers' argument
    which should be a list of tuples, each tuple representing the number and
    width of convolutional layers at each stage of encoder/decoder.
    '''

    masks = []

    io = inputs
    total_layers = 0

    # encoder
    for group in layers:
        layers_no = group[0]
        width = group[1]
        for i in range(layers_no):
            io = Convolution2D(width, (kernel, kernel), padding="same")(io)
            io = BatchNormalization()(io)
            io = Activation("relu")(io)
            total_layers += 1

        io, mask = MaxPoolingWithArgmax2D(pool_size)(io)
        masks.append(mask)

    print("Build enceder done..")

    # decoder
    if shave_off_decoder_end > 0:
        total_layers -= shave_off_decoder_end

    # pdb.set_trace()
    for idx, group in enumerate(reversed(layers)):
        layers_no = group[0]
        width = group[1]

        io = MaxUnpooling2D(pool_size)([io, masks[-1-idx]])

        for i in range(layers_no):
            io = Convolution2D(width, (kernel, kernel), padding="same")(io)
            io = BatchNormalization()(io)
            io = Activation("relu")(io)
            total_layers -= 1
            if total_layers <= 0:
                return io, masks[:-1-idx]


def main(args):
    train_gen = data_gen(args.data_dir, args.batch_size,
                         timestamp_range=(0.1, 0.8), range_in_fractions=True,
                         img_resolution=args.input_shape)
    val_gen = data_gen(args.data_dir, args.batch_size,
                         timestamp_range=(0.8, 0.9), range_in_fractions=True,
                         img_resolution=args.input_shape)


    segnet = CreateSegNet(args.input_shape, args.kernel, args.pool_size, args.output_mode)
    print(segnet.summary())

    segnet.compile(loss=args.loss, optimizer=args.optimizer, metrics=["accuracy"])

    config = tf.ConfigProto(device_count={'GPU': 0})
    sess = tf.Session(config=config)

    sess = K.get_session()
    sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    K.set_session(sess)

    history = segnet.fit_generator(train_gen, steps_per_epoch=args.epoch_steps, epochs=args.n_epochs, validation_data=val_gen, validation_steps=args.val_steps, workers=0)

    pickle.dump(history.history, open(r'history.pickle', 'wb'))


    segnet.save_weights("pretrained/LIP_SegNet"+str(args.n_epochs)+".hdf5")
    print("sava weight done..")

    json_string = segnet.to_json()
    open("pretrained/LIP_SegNet.json", "w").write(json_string)


if __name__ == "__main__":
    # command line argments
    training_data_loc = '/home/kubafyi/Code/SegNet-Tutorial/CamVid'
    parser = argparse.ArgumentParser(description="SegNet LIP dataset")
    parser.add_argument("--data_dir",
            default=os.path.join('/mnt', 'data', 'AirSimCollectedData', 'testing'), help="Training / validation / testing data location")
    parser.add_argument("--batch_size",
            default=10,
            type=int,
            help="batch size")
    parser.add_argument("--n_epochs",
            default=10,
            type=int,
            help="number of epoch")
    parser.add_argument("--epoch_steps",
            default=100,
            type=int,
            help="number of epoch step")
    parser.add_argument("--val_steps",
            default=10,
            type=int,
            help="number of valdation step")
    parser.add_argument("--n_labels",
            default=20,
            type=int,
            help="Number of label")
    parser.add_argument("--input_shape",
            default=(144, 256, 2),
            help="Input images shape")
    parser.add_argument("--kernel",
            default=3,
            type=int,
            help="Kernel size")
    parser.add_argument("--pool_size",
            default=(2, 2),
            help="pooling and unpooling size")
    parser.add_argument("--output_mode",
            default="softmax",
            type=str,
            help="output activation")
    parser.add_argument("--loss",
            default="mean_squared_error",
            type=str,
            help="loss function")
    parser.add_argument("--optimizer",
            default="adadelta",
            type=str,
            help="oprimizer")
    args = parser.parse_args()

    main(args)
