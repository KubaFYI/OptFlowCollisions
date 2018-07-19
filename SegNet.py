# -*- coding: utf-8 -*-
import tensorflow as tf

from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Multiply
from tensorflow.keras.layers import Permute
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint

import tensorflow.keras.backend as K

from tensorflow.python import debug as tf_debug

from Mylayers import MaxPoolingWithArgmax2D
from Mylayers import MaxUnpooling2D
from generator import data_gen

from datetime import datetime
import argparse
import json
import numpy as np
import os
import sys
import pickle


# [Recipe 577058](https://code.activestate.com/recipes/577058/) adopted for Python 3
def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.
    
    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is one of "yes" or "no".
    """
    valid = {"yes":"yes",   "y":"yes",  "ye":"yes",
             "no":"no",     "n":"no"}
    if default == None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while 1:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return default
        elif choice in valid.keys():
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "\
                             "(or 'y' or 'n').\n")


def CreateSegNet(input_shape, kernel=3, pool_size=(2, 2), output_mode="sigmoid", inputz=None):
    # encoder
    K.set_image_data_format('channels_last')

    if inputz is None:
        inputs = Input(shape=(input_shape[1], input_shape[0], input_shape[2]))
    else:
        inputs = tf.stack(inputz[0])

    layers = [(2, 64), (2, 128), (3, 256), (3, 512)]
    enc_dec, masks = SegNetEncoderDecoderGenerator(inputs, layers=layers,
                                kernel=3, pool_size=(2, 2),
                                shave_off_decoder_end=1)

    conv_26 = Convolution2D(1, (1, 1), padding="same")(enc_dec)
    outputs = BatchNormalization()(conv_26)

    # For some reason the final activation sets all tensor elements to 1.0 -- why??
    # outputs = Activation(output_mode)(outputs)

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
    for idx, group in enumerate(layers):
        layers_no = group[0]
        width = group[1]
        for i in range(layers_no):
            io = Convolution2D(width, (kernel, kernel), padding="same")(io)
            io = BatchNormalization()(io)
            io = Activation("relu")(io)
            total_layers += 1

        io, mask = MaxPoolingWithArgmax2D(pool_size)(io)
        masks.append(mask)

    print("Building enceder done..")

    # decoder
    if shave_off_decoder_end > 0:
        total_layers -= shave_off_decoder_end

    for idx, group in enumerate(reversed(layers)):
        layers_no = group[0]
        width = group[1]

        io = MaxUnpooling2D(pool_size)([io, masks[-1-idx]])

        for i in range(layers_no):
            # if this is the last convolution in a series, followed up by
            # unpooling and then a convolution group with a different width, it
            # has to reduce width of the tensor now
            if i == layers_no-1 and idx != len(layers)-1:
                # last layer before UnPooling
                width = layers[-1-idx-1][1]

            io = Convolution2D(width, (kernel, kernel), padding="same")(io)
            io = BatchNormalization()(io)
            io = Activation("relu")(io)
            total_layers -= 1
            if total_layers <= 0:
                return io, masks[:-1-idx]

    print("Building decoder done..")

def main(args):
    auto_checkpoint_name = 'auto-checkpoint'
    end_checkpoint_name = 'end_checkpoint'
    train_gen = data_gen(args.data_dir, args.batch_size,
                         timestamp_range=(0.0, 0.8), range_in_fractions=True,
                         img_resolution=args.input_shape)
    val_gen = data_gen(args.data_dir, args.batch_size,
                         timestamp_range=(0.8, 1.0), range_in_fractions=True,
                         img_resolution=args.input_shape)

    checkpoint = None
    if args.continue_training:
        # Load model and weights
        if not os.path.isfile(end_checkpoint_name):
            if os.path.isfile(auto_checkpoint_name):
                print('Training didn\'t terminate - loading last mid-training checkpoint...')
                checkpoint = auto_checkpoint_name
            else:
                print('No available checkpoints - training from scratch')
        elif os.stat(auto_checkpoint_name).st_mtime > os.stat(end_checkpoint_name).st_mtime:
            print('Available final checkpoint is older than a present mid-training checkpoint.')
            if query_yes_no('Use the newer checkpoint?'):
                checkpoint = auto_checkpoint_name
            else:
                checkpoint = end_checkpoint_name
        else:
            checkpoint = end_checkpoint_name
    
    if checkpoint is None:
        #  Create fresh model
        segnet = CreateSegNet(args.input_shape, args.kernel, args.pool_size, args.output_mode)
    else:
        custom_layers = {'MaxPoolingWithArgmax2D': MaxPoolingWithArgmax2D,
                         'MaxUnpooling2D': MaxUnpooling2D}
        segnet = load_model(checkpoint, custom_objects=custom_layers, compile=False)
    
    segnet.compile(loss=args.loss, optimizer=args.optimizer, metrics=["mean_squared_error"])
    print(segnet.summary())


    # Use below to run CPU-only session
    # config = tf.ConfigProto(device_count={'GPU': 0})
    # sess = tf.Session(config=config)

    # Use below to run with the tensorflow debugger
    sess = K.get_session()
    # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    # K.set_session(sess)

    print("About to train")
    checkpoint_cb = ModelCheckpoint(auto_checkpoint_name, monitor='val_loss', verbose=1, save_best_only=False, mode='min')
    history = segnet.fit_generator(train_gen,
                                   steps_per_epoch=args.epoch_steps,
                                   epochs=args.n_epochs,
                                   validation_data=val_gen,
                                   validation_steps=args.val_steps,
                                   callbacks=[checkpoint_cb])

    pickle.dump(history.history, open(r'history.pickle', 'wb'))

    segnet.save(end_checkpoint_name)

    sess.close()


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
            default="Adam",
            type=str,
            help="optimizer")
    parser.add_argument("--continue_training",
            action="store_true",
            help="Load previously done weights?")
    args = parser.parse_args()

    main(args)