# -*- coding: utf-8 -*-

import time
import argparse
import json
import numpy as np
import os
import sys
import shutil
import pickle
import re
import pdb

parser = argparse.ArgumentParser(description="SegNet LIP dataset")
parser.add_argument("--data_dir",
        action='append',nargs=1,
        help="Training / validation / testing data location")
parser.add_argument("--batch_size",
        default=10,
        type=int,
        help="batch size")
parser.add_argument("--n_epochs",
        default=10,
        type=int,
        help="number of epoch")
parser.add_argument("--epoch_steps",
        default=None,
        type=int,
        help="number of epoch step")
parser.add_argument("--val_steps",
        default=10,
        type=int,
        help="number of valdation step")
parser.add_argument("--input_shape",
        default=(144, 256, 2),
        help="Input images shape")
parser.add_argument("--pool_size",
        default=(2, 2),
        help="pooling and unpooling size")
parser.add_argument("--output_mode",
        default="softmax",
        type=str,
        help="output activation")
parser.add_argument("--loss",
        default=None,
        type=str,
        help="loss function")
parser.add_argument("--optimizer",
        default="SGD",
        type=str,
        help="optimizer")
parser.add_argument("--continue_training",
        default=None,
        help="Load previously done weights?")
parser.add_argument("--force_metadata_refresh",
        action="store_true",
        help="Force updates of dataset metadate upon creation of data generator")
parser.add_argument("--multi_gpu",
        action="store_true",
        help="Use multiple GPUs for training. BROKEN")
parser.add_argument("--incl_ang_vel",
        action="store_true",
        help="Do not compensate OF for angular velocity instead train with it included in the input vector.")
parser.add_argument("--gpu_n",
        default='0',
        help="Default GPU index to use.")
parser.add_argument("-n", "--name",
        default=None,
        help="Name of the run to be used in TensorBoard training record.")
parser.add_argument("-v", "--verbose",
        action="store_true",
        help="Print network summary")
parser.add_argument("-a", "--architecture",
        default="first_guess",
        help="Print network summary")
args = parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_n

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
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Multiply
from tensorflow.keras.layers import Permute
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.activations import softmax
from tensorflow.python.client import device_lib
import tensorflow.keras.backend as K

from tensorflow.python import debug as tf_debug

from Mylayers import MaxPoolingWithArgmax2D
from Mylayers import MaxUnpooling2D
from Mylayers import CombineMotionWithImg
from generator import data_gen, default_bins, calc_class_weights, generator_size, set_incl_ang_vel, get_incl_ang_vel

set_incl_ang_vel(args.incl_ang_vel)


# [Recipe 577058](https://code.activestate.com/recipes/577058/) adopted for Python 3
def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.
    
    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is one of "yes" or "no".
    """
    valid = {"yes":True,   "y":True,  "ye":True,
             "no":False,     "n":False}
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

def softmax_last_axis(x):
    return softmax(x, axis=-1)

def CreateSegNet(input_shapes, nlabels=1, verbose=False):
    K.set_image_data_format('channels_last')
    if args.architecture == 'first_guess':
        net = InitialSegNet(input_shapes, nlabels=nlabels, verbose=verbose)
    elif args.architecture == 'enc-dec':
        net = EncDecArch(input_shapes, nlabels=nlabels, verbose=verbose)
    elif args.architecture == 'big-enc-dec':
        net = BigEncDecArch(input_shapes, nlabels=nlabels, verbose=verbose)
    elif args.architecture == 'simple':
        net = SimplestArch(input_shapes, nlabels=nlabels, verbose=verbose)
    elif args.architecture == '2':
        net = Arch2(input_shapes, nlabels=nlabels, verbose=verbose)
    else:
        raise NameError('Unknown architecture')

    if verbose:
        print(net[0].summary())
        print(net)

    return net

def MotionDataPreNet(input_motion, input_shapes):
    # Tile the motion vector to match the size of image inputs
    with tf.name_scope('vel_info_reshape'):
        input_motion_shaped = UpSampling2D(size=input_shapes[0][:2])(input_motion)
        input_motion_shaped = CombineMotionWithImg()(input_motion_shaped)

    with tf.name_scope('motion-pre'):
        pre_convs = Convolution2D(4*(input_shapes[1][-1]+2), (1, 1), padding='same', activation='relu')(input_motion_shaped)
        pre_convs = Convolution2D(20*(input_shapes[1][-1]+2), (1, 1), padding='same', activation='relu')(pre_convs)
        pre_convs = Convolution2D(40*(input_shapes[1][-1]+2), (1, 1), padding='same', activation='relu')(pre_convs)
        pre_convs = Convolution2D(20*(input_shapes[1][-1]+2), (1, 1), padding='same', activation='relu')(pre_convs)
        pre_convs = Convolution2D(4*(input_shapes[1][-1]), (1, 1), padding='same', activation='relu')(pre_convs)

    return pre_convs

def InitialSegNet(input_shapes, nlabels=1, verbose=False):
    input_optical_flow = Input(shape=input_shapes[0], name='input_optical_flow')
    input_motion = Input(shape=input_shapes[1], name='input_motion')

    pre_convs5 = MotionDataPreNet(input_motion, input_shapes)

    full_input = Concatenate(name="opt_flow_vel_combin", axis=-1)([input_optical_flow, pre_convs5])

    # encoder_decoder_layers = [(2, 32, 5), (3, 64, 5)]
    # enc_dec, masks = SegNetEncoderDecoderGenerator(full_input,
    #                             layers=encoder_decoder_layers,
    #                             shave_off_decoder_end=1)

    with tf.name_scope('conv_size_3'):
        conv_3 = Convolution2D(32, (3, 3), padding="same")(full_input)
        conv_3 = BatchNormalization()(conv_3)
        conv_3 = Activation('relu')(conv_3)
    with tf.name_scope('conv_size_5'):
        conv_5 = Convolution2D(16, (5, 5), padding="same")(full_input)
        conv_5 = BatchNormalization()(conv_5)
        conv_5 = Activation('relu')(conv_5)
    with tf.name_scope('conv_size_1'):
        conv_1 = Convolution2D(64, (1, 1), padding="same")(full_input)
        conv_1 = BatchNormalization()(conv_1)
        conv_1 = Activation('relu')(conv_1)

    with tf.name_scope('conv_combin'):
        post_conv = Concatenate(name="conv_concat", axis=-1)([conv_1, conv_3, conv_5])
        post_conv = Convolution2D(64, (1, 1), padding="same", name='conv_combin_2')(post_conv)
        post_conv = BatchNormalization()(post_conv)
        post_conv = Activation('relu')(post_conv)
        post_conv = Convolution2D(32, (1, 1), padding="same", name='conv_combin_3')(post_conv)
        post_conv = BatchNormalization()(post_conv)
        post_conv = Activation('relu')(post_conv)
        post_conv = Convolution2D(2*nlabels, (1, 1), padding="same", name='conv_combin_4')(post_conv)
        post_conv = BatchNormalization()(post_conv)
        post_conv = Activation('relu')(post_conv)
        post_conv = Convolution2D(nlabels, (1, 1), padding="same", name='conv_consol')(post_conv)

    # For some reason the final activation sets all tensor elements to 1.0 -- why??
    outputs = Reshape((input_shapes[0][0] * input_shapes[0][1], nlabels), input_shape=(input_shapes[0][0], input_shapes[0][1], nlabels), name='out_reshape')(post_conv)

    outputs = Activation(softmax_last_axis)(outputs)

    segnet = Model(inputs=[input_optical_flow, input_motion], outputs=outputs, name="OFNet")
    segnet_for_debug = Model(inputs=[input_optical_flow, input_motion], outputs=[outputs, pre_convs5], name="OFNet")

    return segnet, segnet_for_debug

def EncDecArch(input_shapes, nlabels=1, verbose=False):
    input_optical_flow = Input(shape=input_shapes[0], name='input_optical_flow')
    input_motion = Input(shape=input_shapes[1], name='input_motion')

    pre_convs5 = MotionDataPreNet(input_motion, input_shapes)

    full_input = Concatenate(name="opt_flow_vel_combin", axis=-1)([input_optical_flow, pre_convs5])

    encoder_decoder_layers = [(2, 32, 10), (2, 64, 5), (2, 64, 3)]
    with tf.name_scope('encoder-decoder'):
        enc_dec, masks = SegNetEncoderDecoderGenerator(full_input,
                                layers=encoder_decoder_layers)

    post_enc_dec = Convolution2D(nlabels, (1, 1), padding="same", name='conv_consol')(enc_dec)
    # For some reason the final activation sets all tensor elements to 1.0 -- why??
    outputs = Reshape((input_shapes[0][0] * input_shapes[0][1], nlabels), input_shape=(input_shapes[0][0], input_shapes[0][1], nlabels), name='out_reshape')(post_enc_dec)

    outputs = Activation(softmax_last_axis)(outputs)

    segnet = Model(inputs=[input_optical_flow, input_motion], outputs=outputs, name="OFNet")
    segnet_for_debug = Model(inputs=[input_optical_flow, input_motion], outputs=[outputs, pre_convs5], name="OFNet")

    return segnet, segnet_for_debug

def BigEncDecArch(input_shapes, nlabels=1, verbose=False):
    input_optical_flow = Input(shape=input_shapes[0], name='input_optical_flow')
    input_motion = Input(shape=input_shapes[1], name='input_motion')

    pre_convs5 = MotionDataPreNet(input_motion, input_shapes)

    full_input = Concatenate(name="opt_flow_vel_combin", axis=-1)([input_optical_flow, pre_convs5])

    encoder_decoder_layers = [(2, 32, 3), (2, 64, 3), (2, 128, 3), (2, 256, 3)]
    with tf.name_scope('encoder-decoder'):
        enc_dec, masks = SegNetEncoderDecoderGenerator(full_input,
                                layers=encoder_decoder_layers)

    encoder_decoder_layers2 = [(2, 8, 20), (2, 64, 10)]
    with tf.name_scope('encoder-decoder'):
        enc_dec2, masks = SegNetEncoderDecoderGenerator(full_input,
                                layers=encoder_decoder_layers2)

    full_input = Concatenate(name="enc_decs_combin", axis=-1)([enc_dec, enc_dec2])

    post_enc_dec = Convolution2D(32, (1, 1), padding="same", name='conv_consol1')(full_input)
    post_enc_dec = Convolution2D(16, (1, 1), padding="same", name='conv_consol2')(post_enc_dec)
    post_enc_dec = Convolution2D(nlabels, (1, 1), padding="same", name='conv_consol3')(post_enc_dec)
    # For some reason the final activation sets all tensor elements to 1.0 -- why??
    outputs = Reshape((input_shapes[0][0] * input_shapes[0][1], nlabels), input_shape=(input_shapes[0][0], input_shapes[0][1], nlabels), name='out_reshape')(post_enc_dec)

    outputs = Activation(softmax_last_axis)(outputs)

    segnet = Model(inputs=[input_optical_flow, input_motion], outputs=outputs, name="OFNet")
    segnet_for_debug = Model(inputs=[input_optical_flow, input_motion], outputs=[outputs, pre_convs5], name="OFNet")

    return segnet, segnet_for_debug

def SimplestArch(input_shapes, nlabels=1, verbose=False):
    input_optical_flow = Input(shape=input_shapes[0], name='input_optical_flow')
    input_motion = Input(shape=input_shapes[1], name='input_motion')

    pre_convs5 = MotionDataPreNet(input_motion, input_shapes)

    full_input = Concatenate(name="opt_flow_vel_combin", axis=-1)([input_optical_flow, pre_convs5])

    final_bit = Convolution2D(20*(nlabels), (1, 1), padding='same', activation='relu')(full_input)
    final_bit = Convolution2D(10*(nlabels), (1, 1), padding='same', activation='relu')(final_bit)
    final_bit = Convolution2D(5*(nlabels), (1, 1), padding='same', activation='relu')(final_bit)

    post_enc_dec = Convolution2D(nlabels, (1, 1), padding="same", name='conv_consol')(final_bit)
    # For some reason the final activation sets all tensor elements to 1.0 -- why??
    outputs = Reshape((input_shapes[0][0] * input_shapes[0][1], nlabels), input_shape=(input_shapes[0][0], input_shapes[0][1], nlabels), name='out_reshape')(post_enc_dec)

    outputs = Activation(softmax_last_axis)(outputs)

    segnet = Model(inputs=[input_optical_flow, input_motion], outputs=outputs, name="OFNet")
    segnet_for_debug = Model(inputs=[input_optical_flow, input_motion], outputs=[outputs, pre_convs5], name="OFNet")

    return segnet, segnet_for_debug

def Arch2(input_shapes, nlabels=1, verbose=False):
    input_optical_flow = Input(shape=input_shapes[0], name='input_optical_flow')
    input_motion = Input(shape=input_shapes[1], name='input_motion')

    pre_convs5 = MotionDataPreNet(input_motion, input_shapes)
    with tf.name_scope('motion-metainformation'):
        motion_stuff = Convolution2D(20*(input_shapes[1][-1]+2), (30, 30), strides=(15,15), padding='same', activation='relu')(pre_convs5)
        flat_motion_stuff = Convolution2D(40*(input_shapes[1][-1]+2), (30, 30), strides=(15,15), padding='same', activation='relu')(motion_stuff)
        flat_motion_stuff = AveragePooling2D(pool_size=(1, 2))(flat_motion_stuff)
        flat_upsampled_motion_stuff = UpSampling2D(size=input_shapes[0][:2])(flat_motion_stuff)

    full_input = Concatenate(name="opt_flow_vel_combin", axis=-1)([input_optical_flow, flat_upsampled_motion_stuff])

    encoder_decoder_layers2 = [(2, 128, 3), (2, 256, 5)]
    with tf.name_scope('encoder-decoder'):
        enc_dec2, masks = SegNetEncoderDecoderGenerator(full_input,
                                layers=encoder_decoder_layers2)

    final_bit = Convolution2D(20*(nlabels), (1, 1), padding='same', activation='relu')(enc_dec2)
    final_bit = Convolution2D(10*(nlabels), (1, 1), padding='same', activation='relu')(final_bit)
    final_bit = Convolution2D(5*(nlabels), (1, 1), padding='same', activation='relu')(final_bit)

    post_enc_dec = Convolution2D(nlabels, (1, 1), padding="same", name='conv_consol')(final_bit)
    # For some reason the final activation sets all tensor elements to 1.0 -- why??
    outputs = Reshape((input_shapes[0][0] * input_shapes[0][1], nlabels), input_shape=(input_shapes[0][0], input_shapes[0][1], nlabels), name='out_reshape')(post_enc_dec)

    outputs = Activation(softmax_last_axis)(outputs)

    segnet = Model(inputs=[input_optical_flow, input_motion], outputs=outputs, name="OFNet")
    segnet_for_debug = Model(inputs=[input_optical_flow, input_motion], outputs=[outputs, pre_convs5], name="OFNet")

    return segnet, segnet_for_debug


def SegNetEncoderDecoderGenerator(inputs, layers, pool_size=(2, 2), shave_off_decoder_end=0):
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
    with tf.name_scope('encoder'):
        for idx, group in enumerate(layers):
            layers_no = group[0]
            width = group[1]
            kernel = group[2]
            for i in range(layers_no):
                with tf.name_scope('encoder_{0}x{0}_{1}ch'.format(kernel, width)):
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
    with tf.name_scope('decoder'):
        for idx, group in enumerate(reversed(layers)):
            layers_no = group[0]
            width = group[1]
            kernel = group[2]
    
            io = MaxUnpooling2D(pool_size)([io, masks[-1-idx]])
    
            for i in range(layers_no):
                # if this is the last convolution in a series, followed up by
                # unpooling and then a convolution group with a different width, it
                # has to reduce width of the tensor now
                if i == layers_no-1 and idx != len(layers)-1:
                    # last layer before UnPooling
                    width = layers[-1-idx-1][1]
    
                with tf.name_scope('decoder_{0}x{0}_{1}ch'.format(kernel, width)):
                    io = Convolution2D(width, (kernel, kernel), padding="same")(io)
                    io = BatchNormalization()(io)
                    io = Activation("relu")(io)
                total_layers -= 1
                if total_layers <= 0:
                    return io, masks[:-1-idx]

    print("Building decoder done..")

def weighted_categorical_crossentropy(match_weights, mismatch_weights=None):
    """
    A weighted version of keras.objectives.categorical_crossentropy

    (adapted from gist to work with multiclass-per-sample https://gist.github.com/wassname/ce364fddfc8a025bfab4348cf5de852d)
    
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """
    
    match_weights = tf.constant(match_weights, dtype=tf.float32)
    # if mismatch_weights is not None
    # mismatch_weights = tf.Variable(mismatch_weights, dtype=tf.float32)
        
    def loss(y_true, y_pred):
        # clip to prevent NaN's and Inf's
        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * tf.log(y_pred) * match_weights
        # y_true_opposite = tf.floormod(tf.add(y_true, tf.constant(1.)), tf.constant(2.))
        # loss += - y_true_opposite * tf.log(y_pred) * mismatch_weights
        loss = -tf.reduce_sum(loss, -1)
        pixel_no = tf.shape(y_true)
        pixel_no = tf.cast(pixel_no, tf.float32)
        loss = tf.reduce_sum(loss, -1)
        loss = tf.divide(loss, pixel_no[-2])
        return loss
    
    return loss

def weighted_focal_loss(weights):
    weights = tf.constant(weights, dtype=tf.float32)
    def focal_loss(target, output, gamma=2):
        # output /= K.sum(output, axis=-1, keepdims=True)
        eps = K.epsilon()
        output = K.clip(output, eps, 1. - eps)
        pixel_no = tf.shape(output)
        pixel_no = tf.cast(pixel_no, tf.float32)
        loss = -K.sum(K.sum(K.pow(1. - output, gamma) * target * K.log(output) * weights,
                      axis=-1), axis=-1)
        loss /= pixel_no[-2]
        return loss
    return focal_loss

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

def main(args):
    auto_checkpoint_name = os.path.join(os.getcwd(), 'auto-checkpoint_{}'.format(args.name))
    print('Auto-checkpoint at {}'.format(auto_checkpoint_name))
    auto_checkpoint_w_name = os.path.join(os.getcwd(), 'auto-checkpoint_weights_{}.h5'.format(args.name))
    end_checkpoint_name = os.path.join(os.getcwd(), 'end-checkpoint_{}'.format(args.name))
    end_checkpoint_w_name = os.path.join(os.getcwd(), 'end-checkpoint_weights_{}.h5'.format(args.name))
    bins = default_bins
    seed = 77
    min_max_value = 0.1

    training_size = generator_size(args.data_dir,
                         scrambled_range=(0.0, 0.9, seed), range_in_fractions=True, min_max_value=min_max_value,
                         include_motion_data=True,
                         bins=bins,
                         force_metadata_refresh=args.force_metadata_refresh)
    print('Training using dataset of size {}'.format(training_size))

    if args.epoch_steps is None:
        epoch_steps = int((training_size / args.batch_size) + 1.)
    else:
        epoch_steps = args.epoch_steps

    train_gen = data_gen(args.data_dir, args.batch_size,
                         scrambled_range=(0.0, 0.9, seed), range_in_fractions=True,
                         img_resolution=args.input_shape, min_max_value=min_max_value,
                         include_motion_data=True,
                         bins=bins)
    print('Validation set size {}'.format(int(np.floor(training_size*0.1))))
    val_gen = data_gen(args.data_dir, int(np.floor(training_size*0.1)),
                         scrambled_range=(0.9, 1.0, seed), range_in_fractions=True,
                         img_resolution=args.input_shape, min_max_value=min_max_value,
                         include_motion_data=True,
                         bins=bins)
    if not get_incl_ang_vel():
        motion_data_size = (1, 1, 3)
    else:
        motion_data_size = (1, 1, 6)

    checkpoint = None
    if args.continue_training is not None:
        # Load model and weights
        checkpoint = args.continue_training
        # if not os.path.isfile(end_checkpoint_name):
        #     if os.path.isfile(auto_checkpoint_name):
        #         print('Training didn\'t terminate - loading last mid-training checkpoint...')
        #         checkpoint = auto_checkpoint_name
        #     else:
        #         print('No available checkpoints - training from scratch')
        # elif os.stat(auto_checkpoint_name).st_mtime > os.stat(end_checkpoint_name).st_mtime:
        #     print('Available final checkpoint is older than a present mid-training checkpoint.')
        #     if query_yes_no('Use the newer checkpoint?'):
        #         checkpoint = auto_checkpoint_name
        #     else:
        #         checkpoint = end_checkpoint_name
        # else:
        #     checkpoint = end_checkpoint_name
    
    if args.loss == 'categorical_crossentropy':
        weights = np.ones_like(bins, dtype=np.float32)
        loss_fnctn = weighted_categorical_crossentropy(weights)
    elif args.loss == 'weighted_categorical_crossentropy':
        class_weights, mismatch_weights = calc_class_weights(args.data_dir,
                                           bins=bins,
                                           min_max_value=min_max_value)
        print('Using class weights {}\nAnd mismatch weights {}'.format(class_weights, mismatch_weights))
        loss_fnctn = weighted_categorical_crossentropy(class_weights)
        # y_pred = tf.Variable(np.random.random((10, 15, 2)), dtype=tf.float32)
        # y_true = tf.Variable(np.random.random((10, 15, 2)), dtype=tf.float32)
        # (loss_fnctn(y_true, y_pred)).eval()
    elif args.loss == 'focal':
        print('Using focal loss')
        weights = np.ones_like(bins, dtype=np.float32)
        loss_fnctn = weighted_focal_loss(weights)
    elif args.loss == 'weighted_focal':
        weights, mismatch_weights = calc_class_weights(args.data_dir,
                                           bins=bins,
                                           min_max_value=min_max_value)
        print('Using weighted focal loss ({})'.format(weights))
        loss_fnctn = weighted_focal_loss(weights)
    else:
        loss_fnctn = args.loss

    if checkpoint is None:
        #  Create fresh model
        segnet, segnet_for_debug = CreateSegNet([args.input_shape, motion_data_size], nlabels=len(bins), verbose=args.verbose)
    else:
        custom_layers = {'MaxPoolingWithArgmax2D': MaxPoolingWithArgmax2D,
                         'MaxUnpooling2D': MaxUnpooling2D,
                         'CombineMotionWithImg': CombineMotionWithImg,
                         'loss': loss_fnctn,
                         'softmax_last_axis': softmax_last_axis,
                         'weighted_focal_loss': weighted_focal_loss}
        print('loading {}'.format(checkpoint))
        segnet = load_model(checkpoint, custom_objects=custom_layers, compile=False)

    gpus = get_available_gpus()
    if len(gpus) > 1 and args.multi_gpu:
        gpus_to_use = []
        print('Available GPUs: {}'.format(gpus))
        gpu_idxs = input('Give indexes of GPUs you would like to use...')
        for c in gpu_idxs:
            gpus_to_use.append(int(c))
        segnet_to_use = multi_gpu_model(segnet, gpus=2, cpu_relocation=True)
        segnet_to_use.compile(loss=loss_fnctn, optimizer='adam', metrics=['categorical_crossentropy', 'accuracy'])
    else:
        segnet.compile(loss=loss_fnctn, optimizer='adam', metrics=['categorical_crossentropy', 'accuracy'])
        segnet_to_use = segnet
    # print(segnet_to_use.summary())


    # Use below to run CPU-only session
    # config = tf.ConfigProto(device_count={'GPU': 0})
    sess = K.get_session()
    # config = tf.ConfigProto()  
    # config.gpu_options.allow_growth = True  
    # sess = tf.Session(config=config)

    # Use below to run with the tensorflow debugger
    # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    # K.set_session(sess)

    print("About to train")
    train_callbacks = []
    train_callbacks.append(ModelCheckpoint(auto_checkpoint_name + '{epoch}', monitor='val_loss', verbose=1, save_best_only=False, mode='min', period=100))
    if os.environ['USER'] == 'kuba':
        # Working on group server
        tensorboard_log_dir_base = '/media/disk1/kuba/tensorboard'
    elif os.environ['USER'] == 'kubafyi':
        # Working on own laptop
        tensorboard_log_dir_base = '/home/kubafyi/tensorboard'
    else:
        raise EnvironmentError('Unexpected user.')
    if os.path.isdir(tensorboard_log_dir_base):
        if args.name is not None:
            run_name = args.name
        else:
            run_name = time.strftime('%y-%m-%d_%H-%M-%S')
        tensorboard_log_dir = os.path.join(tensorboard_log_dir_base, run_name)
        if os.path.isdir(tensorboard_log_dir):
            if query_yes_no('Log with this name already exists. Overwrite?'):
                shutil.rmtree(tensorboard_log_dir)
            else:
                max_num = 0
                for subdir in get_immediate_subdirectories(tensorboard_log_dir_base):
                    tb_dir_split = re.search(run_name + '\(([0-9]+)\)', subdir)

                    if tb_dir_split is not None and max_num < int(tb_dir_split.group(1)):
                        max_num = int(tb_dir_split.group(1))

                tensorboard_log_dir = tensorboard_log_dir + '(' + str(max_num+1) + ')'
        print('Logging at {}'.format(tensorboard_log_dir))

        os.makedirs(tensorboard_log_dir)
        train_callbacks.append(TensorBoard(log_dir=tensorboard_log_dir, histogram_freq=0, write_graph=True, write_images=True))

    val_dataset = next(val_gen)
    history = segnet_to_use.fit_generator(train_gen,
                                   steps_per_epoch=epoch_steps,
                                   epochs=args.n_epochs,
                                   validation_data=val_dataset,
                                   validation_steps=args.val_steps,
                                   callbacks=train_callbacks)

    pickle.dump(history.history, open(r'history.pickle', 'wb'))

    segnet.save(end_checkpoint_name)
    segnet.save_weights(end_checkpoint_w_name)

    sess.close()


if __name__ == "__main__":
    # command line argments
    print(get_available_gpus())
    print(args.data_dir)
    with tf.device('/GPU:0'.format(args.gpu_n)):
        main(args)
