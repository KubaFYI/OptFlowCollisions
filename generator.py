# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import cv2
import csv
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img, ImageDataGenerator
from scipy.misc import imresize
import pdb

csv_col_names = ['LV_x', 'LV_y', 'LV_z', 'AV_x', 'AV_y', 'AV_z', 'OQ_w', 'OQ_x', 'OQ_y', 'OQ_z']

def data_gen(data_dir, batch_size,
             timestamp_range=None, range_in_fractions=False,
             img_resolution=None,
             random_order=True,
             include_rgb=False):
    ''' Generator for data batches from AirSim-generated data. '''
    datapoints_csv_name = os.path.join(data_dir, 'airsim_rec.csv')
    datapoints_all = pd.read_csv(datapoints_csv_name,
                             header=None,
                             sep=',',
                             names=csv_col_names,
                             index_col=0)

    # Figure out first and last timestep of the range
    if timestamp_range is None:
        timestamp_range = (datapoints_all.index[0], datapoints_all.index[-1])

    if range_in_fractions:
        start_idx = int(np.floor(timestamp_range[0] * datapoints_all.shape[0]))
        end_idx = int(np.ceil(timestamp_range[1] * datapoints_all.shape[0]))
        start_timestamp = datapoints_all.index[start_idx]
        end_timestamp = datapoints_all.index[end_idx-1]
    else:
        start_timestamp = timestamp_range[0]
        end_timestamp = timestamp_range[1]

    datapoints = datapoints_all.loc[start_timestamp:end_timestamp]
    batch_start = start_idx
    print(end_idx)
    while True:
        inputs = []
        labels = []
        rgbs = []
        if random_order:
            ix = np.random.choice(np.arange(len(datapoints)), batch_size)
        else:
            ix = np.arange(batch_start, batch_start + batch_size)
            ix[ix>=end_idx] = start_idx + ix[ix>=end_idx] % end_idx
            batch_start += batch_size
            if batch_start >= end_idx:
                batch_start = start_idx + batch_start % end_idx
        for i in ix:
            # Get the optical flow input
            opt_flow_n = os.path.join(data_dir, 'images',
                                      'flow_' + str(datapoints.index[i]) + '.npz')
            opt_flow = np.load(opt_flow_n)['opt_flow']

            # Get the 'collision distance' labels
            coll_dist_n = os.path.join(data_dir, 'images',
                                      'misc_' + str(datapoints.index[i]) + '.npz')
            coll_dist = np.load(coll_dist_n)['arr_0']

            if include_rgb:
                rgb_n = os.path.join(data_dir, 'images',
                                     'rgb_' + str(datapoints.index[i]) + '.png')
                rgb = cv2.imread(rgb_n)
                rgbs.append(rgb)

            if img_resolution is not None:
                # print('{} -> {}'.format(opt_flow.shape[:2], img_resolution[:2]))
                opt_flow = cv2.resize(opt_flow, img_resolution[:2])
            inputs.append(opt_flow)
            labels.append(np.expand_dims(coll_dist, axis=0).T)
        inputs = np.array(inputs)
        labels = np.array(labels)
        if include_rgb:
            rgbs = np.array(rgbs)
            yield inputs, labels, rgbs
        else:
            yield inputs, labels
