# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import cv2
import csv
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img, ImageDataGenerator
from scipy.misc import imresize
from matplotlib import pyplot as plt
from tqdm import tqdm
from pyquaternion import Quaternion
import pickle
from flowlib import flow_to_image
# cv2.imshow('qwe', flow_to_image(opt_flow))

default_data_dir = os.path.join('/mnt', 'data', 'AirSimCollectedData', 'testing')
csv_col_names = ['LV_x', 'LV_y', 'LV_z', 'AV_x', 'AV_y', 'AV_z', 'OQ_w', 'OQ_x', 'OQ_y', 'OQ_z']
datapoints_csv_name = os.path.join(default_data_dir, 'airsim_rec.csv')

default_bins = [0, 0.1, 0.25, 0.5, 0.75, 0.9]

# For some reason max values for collision distance seem to be bounded at 0.5
# This multiplier aims to fix that
coll_dist_mul = 2

def world2camera_coords(vect, cam_orient):
    '''
    Rotates a 3D vector from world orientation into camera orientation.
    cam_orient is a quaternion description of camera rotation w.r.t. world
    '''
    ret = np.empty_like(vect)
    if len(vect.shape) > 1:
        for idx, quat in enumerate(cam_orient):
            # To convert velocity from world orientation to camera orientation
            # need to apply rotation inverse to orientation.
            # Formula for applying quaternion rotation to a vector from wikipedia
            # https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
            quat_vec = Quaternion(scalar=0, vector=vect[idx])
            quat_vec_rot = cam_orient[idx].inverse * quat_vec * cam_orient[idx]
            ret[idx] = quat_vec_rot.vector
    else:
        quat_vec = Quaternion(scalar=0, vector=vect)
        ret = cam_orient.inverse * quat_vec * cam_orient
        ret
    return ret

def load_optical_flow_metadata(data_dir, datapoints):
    metadata_file = os.path.join(data_dir, 'optical_flow_metadata.pickle')
    if os.path.isfile(metadata_file):
        f = open(metadata_file, 'rb')
        metadata = pickle.load(f)
    else:
        # No metadata file, need to create one
        print('Creating metadata file.')
        metadata = {}
        metadata['sums'] = []
        metadata['max_vals'] = []
        metadata['camera_vel'] = []
        vel = []
        orient = []
        for timestamp in tqdm(datapoints.index):
            # Calculate sum of collision probability over pixels
            coll_dist_n = os.path.join(data_dir, 'images',
                                      'misc_' + str(timestamp) + '.npz')
            coll_dist = np.load(coll_dist_n)['arr_0']
            coll_dist *= coll_dist_mul  # not sure why the max value seems to be bounded at 0.5...
            metadata['sums'].append(np.sum(coll_dist))
            metadata['max_vals'].append(np.max(coll_dist))
            orient.append(Quaternion(datapoints.loc[timestamp][['OQ_w', 'OQ_x', 'OQ_y', 'OQ_z']]))


        # Calculate the velocity w.r.t. camera
        vel = np.array(datapoints[['LV_x','LV_y','LV_z']])
        metadata['camera_vel'] = world2camera_coords(vel, orient)
        metadata['sums'] = np.array(metadata['sums'])
        metadata['max_vals'] = np.array(metadata['max_vals'])

        f = open(metadata_file, 'wb')
        pickle.dump(metadata, f, protocol=pickle.HIGHEST_PROTOCOL)
    return metadata

def data_gen(data_dir, batch_size, bins=None,
             timestamp_range=None, range_in_fractions=False,
             img_resolution=None,
             random_order=True,
             include_rgb=False,
             include_motion_data=False,
             min_sum_percentile=None,
             min_max_value=None):
    ''' Generator for data batches from AirSim-generated data. '''
    datapoints_csv_name = os.path.join(data_dir, 'airsim_rec.csv')
    datapoints_all = pd.read_csv(datapoints_csv_name,
                             header=None,
                             sep=',',
                             names=csv_col_names,
                             index_col=0)
    if min_sum_percentile or min_max_value or include_motion_data:
        metadata = load_optical_flow_metadata(data_dir, datapoints_all)
    # If needed, take only those datapoints where there is most collision
    # probability
    if min_sum_percentile is not None:
        min_sum = np.percentile(metadata['sums'], min_sum_percentile)
        datapoints_all = datapoints_all[metadata['sums'] > min_sum]
    elif min_max_value is not None:
        datapoints_all = datapoints_all[metadata['max_vals'] > min_max_value]

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
    print('Using dataset with of {} elements'.format(datapoints.shape[0]))
    batch_start = start_idx
    while True:
        timestamps = []
        inputs = []
        vel_inputs = []
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
            coll_dist *= coll_dist_mul  # not sure why the max value seems to be bounded at 0.5...

            if include_rgb:
                rgb_n = os.path.join(data_dir, 'images',
                                     'rgb_' + str(datapoints.index[i]) + '.png')
                rgb = cv2.imread(rgb_n)
                rgbs.append(rgb)

            if img_resolution is not None:
                opt_flow = cv2.resize(opt_flow, (img_resolution[1], img_resolution[0]))
                # print(opt_flow.shape)
            timestamps.append(datapoints.index[i])
            inputs.append(opt_flow)
            labels.append(np.expand_dims(coll_dist, axis=-1))
            # print('cd:{}'.format(coll_dist.shape))
            if include_motion_data:
                vel_inputs.append(metadata['camera_vel'][i])
        timestamps = np.array(timestamps)
        inputs = np.array(inputs)
        if include_motion_data:
            vel_inputs = np.array(vel_inputs)
            # Turn motion vectors into motion 1x1 'images' with vector values as 'channels'
            vel_inputs = np.expand_dims(np.expand_dims(vel_inputs, axis=1), axis=1)
            inputs = [inputs, vel_inputs]
        labels = np.array(labels)
        if include_rgb:
            rgbs = np.array(rgbs)
            yield inputs, bin_pixels(labels, bins), rgbs, timestamps
        elif include_motion_data:
            yield inputs, bin_pixels(labels, bins)
        else:
            yield inputs, bin_pixels(labels, bins)

def bin_pixels(data, bins):
    '''
    Bins the values of 1-channel pixel data into N pre-defined buckets, and
    returns on-hot-encoded N-channel array corresponding to the result.
    '''
    if bins is None:
        return data
    bin_indices = np.digitize(data, bins) - 1
    result = np.zeros((data.shape[0], data.shape[1], data.shape[2], len(bins)))
    
    for bin_idx in range(len(bins)):
        one_hot_indices = np.nonzero(bin_indices==bin_idx)
        one_hot_indices = (one_hot_indices[0],
                           one_hot_indices[1],
                           one_hot_indices[2],
                           bin_idx * np.ones_like(one_hot_indices[0]))
        result[one_hot_indices] = 1
    # print('r{}'.format(result.shape))
    return result

def data_stats(data_dir, bins, number_for_analysis=None):
    datapoints_csv_name = os.path.join(data_dir, 'airsim_rec.csv')
    datapoints = pd.read_csv(datapoints_csv_name,
                             header=None,
                             sep=',',
                             names=csv_col_names,
                             index_col=0)
    if number_for_analysis is None:
        ix = range(datapoints.shape[0])
    else:
        ix = np.random.choice(np.arange(len(datapoints)), number_for_analysis)

    single_maxs = []
    single_mins = []
    single_means = []
    single_std_devs = []
    single_sums = []

    binned_sums = {}
    for bin_idx in range(len(bins)):
        binned_sums[bin_idx] = 0

    for i in tqdm(ix):
        coll_dist_n = os.path.join(data_dir, 'images',
                                  'misc_' + str(datapoints.index[i]) + '.npz')
        coll_dist = np.load(coll_dist_n)['arr_0']
        coll_dist *= 2

        single_maxs.append(np.max(coll_dist))
        single_mins.append(np.min(coll_dist))
        single_means.append(np.mean(coll_dist))
        single_std_devs.append(np.std(coll_dist))
        single_sums.append(np.sum(coll_dist))

        bin_indices = np.digitize(coll_dist, bins) - 1

        for bin_idx in range(len(bins)):
            binned_sums[bin_idx] += np.sum(bin_indices==bin_idx)


    single_maxs = np.array(single_maxs)
    single_mins = np.array(single_mins)
    single_means = np.array(single_means)
    single_std_devs = np.array(single_std_devs)
    single_sums = np.array(single_sums)

    print('Min: {}\n'
          'Max: {}\n'
          'Avg mean: {}\n'
          'Max mean: {}\n'
          'Min mean: {}\n'
          'Avg std: {}\n'
          'Max std: {}\n'
          'Max sum: {}\n'
          'Min sum: {}\n'
          'Mean sum: {}\n'
          'Median sum: {}\n'.format(np.min(single_mins),
                                 np.max(single_maxs),
                                 np.mean(single_means),
                                 np.max(single_means),
                                 np.min(single_means),
                                 np.mean(single_std_devs),
                                 np.max(single_std_devs),
                                 np.max(single_sums),
                                 np.min(single_sums),
                                 np.mean(single_sums),
                                 np.median(single_sums)))

    for bin_idx in range(len(bins)):
        print('#{} bin: {}'.format(bin_idx, binned_sums[bin_idx]))

    perc = 70
    prec_sum = np.percentile(single_sums, perc)
    sums = single_sums[single_sums>prec_sum]
    print('showing sums bigger than {} ({} of them)'.format(prec_sum, np.size(sums)))
    # plt.hist(sums, bins=40)
    # plt.show()

if __name__ == '__main__':
    data_stats(default_data_dir, default_bins)
