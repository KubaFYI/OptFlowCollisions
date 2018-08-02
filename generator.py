# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import cv2
import csv
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img, ImageDataGenerator
from scipy.optimize import minimize
from matplotlib import pyplot as plt
from tqdm import tqdm
from pyquaternion import Quaternion
import pickle
from flowlib import flow_to_image
import pdb
# cv2.imshow('qwe', flow_to_image(opt_flow))

default_data_dir = os.path.join('/mnt', 'data', 'AirSimCollectedData', 'testing')
csv_col_names = ['LV_x', 'LV_y', 'LV_z', 'AV_x', 'AV_y', 'AV_z', 'OQ_w', 'OQ_x', 'OQ_y', 'OQ_z']
datapoints_csv_name = os.path.join(default_data_dir, 'airsim_rec.csv')

default_bins = [0, 0.5]

# For some reason max values for collision distance seem to be bounded at 0.5
# This multiplier aims to fix that
coll_dist_mul = 2

# https://github.com/unrealcv/unrealcv/issues/14
orig_image_size = (270, 480)    # (h,w)
FOV = 90    # in degeres
cam_f = orig_image_size[1] / 2 / np.tan( FOV / 180 * np.pi / 2 )

cam_f = 4.47189418e+00 # calculated by minimization of horizontal flow in a yaw-only dataset
xy_scale_default = 4.78224949e-06

# velocity_input_mode = 'raw'
# velocity_input_mode = 'camera-compensated'
velocity_input_mode = 'angle-magnitude'

incl_ang_vel = True
def set_incl_ang_vel(val):
    global incl_ang_vel
    incl_ang_vel = val

def get_incl_ang_vel():
    global incl_ang_vel
    return incl_ang_vel

def o_f_compensate_for_rotation(opt_flow, rot, focal=cam_f, xy_scaling=xy_scale_default):
    '''
    Removes estimated optical flow due to camera rotation.
    '''
    # Calculate OF due to rotation
    # http://scholarpedia.org/article/Optic_flow#Optic_flow_for_guidance_of_locomotion_and_scene_parsing
    # return opt_flow
    focal = np.abs(focal)
    xy_scaling = np.abs(xy_scaling)
    opt_flow_corrected = np.copy(opt_flow)
    current_size = opt_flow.shape[-3:-1]

    # Prepare a grid of equivalent pixel positions of original size image
    ys = np.expand_dims(np.linspace(-orig_image_size[0]/2, orig_image_size[0]/2, num=current_size[0]), axis=-1)
    xs = np.expand_dims(np.linspace(-orig_image_size[1]/2, orig_image_size[1]/2, num=current_size[1]), axis=0)
    ys = np.tile(ys, (1, current_size[1]))
    ys *= xy_scaling
    xs = np.tile(xs, (current_size[0], 1))
    xs *= xy_scaling

    # In AirSim Coordinates rot[0] is rotation around the central axis (roll)
    # rot[1] is pitch and rot[2] is negative yaw

    # First channel is the horizontal motion component
    opt_flow_corrected[..., 0] -= ys * xs * -rot[1]
    opt_flow_corrected[..., 0] -= -(focal * focal + xs * xs) * -rot[2]
    opt_flow_corrected[..., 0] -= ys * focal * rot[0]
    # Second channel is the vertical motion component
    opt_flow_corrected[..., 1] -= (focal * focal + ys * ys) * -rot[1]
    opt_flow_corrected[..., 1] -= - ys * xs * -rot[2]
    opt_flow_corrected[..., 1] -= - xs * focal * rot[0]

    opt_flow_corrected /= focal

    return opt_flow_corrected




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

def load_optical_flow_metadata(data_dir, datapoints=None, force_metadata_refresh=False, bins=None):
    if datapoints is None:
        datapoints_csv_name = os.path.join(data_dir, 'airsim_rec.csv')
        datapoints = pd.read_csv(datapoints_csv_name,
                             header=None,
                             sep=',',
                             names=csv_col_names,
                             index_col=0)
    if bins is None:
        bins = default_bins
    metadata_file = os.path.join(data_dir, 'optical_flow_metadata.pickle')
    if os.path.isfile(metadata_file) and not force_metadata_refresh:
        f = open(metadata_file, 'rb')
        metadata = pickle.load(f)
    else:
        # No metadata file, need to create one
        print('Creating metadata file.')
        metadata = {}
        metadata['sums'] = []
        metadata['max_vals'] = []
        metadata['cam_lin_vel'] = []
        metadata['cam_ang_vel'] = []
        metadata['timestamp_idx'] = {}
        metadata['class_count'] = []
        vel = []
        orient = []
        idx = 0
        for timestamp in tqdm(datapoints.index):
            # Calculate sum of collision probability over pixels
            coll_dist_n = os.path.join(data_dir, 'images',
                                      'misc_' + str(timestamp) + '.npz')
            coll_dist = np.load(coll_dist_n)['arr_0']
            coll_dist *= coll_dist_mul  # not sure why the max value seems to be bounded at 0.5...
            metadata['sums'].append(np.sum(coll_dist))
            metadata['max_vals'].append(np.max(coll_dist))
            orient.append(Quaternion(datapoints.loc[timestamp][['OQ_w', 'OQ_x', 'OQ_y', 'OQ_z']]))
            metadata['cam_ang_vel'].append(datapoints.loc[timestamp][['AV_x', 'AV_y', 'AV_z']])
            metadata['timestamp_idx'][timestamp] = idx

            class_count = []
            for bin_idx, a_bin in enumerate(bins):
                if bin_idx != len(bins)-1:
                    upper = bins[bin_idx+1]
                else:
                    upper = 1.
                class_count.append(np.sum(np.logical_and(coll_dist > a_bin, coll_dist < upper)))
            metadata['class_count'].append(class_count)

            idx += 1

        # Calculate the velocity w.r.t. camera
        vel = np.array(datapoints[['LV_x','LV_y','LV_z']])
        metadata['cam_lin_vel'] = world2camera_coords(vel, orient)
        metadata['sums'] = np.array(metadata['sums'])
        metadata['max_vals'] = np.array(metadata['max_vals'])
        metadata['cam_ang_vel'] = np.array(metadata['cam_ang_vel'])
        metadata['class_count'] = np.array(metadata['class_count'])

        f = open(metadata_file, 'wb')
        pickle.dump(metadata, f, protocol=pickle.HIGHEST_PROTOCOL)
    return metadata

def calc_class_weights(data_dir, force_metadata_refresh=False, bins=None, min_max_value=0):
    # pdb.set_trace()
    datapoints_csv_name = os.path.join(data_dir, 'airsim_rec.csv')
    datapoints_all = pd.read_csv(datapoints_csv_name,
                             header=None,
                             sep=',',
                             names=csv_col_names,
                             index_col=0)
    metadata = load_optical_flow_metadata(data_dir, datapoints_all,
                                              force_metadata_refresh=force_metadata_refresh,
                                              bins=bins)

    class_totals = np.sum(metadata['class_count'][np.nonzero(metadata['max_vals'] > min_max_value)], axis=0)
    weights = np.sum(class_totals) / (class_totals.shape[0] * class_totals)
    # weights = np.power(weights, 1/5)
    # weights = np.power(weights, 1/5)
    mismatch_weights = np.sum(weights) / weights
    return weights, mismatch_weights

def generator_size(data_dir, bins=None,
             scrambled_range=None,
             timestamp_range=None, range_in_fractions=False,
             include_motion_data=False,
             min_sum_percentile=None,
             min_max_value=None,
             force_metadata_refresh=False):
    ''' Generator for data batches from AirSim-generated data. '''
    datapoints_csv_name = os.path.join(data_dir, 'airsim_rec.csv')
    datapoints_all = pd.read_csv(datapoints_csv_name,
                             header=None,
                             sep=',',
                             names=csv_col_names,
                             index_col=0)
    if min_sum_percentile or min_max_value or include_motion_data:
        metadata = load_optical_flow_metadata(data_dir, datapoints_all,
                                              force_metadata_refresh=force_metadata_refresh,
                                              bins=bins)
    # If needed, take only those datapoints where there is most collision
    # probability
    if min_sum_percentile is not None:
        min_sum = np.percentile(metadata['sums'], min_sum_percentile)
        datapoints_all = datapoints_all[metadata['sums'] > min_sum]
    elif min_max_value is not None:
        datapoints_all = datapoints_all[metadata['max_vals'] > min_max_value]

    if scrambled_range is not None:
        datapoints = datapoints_all.sample(frac=1., random_state=scrambled_range[2])
        datapoints = datapoints.iloc[int(scrambled_range[0] * len(datapoints)):int(scrambled_range[1] * len(datapoints))]
    else:
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

    print('Analytical rotational motion OF compensation is {}'.format('OFF' if incl_ang_vel else 'ON'))
    return datapoints.shape[0]


def data_gen(data_dir, batch_size, bins=None,
             scrambled_range=None,
             timestamp_range=None, range_in_fractions=False,
             img_resolution=None,
             random_order=True,
             include_rgb=False,
             include_motion_data=False,
             min_sum_percentile=None,
             min_max_value=None,
             force_metadata_refresh=False):
    ''' Generator for data batches from AirSim-generated data. '''
    datapoints_csv_name = os.path.join(data_dir, 'airsim_rec.csv')
    datapoints_all = pd.read_csv(datapoints_csv_name,
                             header=None,
                             sep=',',
                             names=csv_col_names,
                             index_col=0)
    if min_sum_percentile or min_max_value or include_motion_data:
        metadata = load_optical_flow_metadata(data_dir, datapoints_all,
                                              force_metadata_refresh=force_metadata_refresh,
                                              bins=bins)
    # If needed, take only those datapoints where there is most collision
    # probability
    if min_sum_percentile is not None:
        min_sum = np.percentile(metadata['sums'], min_sum_percentile)
        datapoints_all = datapoints_all[metadata['sums'] > min_sum]
    elif min_max_value is not None:
        datapoints_all = datapoints_all[metadata['max_vals'] > min_max_value]

    if scrambled_range is not None:
        datapoints = datapoints_all.sample(frac=1., random_state=scrambled_range[2])
        datapoints = datapoints.iloc[int(scrambled_range[0] * len(datapoints)):int(scrambled_range[1] * len(datapoints))]
    else:
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

    while True:
        timestamps = []
        inputs = []
        vel_inputs = []
        labels = []
        rgbs = []
        if random_order or scrambled_range:
            ix = datapoints.sample(n=batch_size).index
        else:
            ix = np.arange(batch_start, batch_start + batch_size)
            ix[ix>=end_idx] = start_idx + ix[ix>=end_idx] % end_idx
            batch_start += batch_size
            if batch_start >= end_idx:
                batch_start = start_idx + batch_start % end_idx
        for timestamp in ix:
            idx = metadata['timestamp_idx'][timestamp]
            # Get the optical flow input
            opt_flow_n = os.path.join(data_dir, 'images',
                                      'flow_' + str(timestamp) + '.npz')
            opt_flow = np.load(opt_flow_n)['opt_flow']

            # Get the 'collision distance' labels
            coll_dist_n = os.path.join(data_dir, 'images',
                                      'misc_' + str(timestamp) + '.npz')
            coll_dist = np.load(coll_dist_n)['arr_0']
            coll_dist *= coll_dist_mul  # not sure why the max value seems to be bounded at 0.5...
            coll_dist = coll_dist.flatten()

            if include_rgb:
                rgb_n = os.path.join(data_dir, 'images',
                                     'rgb_' + str(timestamp) + '.png')
                rgb = cv2.imread(rgb_n)
                rgbs.append(rgb)

            if img_resolution is not None:
                opt_flow = cv2.resize(opt_flow, (img_resolution[1], img_resolution[0]))
                if not incl_ang_vel:
                    opt_flow = o_f_compensate_for_rotation(opt_flow, metadata['cam_ang_vel'][idx])
                # print(opt_flow.shape)
            timestamps.append(timestamp)
            inputs.append(opt_flow)
            labels.append(np.expand_dims(coll_dist, axis=-1))
            # print('cd:{}'.format(coll_dist.shape))
            if include_motion_data:
                if not incl_ang_vel:
                    vel_info = metadata['cam_lin_vel'][idx]
                else:
                    vel_info = np.concatenate((metadata['cam_lin_vel'][idx],
                                               metadata['cam_ang_vel'][idx]),
                                              axis=-1)
                vel_inputs.append(vel_info)
        timestamps = np.array(timestamps)
        inputs = np.array(inputs)
        if include_motion_data:
            vel_inputs = velocity_form(np.array(vel_inputs))
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

def velocity_form(vel_inputs):
    if vel_inputs.ndim == 1:
        vel_inputs = np.expand_dims(vel_inputs, axis=0)
    if velocity_input_mode == 'angle-magnitude':
        mag = np.linalg.norm(vel_inputs[:, 0:3], axis=-1)
        comp_hor = vel_inputs[:, 1] / mag
        comp_ver = vel_inputs[:, 2] / mag
        vel_inputs[:, 0:3] = np.concatenate((np.expand_dims(mag, axis=-1),
                                     np.expand_dims(comp_hor, axis=-1),
                                     np.expand_dims(comp_ver, axis=-1)),
                                    axis=-1)
    return vel_inputs

def bin_pixels(data, bins):
    '''
    Bins the values of 1-channel pixel data into N pre-defined buckets, and
    returns on-hot-encoded N-channel array corresponding to the result.
    '''
    if bins is None:
        return data
    bin_indices = np.digitize(data, bins) - 1
    result = np.zeros((data.shape[0], data.shape[1], len(bins)))
    
    for bin_idx in range(len(bins)):
        one_hot_indices = np.nonzero(bin_indices==bin_idx)
        one_hot_indices = (one_hot_indices[0],
                           one_hot_indices[1],
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

def find_focal_len(data_dir):
    np.random.seed(77)
    starting_estimate = [100, 100]
    number_for_analysis = 100
    datapoints_csv_name = os.path.join(data_dir, 'airsim_rec.csv')
    datapoints = pd.read_csv(datapoints_csv_name,
                             header=None,
                             sep=',',
                             names=csv_col_names,
                             index_col=0)
    ix = np.random.choice(np.arange(len(datapoints)), number_for_analysis)

    x0 = [starting_estimate]
    minim_args = (datapoints.iloc[ix],)
    optimize = True
    res = None
    if optimize:
        res = minimize(avg_horizontal_flow, x0, args=minim_args, method='Nelder-Mead', tol=1e-3, options={'disp': True})
    else:
        # vals = [100, 125, 150, 175]
        vals = [134]
        for val in vals:
            print('F={} --> mean_OF:{}'.format(val, avg_horizontal_flow(val, datapoints.iloc[ix])))
    return res

def avg_horizontal_flow(inp, datapoints):
    corr_opt_flow_cumul = 0
    for i in range(len(datapoints)):
        opt_flow_n = os.path.join(data_dir, 'images',
                                  'flow_' + str(datapoints.index[i]) + '.npz')
        opt_flow = np.load(opt_flow_n)['opt_flow']
        opt_flow = cv2.resize(opt_flow, (256, 144))

        corr_opt_flow_cumul += np.mean(np.abs(o_f_compensate_for_rotation(opt_flow,
                                                datapoints.iloc[i][['AV_x', 'AV_y', 'AV_z']],
                                                focal=inp[0], xy_scaling=inp[1])[...,0]))

    corr_opt_flow_cumul /= len(datapoints)
    return corr_opt_flow_cumul

if __name__ == '__main__':
    data_dir = os.path.join('/mnt', 'data', 'AirSimCollectedData', '18-07-23_23-05-47')
    res = find_focal_len(data_dir)
    print(res)
