import matplotlib as mpl
mpl.use('Agg')
import os, sys
import pickle
import pdb
import flowlib
import cv2
import re
import shutil
import time
import numpy as np
import pandas as pd
import math


from matplotlib import pyplot as plt
from scipy.interpolate import spline
from tensorflow.keras.models import load_model

from Mylayers import MaxPoolingWithArgmax2D, MaxUnpooling2D, CombineMotionWithImg
from SegNet import weighted_categorical_crossentropy, CreateSegNet, weighted_focal_loss, args, softmax_last_axis
from generator import *
from predict import one_hot_to_img

show = False
log_plots = True
do_pdf = True
plotdir = 'plots'
nbins = 20
max_flow = 0

loss_fnctn = weighted_categorical_crossentropy([0,0], [0,0])
custom_layers = {'MaxPoolingWithArgmax2D': MaxPoolingWithArgmax2D,
                 'MaxUnpooling2D': MaxUnpooling2D,
                 'CombineMotionWithImg': CombineMotionWithImg,
                 'loss': loss_fnctn,
                 'softmax_last_axis': softmax_last_axis,
                 'focal_loss': weighted_focal_loss([0,0])}



def plot_histograms(metadata, do_pdf, log_plots, suffix='', bins=10):
    plotnames = {'max_vals': 'Maximum collision likelihood',
                 'mean_vals': 'Average collision likelihood',
                 'sums': 'Total collision likelihood'}
    print('Generating histograms...')
    for elname, plotname in (plotnames.items()):
        for log in ([True, False] if log_plots else [False]):
            fig = plt.figure()
            plt.hist(metadata[elname], bins=bins)
            plt.title("Distribution of {} values{}".format(plotname.lower(),
                                                           ((' (' + suffix + ')') if suffix != '' else suffix)))
            plt.xlabel("{} value".format(plotname))
            if log:
                plt.ylabel("Log number of samples")
                plt.yscale("log")
            else:
                plt.ylabel("Number of samples")
            mpl.style.use('ggplot')

            # fig = plt.figure(figsize=(8,6), dpi=72, facecolor="white")
            # axes = plt.subplot(111)
            # heights, positions, patches = axes.hist(metadata[elname], bins=bins, color='white')
            
            # axes.spines['right'].set_color('none')
            # axes.spines['top'].set_color('none')
            # axes.xaxis.set_ticks_position('bottom')
            
            # # was: axes.spines['bottom'].set_position(('data',1.1*X.min()))
            # axes.spines['bottom'].set_position(('axes', -0.05))
            # axes.yaxis.set_ticks_position('left')
            # axes.spines['left'].set_position(('axes', -0.05))
            
            # axes.set_xlim([np.floor(positions.min()), np.ceil(positions.max())])
            # # axes.set_ylim([0,70])
            # axes.xaxis.grid(False)
            # axes.yaxis.grid(False)
            plt.title("Distribution of {} values{}".format(plotname.lower(), ((' (' + suffix + ')') if suffix != '' else suffix)))
            plt.xlabel("{} value".format(plotname))
            if log:
                plt.ylabel("Log number of samples")
                plt.yscale("log")
            else:
                plt.ylabel("Number of samples")
            
            fig.tight_layout()
            
            figname = os.path.join(plotdir, elname +
                                            ('_log' if log else '') +
                                            (('_' + suffix) if suffix != '' else suffix) +
                                            '_hist')
            plt.savefig(figname + '.png', bbox_inches='tight')
            if do_pdf:
                plt.savefig(figname + '.pdf', bbox_inches='tight')
            if show:
                plt.show()
            plt.close(fig)

def truncated_metadata(metadata, max_val_thres):
    trunc_mdata_name = 'metadata_results_truncated.pickle'
    if os.path.isfile(trunc_mdata_name):
        f = open(trunc_mdata_name, 'rb')
        return pickle.load(f)
    else:
        metadata_over_thresh = {}
        metadata_over_thresh_idx = np.nonzero(metadata['max_vals'] >= max_val_thres)
        # print(metadata.keys())
        for key in metadata.keys():
            if key != 'timestamp_idx':
                metadata_over_thresh[key] = metadata[key][metadata_over_thresh_idx]
            else:
                metadata_over_thresh[key] = {}
                for tmstmp_key in metadata[key]:
                    if metadata[key][tmstmp_key][0] in metadata_over_thresh_idx[0]:
                        metadata_over_thresh[key][tmstmp_key] = metadata[key][tmstmp_key]
        return metadata_over_thresh

def adj_wrong_mulitplication(metadata, factor=1.):
    vals = ['sums', 'mean_vals', 'max_vals']
    for val in vals:
        metadata[val] = metadata[val] / factor
    return metadata

def matching_stats(pred, truth):
    # Convert form one-hot to indexed
    pred = np.argmax(pred, axis=-1)[0,:]
    truth = np.argmax(truth, axis=-1).flatten()
    pixels_no = pred.shape[-1]
    positives_match = np.sum(np.logical_and((pred == 1), (truth == 1))) / pixels_no
    negatives_match = np.sum(np.logical_and((pred == 0), (truth == 0))) / pixels_no
    false_positives = np.sum(np.logical_and((pred == 1), (truth == 0))) / pixels_no
    false_negatives = np.sum(np.logical_and((pred == 0), (truth == 1))) / pixels_no
    # if math.isnan(np.sum((pred[np.nonzero(truth==0)] == 1))/np.sum(truth==0)) or math.isnan(np.sum((pred[np.nonzero(truth==1)] == 1))/np.sum(truth==1)):
    #     pdb.set_trace()
    a = np.sum((pred[np.nonzero(truth==0)] == 1))/np.sum(truth==0)
    b = np.sum((pred[np.nonzero(truth==1)] == 1))/np.sum(truth==1)
    return positives_match + negatives_match, a if not math.isnan(a) else np.sum((pred[np.nonzero(truth==0)] == 1)) / pixels_no, b if not math.isnan(b) else np.sum((pred[np.nonzero(truth==1)] == 1)) / pixels_no

def get_opt_flow(dir, timestamp):
    opt_flow_n = os.path.join(dir, 'images',
                              'flow_' + str(timestamp) + '.npz')
    opt_flow = np.load(opt_flow_n)['opt_flow']
    opt_flow = cv2.resize(opt_flow, (256, 144))
    return opt_flow

def get_opt_flow_rgb(dir, timestamp, datapoints=None, correction=True):
    opt_flow_n = os.path.join(dir, 'images',
                              'flow_' + str(timestamp) + '.npz')
    opt_flow = np.load(opt_flow_n)['opt_flow']
    opt_flow = cv2.resize(opt_flow, (256, 144))

    # optical_flow_correction_stats needs to be rune before to callibrate flowlib rgb conversion
    opt_flow = o_f_compensate_for_rotation(opt_flow, datapoints.loc[timestamp][['AV_x', 'AV_y', 'AV_z']], focal=4.47189418e+00, xy_scaling=4.78224949e-06)
    flowlib.enable_sticky_max_flow(False)
    opt_flow_rgb = flowlib.flow_to_image(opt_flow)
    return opt_flow_rgb

# def plot_motion_stuff(metadata, show=False):
#     fig = plt.figure()
#     plt.hist(metadata[elname], bins=bins)
#     plt.title("Distribution of {} values{}".format(plotname.lower(),
#                                                            ((' (' + suffix + ')') if suffix != '' else suffix)))
#             plt.xlabel("{} value".format(plotname))
#             if log:
#                 plt.ylabel("Log number of samples")
#                 plt.yscale("log")
#             else:
#                 plt.ylabel("Number of samples")
#     plt.savefig(figname + '.png', bbox_inches='tight')
#     if do_pdf:
#         plt.savefig(figname + '.pdf', bbox_inches='tight')
#     if show:
#         plt.show()

def optical_flow_correction_stats(data_dir, batch_size):
    orig_image_size = (270, 480)    # (h,w)
    FOV = 90    # in degeres
    cam_f = orig_image_size[1] / 2 / np.tan( FOV / 180 * np.pi / 2 )
    corr_params = {'raw': None,
                   'optimized': (4.47189418e+00, 4.78224949e-06),
                   'calculated': (orig_image_size[1] / 2 / np.tan( FOV / 180 * np.pi / 2 ), 1)}

    datapoints_csv_name = os.path.join(data_dir, 'airsim_rec.csv')
    datapoints = pd.read_csv(datapoints_csv_name,
                             header=None,
                             sep=',',
                             names=csv_col_names,
                             index_col=0)
    ix = np.random.choice(np.arange(len(datapoints)-1), batch_size) + 1
    flow_to_save = []
    avg_mag = {'raw': 0, 'optimized': 0, 'calculated': 0}
    global max_flow # Yes, I feel bad for using a global but whatever, time shortcuts
    print('Generating optical flow examples...')
    for i, idx in (enumerate(ix)):
        opt_flow = get_opt_flow(data_dir, datapoints.index[idx])
        # print(hash(str(opt_flow)))
        for name, params in corr_params.items():
            if params is not None:
                # corrected optical flow
                opt_flow_alt = o_f_compensate_for_rotation(opt_flow,
                                                       datapoints.iloc[idx][['AV_x', 'AV_y', 'AV_z']],
                                                       focal=params[0], xy_scaling=params[1])
            else:
                opt_flow_alt = opt_flow

            savename = os.path.abspath(os.path.join(plotdir, 'opt_fl_eg', name + '_{}.png'.format(str(i))))
            flow_to_save.append((savename, opt_flow_alt))

            flow_norm = np.linalg.norm(opt_flow_alt, axis=-1)
            avg_mag[name] += np.mean(flow_norm) / batch_size
            print('{}_{}'.format(name, i))
            print(np.max(flow_norm))
            if name != 'calculated':
                # Calculated corrector gives overinflated flow values
                max_flow = np.max(flow_norm) if np.max(flow_norm) > max_flow else max_flow
            
        savename = os.path.abspath(os.path.join(plotdir, 'opt_fl_eg', 'farneback_{}.png'.format(str(i))))
        opt_flow_alt = get_cv2_flow(datapoints, os.path.join(data_dir, 'images'), idx)
        flow_to_save.append((savename, opt_flow_alt))
            
        savename = os.path.abspath(os.path.join(plotdir, 'opt_fl_eg', 'scene_{}.png'.format(str(i))))
        opt_flow_alt = get_scene(datapoints, os.path.join(data_dir, 'images'), idx)
        flow_to_save.append((savename, opt_flow_alt))

    print('Max flow mag encountered: {}'.format(max_flow))
    flowlib.set_max_flow(max_flow)
    for flow in flow_to_save:
        if not os.path.isdir(os.path.dirname(flow[0])):
            os.makedirs(os.path.dirname(flow[0]))
        if flow[1].shape[-1] != 3:
            opt_flow_rgb = flowlib.flow_to_image(flow[1])
        else:
            opt_flow_rgb = flow[1]
        cv2.imwrite(flow[0], opt_flow_rgb)

    print('Max flow mag: {}'.format(max_flow))

    for name, params in corr_params.items():
        print('Correction parameters: {}\nAvg flow: {}\n\n'.format(params, avg_mag[name]))

def get_cv2_flow(datapoints, img_dir, idx):
    curr_n = os.path.join(img_dir, 'rgb_' + str(datapoints.index[idx]) + '.png')
    curr = cv2.imread(curr_n)
    curr = cv2.resize(curr, (256, 144))
    curr = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
    prev_n = os.path.join(img_dir, 'rgb_' + str(datapoints.index[idx-1]) + '.png')
    prev = cv2.imread(prev_n)
    prev = cv2.resize(prev, (256, 144))
    prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    # (prev, next, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags[, flow]) → flow
    start = time.time()
    for i in range(100):
        opt_flow = cv2.calcOpticalFlowFarneback(prev, curr, None, 0.5, 10, 15, 3, 5, 1.1, 0)

    print('timed {}ms'.format((time.time()-start)*10))
    return opt_flow

def get_scene(datapoints, img_dir, idx):
    rgb_n = os.path.join(img_dir, 'rgb_' + str(datapoints.index[idx]) + '.png')
    rgb = cv2.imread(rgb_n)
    rgb = cv2.resize(rgb, (256, 144))
    rgb = increase_brightness(rgb, value=30)
    return rgb

def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also: 

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise(ValueError, "smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise(ValueError, "Input vector needs to be bigger than window size.")


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise(ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y

def plot_smooth_and_normal(xs, ys, color, label, just_normal=False):
    plt.plot(xs, ys, color=color, alpha=0.2)

    if not just_normal:
        # if xs is tuple:
        # x_smth= np.linspace(xs.min(),xs.max(),300) #300 represents number of points to make between T.min and T.max
        x_smth= xs #300 represents number of points to make between T.min and T.max
        window = 11
        ys_smth = smooth(ys,window_len=window)
        plt.plot(x_smth, ys_smth[6:-4], color=color, label=label)
        ymax =np.max(ys_smth[6:-4])
        ymin = np.min(ys_smth[6:-4])

    return (ymin, ymax)

def create_training_plots(do_pdf=True, log=False):
    plotdir = 'plots'
    pickle_name = 'tensorboard.pickle'
    with open(pickle_name, 'rb') as f:
        tdata = pickle.load(f)

    plot_labels = {}
    plot_labels['3convs_fixed_focal_loss_adam_incl_ang_motion'] = ''

    names = {'loss': 'Loss (training)', 'val_loss': 'Loss (validation)', ('loss', 'val_loss'): 'combined Losses'}

    colors = ['b', 'g', 'p']

    if not os.path.isdir(os.path.join(plotdir, 'run_plots')):
            os.makedirs(os.path.join(plotdir, 'run_plots'))
    print('Generating run plots...')
    for run_name in (tdata):
        tdata[run_name]
        if run_name != '3convs_fixed_weighted_focal_loss_adam_incl_ang_motion_rot_comp':
            continue
        to_plot = ['loss', 'val_loss', ('loss', 'val_loss')]
        ymax_best = 0
        for plotting in to_plot:
            fig = plt.figure()
            if type(plotting) is tuple:
                for idx, el in enumerate(plotting):
                    xs = np.array(tdata[run_name][el]['step_nums'])
                    ys = np.array(tdata[run_name][el]['vals']) /1.53

                    if ys.shape[0] <=1:
                        break

                    # plot
                    ymin, ymax = plot_smooth_and_normal(xs, ys, colors[idx], names[el])
                    if ymax_best > ymax:
                        ymax = ymax_best
                    else:
                        ymax_best = ymax
                    if run_name in plot_labels.keys():
                        title = plot_labels[run_name]
                    else:
                        title = run_name
                plt.legend()
            else:
                ymax = None
                xs = np.array(tdata[run_name][plotting]['step_nums'])
                ys = np.array(tdata[run_name][plotting]['vals'])
                if ys.shape[0] <=1:
                        break
                plt.plot(xs, ys)
                if run_name in plot_labels.keys():
                    title = 'Run name: ' + plot_labels[run_name] + names[plotting]
                else:
                    title = 'Run name: ' + run_name + names[plotting]
            title = ''

            # plt.title(title)
            plt.xlabel("Epochs")

            if log:
                plt.ylabel("Log value")
                plt.yscale("log")
            else:
                plt.ylabel("Value")
            mpl.style.use('ggplot')
            
            if type(plotting) is tuple:
                plotting_n = ''
                for el in plotting:
                    plotting_n = plotting_n + '_' + str(el)
            else:
                plotting_n = str(plotting)
            if ymax is not None:
                print(1.1*ymax)
                plt.ylim(0, 1.1*ymax)

            figname = os.path.join(plotdir, 'run_plots', run_name + '_' + plotting_n)
            plt.savefig(figname + '.png', bbox_inches='tight')
            if do_pdf:
                plt.savefig(figname + '.pdf', bbox_inches='tight')
            if show:
                plt.show()
            plt.close(fig)

def get_model(checkpoint_file, epoch='', checkpoint_type='auto', incl_ang_vel=False):
    # checkpoint_file = checkpoint_type + '-checkpoint_' + run_name + epoch
    loaded_model = load_model(checkpoint_file, custom_objects=custom_layers)
    # loaded_model, _ = CreateSegNet([(144, 256, 2), (1, 1, 3) if not incl_ang_vel else (1, 1, 6)], nlabels=len(bins))
    return loaded_model

def create_example_predictiors(data_dir, number, metadata, chkpoints_to_use=None):
    local_metadata = load_optical_flow_metadata([[data_dir]], bins=default_bins)

    dir_contents = os.listdir('checkpoints')
    pat = re.compile('-checkpoint_.*(?<!.h5)$')
    pat2 = re.compile('.h5')
    if chkpoints_to_use is None:
        checkpoints = [os.path.join('checkpoints', x) for x in dir_contents if (pat.search(x) is None) and (pat2.search(x) is None)]
    else:
        checkpoints = chkpoints_to_use

    print('Generating examples of checkpoint behaviour...')
    for checkpoint in (checkpoints):
        with_ang_motion = ['auto-checkpoint_3convs_fixed_focal_loss_adam_incl_ang_motion10',
                           '']

        # wee heuristic

        if checkpoint in with_ang_motion or ((re.search('.*inc.*', checkpoint) is not None) and (re.search('.*rot.*', checkpoint) is None)):
            incl_ang_vel = True
        else:
            incl_ang_vel = False
        print('{} -> {}'.format(checkpoint, incl_ang_vel))
        model = get_model(checkpoint)

        datapoints_csv_name = os.path.join(data_dir, 'airsim_rec.csv')
        datapoints = pd.read_csv(datapoints_csv_name,
                                 header=None,
                                 sep=',',
                                 names=csv_col_names,
                                 index_col=0)
        datapoints = datapoints.iloc[np.nonzero(local_metadata['max_vals'] > 0.3)]
        ix = np.random.choice(np.arange(len(datapoints)-1), number) + 1
        for idx, i in enumerate(ix):
            opt_flow = get_opt_flow(data_dir, datapoints.index[i])
            opt_flow_rgb = get_opt_flow_rgb(data_dir, datapoints.index[i], datapoints)
            coll_dist_n = os.path.join(data_dir, 'images',
                                      'misc_' + str(datapoints.index[i]) + '.npz')
            coll_dist = np.load(coll_dist_n)['arr_0']
            ground_truth = bin_pixels(coll_dist, default_bins)

            if not incl_ang_vel:
                vel_info = metadata['cam_lin_vel'][i]
            else:
                vel_info = np.concatenate((metadata['cam_lin_vel'][i], metadata['cam_ang_vel'][i]), axis=-1)
            vel_info = np.expand_dims(np.expand_dims(vel_info, axis=0), axis=0)
            prediction = model.predict([np.expand_dims(opt_flow, axis=0), np.expand_dims(vel_info, axis=0)])

            ground_truth = one_hot_to_img(np.expand_dims(ground_truth, axis=0), default_bins, (144, 256))
            prediction = one_hot_to_img(np.expand_dims(prediction, axis=0), default_bins, (144, 256))

            pred_dir = os.path.join('plots', 'pred_ex', checkpoint)
            ground_truth_n = os.path.join(pred_dir, '{}_ground_truth.png'.format(idx))
            prediction_n = os.path.join(pred_dir, '{}_prediction.png'.format(idx))
            opt_flow_rgb_n = os.path.join(pred_dir, '{}_opt_flow_rgb.png'.format(idx))
            if not os.path.isdir(pred_dir):
                os.makedirs(pred_dir)
            src = os.path.join(data_dir, 'images', 'rgb_' + str(datapoints.index[i]) + '.png')
            dst = os.path.join(pred_dir, '{}_scene.png'.format(idx))
            shutil.copyfile(src, dst)
            cv2.imwrite(ground_truth_n, np.tile(np.expand_dims(ground_truth[0,...], axis=-1), (1,1,3)))
            cv2.imwrite(prediction_n, np.tile(np.expand_dims(prediction[0,...], axis=-1), (1,1,3)))
            cv2.imwrite(opt_flow_rgb_n, opt_flow_rgb)


def eval_performance(data_dir, metadata, chkpnts_for_benchmark=None):
    if chkpnts_for_benchmark is None:
        chkpnts_for_benchmark = dir_contents = os.listdir('checkpoints')
    local_metadata = load_optical_flow_metadata([[data_dir]], bins=default_bins)
    timing_repeats = 100
    accuracy_samples = 1000
    for checkpoint in (chkpnts_for_benchmark):
        if checkpoint[-3:] == '.h5':
            continue
        print('Evaluating checkpoint {}...'.format(checkpoint))
        sys.stdout.flush()
        with_ang_motion = ['auto-checkpoint_3convs_fixed_focal_loss_adam_incl_ang_motion10',
                           '']

        # wee heuristic
        if checkpoint in with_ang_motion or ((re.search('.*inc.*', checkpoint) is not None) and (re.search('.*rot.*', checkpoint) is None)):
            incl_ang_vel = True
        else:
            incl_ang_vel = False
        print('{} -> {}'.format(checkpoint, incl_ang_vel))
        model = get_model(os.path.join('checkpoints', checkpoint))



        #  timing check
        opt_flow = np.random.rand(1, 144, 256, 2)

        if not incl_ang_vel:
            vel_info = np.random.rand(1, 1, 1, 3)
        else:
            vel_info = np.random.rand(1, 1, 1, 6)
        
        start_time = time.time()
        for i in range(timing_repeats):
            prediction = model.predict([opt_flow, vel_info])
        timing = time.time() - start_time
        sys.stdout.flush()
        print('{} exectues at {}s ({}ms)'.format(checkpoint, timing/timing_repeats, timing/timing_repeats*1000))
        sys.stdout.flush()

        # accuracy check
        datapoints_csv_name = os.path.join(data_dir, 'airsim_rec.csv')
        datapoints = pd.read_csv(datapoints_csv_name,
                                 header=None,
                                 sep=',',
                                 names=csv_col_names,
                                 index_col=0)
        datapoints = datapoints.iloc[np.nonzero(local_metadata['max_vals'] > 0.0)]
        ix = np.random.choice(np.arange(len(datapoints)-1), accuracy_samples) + 1
        tot_acc = 0
        tot_false_pos = 0
        tot_false_neg = 0
        for i in tqdm(ix):
            opt_flow = get_opt_flow(data_dir, datapoints.index[i])
            coll_dist_n = os.path.join(data_dir, 'images',
                                      'misc_' + str(datapoints.index[i]) + '.npz')
            coll_dist = np.load(coll_dist_n)['arr_0']
            ground_truth = bin_pixels(coll_dist, default_bins)
            if not incl_ang_vel:
                vel_info = metadata['cam_lin_vel'][i]
            else:
                vel_info = np.concatenate((metadata['cam_lin_vel'][i], metadata['cam_ang_vel'][i]), axis=-1)
            vel_info = np.expand_dims(np.expand_dims(vel_info, axis=0), axis=0)
            prediction = model.predict([np.expand_dims(opt_flow, axis=0), np.expand_dims(vel_info, axis=0)])
            acc, false_pos, false_neg = matching_stats(prediction, ground_truth)
            tot_acc += acc
            tot_false_pos += false_pos
            tot_false_neg += false_neg

        sys.stdout.flush()

        print('Correctly classified: {}\nFalse positives: {}\nFalse negatives: {}'.format(tot_acc/accuracy_samples, tot_false_pos/accuracy_samples, tot_false_neg/accuracy_samples))

        sys.stdout.flush()

if __name__ == '__main__':
    # Variables
    # plt.xkcd()

    # Load metadata
    mdata_name = 'metadata_results.pickle'
    mdata_f = open(mdata_name, 'rb')
    metadata = pickle.load(mdata_f)
    mdata_f.close()

    metadata = adj_wrong_mulitplication(metadata)

    # Calculate more balanced dataset
    max_val_thres = 0.05
    metadata_over_thresh = truncated_metadata(metadata, max_val_thres)

    # Overall stats
    print('Metadata keys: {}'.format(metadata.keys()))
    print('Datapoints total: {}'.format(len(metadata['max_vals'])))
    print('Datapoints balanced total: {} ({}% of original)'.format(len(metadata_over_thresh['max_vals']),
                                                                    100*len(metadata_over_thresh['max_vals'])/len(metadata['max_vals'])))

    print('Normal data')
    print('Max vals:\n\tmin {}\n\tmax {}\n\tmean {}'.format(np.min(metadata['max_vals']), np.max(metadata['max_vals']), np.mean(metadata['max_vals'])))
    print('Sum vals:\n\tmin {}\n\tmax {}\n\tmean {}'.format(np.min(metadata['sums']), np.max(metadata['sums']), np.mean(metadata['sums'])))

    print(metadata['class_count'].shape)
    print('Class 0 vals:total {}\n\tmin {}\n\tmax {}\n\tmean {}'.format(np.sum(metadata['class_count'][:,0]), np.min(metadata['class_count'][:,0]), np.max(metadata['class_count'][:,0]), np.mean(metadata['class_count'][:,0])))
    print('Class 1 vals:total {}\n\tmin {}\n\tmax {}\n\tmean {}'.format(np.sum(metadata['class_count'][:,1]), np.min(metadata['class_count'][:,1]), np.max(metadata['class_count'][:,1]), np.mean(metadata['class_count'][:,1])))

    presence = np.nonzero(metadata['class_count'][:,1]>0)
    print('{} has class 1 in it'.format(np.sum(metadata['class_count'][:,1]>0)))
    ratios = metadata['class_count'][presence,1]/(np.sum(metadata['class_count'][presence,:], axis=-1))
    print('% class 1:\n\tmin {}\n\tmax {}\n\tmean {}'.format(np.min(ratios), np.max(ratios), np.mean(ratios)))

    print('balanced data:')
    print('Max vals:\n\tmin {}\n\tmax {}\n\tmean {}'.format(np.min(metadata_over_thresh['max_vals']), np.max(metadata_over_thresh['max_vals']), np.mean(metadata_over_thresh['max_vals'])))
    print('Sum vals:\n\tmin {}\n\tmax {}\n\tmean {}'.format(np.min(metadata_over_thresh['sums']), np.max(metadata_over_thresh['sums']), np.mean(metadata_over_thresh['sums'])))

    print(metadata_over_thresh['class_count'].shape)
    print('Class 0 vals:total {}\n\tmin {}\n\tmax {}\n\tmean {}'.format(np.sum(metadata_over_thresh['class_count'][:,0]), np.min(metadata_over_thresh['class_count'][:,0]), np.max(metadata_over_thresh['class_count'][:,0]), np.mean(metadata_over_thresh['class_count'][:,0])))
    print('Class 1 vals:total {}\n\tmin {}\n\tmax {}\n\tmean {}'.format(np.sum(metadata_over_thresh['class_count'][:,1]), np.min(metadata_over_thresh['class_count'][:,1]), np.max(metadata_over_thresh['class_count'][:,1]), np.mean(metadata_over_thresh['class_count'][:,1])))


    presence = np.nonzero(metadata_over_thresh['class_count'][:,1]>0)
    ratios = metadata_over_thresh['class_count'][presence,1]/(np.sum(metadata_over_thresh['class_count'][presence,:], axis=-1))
    print('% class 1:\n\tmin {}\n\tmax {}\n\tmean {}'.format(np.min(ratios), np.max(ratios), np.mean(ratios)))
    # Collision value stats
    # Histograms
    # plot_histograms(metadata, do_pdf, log_plots, bins=20)
    # plot_histograms(metadata_over_thresh, do_pdf, log_plots, suffix='balanced', bins=20)

    # # Plot optical flow examples
    np.random.seed(777)
    data_dir = os.path.join('/mnt', 'data', 'AirSimCollectedData', '18-08-07_10-37-26')
    # data_dir = os.path.join('/media', 'disk1', 'kuba', 'TrainingData', '18-08-07_10-37-26')
    batch_size = 30

    # # Provide stats regarding optical flow correction
    # optical_flow_correction_stats(data_dir, batch_size)
    # pdb.set_trace()
    # create_training_plots()
    chkpoints_to_use = ['auto-checkpoint_enc_dec_1.5root_wght_foc_2dts300',
    'auto-checkpoint_enc_dec_sqrt_wght_foc_2dts300',
    'auto-checkpoint_enc_dec_wght_foc_2dts300',
    'auto-checkpoint_1x1_1.5root_wght_foc_2dts300',
    'end-checkpoint_3convs_fixed_weighted_focal_loss_adam_incl_ang_motion_rot_comp']
    # for i, c in enumerate(chkpoints_to_use):
    #     chkpoints_to_use[i] = os.path.join('checkpoints', c)
    create_example_predictiors(data_dir, 20, metadata, chkpoints_to_use=[os.path.join('checkpoints', 'auto-checkpoint_enc_dec_wght_foc_2dts300')])

    # chkpnts_for_benchmark = ['auto-checkpoint_enc_dec_1.5root_wght_foc_2dts200',
    #                          'auto-checkpoint_enc_dec_1.5root_wght_foc_2dts_inc_ang_vel200']
    eval_performance(data_dir, metadata, chkpnts_for_benchmark=chkpoints_to_use)