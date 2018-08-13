import os, sys
import pickle
import pdb
import numpy as np
from matplotlib import pyplot as plt


def plot_histograms(metadata, do_pdf, log_plots, suffix='', bins=10):
    plotnames = {'max_vals': 'Maximum collision likelihood',
                 'mean_vals': 'Average collision likelihood',
                 'sums': 'Total collision likelihood'}
    for elname, plotname in plotnames.items():

        # if elname == 'mean_vals':
        #     pdb.set_trace()
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
                    # pdb.set_trace()
                    if metadata[key][tmstmp_key][0] in metadata_over_thresh_idx[0]:
                        metadata_over_thresh[key][tmstmp_key] = metadata[key][tmstmp_key]
        return metadata_over_thresh

def adj_wrong_mulitplication(metadata):
    vals = ['sums', 'mean_vals', 'max_vals']
    for val in vals:
        metadata[val] = metadata[val] / 2.
    return metadata

def matching_stats(pred, truth):
    # Convert form one-hot to indexed
    pred = np.argmax(pred, axis=-1)
    truth = np.argmax(truth, axis=-1)
    pixels_no = pred.shape[-2] * pred.shape[-3]
    positives_match = np.sum(np.logical_and((pred == 1), (truth == 1))) / pixels_no
    negatives_match = np.sum(np.logical_and((pred == 0), (truth == 0))) / pixels_no
    false_positives = np.sum(np.logical_and((pred == 1), (truth == 0))) / pixels_no
    false_negatives = np.sum(np.logical_and((pred == 0), (truth == 1))) / pixels_no
    return positives_match + negatives_match, false_positives, false_negatives

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

if __name__ == '__main__':
    # Variables
    show = False
    log_plots = True
    do_pdf = True
    plotdir = 'plots'
    nbins = 20
    plt.close("all")
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

    # Collision value stats
    # Histograms
    plot_histograms(metadata, do_pdf, log_plots, bins=20)
    plot_histograms(metadata_over_thresh, do_pdf, log_plots, suffix='balanced', bins=20)

    # 
