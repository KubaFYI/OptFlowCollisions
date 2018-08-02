from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.activations import softmax

import tensorflow.keras.backend as K
from tensorflow.python import debug as tf_debug


from Mylayers import MaxPoolingWithArgmax2D, MaxUnpooling2D, CombineMotionWithImg
from SegNet import weighted_categorical_crossentropy, CreateSegNet, weighted_focal_loss, args, softmax_last_axis
import generator

import os
import sys
# import argparse
import cv2
import numpy as np
import cv2
import flowlib
import time
import pdb

data_dir = os.path.join('/mnt', 'data', 'AirSimCollectedData', 'testing')
model_dir = os.path.join('pretrained')

batch_size = 1
input_shape = (144, 256, 2)

bins = generator.default_bins

train_gen = generator.data_gen(data_dir, batch_size,
                     timestamp_range=(0.0, 1.0), range_in_fractions=True,
                     img_resolution=input_shape, include_rgb=True, random_order=True,
                     include_motion_data=True, bins=bins, min_max_value=0.6)

sess = K.get_session()
class_weights, mismatch_weights = generator.calc_class_weights(data_dir,
                                    force_metadata_refresh=False,
                                    bins=bins)

print('Using class weights {}\nAnd mismatch weights {}'.format(class_weights, mismatch_weights))
loss_fnctn = weighted_categorical_crossentropy(class_weights, mismatch_weights)
custom_layers = {'MaxPoolingWithArgmax2D': MaxPoolingWithArgmax2D,
                 'MaxUnpooling2D': MaxUnpooling2D,
                 'CombineMotionWithImg': CombineMotionWithImg,
                 'loss': loss_fnctn,
                 'softmax_last_axis': softmax_last_axis,
                 'focal_loss': weighted_focal_loss(class_weights)}

if args.name is not None:
    auto_checkpoint_name = args.name
else: 
    auto_checkpoint_name = 'auto-checkpoint'
end_checkpoint_name = 'end-checkpoint'
model, loaded_model = CreateSegNet([(144, 256, 2), (1, 1, 3)], nlabels=len(bins))


if os.stat(end_checkpoint_name).st_mtime >= os.stat(auto_checkpoint_name).st_mtime:
    print('Loading {}'.format(end_checkpoint_name))
    loaded_model.load_weights(os.path.join(os.getcwd(), 'end-checkpoint_weights.h5'))
    # loaded_model = load_model(end_checkpoint_name, custom_objects=custom_layers)
else:
    print('Loading {}'.format(auto_checkpoint_name))
    loaded_model = model
    loaded_model = load_model(auto_checkpoint_name, custom_objects=custom_layers)

print(loaded_model.summary())

input_data = next(train_gen)
pred = loaded_model.predict(input_data[0])

def lout(no):
    res = sess.run([loaded_model.layers[no].output], feed_dict={'input_1:0':input_data[0]})
    print('{}\n{}'.format(res,loaded_model.layers[no]))


def one_hot_to_img(img, bins, img_shape):
    '''
    Convert a one-hot encoded (or softmaxed probabilities) image of binned
    values into grayscale.
    '''
    bin_idxs = np.argmax(img, axis=-1)
    retimg = np.zeros_like(bin_idxs, dtype=img.dtype)
    for bin_idx, bin_val in enumerate(bins):
        retimg[np.nonzero(bin_idxs==bin_idx)] = bin_val
    retimg *= 255

    # Finally, unflatten

    if len(retimg.shape) > 1:
        unflattented = np.empty((retimg.shape[0], img_shape[0], img_shape[1]), dtype=retimg.dtype)
        for idx in range(retimg.shape[0]):
            unflattented[idx, ...] = np.reshape(retimg[idx, ...], img_shape)
    else:
        unflattented = np.reshape(retimg, img_shape)
    return unflattented



def visualize_prediction(datapoint, display=True, metadata=None):
    inputs, ground_truth, rgbs, timestamps = datapoint

    predictions = loaded_model.predict(inputs)
    if type(predictions) is list:
        predictions, layers_debug = predictions

    inputs = inputs[0]
    
    ground_truth = one_hot_to_img(ground_truth, bins, inputs.shape[-3:-1])
    print(predictions[0, 777, :])
    predictions = one_hot_to_img(predictions, bins, inputs.shape[-3:-1])

    predictions = predictions.astype(np.uint8)
    ground_truth = ground_truth.astype(np.uint8)

    # make into rgb grayscale
    if len(ground_truth.shape) == 3:
        ground_truth = np.expand_dims(ground_truth, axis=-1)
    ground_truth = np.tile(ground_truth, (1,1,1,3))
    if len(predictions.shape) == 3:
        predictions = np.expand_dims(predictions, axis=-1)
    predictions = np.tile(predictions, (1,1,1,3))

    rgb_scaled = np.empty_like(ground_truth)
    optical_flows = np.empty_like(ground_truth)

    for idx, inp in enumerate(inputs):
        rgb_scaled[idx] = cv2.resize(rgbs[idx], (inputs.shape[2], inputs.shape[1]))
        optical_flows[idx] = flowlib.flow_to_image(inputs[idx])
    sep_thickness = 5
    sep_pat = 4
    separator = np.tile(((np.arange(inputs.shape[-2], dtype=np.uint8) % sep_pat) / sep_pat)*255, (sep_thickness, 1))
    separator = np.tile(np.expand_dims(separator,axis=-1), (1,1,3))
    separator = np.expand_dims(separator,axis=0)
    separator = separator.astype(np.uint8)

    collated_img = np.concatenate((rgb_scaled, separator, optical_flows, separator, ground_truth, separator, predictions), axis=1)

    # Present also some metadata information
    if metadata is not None:
        metadata_idx = metadata['timestamp_idx'][timestamps[0]]
        keys_to_display = ['cam_lin_vel', 'cam_ang_vel']
        for key in keys_to_display:
            if key == 'cam_lin_vel':
                data = generator.velocity_form(metadata[key][metadata_idx])
            else:
                data = metadata[key][metadata_idx]
            print(key + ': ' + str(data))

    if display:
        if metadata is not None:
            print("timestamp {}\tmetadata_idx {}".format(timestamps[0], metadata_idx))
        else:
            print("timestamp {}".format(timestamps[0]))
        cv2.imshow("preview", collated_img[0])
        k = cv2.waitKey(0)
        if k == 27:
            return None
        elif k == ord('s'):
            # Save the image example
            filename = get_f_name_no_duplicates(auto_checkpoint_name, 'png', 'saved_preview')
            cv2.imwrite(filename, collated_img[0])
 
    return collated_img

def get_f_name_no_duplicates(name, extension, directory=os.getcwd()):
    path = os.path.join(directory, '%s.%s' % (name, extension))
    uniq = 1
    while os.path.exists(path):
        path = os.path.join(directory, '%s_%d.%s' % (name, uniq, extension))
        uniq += 1
    return path

if __name__ == '__main__':
    np.random.seed(73)
    data = next(train_gen)
    optical_flow = data[0][1]

    metadata = generator.load_optical_flow_metadata(data_dir, force_metadata_refresh=False)
    while True:
        nex_dat = next(train_gen)
        nex_dat = ([nex_dat[0][0], optical_flow], nex_dat[1], nex_dat[2], nex_dat[3])
        if visualize_prediction(nex_dat, metadata=metadata) is None:
            break