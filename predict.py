from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model

import tensorflow.keras.backend as K
from tensorflow.python import debug as tf_debug


from Mylayers import MaxPoolingWithArgmax2D, MaxUnpooling2D, CombineMotionWithImg
import generator

import os
import argparse
import cv2
import numpy as np
import cv2
import flowlib
import time

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

custom_layers = {'MaxPoolingWithArgmax2D': MaxPoolingWithArgmax2D,
                 'MaxUnpooling2D': MaxUnpooling2D,
                 'CombineMotionWithImg': CombineMotionWithImg}
auto_checkpoint_name = 'auto-checkpoint'
end_checkpoint_name = 'end_checkpoint'
if os.stat(end_checkpoint_name).st_mtime >= os.stat(auto_checkpoint_name).st_mtime:
    loaded_model = load_model(end_checkpoint_name, custom_objects=custom_layers)
else:
    loaded_model = load_model(auto_checkpoint_name, custom_objects=custom_layers)

print(loaded_model.summary())

input_data = next(train_gen)
pred = loaded_model.predict(input_data[0])

def lout(no):
    res = sess.run([loaded_model.layers[no].output], feed_dict={'input_1:0':input_data[0]})
    print('{}\n{}'.format(res,loaded_model.layers[no]))


def one_hot_to_img(img, bins):
    '''
    Convert a one-hot encoded (or softmaxed probabilities) image of binned
    values into grayscale.
    '''
    bin_idxs = np.argmax(img, axis=-1)
    retimg = np.zeros_like(bin_idxs, dtype=img.dtype)
    for bin_idx, bin_val in enumerate(bins):
        retimg[np.nonzero(bin_idxs==bin_idx)] = bin_val
    retimg *= 255
    return retimg



def visualize_prediction(datapoint, display=True):
    inputs, ground_truth, rgbs, timestamps = datapoint
    predictions = loaded_model.predict(inputs)
    inputs = inputs[0]

    ground_truth = one_hot_to_img(ground_truth, bins)
    predictions = one_hot_to_img(predictions, bins)

    predictions = predictions.astype(np.uint8)
    ground_truth = ground_truth.astype(np.uint8)

    # make into rgb grayscale
    if len(ground_truth.shape) == 3:
        ground_truth = np.expand_dims(ground_truth, axis=-1)
    ground_truth = np.concatenate((ground_truth, ground_truth, ground_truth), axis=3)
    if len(predictions.shape) == 3:
        predictions = np.expand_dims(predictions, axis=-1)
    predictions = np.concatenate((predictions, predictions, predictions), axis=3)

    rgb_scaled = np.empty_like(ground_truth)
    optical_flows = np.empty_like(ground_truth)

    for idx, inp in enumerate(inputs):
        rgb_scaled[idx] = cv2.resize(rgbs[idx], (inputs.shape[2], inputs.shape[1]))
        optical_flows[idx] = flowlib.flow_to_image(inputs[idx])

    collated_img = np.concatenate((rgb_scaled, optical_flows, ground_truth, predictions), axis=1)

    if display:
        print("timestamp {}".format(timestamps[0]))
        cv2.imshow("preview", collated_img[0])
        k = cv2.waitKey(0)
        if k == 27:
            return None

    return collated_img


if __name__ == '__main__':
    while True:
        if visualize_prediction(next(train_gen)) is None:
            break