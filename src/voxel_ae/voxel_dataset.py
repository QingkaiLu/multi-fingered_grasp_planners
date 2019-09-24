import tensorflow as tf
#tf.enable_eager_execution()
import numpy as np
import random

from os import listdir
from os.path import join, isfile

import pdb

def get_voxel_dataset(tffiles, batch_size=32):
    '''
    Given a list of TFRecord filenames, create a voxel dataset (assume data is in
    voxel partial/full format) and return it.
    '''

    dataset = tf.data.TFRecordDataset(tffiles)
    
    # Setup parsing of objects.
    voxel_feature_description = {
        'partial': tf.FixedLenFeature([], tf.string),
        'full': tf.FixedLenFeature([], tf.string),
    }

    def _parse_voxel_function(example_proto):
        voxel_example = tf.parse_single_example(example_proto, voxel_feature_description)
        return tf.reshape(tf.parse_tensor(voxel_example['partial'], out_type=tf.bool), (32,32,32,1)), tf.reshape(tf.parse_tensor(voxel_example['full'], out_type=tf.bool), (32,32,32,1))

    dataset = dataset.map(_parse_voxel_function)
    dataset = dataset.batch(batch_size)
    dataset = dataset.shuffle(1500) # Shuffle buffer size is size of single TRFormat.
    dataset = dataset.prefetch(8)

    return dataset

if __name__ == '__main__':
    train_folder = '/home/markvandermerwe/ReconstructionData/Train'
    train_files = [join(train_folder, filename) for filename in listdir(train_folder) if ".tfrecord" in filename]
    
    dataset = get_voxel_dataset(train_files)

    for x, y in dataset:
        print x, y
    
