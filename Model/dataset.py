
import os
import tensorflow as tf
import numpy as np
import cv2
import time
import glob
import math
import random
import torch
from torch.utils.data import Dataset, DataLoader
import config
from tfrecord.torch.dataset import TFRecordDataset

CHANNELS = "xyzdr"

IMAGE_HEIGHT = 512
IMAGE_WIDTH = 64
N_CLASSES = 4
N_LEN = 8

def seq_to_idx(seq):
    idx = []
    if "x" in seq:
        idx.append(0)
    if "y" in seq:
        idx.append(1)
    if "z" in seq:
        idx.append(2)
    if "r" in seq:
        idx.append(3)
    if "d" in seq:
        idx.append(4)

    return np.array(idx, dtype=np.intp)

def read_example(dataset, batch_size):
    feature_description = {
        'neighbors': tf.io.FixedLenFeature([], tf.string),
        'points': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.string),
    }
    with tf.device(f'/device:GPU:{config.GPU}'):
        # Create a dataset of records
        #dataset = tf.data.TFRecordDataset([filename])

        # Define the parsing function for each record
        def parse_fn(serialized_example):
            example = tf.io.parse_single_example(serialized_example, feature_description)
            return example

        # Map the parsing function to each record in the dataset
        parsed_dataset = dataset.map(parse_fn)

        # Shuffle and batch the dataset
        buffer_size = 1000
        parsed_dataset = parsed_dataset.shuffle(buffer_size=buffer_size).batch(batch_size)

        # Create an iterator to iterate through the dataset
        iterator = iter(parsed_dataset)

        # Get the next batch of data from the iterator
        batch = next(iterator)

        idx = seq_to_idx(CHANNELS)
        points_raw = tf.io.decode_raw(batch['points'], tf.float32)
        points = tf.reshape(points_raw, [batch_size, IMAGE_HEIGHT * IMAGE_WIDTH, 1 , 5])
        points = tf.gather(points, seq_to_idx(CHANNELS), axis=3)


        neighbors_raw = tf.io.decode_raw(batch['neighbors'], tf.float32)
        neighbors = tf.reshape(neighbors_raw, [batch_size, IMAGE_HEIGHT * IMAGE_WIDTH, N_LEN, 5])
        neighbors = tf.gather(neighbors, seq_to_idx(CHANNELS), axis=3)

        # Decode label
        label_raw = tf.io.decode_raw(batch['label'], tf.float32)
        label = tf.reshape(label_raw, [batch_size, IMAGE_HEIGHT* IMAGE_WIDTH, N_CLASSES + 2])
    
    return points, neighbors, label


    
    

class PointDataset(Dataset):
    def __init__(self, filename, batch_size=1):
        self.filename = filename
        self.batch_size = batch_size
        self.tfrecord_list = glob.glob(filename)
        self.len = len(self.tfrecord_list)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        # Read the example at the given index from the TFRecord file
        
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            filename = self.tfrecord_list[index]
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            filenames = self.tfrecord_list[worker_id::num_workers]
            filename = filenames[index % len(filenames)]
            random.seed(worker_info.seed)

        # Read the example from the TFRecord file
        dataset = tf.data.TFRecordDataset([filename])
        
        
        points, neighbors, label = read_example(dataset, batch_size=self.batch_size)

        # Convert the TensorFlow tensors to PyTorch tensors
        points = torch.from_numpy(points.numpy()).transpose(1,3)
        neighbors = torch.from_numpy(neighbors.numpy()).transpose(1,3)
        label = torch.from_numpy(label.numpy()).transpose(1,2).reshape(self.batch_size,N_CLASSES + 2, IMAGE_HEIGHT, IMAGE_WIDTH)
        #print(points.shape)
        # Return a tuple of the PyTorch tensors
        return points, neighbors, label
    
def get_dataloader(path, batch_size, n_workers):
    
    dataset = PointDataset(path, batch_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers)
    return dataloader
    

if __name__ == "__main__":
    dataset = PointDataset("data/pnl_train.tfrecord")
    print(len(dataset))
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=20)
    batch_points, batch_neighbors, batch_label = next(iter(dataloader))
    print(batch_points.shape, batch_neighbors.shape, batch_label.shape)
    print(len(dataloader))
