import sys

import keras
import pandas as pd
import scipy as sp
import tensorflow as tf
import platform

'''
print(f"Python Platform: {platform.platform()}")
print(f"Tensor Flow Version: {tf.__version__}")
print(f"Keras Version: {keras.__version__}")
print()
print(f"Python {sys.version}")
print(f"Pandas {pd.__version__}")
print(f"SciPy {sp.__version__}")
gpu = len(tf.config.list_physical_devices('GPU'))>0
print("GPU is", "available" if gpu else "NOT AVAILABLE")
'''
import tensorflow as tf

def dataset_test():
 
    dataset = tf.data.Dataset.from_tensor_slices(["apple", "banana", "cherry"])
    # dataset = tf.data.TextLineDataset(["file1.txt", "file2.txt"]).range(10)             # same output
    # dataset = tf.data.TFRecordDataset(["file1.tfrecords", "file2.tfrecords"]).range(10) # same output
    # dataset = tf.data.Dataset.from_tensors([1,2,3]).range(10)                           # same output

    print(list(dataset.as_numpy_iterator()))

if __name__ == "__main__":
    dataset_test()