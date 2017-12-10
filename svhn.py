""" 
SVHN Dataset Loading Functions

Dataset available at: http://ufldl.stanford.edu/housenumbers/
Based on code from: https://github.com/tflearn/tflearn/blob/master/tflearn/datasets/cifar10.py

"""

from __future__ import absolute_import, print_function

import os
import sys
from six.moves import urllib
import tarfile
import scipy.io as sio

import numpy as np
import pickle

#from ..data_utils import to_categorical


def load_data(dirname="SVHN", one_hot=False):
    '''
    Main function to load the SVHN dataset. Returns the examples as NumPy array and labels as a list
    
    '''
    
    # Download data if it does not exist yet
    tarpath = maybe_download("train_32x32.mat", "http://ufldl.stanford.edu/housenumbers/", dirname)
    
    tarpath = maybe_download("test_32x32.mat", "http://ufldl.stanford.edu/housenumbers/", dirname)

    
    fpath = os.path.join(dirname, 'train_32x32.mat')
    X_train, Y_train = load_batch(fpath)
    
    # make X channel last
    X_train = np.reshape(X_train, [-1, 32, 32, 3])
    # convert "0" labels to be value 0, convert to list
    Y_train = np.ndarray.tolist(np.squeeze(np.mod(Y_train, 10)))
    
    
    fpath = os.path.join(dirname, 'test_32x32.mat')
    X_test, Y_test = load_batch(fpath)
    
    # make X channel last 
    X_test = np.reshape(X_test, [-1, 32, 32, 3])
    # convert "0" labels to be value 0, convert to list
    Y_test = np.ndarray.tolist(np.squeeze(np.mod(Y_test, 10)))

    return (X_train, Y_train), (X_test, Y_test)


def load_batch(fpath):
    with open(fpath, 'rb') as f:
        d = sio.loadmat(f)
    data = d["X"]
    labels = d["y"]
    return data, labels


def maybe_download(filename, source_url, work_directory):
    if not os.path.exists(work_directory):
        os.mkdir(work_directory)
    filepath = os.path.join(work_directory, filename)
    if not os.path.exists(filepath):
        print("Downloading SVHN, Please wait...")
        filepath, _ = urllib.request.urlretrieve(source_url + filename,
                                                 filepath, reporthook)
        statinfo = os.stat(filepath)
        print(('Succesfully downloaded', filename, statinfo.st_size, 'bytes.'))
    return filepath

#reporthook from stackoverflow #13881092
def reporthook(blocknum, blocksize, totalsize):
    readsofar = blocknum * blocksize
    if totalsize > 0:
        percent = readsofar * 1e2 / totalsize
        s = "\r%5.1f%% %*d / %d" % (
            percent, len(str(totalsize)), readsofar, totalsize)
        sys.stderr.write(s)
        if readsofar >= totalsize: # near the end
            sys.stderr.write("\n")
    else: # total size is unknown
        sys.stderr.write("read %d\n" % (readsofar,))
