### class to load neurosynth data into pytorch
#
#
try:
   import cPickle as pickle
except:
   import pickle

import numpy as np
import keras

class NeurosynthGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, pickle_file, batch_size=32, n_channels=1,
                 shuffle=True):
        'Initialization'
        # self.dim = dim
        self.batch_size = batch_size

        self.n_channels = n_channels
        self.shuffle = shuffle

        # Load data from file
        self.dat = pickle.load(open(pickle_file, 'rb'))
        self.pids = self.dat.keys()
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.pids) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        pids_temp = [self.pids[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(pids_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.pids))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, pids):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)      

        # Get longest sequence in batch
        l_max = 0
        for pid in pids:
            l = self.dat[pid]['wordvec'].shape[0]
            l_max = max(l, l_max)      

        # Initialise arrays        
        wordMat = np.zeros((self.batch_size, l_max, 200))
        imageMat = np.zeros((self.batch_size, 400))

        # Populate
        for i,pid in enumerate(pids):
            wordvec = self.dat[pid]['wordvec']
            image = self.dat[pid]['image']
            l = wordvec.shape[0]
            wordMat[i][-l:,:] = wordvec
            imageMat[i] = image.flatten()

        return wordMat, imageMat
