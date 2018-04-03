import os 
from data_loader import NeurosynthGenerator
from time import gmtime, strftime
try:
   import cPickle as pickle
except:
   import pickle

from models import rnn_model

def getopts(argv):
    opts = {}  # Empty dictionary to store key-value pairs.
    while argv:  # While there are arguments left to parse...
        if argv[0][0] == '-':  # Found a "-name value" pair.
            opts[argv[0]] = argv[1]  # Add key and value to the dictionary.
        argv = argv[1:]  # Reduce the argument list by copying it starting from index 1.
    return opts

if __name__ == '__main__':
    from sys import argv
    myargs = getopts(argv)

    # set gpu before importing any TF stuff
    if '-gpu' in myargs:
        os.environ["CUDA_VISIBLE_DEVICES"] = myargs['-gpu']

    # belated imports
    from keras.layers import *
    from keras.models import Model, Sequential
    from keras import losses 
    from keras.callbacks import TensorBoard, ModelCheckpoint

    # Set parameters
    dropout = 0
    lr = 3.25
    bs = 32
    use_gru=False
    if '-d' in myargs: 
        dropout = float(myargs['-d'])    
    if '-l' in myargs: 
        lr = float(myargs['-l'])
    if '-bs' in myargs: 
        bs = float(myargs['-bs'])
    if '-gru' in myargs: 
        use_gru = bool(myargs['-gru'])

    model = rnn_model(dropout=dropout, use_gru=use_gru)
    model.summary()

    model.compile(loss=losses.mean_squared_error,
                  optimizer='Adam')    
    model.optimizer.lr = 10**(-lr)

    # Load in neurosynth dataset:
    data_dir='/nfs/data/kirstym'
    bs = 32
    training_generator = NeurosynthGenerator(os.path.join(data_dir,'MatrixFormatedFullVectors_kernsize_10_pubmedVectors_training.p'), batch_size=bs)
    validation_generator = NeurosynthGenerator(os.path.join(data_dir,'MatrixFormatedFullVectors_kernsize_10_pubmedVectors_testing.p'), batch_size=bs)

    # Set up callbacksdws
    now = strftime("%Y%m%d_%H%M%S", gmtime())
    tmp_name = 'rnn2_{}_bs{}_lr{}_d{}_gru{}'.format(now, bs, lr, dropout,use_gru)
    tensorboard = TensorBoard(log_dir=os.path.join('logs', tmp_name), histogram_freq=0, batch_size=bs, 
                                write_graph=True, write_grads=False, write_images=False)

    os.makedirs(os.path.join('tmp', tmp_name))
    save_name = os.path.join('tmp', tmp_name, 'checkpoint_e{epoch:02d}-val{val_loss:.2f}.hdf5')
    print('Saving checkpoints to {}'.format(save_name))
    save_checkpoint = ModelCheckpoint(save_name, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)

    # Train!
    model.fit_generator(generator=training_generator,
                        epochs=20,
                        callbacks = [tensorboard, save_checkpoint],
                        validation_data=validation_generator,
                        use_multiprocessing=False)



