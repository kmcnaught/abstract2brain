from keras.layers import *
import os 
from data_loader import NeurosynthGenerator
from keras.models import Model, Sequential
from keras import losses 
from keras.callbacks import TensorBoard, ModelCheckpoint
from time import gmtime, strftime
from models import rnn_model

chk_path = 'rnn_20180403_190017_bs32_lr3.25_d0.5_gruTrue'
chk_name = 'checkpoint_e15-val0.03.hdf5'

weights_file = os.path.join('tmp', chk_path, chk_name)

dropout = 0.5
use_gru = True
model = rnn_model(dropout=dropout, use_gru=use_gru)

model.load_weights(weights_file)

data_dir='/nfs/data/kirstym'
bs = 32
training_generator = NeurosynthGenerator(os.path.join(data_dir,'MatrixFormatedFullVectors_kernsize_10_pubmedVectors_training.p'), batch_size=bs)
validation_generator = NeurosynthGenerator(os.path.join(data_dir,'MatrixFormatedFullVectors_kernsize_10_pubmedVectors_testing.p'), batch_size=bs)

# score_train = self.model.evaluate_generator(generator=training_generator)
# pred_train  = self.model.predict_generator(generator=training_generator, verbose=1)

score_test = self.model.evaluate_generator(generator=validation_generator)
pred_test  = self.model.predict_generator(generator=validation_generator, verbose=1)




