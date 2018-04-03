from keras.layers import *
from keras.models import Model

def rnn_model(num_conv_feats=8, lstm_output_dim = 200, num_conv_layers = 3, dropout=0, use_gru=False):

    input_dim = 200
    output_dim = 400    
    num_conv_layers = 3    

    model_input = Input(
                  shape=(None, input_dim),
                  name='Input')
    if use_gru:
        rnn_output = GRU(lstm_output_dim, dropout=dropout)(model_input)        
    else:
        rnn_output = LSTM(lstm_output_dim, dropout=dropout)(model_input)

    hidden = Dense(units=lstm_output_dim, activation=K.relu)(rnn_output)
    conv_input = Dense(units=output_dim, activation=K.relu)(hidden)
    conv_input = Reshape((20, 20, 1))(conv_input)

    layer = conv_input
    for i in range(num_conv_layers):          
        layer = Conv2D(num_conv_feats, kernel_size=(3, 3),
                       activation='elu',
                       padding='same')(layer)    
        if dropout > 0.0:            
          layer = Dropout(dropout)(layer)

    image = Conv2D(1, kernel_size=(3, 3),
                   activation='elu',
                   padding='same')(layer)    

    image = Flatten()(image)

    model = Model(
       inputs=model_input,
       outputs=image)

    return model