import tensorflow as tf
import numpy as np
import random

from tensorflow.keras.layers import Reshape, LSTM, Dense, Input, RepeatVector
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical


class lstm_model:
    def __init__(self, n_classes, n_timestep, n_activation_units) -> Model:
        self.n_classes = n_classes
        self.n_timestep = n_timestep
        self.n_activation_units = n_activation_units
        
        self.init_layers()
        
        random.seed(0)

    def init_layers(self, ):
        self.reshape_layer = Reshape((1, self.n_classes))
        self.lstm_cell = LSTM(self.n_activation_units, return_state = True)
        self.dense_layer = Dense(self.n_classes, activation='softmax')
        
        
    def init_training_model(self, ):
        
        # Define the input layer and specify the shape
        X = Input(shape=(self.n_timestep, self.n_classes)) 
        # Define the initial hidden state a0 and initial cell state c0
        a0 = Input(shape=(self.n_activation_units,), name='a0')
        c0 = Input(shape=(self.n_activation_units,), name='c0')
        
        a = a0
        c = c0
        # Create empty list to append the outputs while you iterate
        outputs = []
        
        for t in range(self.n_timestep):
            # Select the "t"th time step vector from X. 
            x = X[:,t,:]
            # Use reshaper to reshape x to be (1, n_values) (≈1 line)
            x = self.reshape_layer(x)
            # Perform one step of the LSTM_cell
            a, _, c = self.lstm_cell(x, initial_state=[a, c])
            # Apply densor to the hidden state output of LSTM_Cell
            out = self.dense_layer(a)
            # Append the output
            outputs.append(out)
            
        # Create model instance
        model = Model(inputs=[X, a0, c0], outputs=outputs)
        
        return model
    
    def init_inference_model(self, ):
        
        # Define the input of your model with a shape 
        x = Input(shape=(1, self.n_classes))
        
        # Define s0, initial hidden state for the decoder LSTM
        a0 = Input(shape=(self.n_activation_units,), name='a0')
        c0 = Input(shape=(self.n_activation_units,), name='c0')

        a = a0
        c = c0
        # Create empty list to append the outputs while you iterate
        outputs = []
        
        for t in range(self.n_timestep):
            # Perform one step of the LSTM_cell
            _, a, c = self.lstm_cell(x, initial_state=[a, c])
            # Apply densor to the hidden state output of LSTM_Cell
            out = self.dense_layer(a)
            # Append the output
            outputs.append(out)
            
            # Select the next input
            idx = tf.math.argmax(out, axis=-1)
            # Set "x" to be the one-hot representation of the selected value
            x = tf.one_hot(idx, self.n_classes)
            # Step 2.E: 
            # Convert x into a tensor with shape=(None, 1, n_classes)
            x = RepeatVector(1)(x)
        
        # Create model instance
        model = Model(inputs=[x, a0, c0], outputs=outputs)
        
        return model
    
    
def predict_and_sample(inference_model, input, hidden_state0, hidden_cell0):
    
    n_classes = input.shape[2]
    
    pred = inference_model.predict([input, hidden_state0, hidden_cell0])
    # Convert "pred" into an np.array() of indices with the maximum probabilities
    indices = np.argmax(pred, axis=-1)
    # Convert indices to one-hot vectors, the shape of the results should be (timesteps, n_classes)
    results = to_categorical(indices, num_classes=n_classes)
    
    return results, indices
    