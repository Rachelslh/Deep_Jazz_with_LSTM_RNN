
from tensorflow.keras.layers import Reshape, LSTM, Dense, Input, Lambda
from tensorflow.keras.models import Model


class lstm_model:
    def __init__(self, n_classes, n_timestep, n_activation_units) -> Model:
        self.n_classes = n_classes
        self.n_timestep = n_timestep
        self.n_activation_units = n_activation_units
        
        self.init_layers()

    def init_layers(self, ):
        self.reshape_layer = Reshape((1, self.n_classes))
        self.lstm_cell = LSTM(self.n_activation_units, return_state = True)
        self.dense_layer = Dense(self.n_classes, activation='softmax')
        
        
    def create_model(self, ):
        
        # Define the input layer and specify the shape
        X = Input(shape=(self.n_timestep, self.n_classes)) 
        # Define the initial hidden state a0 and initial cell state c0
        a0 = Input(shape=(self.n_activation_units,), name='a0')
        c0 = Input(shape=(self.n_activation_units,), name='c0')
        
        # Create empty list to append the outputs while you iterate
        outputs = []
        
        for t in range(self.n_timestep):
            # Select the "t"th time step vector from X. 
            x = X[:,t,:]
            # Use reshaper to reshape x to be (1, n_values) (≈1 line)
            x = self.reshape_layer(x)
            # Perform one step of the LSTM_cell
            _, a, c = self.lstm_cell(x)
            # Apply densor to the hidden state output of LSTM_Cell
            out = self.dense_layer(a)
            # Append the output
            outputs.append(out)
            
        # Create model instance
        model = Model(inputs=[X, a0, c0], outputs=outputs)
        
        return model
    