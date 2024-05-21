from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard

from omegaconf import OmegaConf
import numpy as np
from datetime import datetime

from src.data.data import load_music_utils
from src.models.model import lstm_model


config = OmegaConf.load("src/configs/config.yaml")
batch_size = config['batch_size']
activation_units = config['architecture']['n_activation_units']

X, Y, n_values, indices_values, chords = load_music_utils('data/original_metheny.mid', batch_size, config['architecture']['n_timestep'])
a0 = np.zeros((batch_size, activation_units))
c0 = np.zeros((batch_size, activation_units))

deep_jazz_network = lstm_model(**config['architecture'])
model = deep_jazz_network.create_model()
print(model.summary())

opt = Adam(**config['optimizer'])

log_dir = config['log_dir'] + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

model.compile(optimizer=opt, loss=config['loss'], metrics=['accuracy'])

model.fit([X, a0, c0], list(Y), epochs=config['epochs'], callbacks=[tensorboard_callback], verbose = 1)
