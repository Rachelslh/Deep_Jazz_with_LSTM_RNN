from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping

from omegaconf import OmegaConf
import numpy as np
from datetime import datetime
import pickle

from src.data.data import load_music_utils
from src.models.model import lstm_model


config = OmegaConf.load("src/configs/config.yaml")
batch_size = config['batch_size']
activation_units = config['architecture']['n_activation_units']

log_dir = config['log_dir'] + datetime.now().strftime("%Y%m%d-%H%M%S")
weights_path = "src/models/weights/model_weights"

X, Y, n_values, indices_values, chords = load_music_utils('data/original_metheny.mid', batch_size, config['architecture']['n_training_timestep'])

a0 = np.zeros((batch_size, activation_units))
c0 = np.zeros((batch_size, activation_units))

deep_jazz_network = lstm_model(**config['architecture'])
model = deep_jazz_network.init_training_model()
print(model.summary())

opt = Adam(**config['optimizer'])

tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
earlystopping_cp = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)

model.compile(optimizer=opt, loss=config['loss'], metrics=['accuracy'])

model.fit([X, a0, c0], list(Y), epochs=config['epochs'], callbacks=[earlystopping_cp, tensorboard_callback], verbose = 1)

# Save model vocabulary and weights
with open('data/vocabulary.p', 'wb') as fp:
    pickle.dump(indices_values, fp, protocol=pickle.HIGHEST_PROTOCOL)
with open('data/chords.p', 'wb') as fp:
    pickle.dump(chords, fp, protocol=pickle.HIGHEST_PROTOCOL)   
model.save_weights(weights_path)
