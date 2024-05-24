from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping

from omegaconf import OmegaConf
import numpy as np
from datetime import datetime
import pickle

from src.data.data import load_music_utils
from src.models.model import lstm_model


config = OmegaConf.load("src/configs/config.yaml")
training_config = config["train"]

batch_size = training_config['batch_size']
activation_units = training_config['n_activation_units']

log_dir = training_config['log_dir'] + datetime.now().strftime("%Y%m%d-%H%M%S")
weights_path = training_config['weights_dir']

X, Y, n_values, indices_values, chords = load_music_utils(training_config['data'], batch_size, training_config['n_timestep'])

a0 = np.zeros((batch_size, activation_units))
c0 = np.zeros((batch_size, activation_units))

deep_jazz_network = lstm_model(**training_config)
model = deep_jazz_network.init_training_model()
print(model.summary())

opt = Adam(**training_config['optimizer'])

tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
earlystopping_cp = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)

model.compile(optimizer=opt, loss=training_config['loss'], metrics=['accuracy'])

model.fit([X, a0, c0], list(Y), epochs=training_config['epochs'], callbacks=[earlystopping_cp, tensorboard_callback])

# Save model vocabulary and weights
with open(config['notes_vocabulary_path'], 'wb') as fp:
    pickle.dump(indices_values, fp, protocol=pickle.HIGHEST_PROTOCOL)
with open(config['chords_vocabulary_path'], 'wb') as fp:
    pickle.dump(chords, fp, protocol=pickle.HIGHEST_PROTOCOL)   
model.save_weights(weights_path)
