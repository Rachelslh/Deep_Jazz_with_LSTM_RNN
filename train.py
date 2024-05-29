from tensorflow.keras.optimizers.legacy import Adam, SGD
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping

from omegaconf import OmegaConf
import numpy as np
from datetime import datetime
import pickle
import matplotlib.pyplot as plt

from src.data_processing.data import load_music_utils
from src.models.model import lstm_model


config = OmegaConf.load("src/configs/config.yaml")
data_config = config["data"]
training_config = config["train"]
dataset_size = sum(data_config['sequences_per_offset_interval'])

batch_size = training_config['batch_size']
activation_units = training_config['model']['n_activation_units']
num_timesteps = training_config['model']['n_timestep']

log_dir = training_config['log_dir'] + datetime.now().strftime("%Y%m%d-%H%M%S")
weights_path = training_config['weights_dir']

X, Y, tones, indices_tones, chords = load_music_utils(**data_config, n_timestep=num_timesteps)

a0 = np.zeros((dataset_size, activation_units))
c0 = np.zeros((dataset_size, activation_units))

deep_jazz_network = lstm_model(**training_config['model'])
model = deep_jazz_network.init_training_model()
print(model.summary())

opt = Adam(**training_config['optimizer'])

tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
earlystopping_cp = EarlyStopping(monitor='loss', patience=100, restore_best_weights=True)

model.compile(optimizer=opt, loss=training_config['loss'], metrics=['accuracy'])

history = model.fit([X, a0, c0], list(Y), epochs=training_config['epochs'], batch_size=batch_size, callbacks=[tensorboard_callback])

print(f"loss at epoch 1: {history.history['loss'][0]}")
print(f"loss at last epoch: {history.history['loss'][training_config['epochs'] - 1]}")

# Save model vocabulary and weights
with open(config['notes_vocabulary_path'], 'wb') as fp:
    pickle.dump(indices_tones, fp, protocol=pickle.HIGHEST_PROTOCOL)
with open(config['chords_vocabulary_path'], 'wb') as fp:
    pickle.dump(chords, fp, protocol=pickle.HIGHEST_PROTOCOL)
model.save_weights(weights_path)

plt.plot(history.history['loss'])
plt.show()