from omegaconf import OmegaConf
import numpy as np
import pickle

from src.models.model import lstm_model, predict_and_sample
from src.data_processing.postprocess import generate_music


config = OmegaConf.load("src/configs/config.yaml")
inference_config = config['infer']

batch_size = inference_config['batch_size']
n_classes = inference_config['model']['n_classes']
activation_units = inference_config['model']['n_activation_units']

with open(config['notes_vocabulary_path'], 'rb') as fp:
    note_vocabulary = pickle.load(fp)
with open(config['chords_vocabulary_path'], 'rb') as fp:
    chords = pickle.load(fp)
    
input0 = np.zeros((batch_size, 1, n_classes))
hidden_state0 = np.zeros((batch_size, activation_units))
hidden_cell0 = np.zeros((batch_size, activation_units))

deep_jazz_network = lstm_model(**inference_config['model'])
model = deep_jazz_network.init_inference_model()
# Restore the weights
model.load_weights(inference_config["weights"])
print(model.summary())

# Testing
results, indices = predict_and_sample(model, input0, hidden_state0, hidden_cell0)

print("np.argmax(results[12]) =", np.argmax(results[12]))
print("np.argmax(results[17]) =", np.argmax(results[17]))
print("list(indices[12:18]) =", list(indices[12:18]))

out_stream = generate_music(model, input0, hidden_state0, hidden_cell0, note_vocabulary, chords)

#TODO convert to mp3 and store