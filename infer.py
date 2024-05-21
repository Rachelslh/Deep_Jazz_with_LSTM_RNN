from omegaconf import OmegaConf
import numpy as np

from src.models.model import lstm_model, predict_and_sample


config = OmegaConf.load("src/configs/config.yaml")
batch_size = config['batch_size']
n_classes = config['architecture']['n_classes']
activation_units = config['architecture']['n_activation_units']

checkpoint_path = "checkpoint.ckpt"

input0 = np.zeros((1, 1, n_classes))
hidden_state0 = np.zeros((1, activation_units))
hidden_cell0 = np.zeros((1, activation_units))

deep_jazz_network = lstm_model(**config['architecture'])
model = deep_jazz_network.init_inference_model()
print(model.summary())

results, indices = predict_and_sample(model, input0, hidden_state0, hidden_cell0)

print("np.argmax(results[12]) =", np.argmax(results[12]))
print("np.argmax(results[17]) =", np.argmax(results[17]))
print("list(indices[12:18]) =", list(indices[12:18]))

#TODO Post-process results 