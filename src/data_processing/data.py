from functools import reduce

import numpy as np
from .preprocess import get_musical_data, get_corpus_data


def data_processing(data, num_tones, values_indices, m, Tx):
    # cut the corpus into semi-redundant sequences of Tx values
    np.random.seed(0)
    X = np.zeros((m, Tx, num_tones), dtype=np.bool)
    Y = np.zeros((m, Tx, num_tones), dtype=np.bool)
    for i in range(m):
#         for t in range(1, Tx):
        random_idx = np.random.choice(len(data) - Tx)
        sequence = data[random_idx:(random_idx + Tx)]
        for j in range(Tx):
            idx = values_indices[sequence[j]]
            if j != 0:
                X[i, j, idx] = 1
                Y[i, j-1, idx] = 1
            if j == Tx - 1:
                Y[i, j, idx] = 1
                
    Y = np.swapaxes(Y, 0, 1)
    Y = Y.tolist()
    return np.asarray(X), np.asarray(Y)


def load_music_utils(midi_path, offsets, sequences_per_offset_interval, n_timestep):
    chords_unified, abstract_grammars_unified = [], []
    for offset in offsets:
        chords, abstract_grammars = get_musical_data(midi_path, offset)
        chords_unified.extend(chords)
        abstract_grammars_unified.append(abstract_grammars)
        
    X, Y = [], []
    tones, tones_indices, indices_tones = get_corpus_data(list(reduce(lambda x, y: x + y, abstract_grammars_unified, [])))
    for i, offset in enumerate(offsets):
        data = [x for sublist in abstract_grammars_unified[i] for x in sublist.split(' ')]
        x, y = data_processing(data, len(tones), tones_indices, sequences_per_offset_interval[i], n_timestep) 
        # Append to X and Y lists
        X.append(x)
        Y.append(y)
        
    X = np.vstack(X)
    Y = np.hstack(Y)
        
    return (X, Y, tones, indices_tones, chords_unified)


if __name__ == '__main__':
    load_music_utils('data/original_metheny.mid')