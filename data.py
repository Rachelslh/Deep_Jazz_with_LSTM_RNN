
import numpy as np
from preprocess import get_musical_data, get_corpus_data


def data_processing(corpus, values_indices, m, Tx):
    # cut the corpus into semi-redundant sequences of Tx values
    n_values = len(set(corpus))
    np.random.seed(0)
    X = np.zeros((m, Tx, n_values), dtype=np.bool)
    Y = np.zeros((m, Tx, n_values), dtype=np.bool)
    for i in range(m):
#         for t in range(1, Tx):
        random_idx = np.random.choice(len(corpus) - Tx)
        corp_data = corpus[random_idx:(random_idx + Tx)]
        for j in range(Tx):
            idx = values_indices[corp_data[j]]
            if j != 0:
                X[i, j, idx] = 1
                Y[i, j-1, idx] = 1
    
    Y = np.swapaxes(Y, 0, 1)
    Y = Y.tolist()
    return np.asarray(X), np.asarray(Y), n_values 


def load_music_utils(file, batch_size, n_timestep):
    chords, abstract_grammars = get_musical_data(file)
    corpus, tones, tones_indices, indices_tones = get_corpus_data(abstract_grammars)
    X, Y, N_tones = data_processing(corpus, tones_indices, batch_size, n_timestep)   
    return (X, Y, N_tones, indices_tones, chords)


if __name__ == '__main__':
    load_music_utils('data/original_metheny.mid')