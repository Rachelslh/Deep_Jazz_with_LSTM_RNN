# Generate Jazz Solo with LSTM RNN

This repository contains code for generating jazz solos using Long Short-Term Memory (LSTM) Recurrent Neural Networks (RNN).

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Model](#model)
- [Results](#results)
- [Contributing](#contributing)

## Introduction
This project aims to generate jazz solos using a neural network model based on LSTM RNN. The model is trained on MIDI files of jazz music and generates new sequences that mimic the style of the training data.

## Installation
To get started with the project, follow these steps:

1. Clone the repository:
    ```sh
    git clone https://github.com/Rachelslh/Jazz-Solo-with-LSTM-RNN.git
    cd Jazz-Solo-with-LSTM-RNN
    ```

2. Create and activate a virtual environment (optional but recommended):
    ```sh
    python -m venv venv
    source venv/bin/activate   # On Windows use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage
To train the model and generate jazz solos, follow these steps:

1. Preprocess the data and train the model:
    ```sh
    python train.py
    ```

1. Generate music:
    ```sh
    python infer.py
    ```

## Data
The dataset consists of MIDI files containing jazz solos. The preprocessing script converts these files into a format suitable for training the LSTM model following these steps:
- Restore zero lengths to at least 0.25 beats (1/4 beat): Due to quantization applied during the conversion process to MIDI files, some tone lengths may be zeroed-out.
- Transposition: Change the key signature to adhere to major keys.
- For each offset interval specified in the configuration:
- Extract tones from the acoustic guitar part.
- Extract chords from the piano part.
- Group tones and chords into measures of 4 beats, matching the time signature of the music used here.
- Parse the tones into a grammar consisting of: rests, chord tones, scale tones, approach tones, and arbitrary tones, and retrieve tone lengths along with the maximum upward and downward intervals.
- Define the musical tone corpus based on the occurring tones and intervals.
- Extract semi-redundant sequences from the grammar, with each sequence being of length Tx (number of timesteps).

## Model
The model is based on an LSTM RNN architecture, which is well-suited for sequence generation tasks. The key components of the model are:

- **Input Layer**: Takes the processed MIDI data i.e. music grammar.
- **LSTM Layers**: Single-layer, captures temporal dependencies in the music.
- **Dense Layer**: Outputs the probabilities for the next note in the sequence using a Softmax activation function.

## Results
After training, the model generates jazz solos that are stylistically similar to the training data. Here are some sample results:

- Sample (to be generated)

## Contributing
Contributions are welcome! Please open an issue or submit a pull request if you have any improvements or bug fixes.
