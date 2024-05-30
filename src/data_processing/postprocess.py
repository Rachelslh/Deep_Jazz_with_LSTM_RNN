from music21 import *
from random import random

from ..models.model import predict_and_sample
from .grammar import unparse_grammar


def generate_music(inference_model, input0, hidden_state0, hidden_cell0, indices_tones, chords, diversity = 0.5):
    
    # set up audio stream
    out_stream = stream.Stream()
    
    # Initialize chord variables
    curr_offset = 0.0    # variable used to write sounds to the Stream.
    num_chords = int(len(chords) / 3)  # number of different set of chords
    
    # Loop over all 18 set of chords. At each iteration generate a sequence of tones
    # and use the current chords to convert it into actual sounds 
    for i in range(1, num_chords):
        
        # Retrieve current chord from stream
        curr_chords = stream.Voice()
        
        # Loop over the chords of the current set of chords
        for j in chords[i]:
            # Add chord to the current chords with the adequate offset
            curr_chords.insert((j.offset % 4), j)
        
        # Generate a sequence of tones using the model
        _, indices = predict_and_sample(inference_model, input0, hidden_state0, hidden_cell0)
        indices = list(indices.squeeze())
        pred = [indices_tones[p] for p in indices]
        
        predicted_tones = 'C,0.25 '
        for k in range(len(pred) - 1):
            predicted_tones += pred[k] + ' ' 
        
        predicted_tones +=  pred[-1]
        
        # Post-processing phase
        # We will consider "A" and "X" as "C" tones. It is a common choice.
        predicted_tones = predicted_tones.replace(' A',' C').replace(' X',' C')

        # Pruning #1: smoothing measure
        predicted_tones = smooth_out_notes_lengths(predicted_tones)
        
        # Use predicted tones and current chords to generate sounds
        sounds = unparse_grammar(predicted_tones, curr_chords)

        # Pruning #2: removing repeated and too close together sounds
        sounds = prune_notes(sounds)

        # Quality assurance: clean up sounds
        sounds = clean_up_notes(sounds)

        # Print number of tones/notes in sounds
        print('Generated %s sounds using the predicted values for the set of chords ("%s") and after pruning' % (len([k for k in sounds if isinstance(k, note.Note)]), i))
        
        # Insert sounds into the output stream
        for m in sounds:
            out_stream.insert(curr_offset + m.offset, m)
        for mc in curr_chords:
            out_stream.insert(curr_offset + mc.offset, mc)

        curr_offset += 4.0
        
    # Initialize tempo of the output stream with 130 bit per minute
    out_stream.insert(0.0, tempo.MetronomeMark(number=130))

    # Save audio stream to fine
    mf = midi.translate.streamToMidiFile(out_stream)
    mf.open("output/my_music.midi", 'wb')
    mf.write()
    print("Your generated music is saved in output/my_music.midi")
    mf.close()
    
    # Play the final stream through output (see 'play' lambda function above)
    # play = lambda x: midi.realtime.StreamPlayer(x).play()
    # play(out_stream)
    
    return out_stream


''' Smooth the measure, ensuring that everything is in standard note lengths 
    (e.g., 0.125, 0.250, 0.333 ... ). '''
def smooth_out_notes_lengths(curr_grammar):
    pruned_grammar = curr_grammar.split(' ')

    for ix, gram in enumerate(pruned_grammar):
        terms = gram.split(',')
        terms[1] = str(__roundUpDown(float(terms[1]), 0.250, 
            random.choice([-1, 1])))
        pruned_grammar[ix] = ','.join(terms)
    pruned_grammar = ' '.join(pruned_grammar)

    return pruned_grammar


''' Helper function to down num to the nearest multiple of mult. '''
def __roundDown(num, mult):
    return (float(num) - (float(num) % mult))


''' Helper function to round up num to nearest multiple of mult. '''
def __roundUp(num, mult):
    return __roundDown(num, mult) + mult


''' Helper function that, based on if upDown < 0 or upDown >= 0, rounds number 
    down or up respectively to nearest multiple of mult. '''
def __roundUpDown(num, mult, upDown):
    if upDown < 0:
        return __roundDown(num, mult)
    else:
        return __roundUp(num, mult)
    

''' Remove repeated notes, and notes that are too close together. '''
def prune_notes(curr_notes):
    for n1, n2 in __grouper(curr_notes, n=2):
        if n2 == None: # corner case: odd-length list
            continue
        if isinstance(n1, note.Note) and isinstance(n2, note.Note):
            if n1.nameWithOctave == n2.nameWithOctave:
                curr_notes.remove(n2)

    return curr_notes

''' Perform quality assurance on notes '''
def clean_up_notes(curr_notes):
    removeIxs = []
    for ix, m in enumerate(curr_notes):
        # QA1: ensure nothing is of 0 quarter note len, if so changes its len
        if (m.quarterLength == 0.0):
            m.quarterLength = 0.250
        # QA2: ensure no two melody notes have same offset, i.e. form a chord.
        # Sorted, so same offset would be consecutive notes.
        if (ix < (len(curr_notes) - 1)):
            if (m.offset == curr_notes[ix + 1].offset and
                isinstance(curr_notes[ix + 1], note.Note)):
                removeIxs.append((ix + 1))
    curr_notes = [i for ix, i in enumerate(curr_notes) if ix not in removeIxs]

    return curr_notes