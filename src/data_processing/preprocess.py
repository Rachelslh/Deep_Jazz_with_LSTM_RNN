
from collections import OrderedDict
from itertools import groupby

from music21 import *

from .grammar import parse_melody

''' Helper function to parse a MIDI file into its measures and chords '''
def __parse_midi(data_fn, offset):
    # Parse the MIDI data for separate melody and accompaniment parts.
    midi_data = converter.parse(data_fn)
    # A part respresents either a single instrument or a vocal part
    # The melody part is chosen to be the Electric Guitar part here for this specific midi file
    # Get melody part, compress into single voice.
    melody_voice = midi_data[6]
    
    '''
    Outdated music21 version
    # We have 2 voices in this stream, melody and harmony
    melody1, melody2 = melody_stream.getElementsByClass(stream.Voice)
    # Compressing both melody and harmpony voices together, respecting the harmony notes' offsets
    for j in melody2:
        melody1.insert(j.offset, j)
    melody_voice = melody1
    '''
    
    # When Midi files are compressed, durations get quantized and thus smaller durations are not preserved, 
    # my guess here is that we're restoring the smaller durations that got surpressed
    for i in melody_voice:
        if i.quarterLength == 0.0:
            i.quarterLength = 0.25 # 1/16 th note
            
    # Transposition
    # Change key signature to adhere to comp_stream (1 sharp, mode = major).
    melody_voice.insert(0, instrument.ElectricGuitar())
    melody_voice.insert(0, key.KeySignature(sharps=1))
    
    # The accompaniment parts. Take only the best subset of parts from
    # the original data. Maybe add more parts, hand-add valid instruments.
    # Should add least add a string part (for sparse solos).
    # Verified are good parts: 0, 1, 6, 7 '''
    partIndices = [1]
    comp_stream = stream.Voice()
    comp_stream.append([j.flatten() for i, j in enumerate(midi_data) 
        if i in partIndices])
    
    # Full stream containing both the melody and the accompaniment.
    full_stream = stream.Voice()
    for i in range(len(comp_stream)):
        full_stream.append(comp_stream[i])
    full_stream.append(melody_voice.flatten())
    
    # Extract solo stream, assuming you know the positions ..ByOffset(i, j).
    # Note that for different instruments (with stream.flat), you NEED to use
    # stream.Part(), not stream.Voice().
    # Accompanied solo is in range [478, 548)
    solo_stream = stream.Voice()
    for part in full_stream:
        curr_part = stream.Part()
        for instr in part.getElementsByClass(instrument.Instrument):
            if instr.instrumentName not in [el.instrumentName for el in curr_part.getElementsByClass(instrument.Instrument)]:
                curr_part.insert(0, instr)
        for k in part.getElementsByClass(key.KeySignature):
            curr_part.insert(0, k)
        curr_part.insert(0, *part.getElementsByClass(tempo.MetronomeMark))
        curr_part.insert(0, *part.getElementsByClass(meter.TimeSignature))
        
        for note in part.getElementsByOffset(offset[0], offset[1], includeEndBoundary=True):
            curr_part.insert(note.offset, note)
        cp = curr_part.flatten()
        solo_stream.insert(cp)
        
    # Group by measure so you can classify. 
    # Note that measure 0 is for the time signature, metronome, etc. which have
    # an offset of 0.0.
    # Measure is a Bar, a unit which the length is = time signature, here it's supposed to be 4 beats
    melody_stream = solo_stream[-1]
    notes = OrderedDict()
    offsetTuples = [(int(n.offset / 4), n) for n in melody_stream]
    for m, group in groupby(offsetTuples, lambda x: x[0]):
        notes[m] = [n[1] for n in group]
        
    # Piano stream has chords
    # Get the stream of chords.
    # offsetTuples_chords: group chords by measure number.
    chordStream = solo_stream[0]
    chordStream.removeByClass('Rest')
    chordStream.removeByClass('Note')
    chordStream.removeByClass('Key')
    chordStream.removeByClass('Instrument')
    chordStream.removeByClass('MetronomeMark')
    chordStream.removeByClass('TimeSignature')
    offsetTuples_chords = [(int(n.offset / 4), n) for n in chordStream]

    # Generate the chord structure. Use just track 1 (piano) since it is
    # the only instrument that has chords. 
    # Group into 4s, just like before. 
    chords = OrderedDict()
    for m, group in groupby(offsetTuples_chords, lambda x: x[0]):
        chords[m] = [n[1] for n in group]
        
    return notes, chords
    
    
''' Helper function to get the grammatical data from given musical data. '''
def __get_abstract_grammars(measures, chords):
    # extract grammars
    abstract_grammars = []
    for ix in measures.keys():
        if ix > 0 and ix in chords.keys(): 
            m = stream.Voice()
            for i in measures[ix]:
                m.insert(i.offset, i)
            c = stream.Voice()
            for j in chords[ix]:
                c.insert(j.offset, j)
            parsed = parse_melody(m, c)
            abstract_grammars.append(parsed)

    return abstract_grammars


#----------------------------PUBLIC FUNCTIONS----------------------------------#

''' Get musical data from a MIDI file '''
def get_musical_data(data_fn, offset):
    
    measures, chords = __parse_midi(data_fn, offset)
    abstract_grammars = __get_abstract_grammars(measures, chords)

    return chords, abstract_grammars


''' Get corpus data from grammatical data '''
def get_corpus_data(abstract_grammars):
    corpus = [x for sublist in abstract_grammars for x in sublist.split(' ')]
    values = set(corpus)
    val_indices = dict((v, i) for i, v in enumerate(values))
    indices_val = dict((i, v) for i, v in enumerate(values))

    return values, val_indices, indices_val


if __name__ == '__main__':
    get_musical_data('data/original_metheny.mid')