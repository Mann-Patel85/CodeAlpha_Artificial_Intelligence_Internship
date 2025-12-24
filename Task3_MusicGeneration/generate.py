import numpy as np
import music21
import glob
from tensorflow.keras.models import load_model #type: ignore

# --- 1. Load the Data Again (To get the "Style") ---
# Updated get_notes for generate.py
import os # Don't forget this at the top!

def get_notes():
    notes = []
    # Smart Path Fix
    current_dir = os.path.dirname(__file__)
    path_to_search = os.path.join(current_dir, "music_data", "*.mid")
    
    files = glob.glob(path_to_search)
    
    for file in files:
        midi = music21.converter.parse(file)
        elements_to_parse = midi.flat.notes
        for element in elements_to_parse:
            if isinstance(element, music21.note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, music21.chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))
    return notes

# --- 2. Generate the Music ---
def generate():
    # Load the notes to understand the "Language"
    notes = get_notes()
    pitchnames = sorted(set(item for item in notes))
    n_vocab = len(pitchnames)
    
    # Map back from numbers to notes
    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
    
    # Prepare input sequence
    sequence_length = 50
    network_input = []
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
    
    # Load the Trained Brain
    print("🧠 Loading the AI Brain...")
    model = load_model('music_brain.h5')
    
    # Pick a random starting point
    start = np.random.randint(0, len(network_input)-1)
    pattern = network_input[start]
    
    prediction_output = []
    
    print("🎹 AI is composing new music...")
    # Generate 100 notes
    for note_index in range(100):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab)
        
        prediction = model.predict(prediction_input, verbose=0)
        index = np.argmax(prediction)
        result = int_to_note[index]
        prediction_output.append(result)
        
        pattern.append(index)
        pattern = pattern[1:len(pattern)]

    # --- 3. Save as MIDI File ---
    offset = 0
    output_notes = []
    
    for pattern in prediction_output:
        # If it's a chord (multiple notes)
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = music21.note.Note(int(current_note))
                new_note.storedInstrument = music21.instrument.Piano()
                notes.append(new_note)
            new_chord = music21.chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        # If it's a single note
        else:
            new_note = music21.note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = music21.instrument.Piano()
            output_notes.append(new_note)
        
        offset += 0.5 # Speed of the song

    midi_stream = music21.stream.Stream(output_notes)
    midi_stream.write('midi', fp='ai_generated_song.mid')
    print("✅ Success! Music saved as 'ai_generated_song.mid'")

if __name__ == "__main__":
    generate()