import numpy as np
import music21
import glob
from tensorflow.keras.models import Sequential #type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout #type: ignore
from tensorflow.keras.utils import to_categorical #type: ignore

# Updated get_notes function for music_ai.py
import os  # Make sure to add this import at the top!

def get_notes():
    notes = []
    # 1. Get the folder where THIS python file is located
    current_dir = os.path.dirname(__file__)
    
    # 2. Join it with the music_data folder
    # This creates a full path like: C:\Users\mann2\...\Task3\music_data\*.mid
    path_to_search = os.path.join(current_dir, "music_data", "*.mid")
    
    files = glob.glob(path_to_search)
    
    if not files:
        print(f"❌ Error: No .mid files found at: {path_to_search}")
        return []

    print(f"🎵 Finding Notes in {len(files)} song(s)...")
    for file in files:
        midi = music21.converter.parse(file)
        elements_to_parse = midi.flat.notes
        
        for element in elements_to_parse:
            if isinstance(element, music21.note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, music21.chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))
    
    print(f"✅ Success! Found {len(notes)} notes.")
    return notes

# --- PART 2: BUILDING THE BRAIN (LSTM) ---
def create_model(n_vocab, input_shape):
    model = Sequential()
    # LSTM Layer 1
    model.add(LSTM(256, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.3))
    # LSTM Layer 2
    model.add(LSTM(256))
    model.add(Dropout(0.3))
    # Output Layer
    model.add(Dense(n_vocab, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    return model

if __name__ == "__main__":
    # 1. Load Data
    notes = get_notes()
    
    if not notes:
        exit()

    # 2. Prepare Data for AI
    # Sort all unique notes (like a dictionary)
    pitchnames = sorted(set(item for item in notes))
    n_vocab = len(pitchnames)
    
    # Map notes to numbers (AI understands numbers, not "C#")
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
    
    sequence_length = 50 # Look at 50 notes to predict the next one
    network_input = []
    network_output = []

    # Create sequences
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)
    
    # Reshape for LSTM
    network_input_reshaped = np.reshape(network_input, (n_patterns, sequence_length, 1))
    network_input_reshaped = network_input_reshaped / float(n_vocab) # Normalize
    
    network_output = to_categorical(network_output)

    # 3. Train the Model
    print("\n🧠 Building AI Model...")
    model = create_model(n_vocab, (network_input_reshaped.shape[1], network_input_reshaped.shape[2]))
    
    print(f"\n🏋️ Training on {len(pitchnames)} unique notes...")
    # We use 5 epochs just to show it works (Real training takes hours)
    model.fit(network_input_reshaped, network_output, epochs=5, batch_size=64)
    
    print("\n✅ Training Complete! Model saved as 'music_brain.h5'")
    model.save('music_brain.h5')