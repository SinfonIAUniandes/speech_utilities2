import os
import numpy as np
import soundfile as sf

def save_recording(buffer, saving_path, file_name, sample_rate):
    """
    Input:
    buffer: audio buffer
    saving_path: path 
    file_name: name of the audio file
    sample_rate: sample rate of the audio
    ---
    Saves the audio buffer in saving_path/speech_recordings/file_name
    """
    # Ensure the 'speech_recordings' folder exists
    final_folder = os.path.join(saving_path, "speech_recordings")
    os.makedirs(final_folder, exist_ok=True)

    # Full file path
    full_path = os.path.join(final_folder, file_name)

    # Convert the audioBuffer into a proper array
    audio_buffer = np.array(buffer)
    audio_buffer = audio_buffer.astype(float)
    audio_buffer = np.asarray(audio_buffer, dtype=np.int16)

    # Save the audio
    sf.write(full_path, audio_buffer, sample_rate, closefd=True)
