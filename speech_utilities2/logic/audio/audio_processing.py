"""Write captured microphone audio to a WAV file."""

import os

import numpy as np
import soundfile as sf

#: Sub-directory added inside the caller's path.
RECORDINGS_DIR = 'speech_recordings'


def save_recording(buffer, saving_path, file_name, sample_rate):
    """
    Write 'buffer' to 'saving_path/speech_recordings/file_name'.

    'buffer' holds signed 16-bit samples. Note the sub-directory is added by
    this function, so '/tmp/' becomes '/tmp/speech_recordings/'.
    """
    final_folder = os.path.join(saving_path, RECORDINGS_DIR)
    os.makedirs(final_folder, exist_ok=True)
    full_path = os.path.join(final_folder, file_name)

    audio_buffer = np.array(buffer)
    audio_buffer = audio_buffer.astype(float)
    audio_buffer = np.asarray(audio_buffer, dtype=np.int16)

    sf.write(full_path, audio_buffer, sample_rate, closefd=True)
