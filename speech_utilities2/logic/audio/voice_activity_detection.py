"""
Voice activity detection with Silero VAD.

The model loads when this module is imported, so importing it downloads from
the Torch hub on a cold cache and fails offline. Making that lazy is tracked
separately.
"""

import numpy as np
import torch

torch.set_num_threads(1)

REPO_OR_DIR = 'snakers4/silero-vad:v4.0'

MODEL_NAME = 'silero_vad'

#: Confidence above which a chunk counts as speech.
SPEECH_CONFIDENCE_THRESHOLD = 0.57

try:
    vad_model, utils = torch.hub.load(
        repo_or_dir=REPO_OR_DIR, model=MODEL_NAME, force_reload=False)
except Exception as error:
    print(f'Modelo no encontrado localmente, descargando: {error}')
    vad_model, utils = torch.hub.load(
        repo_or_dir=REPO_OR_DIR, model=MODEL_NAME, force_reload=True)


def int2float(sound):
    """Convert signed 16-bit samples to floats between -1 and 1."""
    abs_max = np.abs(sound).max()
    sound = sound.astype('float32')
    if abs_max > 0:
        sound *= 1 / 32768
    sound = sound.squeeze()
    return sound


def detect_speech(audio_buffer, sample_rate, last_speaking_instance):
    """
    Count how many consecutive chunks have passed without speech.

    Returns zero when this chunk contains speech, otherwise the previous count
    plus one. Callers read zero as "someone is speaking right now".
    """
    audio_data = np.array(audio_buffer, dtype=np.int16)
    audio_int16 = np.frombuffer(audio_data.tobytes(), np.int16)
    audio_float32 = int2float(audio_int16)
    audio_rescaled = torch.from_numpy(audio_float32)
    new_confidence = vad_model(audio_rescaled, sample_rate).item()
    if new_confidence > SPEECH_CONFIDENCE_THRESHOLD:
        return 0
    return last_speaking_instance + 1
