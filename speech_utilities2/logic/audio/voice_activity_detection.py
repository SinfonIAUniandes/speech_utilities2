import torch
import numpy as np
torch.set_num_threads(1)

repo_or_dir = 'snakers4/silero-vad:v4.0'
model_name = 'silero_vad'
model_dir = torch.hub.get_dir()

try:
    vad_model, utils = torch.hub.load(repo_or_dir=repo_or_dir, model=model_name, force_reload=False)
except Exception as e:
    print(f"Modelo no encontrado localmente, descargando: {e}")
    vad_model, utils = torch.hub.load(repo_or_dir=repo_or_dir, model=model_name, force_reload=True)

def int2float(sound):
    abs_max = np.abs(sound).max()
    sound = sound.astype('float32')
    if abs_max > 0:
        sound *= 1/32768
    sound = sound.squeeze()  # depends on the use case
    return sound

def detect_speech(audio_buffer, sample_rate, last_speaking_instance):
    audio_data = np.array(audio_buffer, dtype=np.int16)
    audio_int16 = np.frombuffer(audio_data.tobytes(), np.int16)
    audio_float32 = int2float(audio_int16)
    audio_rescaled = torch.from_numpy(audio_float32)
    new_confidence = vad_model(audio_rescaled, sample_rate).item()
    if new_confidence>0.57:
        # If the person just spoke, this is the last instance of speaking
        return 0
    else:
        # If the person has not spoken in a while the number grows
        return last_speaking_instance + 1