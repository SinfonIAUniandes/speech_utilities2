from functools import lru_cache
import time
import torch
import whisper
import os
from ament_index_python.packages import get_package_share_directory

@lru_cache
def load_model(model_size = "small"):
    package_name = 'speech_utilities2'
    package_path = get_package_share_directory(package_name)
    PATH_DATA = package_path+'/data'
    in_memory = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Transcribing audio in {device.upper()}")
    if not os.path.exists(PATH_DATA+f"/{model_size}.pt"): 
        print("Model not found, downloading it")
        in_memory = False
    model_whisp = whisper.load_model(model_size, in_memory= in_memory, download_root = PATH_DATA)
    return model_whisp, device

speech_2_text_model,device = load_model("base.en")

def transcribe(file_path,language="en"):
    """
    Input:
    file_path: path of the .wav file to transcribe
    model: Whisper model instance to transcribe audio
    ---
    Output:
    response of the local model with the transcription of the audio
    ---
    Use the local version of whisper for transcribing short audios
    """
    t1 = float(time.perf_counter() * 1000)
    if language=="en":
        speech_2_text_model,device = load_model("base.en")
    else:
        speech_2_text_model,device = load_model("base")
    torch.cuda.empty_cache()
    result = speech_2_text_model.transcribe(file_path,language=language)
    torch.cuda.empty_cache()
    t2 = float(time.perf_counter() * 1000)
    print("Local [ms]: ", float(t2-t1))
    return result["text"]