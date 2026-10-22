"""
Local speech-to-text with Whisper.

Weights are cached outside the package, under the user's cache directory, so
they survive a clean rebuild and never land in the source or install tree.
Override with 'SPEECH_UTILITIES_MODEL_CACHE' or the 'cache_dir' argument.

A model still loads when this module is imported. Making that lazy is tracked
separately.
"""

from functools import lru_cache
import os
import time

import torch
import whisper

#: Environment variable overriding the cache location.
CACHE_DIR_ENV = 'SPEECH_UTILITIES_MODEL_CACHE'

#: Folder created under the user's cache directory.
CACHE_SUBDIR = os.path.join('speech_utilities2', 'whisper')

#: Whisper checkpoints: English-only, and everything else.
ENGLISH_MODEL = 'base.en'
MULTILINGUAL_MODEL = 'base'


def default_cache_dir():
    """
    Return the directory Whisper weights are cached in.

    Honours 'SPEECH_UTILITIES_MODEL_CACHE', then 'XDG_CACHE_HOME', then
    '~/.cache'. The directory is not created here.
    """
    override = os.environ.get(CACHE_DIR_ENV)
    if override:
        return os.path.expanduser(override)
    base = os.environ.get('XDG_CACHE_HOME') or os.path.join(os.path.expanduser('~'), '.cache')
    return os.path.join(base, CACHE_SUBDIR)


@lru_cache
def load_model(model_size='base.en', cache_dir=None):
    """Load a Whisper checkpoint, caching it for the life of the process."""
    cache_dir = cache_dir or default_cache_dir()
    os.makedirs(cache_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Transcribing audio in {device.upper()}')
    in_memory = os.path.exists(os.path.join(cache_dir, f'{model_size}.pt'))
    if not in_memory:
        print(f'Model {model_size} not cached in {cache_dir}, downloading it')
    model_whisp = whisper.load_model(model_size, in_memory=in_memory, download_root=cache_dir)
    return model_whisp, device


speech_2_text_model, device = load_model(ENGLISH_MODEL)


def transcribe(file_path, language='en', cache_dir=None):
    """
    Transcribe a short WAV file.

    'language' is passed to Whisper and also picks the checkpoint: English uses
    the English-only model, anything else uses the multilingual one.
    """
    started_ms = float(time.perf_counter() * 1000)
    model_size = ENGLISH_MODEL if language == 'en' else MULTILINGUAL_MODEL
    model, _device = load_model(model_size, cache_dir)
    torch.cuda.empty_cache()
    result = model.transcribe(file_path, language=language)
    torch.cuda.empty_cache()
    finished_ms = float(time.perf_counter() * 1000)
    print('Local [ms]: ', float(finished_ms - started_ms))
    return result['text']
