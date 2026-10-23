import os
from functools import cache

from langchain_openai import ChatOpenAI

# acepta la URL con o sin /v1: conversation_node (ChatOllama) usa la misma variable sin /v1
BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/").removesuffix("/v1") + "/v1"
MODEL = os.environ.get("SPEECH_MODEL", "llama3.2:3b")


@cache  # un cliente (y su pool HTTP) por alias+kwargs; tras cambiar MODEL llama chat.cache_clear()
def chat(alias: str, **kwargs) -> ChatOpenAI:
    """Cliente de chat contra Ollama. alias (sinfonia-reasoning | sinfonia-fast | sinfonia-local) indica el rol;
    hoy todos usan SPEECH_MODEL."""
    # ponytail: un solo modelo para todos los roles; mapea alias -> modelo cuando haya más de uno.
    return ChatOpenAI(model=MODEL, base_url=BASE_URL, api_key="ollama", **kwargs)  # Ollama ignora la key
