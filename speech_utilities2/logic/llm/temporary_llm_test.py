"""
create_llm(model_name, temperature, max_tokens)

- NO usa cache: cada llamada crea una nueva instancia del LLM.
- MODEL_REGISTRY contiene todas las credenciales/configs por modelo.
- Único parámetro para la función: model_name, temperature, max_tokens.
"""

from typing import Optional, Dict, Any
import os

# Intentar importar adaptadores (si no están instalados, se lanza error en tiempo de ejecución)
try:
    from langchain.chat_models import AzureChatOpenAI, ChatOpenAI
except Exception:
    AzureChatOpenAI = None
    ChatOpenAI = None

try:
    from langchain_ollama import ChatOllama
except Exception:
    ChatOllama = None

# -------------------------
# REGISTRO DE MODELOS (configura esto según tu infra)
# -------------------------
# Cada entrada puede incluir:
#   provider: "azure" | "ollama" | "openai"
#   api_key, endpoint, api_version, deployment_name, model, base_url, extra, ...
# RECUERDA: no subir claves a repos públicos.
MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "gpt-4o-azure": {
        "provider": "azure",
        "api_key": "<AZURE_OPENAI_API_KEY_AQUI>",          # opcional, mejor usar vault/vars de entorno
        "endpoint": "https://mi-endpoint.openai.azure.com",
        "api_version": "2024-07-31-preview",
        "deployment_name": "gpt-4o-deployment",
        "extra": {},
    },
    "llama3.1-local": {
        "provider": "ollama",
        "model": "llama3.1",
        "base_url": "http://localhost:11434",
        "extra": {"validate_model_on_init": True},
    },
    "gpt-4o-openai": {
        "provider": "openai",
        "api_key": "<OPENAI_API_KEY_AQUI>",
        "model": "gpt-4o-mini",
        "extra": {},
    },
}

# -------------------------
# Función principal (sin cache)
# -------------------------
def create_llm(model_name: str, temperature: Optional[float], max_tokens: Optional[int]):
    """
    Crea y devuelve una nueva instancia del LLM según MODEL_REGISTRY.
    Solo parámetros aceptados: model_name, temperature, max_tokens.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Modelo '{model_name}' no encontrado en MODEL_REGISTRY.")

    entry = MODEL_REGISTRY[model_name]
    provider = entry.get("provider", "").lower()
    extra = dict(entry.get("extra", {}))  # copia de extras

    # kwargs comunes
    llm_kwargs: Dict[str, Any] = {}
    if temperature is not None:
        llm_kwargs["temperature"] = float(temperature)
    if max_tokens is not None:
        llm_kwargs["max_tokens"] = int(max_tokens)

    # Si el registry trae api_key / endpoint, exportarlas a env vars (solo si están)
    if "api_key" in entry and entry["api_key"]:
        if provider == "azure":
            os.environ.setdefault("AZURE_OPENAI_API_KEY", str(entry["api_key"]))
        elif provider == "openai":
            os.environ.setdefault("OPENAI_API_KEY", str(entry["api_key"]))
        elif provider == "ollama":
            os.environ.setdefault("OLLAMA_API_KEY", str(entry["api_key"]))

    if "endpoint" in entry and entry["endpoint"]:
        os.environ.setdefault("OPENAI_API_BASE", str(entry["endpoint"]))
        os.environ.setdefault("AZURE_OPENAI_ENDPOINT", str(entry["endpoint"]))

    if "api_version" in entry and entry["api_version"]:
        os.environ.setdefault("AZURE_OPENAI_API_VERSION", str(entry["api_version"]))

    # Construcción por provider
    if provider == "azure":
        if AzureChatOpenAI is None:
            raise RuntimeError("AzureChatOpenAI no disponible: instala langchain y dependencias.")
        deployment = entry.get("deployment_name") or entry.get("deployment")
        if not deployment:
            raise ValueError(f"Entrada para '{model_name}' necesita 'deployment_name' o 'deployment'.")
        azure_kwargs = dict(llm_kwargs)
        azure_kwargs["deployment_name"] = deployment
        azure_kwargs.update(extra)
        llm = AzureChatOpenAI(**azure_kwargs)

    elif provider == "ollama":
        if ChatOllama is None:
            raise RuntimeError("ChatOllama no disponible: instala langchain-ollama.")
        model = entry.get("model")
        if not model:
            raise ValueError(f"Entrada para '{model_name}' necesita 'model' (nombre en Ollama).")
        ollama_kwargs = dict(llm_kwargs)
        ollama_kwargs["model"] = model
        if "base_url" in entry and entry["base_url"]:
            ollama_kwargs["base_url"] = entry["base_url"]
        # Mapear max_tokens -> num_predict si aplica para Ollama
        if "max_tokens" in ollama_kwargs:
            ollama_kwargs["num_predict"] = ollama_kwargs.pop("max_tokens")
        ollama_kwargs.update(extra)
        llm = ChatOllama(**ollama_kwargs)

    elif provider == "openai":
        if ChatOpenAI is None:
            raise RuntimeError("ChatOpenAI no disponible: instala langchain y dependencias.")
        model = entry.get("model")
        openai_kwargs = dict(llm_kwargs)
        if model:
            openai_kwargs["model"] = model
        openai_kwargs.update(extra)
        llm = ChatOpenAI(**openai_kwargs)

    else:
        raise ValueError(f"Provider desconocido '{provider}' en registro para '{model_name}'.")

    return llm

# -------------------------
# Ejemplos de uso
# -------------------------
if __name__ == "__main__":
    # Cada llamada crea una instancia nueva (sin cache).
    try:
        llm_azure = create_llm("gpt-4o-azure", temperature=0.0, max_tokens=1024)
        print("Azure LLM instanciado:", type(llm_azure))
    except Exception as e:
        print("Error (Azure):", e)

    try:
        llm_ollama = create_llm("llama3.1-local", temperature=0.7, max_tokens=512)
        print("Ollama LLM instanciado:", type(llm_ollama))
    except Exception as e:
        print("Error (Ollama):", e)

    try:
        llm_openai = create_llm("gpt-4o-openai", temperature=0.2, max_tokens=800)
        print("OpenAI LLM instanciado:", type(llm_openai))
    except Exception as e:
        print("Error (OpenAI):", e)
