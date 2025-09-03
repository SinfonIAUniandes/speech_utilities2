from typing import Optional, Dict, Any
import os

# Importar clases de lógica interna
from llm_settings import LLMSettings
from llm_memory import LLMMemory
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Importar adaptadores de LangChain (o el framework que prefieras)
try:
    from langchain_community.chat_models import AzureChatOpenAI
    from langchain_community.chat_models import ChatOpenAI
    from langchain_ollama import ChatOllama
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
except ImportError:
    print("Advertencia: Faltan dependencias de LangChain. Instálalas con 'pip install langchain langchain-openai langchain-community langchain-ollama'")
    AzureChatOpenAI, ChatOpenAI, ChatOllama = None, None, None
    AIMessage, HumanMessage, SystemMessage = None, None, None


# Este registro se puede mover a un archivo de configuración YAML/JSON si se vuelve complejo
MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "gpt-4-azure": {
        "provider": "azure",
        "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
        "endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
        "api_version": os.getenv("AZURE_OPENAI_API_VERSION"),
        "deployment_name": os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
    }
}

class LLMHandler:
    """
    Clase orquestadora para interactuar con un LLM.
    Gestiona la configuración, la memoria y la creación de clientes LLM.
    """
    def __init__(self, initial_settings: Optional[Dict[str, Any]] = None):
        self.settings = LLMSettings()
        if initial_settings:
            self.settings.update_from_dict(initial_settings)

        self.memory = LLMMemory(self.settings.context)
        self.llm_client = None
        self._recreate_client()

    def _recreate_client(self):
        """
        Crea (o recrea) la instancia del cliente LLM basada en la configuración actual.
        """
        model_name = self.settings.model_name
        if model_name not in MODEL_REGISTRY:
            print(f"Error: Modelo '{model_name}' no encontrado en MODEL_REGISTRY.")
            self.llm_client = None
            return

        config = MODEL_REGISTRY[model_name]
        provider = config.get("provider")

        # --- Lógica de creación de cliente (inspirada en temporary_llm_test.py) ---
        # Aquí iría la lógica para instanciar AzureChatOpenAI, ChatOllama, etc.
        # Por ahora, es un placeholder.
        print(f"Recreando cliente para el modelo '{model_name}' con el proveedor '{provider}'...")
        # self.llm_client = create_llm(...) # Esta sería la llamada a una función de fábrica
        self.llm_client = f"Cliente_simulado_para_{model_name}" # Simulación

    def update_settings(self, new_settings: Dict[str, Any]):
        """
        Actualiza la configuración del LLM y recrea el cliente si es necesario.
        """
        print(f"Actualizando configuración con: {new_settings}")
        self.settings.update_from_dict(new_settings)
        self.memory.set_context(self.settings.context)
        self._recreate_client()
        print("Configuración actualizada y cliente recreado.")

    def get_response(self, prompt: str) -> str:
        """
        Envía un prompt al LLM, gestiona el historial y devuelve la respuesta.
        """
        if not self.llm_client:
            return "Error: El cliente LLM no está inicializado."

        self.memory.add_message("user", prompt)
        
        # --- Lógica de invocación del LLM ---
        # Aquí se convertiría self.memory.get_history() al formato que espera LangChain
        # y se llamaría a self.llm_client.invoke(...)
        
        # Simulación de respuesta
        print(f"Enviando al LLM (simulado): {self.memory.get_history()}")
        simulated_answer = f"Respuesta simulada para el prompt: '{prompt}'"
        
        self.memory.add_message("assistant", simulated_answer)
        return simulated_answer

    def clear_history(self):
        """
        Limpia el historial de la conversación.
        """
        print("Limpiando historial de conversación.")
        self.memory.clear()
        
# Ejemplo de cómo se podría usar
if __name__ == '__main__':
    handler = LLMHandler({"model_name": "llama3.1-local"})
    
    respuesta1 = handler.get_response("Hola, ¿cómo estás?")
    print(f"Respuesta del LLM: {respuesta1}\n")

    respuesta2 = handler.get_response("¿Recuerdas mi pregunta anterior?")
    print(f"Respuesta del LLM: {respuesta2}\n")

    handler.update_settings({
        "model_name": "gpt-4o-azure",
        "context": "Eres un asistente muy formal."
    })

    respuesta3 = handler.get_response("Y ahora, ¿quién eres?")
    print(f"Respuesta del LLM: {respuesta3}\n")