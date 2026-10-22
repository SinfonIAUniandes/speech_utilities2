"""
Drive a language-model client: configuration, conversation and requests.

'MODEL_REGISTRY' maps a friendly name to a provider and the names of the
environment variables holding its credentials. No secret is stored here.
"""

import copy
import logging
import os
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain_openai import AzureChatOpenAI, ChatOpenAI

from .llm_memory import LLMMemory
from .llm_settings import LLMSettings

logger = logging.getLogger(__name__)

load_dotenv()

#: Known models. Each entry names a provider and where to find its settings.
MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    'gpt-4o-azure': {
        'provider': 'azure',
        'deployment_name': 'GPT-4o',
        'AZURE_OPENAI_API_KEY': os.getenv('AZURE_OPENAI_API_KEY'),
        'AZURE_OPENAI_ENDPOINT': os.getenv('AZURE_OPENAI_ENDPOINT'),
        'api_version': os.getenv('AZURE_OPENAI_API_VERSION'),
    },
    'gpt-4.1-azure': {
        'provider': 'azure',
        'deployment_name': 'gpt-4.1',
        'AZURE_OPENAI_API_KEY': os.getenv('AZURE_OPENAI_API_KEY'),
        'AZURE_OPENAI_ENDPOINT': os.getenv('AZURE_OPENAI_ENDPOINT'),
        'api_version': os.getenv('AZURE_OPENAI_API_VERSION'),
    },
    'gpt-4o-mini': {
        'provider': 'openai',
        'model': 'gpt-4o-mini',
        # ChatOpenAI reads OPENAI_API_KEY from the environment itself.
    },
    'llama3.1-local': {
        'provider': 'ollama',
        'model': 'llama3.1',
        'base_url': os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434'),
    },
    'gemma3:1b': {
        'provider': 'ollama',
        'model': 'gemma3:1b',
        'base_url': os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434'),
    },
    'gemma3:270m': {
        'provider': 'ollama',
        'model': 'gemma3:270m',
        'base_url': os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434'),
    },
}

#: Role in the stored history -> LangChain message class.
_MESSAGE_TYPES = {
    'system': SystemMessage,
    'user': HumanMessage,
    'assistant': AIMessage,
}


class LLMError(RuntimeError):
    """Raised when the model cannot be configured or cannot answer."""


class LLMHandler:
    """
    Own the settings, the conversation and the provider client.

    This class decides how to spell each provider's arguments, so nothing else
    needs to know that Ollama calls the output limit 'num_predict'.
    """

    def __init__(self, initial_settings: Optional[Dict[str, Any]] = None):
        """
        Build the handler and its first client.

        Raises 'LLMError' if the model is unknown or its client cannot be built.
        """
        self.settings = LLMSettings()
        if initial_settings:
            initial_settings = dict(initial_settings)
            initial_settings.setdefault('model_name', next(iter(MODEL_REGISTRY)))
            self.settings.update_from_dict(initial_settings)

        self.memory = LLMMemory(self.settings.context)
        self.llm_client = None
        self._recreate_client()

    def _recreate_client(self) -> None:
        """Build the provider client for the current settings."""
        model_name = self.settings.model_name
        config = MODEL_REGISTRY.get(model_name)
        if config is None:
            known = ', '.join(sorted(MODEL_REGISTRY))
            raise LLMError(f"Unknown model '{model_name}'. Known models: {known}")

        provider = config.get('provider')
        logger.info("Building client for model '%s' (provider '%s').", model_name, provider)
        try:
            if provider == 'azure':
                self.llm_client = AzureChatOpenAI(
                    azure_deployment=config['deployment_name'],
                    openai_api_version=config['api_version'],
                    api_key=config['AZURE_OPENAI_API_KEY'],
                    azure_endpoint=config['AZURE_OPENAI_ENDPOINT'],
                    temperature=self.settings.temperature,
                    max_tokens=self.settings.max_tokens,
                )
            elif provider == 'openai':
                self.llm_client = ChatOpenAI(
                    model=config['model'],
                    temperature=self.settings.temperature,
                    max_tokens=self.settings.max_tokens,
                )
            elif provider == 'ollama':
                # Ollama spells the output limit 'num_predict'.
                self.llm_client = ChatOllama(
                    model=config['model'],
                    base_url=config['base_url'],
                    temperature=self.settings.temperature,
                    num_predict=self.settings.max_tokens,
                )
            else:
                raise LLMError(f"Unsupported provider '{provider}' for model '{model_name}'.")
        except LLMError:
            raise
        except Exception as error:
            raise LLMError(f"Could not build the client for '{model_name}': {error}") from error
        logger.info('LLM client ready.')

    def update_settings(self, new_settings: Dict[str, Any]) -> None:
        """
        Apply new settings and rebuild the client.

        The previous settings and client are restored if the new ones fail, so a
        rejected request leaves a working handler behind.
        """
        logger.info('Updating LLM settings: %s', ', '.join(sorted(new_settings)))
        logger.debug('New setting values: %s', new_settings)

        previous_settings = copy.copy(self.settings)
        previous_client = self.llm_client
        try:
            self.settings.update_from_dict(new_settings)
            self._recreate_client()
        except Exception:
            self.settings = previous_settings
            self.llm_client = previous_client
            logger.error('LLM settings rejected; the previous configuration is still in use.')
            raise

        if 'context' in new_settings:
            self.memory.set_context(self.settings.context)
        logger.info('LLM settings updated and client rebuilt.')

    def get_response(self, prompt: str) -> str:
        """
        Send 'prompt' to the model and return its answer.

        The prompt is removed from the conversation again if the call fails or comes
        back empty, so an unanswered question does not bias the next one. An empty
        completion returns 'settings.fallback_answer'.
        """
        if not self.llm_client:
            raise LLMError('The LLM client is not initialised; check the configuration.')

        self.memory.add_message('user', prompt)
        history = [
            _MESSAGE_TYPES[message['role']](content=message['content'])
            for message in self.memory.get_history()
            if message['role'] in _MESSAGE_TYPES
        ]

        try:
            logger.info(
                "Sending to '%s': %d messages, %d characters.",
                self.settings.model_name, len(history), sum(len(m.content) for m in history),
            )
            logger.debug('Conversation sent: %s', history)
            answer = self.llm_client.invoke(history).content
        except Exception as error:
            self.memory.drop_last()
            raise LLMError(f'The LLM call failed: {error}') from error

        if not (answer or '').strip():
            # A small model can emit its end-of-turn token immediately and
            # return nothing. That is a failure, not an answer, so it is
            # neither stored nor handed back as if it were one.
            logger.warning("Model '%s' returned an empty answer.", self.settings.model_name)
            self.memory.drop_last()
            return self.settings.fallback_answer

        self.memory.add_message('assistant', answer)
        return answer

    def clear_history(self) -> None:
        """Forget the conversation, keeping the system prompt."""
        logger.info('Clearing the conversation history.')
        self.memory.clear()
