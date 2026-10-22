"""Settings for one language-model client."""

from dataclasses import dataclass
from typing import Any, Dict, Optional

#: Fields converted to a specific type when they arrive as strings.
_COERCIONS = {'temperature': float, 'max_tokens': int}


@dataclass
class LLMSettings:
    """
    Everything needed to build and drive a language-model client.

    'model_name' is a key into the model registry, not a provider's own model
    id. 'fallback_answer' is spoken when the model returns nothing.
    """

    model_name: str = 'default_model'
    temperature: float = 0.7
    max_tokens: int = 500
    context: Optional[str] = None
    fallback_answer: str = 'Sorry, I could not produce an answer right now.'

    def update_from_dict(self, settings_dict: Dict[str, Any]) -> None:
        """
        Apply the recognised keys of 'settings_dict', ignoring the rest.

        Numeric fields are coerced; a bad value raises 'ValueError' naming it.
        """
        for key, value in settings_dict.items():
            if not hasattr(self, key):
                continue
            coerce = _COERCIONS.get(key)
            if coerce is not None:
                try:
                    value = coerce(value)
                except (TypeError, ValueError) as error:
                    raise ValueError(f"Invalid value for '{key}': {value!r} ({error})") from error
            setattr(self, key, value)
