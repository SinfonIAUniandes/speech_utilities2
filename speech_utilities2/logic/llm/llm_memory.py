"""Hold the running conversation that is sent to the language model."""

from typing import Dict, List, Optional

#: Roles a caller may append. The system role is reserved.
CONVERSATION_ROLES = ('user', 'assistant')

SYSTEM_ROLE = 'system'


class LLMMemory:
    """
    Keep the conversation history, with the system context pinned in front.

    The system message is owned by this class, so a failed turn can never strip
    the robot of its persona.
    """

    def __init__(self, system_context: Optional[str] = None):
        """Start an empty conversation seeded with 'system_context'."""
        self._history: List[Dict[str, str]] = []
        self._system_context = system_context
        self.clear()

    def add_message(self, role: str, content: str) -> None:
        """
        Append a 'user' or 'assistant' turn.

        Raises 'ValueError' for any other role. The system role is rejected on
        purpose; use :meth:`set_context`.
        """
        if role not in CONVERSATION_ROLES:
            allowed = ' or '.join(f"'{item}'" for item in CONVERSATION_ROLES)
            raise ValueError(f"Role must be {allowed}, got '{role}'.")
        self._history.append({'role': role, 'content': content})

    def get_history(self) -> List[Dict[str, str]]:
        """
        Return a copy of the full conversation.

        The copy is deliberate: callers must not mutate the history directly.
        """
        return list(self._history)

    def drop_last(self) -> Optional[Dict[str, str]]:
        """
        Remove and return the most recent turn, never the system message.

        Returns None when only the system message is left.
        """
        if len(self._history) <= (1 if self._system_context else 0):
            return None
        return self._history.pop()

    def clear(self) -> None:
        """Drop every turn, keeping the system context if there is one."""
        self._history.clear()
        if self._system_context:
            self._history.append({'role': SYSTEM_ROLE, 'content': self._system_context})

    def set_context(self, system_context: str) -> None:
        """Replace the system context and start a fresh conversation."""
        self._system_context = system_context
        self.clear()
