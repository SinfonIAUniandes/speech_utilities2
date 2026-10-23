import pytest

from speech_utilities2.logic.speech_agent import prompts


@pytest.fixture(autouse=True)
def limpio(monkeypatch):
    """Sin red ni keys de Langfuse: el catálogo sale del snapshot; estado del módulo reiniciado."""
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("SPEECH_PERFIL_FORCE", raising=False)
    monkeypatch.setattr(prompts, "_cat", None)
    monkeypatch.setattr(prompts, "_ts", 0.0)
    monkeypatch.setattr(prompts, "_evento", None)
    # con ROS en el entorno, el plugin launch_testing importa `launch`, que crea todos los loggers con
    # propagate=False (LaunchLogger) y caplog no los vería
    monkeypatch.setattr(prompts.log, "propagate", True)
