import os

import pytest

from speech_utilities2.logic.speech_agent import prompts, router

pytestmark = pytest.mark.skipif(not (os.environ.get("SPEECH_LIVE") and os.environ.get("LANGFUSE_PUBLIC_KEY")),
                                reason="requiere Ollama (SPEECH_LIVE=1) y Langfuse")

CASOS = [
    ("hola, ¿cómo te llamas?", "general", "es"),
    ("¿a qué hora es la charla de robótica?", "informativo", "es"),
    ("llévame al laboratorio", "accion", "es"),
    ("me siento muy mal, creo que me voy a desmayar", "delicado", "es"),
    ("hi, what can you do?", "general", "en"),
]


def test_clasifica_4_de_5(monkeypatch):
    monkeypatch.setenv("SPEECH_PROMPT_LABEL", "staging")
    monkeypatch.setattr(prompts, "LABEL", "staging")
    prompts.set_evento("ejemplo")
    aciertos = 0
    for texto, perfil, idioma in CASOS:
        r = router.analyze(texto)
        aciertos += r.perfil.id == perfil and r.analisis.get("idioma", idioma) == idioma
    assert aciertos >= 4
