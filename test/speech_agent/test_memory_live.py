import os
import time

import pytest

from speech_utilities2.logic.speech_agent import llm, memory, router

pytestmark = pytest.mark.skipif(not os.environ.get("SPEECH_LIVE"), reason="requiere Ollama arriba (SPEECH_LIVE=1)")


@pytest.mark.parametrize("modo", ["extractor", "agente"])
def test_recuerda_entre_sesiones(modo, tmp_path, monkeypatch):
    if modo == "agente" and llm.MODEL == "llama3.2:3b":
        pytest.skip("llama3.2:3b no llama bien a write_file/edit_file; usa SPEECH_MODEL=qwen3:4b-instruct-2507-q4_K_M")
    monkeypatch.setattr(router, "MEMORIA", modo)
    monkeypatch.setattr(router, "_stores", memory.open_stores(tmp_path))
    monkeypatch.setattr(router, "_agents", router.OrderedDict())
    monkeypatch.setattr(router, "_prev", {})
    _, store = router.stores()

    router.responder("hola, me llamo Ana y me encanta el café", "u1", "s1")
    for _ in range(60):  # en modo extractor la memoria se guarda en segundo plano
        if "café" in memory.recall("u1", store):
            break
        time.sleep(0.5)
    assert "Ana" in memory.recall("u1", store) and "café" in memory.recall("u1", store)

    _, respuesta = router.responder("¿qué bebida me gusta?", "u1", "s2")  # otra sesión: solo memoria larga
    assert "café" in respuesta.lower()
