import os

import pytest

from speech_utilities2.logic.speech_agent import llm

pytestmark = pytest.mark.skipif(not os.environ.get("SPEECH_LIVE"), reason="requiere Ollama arriba (SPEECH_LIVE=1)")


def test_chat_local():
    assert llm.chat("sinfonia-local").invoke("di hola").content
