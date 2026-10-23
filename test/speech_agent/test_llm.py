from speech_utilities2.logic.speech_agent import llm


def test_chat_apunta_a_ollama():
    c = llm.chat("sinfonia-fast")
    assert c.model_name == llm.MODEL
    assert c.openai_api_base.endswith(":11434/v1")


def test_kwargs_llegan():
    assert llm.chat("sinfonia-fast", temperature=0).temperature == 0
