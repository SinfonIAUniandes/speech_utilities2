import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from speech_utilities2.logic.speech_agent import memory, prompts, router
from speech_utilities2.logic.speech_agent.router import Analisis

GUARDAR = router._guardar  # el fixture `aislado` lo apaga; el test del modo agente usa el real


def fake(perfil="informativo", conf=0.9, area="speech"):
    return RunnableLambda(lambda t: Analisis(intencion="x", idioma="es", entidades=[], area=area, perfil=perfil, confianza=conf))


def boom(_):
    raise TimeoutError


def test_llm_y_regla_de_area():
    r = router.analyze("hola", llm=fake("informativo", 0.9))
    assert (r.perfil.id, r.fuente) == ("informativo", "llm")
    r = router.analyze("ven", llm=fake("general", 0.9, area="navigation"))
    assert (r.perfil.id, r.fuente) == ("accion", "regla")


def test_sticky():
    r = router.analyze("mm", prev="informativo", llm=fake("general", 0.4))
    assert (r.perfil.id, r.fuente) == ("informativo", "sticky")
    r = router.analyze("mal", prev="informativo", llm=fake("delicado", 0.3))
    assert (r.perfil.id, r.fuente) == ("delicado", "llm")


@pytest.mark.parametrize("llm,prev,esperado", [
    (RunnableLambda(boom), None, "general"),
    (RunnableLambda(boom), "accion", "accion"),
    (fake("inexistente"), None, "general"),
])
def test_fallback_nunca_lanza(llm, prev, esperado):
    r = router.analyze("hola", prev=prev, llm=llm)
    assert (r.perfil.id, r.fuente) == (esperado, "fallback")


def test_forzado_no_llama_al_llm(monkeypatch):
    monkeypatch.setenv("SPEECH_PERFIL_FORCE", "accion")
    r = router.analyze("hola", llm=RunnableLambda(boom))
    assert (r.perfil.id, r.fuente) == ("accion", "forzado")


class Eco(GenericFakeChatModel):
    """Modelo falso que anota los mensajes que recibe."""
    vistos: list = []

    def bind_tools(self, *a, **k):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        self.vistos.append(messages)
        return super()._generate(messages, stop, run_manager, **kw)


@pytest.fixture
def aislado(tmp_path, monkeypatch):
    """Router con stores en tmp, sin agentes cacheados ni memoria en segundo plano."""
    monkeypatch.setattr(router, "_stores", memory.open_stores(tmp_path))
    monkeypatch.setattr(router, "_agents", router.OrderedDict())
    monkeypatch.setattr(router, "_prev", {})
    monkeypatch.setattr(router, "_guardar", lambda *a: None)
    return router.stores()[1]


def usar(monkeypatch, *respuestas):
    m = Eco(messages=iter(respuestas))
    m.vistos.clear()
    monkeypatch.setattr(router._llm, "chat", lambda *a, **k: m)
    return m


def test_agent_for_cache_y_historia_compartida(aislado, monkeypatch):
    usar(monkeypatch, *(AIMessage(content=f"r{i}") for i in range(9)))
    perfiles = prompts.catalog().perfiles
    a1, a2 = router.agent_for(perfiles["general"]), router.agent_for(perfiles["accion"])
    assert router.agent_for(perfiles["general"]) is a1 and a2 is not a1
    cfg = {"configurable": {"thread_id": "t1"}}
    a1.invoke({"messages": [("user", "uno")]}, config=cfg, context=memory.Ctx("u1", "t1"))
    assert any(m.content == "uno" for m in a2.get_state(cfg).values["messages"])  # cambiar de perfil conserva la historia


def test_callbacks_sin_langfuse():
    assert router.callbacks() == []


def test_responder_historia_memoria_y_recorte(aislado, monkeypatch):
    m = usar(monkeypatch, *(AIMessage(content=f"r{i}") for i in range(1, 9)))
    prevs = []
    monkeypatch.setattr(router, "analyze", lambda t, prev=None: prevs.append(prev) or router._fallback("informativo"))
    monkeypatch.setattr(router, "HISTORIA", 3)
    aislado.put(memory.namespace("u1"), memory.KEY, {"content": "- le gusta el café", "encoding": "utf-8"})
    assert router.responder("uno", "u1", "t1")[1] == "r1"
    assert router.responder("dos", "u1", "t1")[1] == "r2"
    assert router.responder("tres", "u1", "t1")[1] == "r3"
    sistema, *hist = m.vistos[2]
    assert "le gusta el café" in sistema.content  # memoria larga en el system prompt
    assert [x.content for x in hist] == ["dos", "r2", "tres"]  # corto plazo, recortado a HISTORIA
    assert prevs == [None, "informativo", "informativo"]
    assert router.responder("otra sesión", "u1", "t2")[1] == "r4" and len(m.vistos[3]) == 2  # hilo nuevo: sin historia


def test_modo_agente_escribe_perfil_y_lo_sanea(aislado, monkeypatch):
    call = {"name": "write_file", "args": {"file_path": memory.ARCHIVO, "content": "- se llama Ana\n- tiene diabetes"}, "id": "c1"}
    usar(monkeypatch, AIMessage(content="", tool_calls=[call]), AIMessage(content="ok"))
    monkeypatch.setattr(router, "MEMORIA", "agente")
    monkeypatch.setattr(router, "_escritor", None)
    GUARDAR("u1", "t1", "me llamo Ana", aislado)  # lo que responder() corre en segundo plano
    assert memory.recall("u1", aislado) == "- se llama Ana"
