import pytest
from deepagents import FilesystemMiddleware
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from speech_utilities2.logic.speech_agent import memory


class FakeModel(GenericFakeChatModel):
    def bind_tools(self, *a, **k):
        return self


def write_in_thread(tmp_path, user, thread, path, text):
    """Corre un agente (como el del modo agente) cuyo modelo solo llama write_file(path, text)."""
    saver, store = memory.open_stores(tmp_path)
    call = {"name": "write_file", "args": {"file_path": path, "content": text}, "id": "c1"}
    model = FakeModel(messages=iter([AIMessage(content="", tool_calls=[call]), AIMessage(content="listo")]))
    agent = create_agent(model, middleware=[FilesystemMiddleware(backend=memory.make_backend(store))],
                         context_schema=memory.Ctx, store=store, checkpointer=saver)
    agent.invoke({"messages": [("user", "x")]}, config={"configurable": {"thread_id": thread}},
                 context=memory.Ctx(user, thread))
    return store


def test_open_stores_crea_archivos(tmp_path):
    memory.open_stores(tmp_path / "nuevo")
    assert (tmp_path / "nuevo" / "checkpoints.sqlite").exists()
    assert (tmp_path / "nuevo" / "store.sqlite").exists()


def test_memories_persiste_entre_hilos_y_reinicios(tmp_path):
    write_in_thread(tmp_path, "u1", "A", memory.ARCHIVO, "- se llama Ana")
    _, store = memory.open_stores(tmp_path)  # "reinicio"
    assert "Ana" in memory.recall("u1", store, "B")
    assert memory.recall("u2", store) == ""  # cada persona solo ve lo suyo


@pytest.mark.parametrize("path", ["/notes.md", "/persona.md", "/context/x.md"])
def test_resto_es_efimero(tmp_path, path):
    store = write_in_thread(tmp_path, "u1", "A", path, "hola")
    assert store.search(("speech",)) == []


def test_anonimo_tiene_un_namespace_por_hilo():
    assert memory.namespace("anonimo", "t.1") == ("speech", "anon", "t_1")
    assert memory.namespace("anonimo", "t.1") != memory.namespace("anonimo", "t2")


def fake(*hechos):
    return RunnableLambda(lambda t: memory.Extraccion(hechos=[memory.Hecho(tipo=a, texto=b) for a, b in hechos]))


@pytest.mark.parametrize("uid", ["../x", "", "a" * 65])
def test_user_id_invalido(tmp_path, uid):
    _, store = memory.open_stores(tmp_path)
    with pytest.raises(ValueError):
        memory.forget_user(uid, store)
    with pytest.raises(ValueError):
        memory.remember(uid, "hola", store, llm=fake())


def test_forget_user_solo_borra_a_ese_usuario(tmp_path):
    _, store = memory.open_stores(tmp_path)
    for u in ("u1", "u2"):
        store.put(memory.namespace(u), memory.KEY, {"content": u, "encoding": "utf-8"})
    memory.forget_user("u1", store)
    assert [i.namespace for i in store.search(("speech",))] == [("speech", "users", "u2")]


def test_ruido_no_toca_el_archivo(tmp_path):
    _, store = memory.open_stores(tmp_path)
    assert memory.remember("u1", "hola, qué tal", store, llm=fake()) == "ok: nada que guardar"
    assert memory.recall("u1", store) == ""


def test_guarda_y_recuerda(tmp_path):
    _, store = memory.open_stores(tmp_path)
    memory.remember("u1", "x", store, hoy="2026-10-18",
                    llm=fake(("nombre", "Ana"), ("gusto", "le gusta el café"), ("fecha", "cumpleaños el 3 de mayo")))
    memory.remember("u1", "x", store, hoy="2026-10-19", llm=fake(("gusto", "le gusta el café"), ("dato", "tiene un perro")))
    d = memory._parse(memory.recall("u1", store))
    assert d["nombre"] == "Ana" and d["prefs"] == ["le gusta el café"] and d["hechos"] == ["cumpleaños el 3 de mayo", "tiene un perro"]
    assert len(d["convs"]) == 2


def test_sensible_se_descarta_y_anonimo_no_guarda(tmp_path):
    _, store = memory.open_stores(tmp_path)
    assert memory.remember("u1", "x", store, llm=fake(("dato", "tiene diabetes"), ("dato", "su contraseña es 1234"))) == "ok: nada que guardar"
    assert memory.remember("anonimo", "x", store, llm=fake(("nombre", "Ana"))).startswith("ok: usuario anónimo")
    assert store.search(("speech",)) == []


def test_tope_de_2kb_descarta_lo_mas_viejo(tmp_path):
    _, store = memory.open_stores(tmp_path)
    for i in range(80):
        memory.remember("u1", "x", store, hoy=f"2026-01-{i % 28 + 1:02d}", llm=fake(("dato", f"dato número {i} " + "x" * 30)))
    txt = memory.recall("u1", store)
    assert len(txt.encode()) <= memory.MAX_BYTES and "número 79" in txt and "número 0 " not in txt


def test_sanear_quita_sensible_y_recorta(tmp_path):
    texto = "# Ana\n- tiene diabetes\n" + "".join(f"- dato {i} {'x' * 40}\n" for i in range(80))
    store = write_in_thread(tmp_path, "u1", "A", memory.ARCHIVO, texto)
    limpio = memory.sanear("u1", store)
    assert "diabetes" not in limpio and len(limpio.encode()) <= memory.MAX_BYTES + 1
    assert limpio.startswith("# Ana") and "dato 79" in limpio and "dato 0 " not in limpio
    assert memory.recall("u1", store) == limpio.strip()
