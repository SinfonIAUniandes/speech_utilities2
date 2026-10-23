"""Analyzer/router del Speech Agent (ver prompt_router.spec.md): elige el perfil del turno y arma el agente."""
import logging
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Literal

from deepagents import FilesystemMiddleware
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call
from langchain_core.messages import HumanMessage, SystemMessage, trim_messages
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel

from . import llm as _llm
from . import memory, prompts

log = logging.getLogger(__name__)

TIMEOUT = float(os.environ.get("SPEECH_ROUTER_TIMEOUT", "2.0"))
STICKY = float(os.environ.get("SPEECH_ROUTER_STICKY", "0.6"))
MAX_AGENTS = 16

MEMORIA = os.environ.get("SPEECH_MEMORIA", "extractor")  # quién escribe perfil.md: "extractor" | "agente"
HISTORIA = int(os.environ.get("SPEECH_HISTORIA", "20"))  # mensajes del hilo que ve el modelo (el checkpointer guarda todos)
CONTEXTO = "\n\n## Lo que ya sabes de la persona con la que hablas (es verdad, úsalo con naturalidad)\n"
PROMPT_AGENTE = f"""Eres la memoria de un robot social. Lees lo que acaba de decir una persona y mantienes su archivo
{memory.ARCHIVO} (solo de ella).
- Si cuenta algo duradero de sí misma (nombre, gustos, estudios, trabajo, familia, mascotas, fechas, compromisos), guárdalo:
  si la sección "Lo que ya sabes de la persona" no aparece abajo, crea el archivo con write_file; si aparece, usa
  edit_file para agregar líneas o corregir las que cambiaron (si ya no le gusta algo, reemplaza esa línea).
- Cada dato en una línea "- frase corta en tercera persona". Ejemplo: "me llamo Ana y amo el café" ->
  write_file(file_path="{memory.ARCHIVO}", content="- se llama Ana\\n- le gusta el café")
- Saludos, preguntas, pedidos, estados pasajeros o datos sensibles (salud, contraseñas, documentos, dinero,
  direcciones): no llames ninguna herramienta.
Al terminar responde solo "ok"."""


class Analisis(BaseModel):
    intencion: str
    idioma: Literal["es", "en"]
    entidades: list[str]
    area: Literal["speech", "navigation", "manipulation", "perception", "task"]
    perfil: str  # id del catálogo; se valida tras la llamada
    confianza: float  # 0..1


@dataclass(frozen=True)
class Route:
    perfil: prompts.Perfil
    fuente: str  # "forzado" | "llm" | "regla" | "sticky" | "fallback"
    confianza: float
    analisis: dict  # {} en fallback


def _catalogo(perfiles: dict) -> str:
    return "\n".join(f'- {p.id}: {p.cuando} Ej: ' + ", ".join(f'"{e}"' for e in p.ejemplos) for p in perfiles.values())


def _structured(system: str, piece_obj):
    """system + texto -> Analisis. Liga la generación al prompt de Langfuse con la metadata del padre."""
    model = _llm.chat("sinfonia-fast", temperature=0, timeout=TIMEOUT, max_retries=0).with_structured_output(Analisis)
    ligar = RunnableLambda(lambda x: x).with_config(metadata={"langfuse_prompt": piece_obj} if piece_obj else {})
    return ligar | RunnableLambda(lambda t: [SystemMessage(system), HumanMessage(t)]) | model


def analyze(texto: str, prev: str | None = None, llm=None) -> Route:
    """Nunca lanza. `llm`: runnable texto -> Analisis (solo tests)."""
    t0 = time.monotonic()
    try:
        r = _analyze(texto, prev, llm)
    except Exception as e:  # el turno no puede romperse por el router
        log.warning("analyzer falló (%s); fallback", e)
        r = _fallback(prev)
    try:
        huella = prompts.compose(r.perfil).huella
    except Exception:
        huella = "?"
    log.info("analyzer perfil=%s fuente=%s conf=%.2f ms=%d huella=%s",
             r.perfil.id, r.fuente, r.confianza, (time.monotonic() - t0) * 1000, huella)
    return r


def _fallback(prev: str | None) -> Route:
    perfiles = prompts.catalog().perfiles
    return Route(perfiles.get(prev) or perfiles["general"], "fallback", 0.0, {})


def _analyze(texto, prev, llm) -> Route:
    cat = prompts.catalog()
    perfiles = cat.perfiles
    if (forced := os.environ.get("SPEECH_PERFIL_FORCE")):
        if forced in perfiles:
            return Route(perfiles[forced], "forzado", 1.0, {})
        log.error("SPEECH_PERFIL_FORCE=%s no existe", forced)
    piece = cat.pieces["speech/analyzer"]
    chain = llm or _structured(prompts.render(piece.prompt, catalogo=_catalogo(perfiles)), piece.obj)
    a = chain.invoke(texto)
    if a.perfil not in perfiles:
        raise ValueError(f"perfil inexistente: {a.perfil}")
    chosen, fuente = perfiles[a.perfil], "llm"
    if a.area != "speech" and chosen.sticky and "accion" in perfiles:  # un perfil no sticky (delicado) no se pisa
        chosen, fuente = perfiles["accion"], "regla"
    elif (p := perfiles.get(prev)) and p.sticky and chosen.sticky and a.confianza < STICKY and chosen.id != p.id:
        chosen, fuente = p, "sticky"
    return Route(chosen, fuente, a.confianza, a.model_dump())


# --- agente por prompt compuesto ---
_agents: OrderedDict = OrderedDict()
_stores = None


def stores():
    """(checkpointer, store) compartidos por todo el proceso; se abren la primera vez."""
    global _stores
    if _stores is None:
        _stores = memory.open_stores()
    return _stores


@wrap_model_call
def _contexto(request, handler):
    """En cada llamada al modelo: agrega perfil.md al system prompt y recorta la historia (no toca el checkpointer)."""
    ctx = request.runtime.context
    sabido = memory.recall(ctx.user_id, request.runtime.store, ctx.thread_id)
    return handler(request.override(
        system_message=SystemMessage(request.system_prompt + (CONTEXTO + sabido if sabido else "")),
        messages=trim_messages(request.messages, strategy="last", token_counter=len, max_tokens=HISTORIA,
                               start_on="human")))


def agent_for(perfil: prompts.Perfil):
    """Agente con el prompt compuesto, checkpointer (corto plazo) y perfil.md (largo plazo); LRU por huella."""
    c = prompts.compose(perfil)
    if c.huella in _agents:
        _agents.move_to_end(c.huella)
        return _agents[c.huella]
    saver, store = stores()
    agent = create_agent(_llm.chat("sinfonia-reasoning", max_tokens=MAX_TOKENS), system_prompt=c.texto,
                         middleware=[_contexto], context_schema=memory.Ctx, store=store, checkpointer=saver)
    _agents[c.huella] = agent
    if len(_agents) > MAX_AGENTS:
        _agents.popitem(last=False)
    return agent


_escritor = None


def escritor():
    """Modo agente: agente aparte que mantiene perfil.md con write_file/edit_file sobre el CompositeBackend.
    Va separado del que conversa porque, con un prompt corto y solo estas herramientas, un modelo de 4B sí decide
    guardar; enterrado bajo la persona y el evento casi nunca lo hace."""
    global _escritor
    if _escritor is None:
        _, store = stores()
        fs = FilesystemMiddleware(backend=memory.make_backend(store), tools=["read_file", "write_file", "edit_file"])
        _escritor = create_agent(_llm.chat("sinfonia-fast", temperature=0), system_prompt=PROMPT_AGENTE,
                                 middleware=[_contexto, fs], context_schema=memory.Ctx, store=store)
    return _escritor


def callbacks() -> list:
    if not os.environ.get("LANGFUSE_PUBLIC_KEY"):
        return []
    from langfuse.langchain import CallbackHandler
    return [CallbackHandler()]


# --- turno completo (lo que hace agent_node por cada frase) ---
MAX_TOKENS: int | None = None  # límite de la respuesta hablada; None = el del modelo
_prev: dict[str, str] = {}  # thread_id -> perfil del turno anterior (sticky)


def responder(texto: str, user_id: str = "anonimo", thread_id: str = "") -> tuple[Route, str]:
    """Analiza, responde con historia + perfil.md y actualiza la memoria según MEMORIA. Devuelve (ruta, respuesta)."""
    thread_id = thread_id or user_id
    ctx = memory.Ctx(user_id, thread_id)
    memory.namespace(user_id, thread_id)  # valida user_id antes de llamar al LLM
    r = analyze(texto, prev=_prev.get(thread_id))
    _prev[thread_id] = r.perfil.id
    cfg = {"configurable": {"thread_id": thread_id}, "callbacks": callbacks(),
           "metadata": {"langfuse_session_id": thread_id, "langfuse_user_id": user_id,
                        "langfuse_tags": [f"evento:{prompts.evento()}", f"robot:{prompts.ROBOT}", f"memoria:{MEMORIA}"]}}
    res = agent_for(r.perfil).invoke({"messages": [HumanMessage(texto)]}, config=cfg, context=ctx)
    # la persona no espera a que se actualice la memoria (otra llamada al LLM)
    threading.Thread(target=_guardar, args=(user_id, thread_id, texto, stores()[1]), daemon=True).start()
    return r, res["messages"][-1].content


def _guardar(user_id: str, thread_id: str, texto: str, store) -> None:
    """Actualiza perfil.md según MEMORIA. Nunca lanza: un fallo de memoria no rompe la conversación."""
    try:
        if MEMORIA == "agente":
            if user_id == "anonimo":
                return
            escritor().invoke({"messages": [HumanMessage(texto)]}, context=memory.Ctx(user_id, thread_id),
                              config={"callbacks": callbacks()})
            log.info("memoria %s (agente): %r", user_id, memory.sanear(user_id, store, thread_id))
        else:
            log.info("memoria %s (extractor): %s", user_id, memory.remember(user_id, texto, store))
    except Exception as e:
        log.warning("memoria %s falló: %s", user_id, e)


def configurar(model: str = "", max_tokens: int = 0) -> None:
    """Cambia modelo y/o límite de la respuesta en caliente ("" / 0 = no cambia)."""
    global MAX_TOKENS, _escritor
    if model:
        _llm.MODEL = model
    if max_tokens > 0:
        MAX_TOKENS = max_tokens
    _llm.chat.cache_clear()
    _agents.clear()  # los agentes cacheados tienen el cliente viejo
    _escritor = None
