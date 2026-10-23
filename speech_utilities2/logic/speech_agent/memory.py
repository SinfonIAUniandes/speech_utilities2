"""Memoria del Speech Agent.

- Corto plazo: el checkpointer (historia del hilo). thread_id = uno por *sesión de conversación*
  (p. ej. f"{user_id}-{inicio_iso}"), lo decide quien llama a agent_node.
- Largo plazo: /memories/perfil.md, un archivo por persona en el store (namespace por user_id: el agente solo
  ve el de quien le habla). Lo escribe `remember()` (modo extractor) o el propio agente con edit_file (modo agente).
- SPEECH_DATA_DIR (default ~/.local/share/speech_agent) guarda checkpoints.sqlite y store.sqlite.
  En Docker debe ser un volumen, si no la memoria se pierde al recrear el contenedor.
"""
import os
import re
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Literal

from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.sqlite import SqliteStore
from pydantic import BaseModel

from . import llm as _llm

DATA_DIR = Path(os.environ.get("SPEECH_DATA_DIR", "~/.local/share/speech_agent")).expanduser()
ARCHIVO = "/memories/perfil.md"  # lo que ve el agente
KEY = "/perfil.md"               # clave en el store (CompositeBackend quita el prefijo /memories)
USER_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

EXTRACT_PROMPT = """Tu trabajo: leer lo que dijo una persona a un robot y sacar SOLO los datos personales duraderos que valga la pena recordar la próxima vez que hablen.

GUARDA (tipo -> ejemplo):
- nombre: cómo se llama LA PERSONA que habla (no otra gente que menciona). "me llamo Ana" -> nombre: Ana
- gusto: lo que le gusta o no le gusta ("le gusta X", "no le gusta X"). "me encanta el café" -> gusto: le gusta el café
- dato: algo estable de su vida (estudios, trabajo, familia, mascotas, ciudad, equipo). "estudio ingeniería" -> dato: estudia ingeniería
- fecha: una fecha que importa. "mi cumpleaños es el 3 de mayo" -> fecha: cumpleaños el 3 de mayo
- compromiso: algo que prometió o quedó pendiente. "mañana te traigo el cargador" -> compromiso: traerá el cargador mañana

NO GUARDES (devuelve lista vacía):
- saludos, charla y relleno: "hola", "qué tal", "jaja", "ok", "gracias", "no sé"
- preguntas y pedidos al robot: "¿qué hora es?", "llévame al laboratorio", "cuéntame un chiste"
- opiniones del momento o cosas pasajeras: "hoy hace frío", "estoy cansado", "esto está lleno"
- lo que dice el robot: SOLO cuenta lo que dice la persona sobre sí misma
- datos sensibles: salud, enfermedades, contraseñas, documentos de identidad, dinero, tarjetas, direcciones exactas

REGLAS:
- Cada dato en una frase corta, en tercera persona y en tiempo PRESENTE ("le gusta", "trabaja"), sin inventar nada que la persona no dijo.
- Si dudas, no lo guardes.
- Responde SOLO el JSON {"hechos": [{"tipo": ..., "texto": ...}]}.

EJEMPLOS
Persona: "hola, ¿cómo estás?" -> {"hechos": []}
Persona: "me llamo Ana y me gusta el café" -> {"hechos": [{"tipo": "nombre", "texto": "Ana"}, {"tipo": "gusto", "texto": "le gusta el café"}]}
Persona: "llévame al baño" -> {"hechos": []}
Persona: "soy Luis, trabajo en Ecopetrol y mi cumpleaños es el 12 de junio" -> {"hechos": [{"tipo": "nombre", "texto": "Luis"}, {"tipo": "dato", "texto": "trabaja en Ecopetrol"}, {"tipo": "fecha", "texto": "cumpleaños el 12 de junio"}]}
Persona: "uf hoy estoy muy cansado, tuve diabetes desde niño" -> {"hechos": []}
Persona: "mañana te traigo el cargador" -> {"hechos": [{"tipo": "compromiso", "texto": "traerá el cargador mañana"}]}
Persona: "odio el brócoli pero amo el fútbol" -> {"hechos": [{"tipo": "gusto", "texto": "no le gusta el brócoli"}, {"tipo": "gusto", "texto": "le gusta el fútbol"}]}
Persona: "mi amiga Sofía cumple años el 5 de agosto" -> {"hechos": [{"tipo": "fecha", "texto": "su amiga Sofía cumple años el 5 de agosto"}]}
Persona: "tengo un perro que se llama Max" -> {"hechos": [{"tipo": "dato", "texto": "tiene un perro llamado Max"}]}"""

SECCION = {"gusto": "prefs", "dato": "hechos", "fecha": "hechos", "compromiso": "hechos"}
SENSIBLE = re.compile(
    r"\b(diabet|cáncer|cancer|enferm|depresi|ansiedad|medicin|tratamiento|contraseña|password|clave|cédula|cedula|"
    r"pasaporte|tarjeta|cuenta bancaria|iban|salario|sueldo|deuda|vive en la (calle|carrera|cra)|dirección)", re.I)
MAX_BYTES = 2048


class Hecho(BaseModel):
    tipo: Literal["nombre", "gusto", "dato", "fecha", "compromiso"]
    texto: str


class Extraccion(BaseModel):
    hechos: list[Hecho] = []


def extract(usuario: str, llm=None) -> list[Hecho]:
    """Hechos duraderos de lo que dijo la persona. `llm`: runnable texto -> Extraccion (tests)."""
    if llm is None:
        model = _llm.chat("sinfonia-fast", temperature=0).with_structured_output(Extraccion)
        llm = RunnableLambda(lambda t: [SystemMessage(EXTRACT_PROMPT), HumanMessage(f'Persona: "{t}"')]) | model
    seen, out = set(), []
    for h in llm.invoke(usuario).hechos:
        t = re.sub(r"(?<=\w)¡(?=\w)", "í", h.texto.strip().rstrip("."))  # glitch de llama3.2: "ingenier¡a"
        if not 2 <= len(t) <= 120 or SENSIBLE.search(t) or t.lower() in seen:
            continue  # ruido del modelo pequeño o dato sensible: no se guarda
        seen.add(t.lower())
        out.append(Hecho(tipo=h.tipo, texto=t))
    return out


@dataclass
class Ctx:
    """Contexto de cada invocación del agente (context_schema)."""
    user_id: str = "anonimo"
    thread_id: str = ""


def _check(user_id: str) -> None:
    if not USER_ID.match(user_id):
        raise ValueError(f"user_id inválido: {user_id!r}")


def namespace(user_id: str, thread_id: str = "") -> tuple[str, ...]:
    """Un namespace por persona. anonimo -> uno por sesión, para no mezclar desconocidos."""
    _check(user_id)
    if user_id == "anonimo":
        return ("speech", "anon", re.sub(r"[^A-Za-z0-9_-]", "_", thread_id) or "sin-hilo")
    return ("speech", "users", user_id)


def _parse(text: str) -> dict:
    d = {"nombre": "", "idioma": "es", "hechos": [], "prefs": [], "convs": []}
    sec = None
    for line in text.splitlines():
        if line.startswith("nombre:"):
            d["nombre"] = line[7:].strip()
        elif line.startswith("idioma:"):
            d["idioma"] = line[7:].strip() or "es"
        elif line.startswith("## "):
            sec = {"Hechos": "hechos", "Preferencias": "prefs", "Conversaciones": "convs"}.get(line[3:].strip())
        elif line.startswith("- ") and sec:
            d[sec].append(line[2:].strip())
    return d


def _render(user_id: str, d: dict) -> str:
    def block(title, items):
        return [f"## {title}"] + [f"- {i}" for i in items]
    return "\n".join([f"# {user_id}", f"nombre: {d['nombre']}", f"idioma: {d['idioma']}",
                      *block("Hechos", d["hechos"]), *block("Preferencias", d["prefs"]),
                      *block("Conversaciones", d["convs"])]) + "\n"


def _fit(user_id: str, d: dict) -> str:
    """Render dentro de MAX_BYTES: descarta primero lo más viejo de Conversaciones, luego de Preferencias y Hechos."""
    for sec in ("convs", "prefs", "hechos"):
        while len(_render(user_id, d).encode()) > MAX_BYTES and d[sec]:
            d[sec].pop(0)
    return _render(user_id, d)


def recall(user_id: str, store, thread_id: str = "") -> str:
    """Contenido de /memories/perfil.md de la persona, o "" si no hay nada."""
    item = store.get(namespace(user_id, thread_id), KEY)
    return item.value["content"].strip() if item else ""


def _put(ns: tuple, store, content: str, item) -> None:
    now = datetime.now(timezone.utc).isoformat()
    store.put(ns, KEY, {"content": content, "encoding": "utf-8",
                        "created_at": item.value["created_at"] if item else now, "modified_at": now})


def remember(user_id: str, usuario: str, store, llm=None, hoy: str | None = None) -> str:
    """Modo extractor: saca hechos de lo que dijo la persona y los fusiona en su perfil.md. Sin hechos no lo toca."""
    _check(user_id)
    if user_id == "anonimo":
        return "ok: usuario anónimo, nada que guardar"
    hechos = extract(usuario, llm)
    if not hechos:
        return "ok: nada que guardar"
    ns = namespace(user_id)
    item = store.get(ns, KEY)
    d = _parse(item.value["content"]) if item else _parse("")
    for h in hechos:
        if h.tipo == "nombre":
            d["nombre"] = h.texto
            continue
        lista = d[SECCION[h.tipo]]
        if h.texto.lower() not in (x.lower() for x in lista):
            lista.append(h.texto)
    d["convs"].append(f"{hoy or date.today().isoformat()}: " + "; ".join(h.texto for h in hechos))
    _put(ns, store, _fit(user_id, d), item)
    return "ok: " + "; ".join(h.texto for h in hechos)


def sanear(user_id: str, store, thread_id: str = "") -> str:
    """Modo agente: el LLM escribe libre en perfil.md; aquí se quitan líneas sensibles y se recorta a MAX_BYTES."""
    ns = namespace(user_id, thread_id)
    item = store.get(ns, KEY)
    if item is None:
        return ""
    txt = item.value["content"]
    lineas = [x for x in txt.splitlines() if not SENSIBLE.search(x)]
    # ponytail: recorta desde la 2.ª línea (lo más viejo si el agente agrega al final); sin estructura fija no hay más que hacer
    while len("\n".join(lineas).encode()) > MAX_BYTES and len(lineas) > 1:
        lineas.pop(1)
    limpio = "\n".join(lineas) + "\n"
    if limpio != txt:
        _put(ns, store, limpio, item)
    return limpio


def make_backend(store=None) -> CompositeBackend:
    """/memories/ -> store persistente con namespace de la persona (Ctx del runtime); el resto efímero por hilo.
    `store` None = el que pasa create_agent(store=...)."""
    persistent = StoreBackend(namespace=lambda rt: namespace(rt.context.user_id, rt.context.thread_id), store=store)
    return CompositeBackend(default=StateBackend(), routes={"/memories/": persistent})


def open_stores(data_dir: Path = DATA_DIR) -> tuple[SqliteSaver, SqliteStore]:
    """Checkpointer (estado de hilos) y store (archivos persistentes) en SQLite. Crea data_dir."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    # check_same_thread=False: el nodo ROS invoca desde hilos del executor.
    saver = SqliteSaver(sqlite3.connect(data_dir / "checkpoints.sqlite", check_same_thread=False))
    store = SqliteStore(sqlite3.connect(data_dir / "store.sqlite", check_same_thread=False, isolation_level=None))
    saver.setup()
    store.setup()
    return saver, store


def forget_user(user_id: str, store) -> None:
    """Borra el perfil.md de la persona (pedido de borrado de datos personales)."""
    store.delete(namespace(user_id), KEY)
