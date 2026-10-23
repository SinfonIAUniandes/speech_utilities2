"""System prompt del robot en piezas de Langfuse (ver prompt_router.spec.md).

Sin Speech adentro: otra área puede copiar este módulo.
# ponytail: mover a un paquete común cuando otra área lo use; no antes.

Nunca depende de que Langfuse esté arriba: si falla, usa prompts_snapshot.json (`python3 -m speech_utilities2.logic.speech_agent.prompts snapshot`).
"""
import json
import logging
import os
import re
import sys
import threading
import time
from dataclasses import dataclass
from functools import cached_property
from importlib import resources

log = logging.getLogger(__name__)

LABEL = os.environ.get("SPEECH_PROMPT_LABEL", "production")
TTL = int(os.environ.get("SPEECH_PROMPTS_TTL", "60"))
ROBOT = os.environ.get("SINFONIA_ROBOT", "Pepper")
SNAPSHOT = "prompts_snapshot.json"
TAGS = ("sinfonia-persona", "sinfonia-evento", "speech-sistema", "speech-perfil")
SLUG = re.compile(r"^[a-z0-9][a-z0-9_-]{0,47}$")
REQUIRED = ("sinfonia/persona", "speech/analyzer", "speech/response-base", "speech/perfil/general")
MAX_LEN = {"sinfonia/persona": 3000, "sinfonia/evento/": 4000, "speech/response-base": 1500, "speech/perfil/": 1500}


@dataclass(frozen=True)
class Piece:
    name: str
    version: int
    prompt: str
    config: dict
    obj: object = None  # objeto de Langfuse (para ligarlo a la traza); None si viene del snapshot


@dataclass(frozen=True)
class Perfil:
    id: str
    cuando: str
    ejemplos: tuple
    sticky: bool
    piece: Piece


@dataclass(frozen=True)
class Catalog:
    pieces: dict  # nombre -> Piece
    origen: str   # "langfuse" | "snapshot"

    @cached_property  # el Catalog es inmutable: se calcula una vez
    def perfiles(self) -> dict:
        return {n.rsplit("/", 1)[1]: Perfil(n.rsplit("/", 1)[1], p.config["cuando"], tuple(p.config.get("ejemplos", ())),
                                            p.config.get("sticky", True), p)
                for n, p in self.pieces.items() if n.startswith("speech/perfil/")}

    @cached_property
    def eventos(self) -> dict:
        return {n.rsplit("/", 1)[1]: p for n, p in self.pieces.items() if n.startswith("sinfonia/evento/")}


@dataclass(frozen=True)
class Composed:
    texto: str
    huella: str   # "persona@v3+evento/feria@v7+base@v2+perfil/accion@v1"
    piezas: tuple


def render(text: str, **vars) -> str:
    return re.sub(r"\{\{\s*(\w+)\s*\}\}", lambda m: str(vars.get(m.group(1), m.group(0))), text)


def _problem(name: str, prompt: str, config: dict) -> str | None:
    """Motivo por el que una pieza no es válida, o None."""
    if name.startswith(("sinfonia/evento/", "speech/perfil/")) and not SLUG.match(name.rsplit("/", 1)[1]):
        return "slug inválido"
    limit = next((v for k, v in MAX_LEN.items() if name.startswith(k)), None)
    if limit and len(prompt) > limit:
        return f"{len(prompt)} caracteres (máx {limit})"
    if name.startswith("speech/perfil/"):
        if not config.get("cuando"):
            return "config.cuando es obligatorio"
        if len(config.get("ejemplos", ())) > 5:
            return "más de 5 ejemplos"
    return None


def _langfuse():
    if not os.environ.get("LANGFUSE_PUBLIC_KEY"):
        return None
    from langfuse import Langfuse
    return Langfuse()


def _fetch(lf, label: str) -> dict:
    """Piezas válidas de Langfuse. Lanza si Langfuse no responde al listar."""
    names = set()
    for tag in TAGS:
        page = 1
        while True:
            r = lf.api.prompts.list(tag=tag, label=label, page=page, request_options={"timeout_in_seconds": 2})
            names.update(m.name for m in r.data)
            if page >= r.meta.total_pages:
                break
            page += 1
    out = {}
    for name in sorted(names):
        try:
            p = lf.get_prompt(name, label=label, type="text", cache_ttl_seconds=TTL, fetch_timeout_seconds=2)
        except Exception as e:
            log.warning("prompt %s no se pudo traer: %s", name, e)
            continue
        cfg = p.config or {}
        if motivo := _problem(name, p.prompt, cfg):
            log.warning("prompt %s descartado: %s", name, motivo)
            continue
        out[name] = Piece(name, p.version, p.prompt, cfg, p)
    return out


def _snapshot() -> dict:
    raw = json.loads(resources.files(__package__).joinpath(SNAPSHOT).read_text("utf-8"))
    return {n: Piece(n, d["version"], d["prompt"], d["config"]) for n, d in raw.items()}


def load(lf=None, label: str = LABEL) -> Catalog:
    lf = lf or _langfuse()
    pieces, origen = {}, "langfuse"
    if lf is None:
        origen = "snapshot"
    else:
        try:
            pieces = _fetch(lf, label)
        except Exception as e:
            log.warning("Langfuse no responde (%s); usando snapshot", e)
            origen = "snapshot"
    if origen == "snapshot":
        pieces = _snapshot()
    elif missing := [n for n in REQUIRED if n not in pieces]:
        snap = _snapshot()
        for n in missing:
            log.warning("falta %s en Langfuse; usando el del snapshot", n)
            if n in snap:
                pieces[n] = snap[n]
    if missing := [n for n in REQUIRED if n not in pieces]:
        raise ValueError(f"faltan prompts obligatorios: {missing}")
    return Catalog(pieces, origen)


# --- catálogo vigente y evento activo ---
_lock = threading.Lock()
_cat: Catalog | None = None
_ts = 0.0
_evento: str | None = os.environ.get("SINFONIA_EVENTO") or None


def set_evento(slug: str | None) -> None:
    global _evento
    _evento = slug or None


def evento() -> str | None:
    return _evento


def catalog() -> Catalog:
    global _cat, _ts
    with _lock:
        if _cat is None:
            _cat, _ts = load(), time.monotonic()
        elif time.monotonic() - _ts > TTL:
            _ts = time.monotonic()  # también tras un fallo: no pagar el timeout en cada turno
            try:
                new = load()
                if not (new.origen == "snapshot" and _cat.origen == "langfuse"):
                    _cat = new
            except Exception as e:
                log.warning("refresco de prompts falló, sigo con el anterior: %s", e)
        return _cat


def compose(perfil: Perfil, *, robot: str = ROBOT) -> Composed:
    c = catalog()
    p = c.pieces
    ev = None
    if _evento:
        ev = c.eventos.get(_evento)
        if ev is None:
            log.error("evento %r no existe en el catálogo; se compone sin evento", _evento)
    partes = [render(p["sinfonia/persona"].prompt, robot=robot)]
    usadas = [p["sinfonia/persona"]]
    if ev:
        partes.append("## Evento actual\n" + render(ev.prompt, robot=robot))
        usadas.append(ev)
    base = p["speech/response-base"]
    partes.append("## Cómo respondes\n" + base.prompt)
    partes.append(f"## Perfil de este turno: {perfil.id}\n" + perfil.piece.prompt)
    usadas += [base, perfil.piece]
    huella = "+".join(f"{x.name.removeprefix('sinfonia/').removeprefix('speech/')}@v{x.version}" for x in usadas)
    return Composed("\n\n".join(partes), huella, tuple(x.obj for x in usadas))


def _write_snapshot() -> None:
    lf = _langfuse()
    if lf is None:
        sys.exit("faltan LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY")
    pieces = _fetch(lf, LABEL)
    data = {n: {"version": p.version, "prompt": p.prompt, "config": p.config} for n, p in sorted(pieces.items())}
    path = resources.files(__package__).joinpath(SNAPSHOT)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", "utf-8")
    print(f"{len(data)} prompts (label={LABEL}) -> {path}")


if __name__ == "__main__":
    if sys.argv[1:] == ["snapshot"]:
        _write_snapshot()
    else:
        sys.exit("uso: python3 -m speech_utilities2.logic.speech_agent.prompts snapshot")
