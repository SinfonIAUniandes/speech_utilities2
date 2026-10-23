import logging
from types import SimpleNamespace as NS

import pytest

from speech_utilities2.logic.speech_agent import prompts


class FakeLF:
    """Langfuse falso: {nombre: (tag, version, prompt, config)}."""

    def __init__(self, data, boom=False):
        self.data, self.boom = data, boom
        self.api = NS(prompts=NS(list=self._list))

    def _list(self, tag, label, page, **kw):
        if self.boom:
            raise ConnectionError("sin red")
        return NS(data=[NS(name=n) for n, d in self.data.items() if d[0] == tag], meta=NS(total_pages=1))

    def get_prompt(self, name, **kw):
        _, v, text, cfg = self.data[name]
        return NS(name=name, version=v, prompt=text, config=cfg)


def full(**over):
    d = {
        "sinfonia/persona": ("sinfonia-persona", 3, "Soy {{robot}}", {}),
        "sinfonia/evento/feria": ("sinfonia-evento", 7, "Feria en {{robot}} land", {}),
        "speech/analyzer": ("speech-sistema", 1, "analiza {{catalogo}}", {}),
        "speech/response-base": ("speech-sistema", 2, "base", {}),
        "speech/perfil/general": ("speech-perfil", 1, "gen", {"cuando": "charla"}),
        "speech/perfil/accion": ("speech-perfil", 1, "acc", {"cuando": "pedidos"}),
    }
    d.update(over)
    return d


def test_load_trae_piezas_con_version():
    c = prompts.load(FakeLF(full()))
    assert c.origen == "langfuse"
    assert c.pieces["sinfonia/persona"].version == 3
    assert set(c.perfiles) == {"general", "accion"} and set(c.eventos) == {"feria"}


def test_piezas_invalidas_se_descartan(caplog):
    lf = FakeLF(full(**{
        "speech/perfil/sin-cuando": ("speech-perfil", 1, "x", {}),
        "speech/perfil/Mal Slug": ("speech-perfil", 1, "x", {"cuando": "a"}),
        "sinfonia/evento/largo": ("sinfonia-evento", 1, "x" * 4001, {}),
    }))
    with caplog.at_level(logging.WARNING):
        c = prompts.load(lf)
    assert set(c.perfiles) == {"general", "accion"} and set(c.eventos) == {"feria"}
    assert "sin-cuando" in caplog.text and "largo" in caplog.text


def test_langfuse_caido_usa_snapshot():
    c = prompts.load(FakeLF(full(), boom=True))
    assert c.origen == "snapshot" and "sinfonia/persona" in c.pieces


def test_falta_persona_toma_la_del_snapshot():
    d = full()
    del d["sinfonia/persona"]
    c = prompts.load(FakeLF(d))
    assert c.origen == "langfuse" and c.pieces["sinfonia/persona"].obj is None


def test_snapshot_del_paquete_tiene_obligatorias():
    c = prompts.load()
    assert all(n in c.pieces for n in prompts.REQUIRED)


def use(monkeypatch, lf):
    monkeypatch.setattr(prompts, "_cat", prompts.load(lf))
    monkeypatch.setattr(prompts, "_ts", 1e18)  # no vence


def test_compose_con_y_sin_evento(monkeypatch):
    use(monkeypatch, FakeLF(full()))
    g = prompts.catalog().perfiles["general"]
    sin = prompts.compose(g, robot="Nao")
    assert "Evento actual" not in sin.texto and sin.texto.startswith("Soy Nao")
    prompts.set_evento("feria")
    con = prompts.compose(g, robot="Nao").texto
    assert con.index("Soy Nao") < con.index("## Evento actual") < con.index("## Cómo respondes") < con.index("## Perfil de este turno: general")
    assert "Feria en Nao land" in con


def test_evento_inexistente_y_luego_aparece(monkeypatch, caplog):
    use(monkeypatch, FakeLF(full()))
    g = prompts.catalog().perfiles["general"]
    prompts.set_evento("nueva")
    with caplog.at_level(logging.ERROR):
        assert "Evento actual" not in prompts.compose(g).texto
    assert "nueva" in caplog.text
    use(monkeypatch, FakeLF(full(**{"sinfonia/evento/nueva": ("sinfonia-evento", 1, "hola nueva", {})})))  # "refresco"
    assert "hola nueva" in prompts.compose(g).texto


def test_huella_cambia_con_la_version(monkeypatch):
    use(monkeypatch, FakeLF(full()))
    h1 = prompts.compose(prompts.catalog().perfiles["general"]).huella
    use(monkeypatch, FakeLF(full(**{"speech/response-base": ("speech-sistema", 3, "base", {})})))
    assert prompts.compose(prompts.catalog().perfiles["general"]).huella != h1


def test_refresco_con_langfuse_caido_conserva_el_anterior(monkeypatch):
    use(monkeypatch, FakeLF(full()))
    monkeypatch.setattr(prompts, "_ts", 0.0)  # vencido
    monkeypatch.setattr(prompts, "_langfuse", lambda: FakeLF(full(), boom=True))
    assert prompts.catalog().origen == "langfuse"
