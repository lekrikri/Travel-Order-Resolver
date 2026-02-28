"""Engine registry -- discovers and provides access to all NLP and graph engines."""

from __future__ import annotations

from src.nlp.base import NLPEngine
from src.graph.base import GraphEngine

_nlp_engines: dict[str, NLPEngine] = {}
_graph_engines: dict[str, GraphEngine] = {}


def register_nlp(engine: NLPEngine) -> None:
    _nlp_engines[engine.name] = engine


def register_graph(engine: GraphEngine) -> None:
    _graph_engines[engine.name] = engine


def get_nlp(name: str) -> NLPEngine:
    return _nlp_engines[name]


def get_graph(name: str) -> GraphEngine:
    return _graph_engines[name]


def list_nlp() -> list[NLPEngine]:
    return list(_nlp_engines.values())


def list_graph() -> list[GraphEngine]:
    return list(_graph_engines.values())
