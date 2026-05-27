from contextvars import ContextVar
from contextlib import contextmanager
from typing import Dict, List, Tuple
import logging

import numpy as np
from numpy import ndarray

logger = logging.getLogger(__name__)

_layer_state: ContextVar[Tuple[Dict[str, ndarray], Dict[str, int]] | None] = ContextVar("_layer_state", default=None)


def _get_state() -> Tuple[Dict[str, ndarray], Dict[str, int]]:
    state = _layer_state.get()
    if state is None:
        state = ({}, {})
        _layer_state.set(state)
    return state


@contextmanager
def isolated_layer_context():
    token = _layer_state.set(({}, {}))
    try:
        yield
    finally:
        _layer_state.reset(token)


def register_layer(name: str, layer: ndarray) -> None:
    layers, access_count = _get_state()

    if name in layers:
        raise KeyError(f"Layer already registered: {name}")

    if not isinstance(layer, np.ndarray):
        raise TypeError(f"Layer '{name}' must be a numpy.ndarray, got {type(layer)}")

    layers[name] = layer
    access_count[name] = 0


def set_layer(name: str, layer: ndarray) -> None:
    layers, access_count = _get_state()

    if not isinstance(layer, np.ndarray):
        raise TypeError(f"Layer '{name}' must be a numpy.ndarray, got {type(layer)}")

    layers[name] = layer
    if name not in access_count:
        access_count[name] = 0


def get_layer(name: str) -> ndarray:
    layers, access_count = _get_state()

    if name not in layers:
        raise KeyError(f"The given layer name not registered: {name}")
    access_count[name] += 1
    return layers[name]


def delete_layer(name: str) -> None:
    layers, access_count = _get_state()
    if name in layers:
        del layers[name]
        del access_count[name]


def list_layers() -> List[str]:
    layers, _ = _get_state()
    return list(layers.keys())


def show_access_count() -> None:
    _, access_count = _get_state()
    logger.info("Layer access count: %s", access_count)


def clear_all_layers() -> None:
    _layer_state.set(({}, {}))