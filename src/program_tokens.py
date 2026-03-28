"""Helpers for serializing and materializing Program token lists."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from src import function as function_module
from src.function import Operator, Terminal
from src.program import Program


TOKEN_REGISTRY = {
    name: value
    for name, value in vars(function_module).items()
    if isinstance(value, (Operator, Terminal))
}


def materialize_token_program(token_names: Iterable[str]) -> list[Operator | Terminal]:
    program = []
    missing = []
    for token_name in token_names:
        token = TOKEN_REGISTRY.get(token_name)
        if token is None:
            missing.append(token_name)
            continue
        program.append(token)

    if missing:
        raise KeyError(f"Unknown token names: {', '.join(missing)}")
    return program


def serialize_token_program(program: Iterable[Operator | Terminal]) -> list[str]:
    reverse_registry = {id(token): name for name, token in TOKEN_REGISTRY.items()}
    names = []
    missing = []
    for token in program:
        token_name = reverse_registry.get(id(token))
        if token_name is None:
            missing.append(str(token))
            continue
        names.append(token_name)
    if missing:
        raise KeyError(f"Unregistered token objects: {', '.join(missing)}")
    return names


def validate_token_program(
    token_names: Iterable[str],
    *,
    max_depth: int = 6,
    max_operators: int = 6,
    metric=None,
) -> Program:
    compiled_program = materialize_token_program(token_names)
    metric = metric or (lambda _expression: 0.0)
    return Program(
        max_depth=max_depth,
        max_operators=max_operators,
        random_state=np.random.RandomState(0),
        metric=metric,
        program=compiled_program,
    )


def render_token_program(
    token_names: Iterable[str],
    *,
    max_depth: int = 6,
    max_operators: int = 6,
) -> str:
    return str(
        validate_token_program(
            token_names,
            max_depth=max_depth,
            max_operators=max_operators,
        )
    )
