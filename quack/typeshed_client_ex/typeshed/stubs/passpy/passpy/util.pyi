from collections.abc import Callable
from typing import Any, TypeVar

_C = TypeVar("_C", bound=Callable[..., Any])

# Technically, the first argument of `_C` must be `Store`,
# but for now we leave it simple:
def initialised(func: _C) -> _C: ...
def trap(path_index: str | int) -> Callable[[_C], _C]: ...
def gen_password(length: int, symbols: bool = ...) -> str: ...
def copy_move(
    src: str, dst: str, force: bool = ..., move: bool = ..., interactive: bool = ..., verbose: bool = ...
) -> str | None: ...
