import ast
import types
import typing


# Module
Module = types.ModuleType

# Class
RuntimeClass = type

Class = RuntimeClass

# Unwrapped Runtime Function
# Unwrap decorated functions (staticmethod, functools._lru_cache_wrapper, etc.) through the `__wrapped__` attribute
# Based on https://stackoverflow.com/questions/1166118/how-to-strip-decorators-from-a-function-in-python
UnwrappedRuntimeFunction = typing.Union[
    types.FunctionType,
    types.BuiltinFunctionType,
    types.WrapperDescriptorType,
    types.MethodDescriptorType,
    types.ClassMethodDescriptorType
]

NamedFunctionDefinition = typing.Union[
    ast.FunctionDef,
    ast.AsyncFunctionDef
]

FunctionDefinition = typing.Union[
    NamedFunctionDefinition,
    ast.Lambda
]

Function = typing.Union[
    UnwrappedRuntimeFunction,
    FunctionDefinition
]
