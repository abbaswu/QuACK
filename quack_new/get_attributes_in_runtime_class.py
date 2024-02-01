from dis import Bytecode
from enum import Enum, auto
from functools import lru_cache
from types import CodeType, FunctionType
from typing import Generator

from get_unwrapped_constructor import get_unwrapped_constructor
from type_definitions import RuntimeClass


# looks for the following bytecode sequences
# uses the state machine design pattern
# LOAD_FAST                0 (self)
# LOAD_ATTR                2 (update)
#
# LOAD_FAST                0 (self)
# STORE_ATTR               5 (_io_refs)
#
# LOAD_GLOBAL              0 (setattr)
# LOAD_FAST                0 (self)
# LOAD_CONST               1 ('x')
#
# LOAD_GLOBAL              6 (super)
# CALL_FUNCTION            0
# LOAD_METHOD              7 (__setattr__)
# LOAD_CONST               4 ('a')
#
# LOAD_GLOBAL              8 (object)
# LOAD_METHOD              7 (__setattr__)
# LOAD_FAST                0 (self)
# LOAD_CONST               5 ('b')
#
# LOAD_FAST                0 (self)
# LOAD_METHOD              7 (__setattr__)
# LOAD_CONST               6 ('c')
class State(Enum):
    START = auto()
    AFTER_LOAD_FAST_0 = auto()
    AFTER_LOADING_FUNCTION_OR_METHOD = auto()
    AFTER_LOADING_FUNCTION_OR_METHOD_AND_POSSIBLY_PROVIDING_SELF = auto()


def get_attributes_accessed_on_self_in_method(code: CodeType) -> Generator[str, None, None]:
    state = State.START

    for instruction in Bytecode(code):
        if state == State.START:
            # Instruction(opname='LOAD_FAST', opcode=124, arg=0, argval='self', argrepr='self', offset=66, starts_line=None, is_jump_target=False)
            if instruction.opname == 'LOAD_FAST' and instruction.arg == 0:
                state = State.AFTER_LOAD_FAST_0
            # Instruction(opname='LOAD_GLOBAL', opcode=116, arg=0, argval='setattr', argrepr='setattr', offset=0, starts_line=3, is_jump_target=False)
            elif instruction.opname == 'LOAD_GLOBAL' and instruction.argval in (
            'setattr', 'getattr', 'delattr', 'hasattr'):
                state = State.AFTER_LOADING_FUNCTION_OR_METHOD
            # Instruction(opname='LOAD_METHOD', opcode=160, arg=7, argval='__setattr__', argrepr='__setattr__', offset=80, starts_line=None, is_jump_target=False)
            elif instruction.opname == 'LOAD_METHOD' and instruction.argval in (
            '__setattr__', '__getattr__', '__delattr__', '__hasattr__'):
                state = State.AFTER_LOADING_FUNCTION_OR_METHOD_AND_POSSIBLY_PROVIDING_SELF
            else:
                state = State.START
        elif state == State.AFTER_LOAD_FAST_0:
            # Instruction(opname='LOAD_ATTR', opcode=106, arg=2, argval='update', argrepr='update', offset=12, starts_line=None, is_jump_target=False)
            # Instruction(opname='STORE_ATTR', opcode=95, arg=5, argval='_io_refs', argrepr='_io_refs', offset=68, starts_line=None, is_jump_target=False)
            if instruction.opname in ('LOAD_ATTR', 'STORE_ATTR'):
                yield instruction.argval
                state = State.START
            # Instruction(opname='LOAD_METHOD', opcode=160, arg=7, argval='__setattr__', argrepr='__setattr__', offset=80, starts_line=None, is_jump_target=False)
            elif instruction.opname == 'LOAD_METHOD' and instruction.argval in (
            '__setattr__', '__getattr__', '__delattr__', '__hasattr__'):
                state = State.AFTER_LOADING_FUNCTION_OR_METHOD_AND_POSSIBLY_PROVIDING_SELF
            else:
                state = State.START
        elif state == State.AFTER_LOADING_FUNCTION_OR_METHOD:
            # Instruction(opname='LOAD_FAST', opcode=124, arg=0, argval='self', argrepr='self', offset=2, starts_line=None, is_jump_target=False)
            if instruction.opname == 'LOAD_FAST' and instruction.arg == 0:
                state = State.AFTER_LOADING_FUNCTION_OR_METHOD_AND_POSSIBLY_PROVIDING_SELF
            else:
                state = State.START
        elif state == State.AFTER_LOADING_FUNCTION_OR_METHOD_AND_POSSIBLY_PROVIDING_SELF:
            # Instruction(opname='LOAD_CONST', opcode=100, arg=1, argval='x', argrepr="'x'", offset=4, starts_line=None, is_jump_target=False)
            if instruction.opname == 'LOAD_CONST':
                yield instruction.argval
                state = State.START
            # Instruction(opname='LOAD_FAST', opcode=124, arg=0, argval='self', argrepr='self', offset=2, starts_line=None, is_jump_target=False)
            elif instruction.opname == 'LOAD_FAST' and instruction.arg == 0:
                state = State.AFTER_LOADING_FUNCTION_OR_METHOD_AND_POSSIBLY_PROVIDING_SELF
            else:
                state = State.START


@lru_cache(maxsize=None)
def get_dynamic_attributes_in_runtime_class(runtime_class: RuntimeClass) -> set[str]:
    dynamic_attributes_in_runtime_class: set[str] = set()

    for mro_class in runtime_class.__mro__:
        unwrapped_constructor = get_unwrapped_constructor(mro_class)
        if isinstance(unwrapped_constructor, FunctionType):
            dynamic_attributes_in_runtime_class.update(get_attributes_accessed_on_self_in_method(
                unwrapped_constructor.__code__
            ))

    return dynamic_attributes_in_runtime_class


@lru_cache(maxsize=None)
def get_non_dynamic_attributes_in_runtime_class(runtime_class: RuntimeClass) -> set[str]:
    return set(dir(runtime_class)) - {
        # https://peps.python.org/pep-3119/
        # https://github.com/python/cpython/blob/main/Modules/_abc.c
        '__abstractmethods__',
        '_abc_impl',
        # https://peps.python.org/pep-0544/
        '_is_protocol',
        '_is_runtime_protocol',
        # https://peps.python.org/pep-0560/
        '__class_getitem__',
        '__orig_bases__',
        # https://docs.python.org/3/reference/datamodel.html#custom-classes
        '__name__',
        '__module__',
        '__dict__',
        '__weakref__',
        '__bases__',
        '__doc__',
        '__annotations__',
        '__type_params__',
        # https://docs.python.org/3/reference/datamodel.html#slots
        '__slots__',
        # https://peps.python.org/pep-0585/
        '__parameters__',
        # https://docs.python.org/3/library/pickle.html
        '__getstate__',
        '__setstate__',
        # https://docs.python.org/3/reference/datamodel.html#object.__length_hint__
        # This method is purely an optimization and is never required for correctness.
        '__length_hint__',
    } | {
        '__ne__',
        '__reduce__',
        '__setattr__',
        '__eq__',
        '__doc__',
        '__lt__',
        '__class__',
        '__gt__',
        '__module__',
        '__le__',
        '__repr__',
        '__dict__',
        '__sizeof__',
        '__init__',
        '__str__',
        '__dir__',
        '__getattribute__',
        '__format__',
        '__delattr__',
        '__init_subclass__',
        '__ge__',
        '__subclasshook__',
        '__hash__',
        '__reduce_ex__',
        '__new__',
        '__annotations__'
    } # all attributes in the Typeshed definition of `object`


@lru_cache(maxsize=None)
def get_attributes_in_runtime_class(runtime_class: RuntimeClass) -> set[str]:
    return get_dynamic_attributes_in_runtime_class(runtime_class) | get_non_dynamic_attributes_in_runtime_class(runtime_class)
