import typing

from type_definitions import Module, Class, Function


class UnboundMethod:
    __slots__ = ('class_', 'function')
    
    def __init__(self, class_: Class, function: Function):
        self.class_ = class_
        self.function = function

    def __hash__(self) -> int:
        return hash((self.class_, self.function))

    def __repr__(self):
        return f'UnboundMethod({self.class_}, {self.function})'
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, UnboundMethod):
            return False
        return self.class_ == other.class_ and self.function == other.function


class Instance:
    __slots__ = ('class_',)
    
    def __init__(self, class_: Class):
        self.class_ = class_
    
    def __hash__(self) -> int:
        return hash(self.class_)

    def __repr__(self):
        return f'Instance({self.class_})'
    
    def __eq__(self, other: object) -> int:
        if not isinstance(other, Instance):
            return False
        return self.class_ == other.class_


class BoundMethod:
    __slots__ = ('instance', 'function')
    
    def __init__(self, instance: Instance, function: Function):
        self.instance = instance
        self.function = function
    
    def __hash__(self) -> int:
        return hash((self.instance, self.function))

    def __repr__(self):
        return f'BoundMethod({self.instance}, {self.function})'
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BoundMethod):
            return False
        return self.instance == other.instance and self.function == other.function


RuntimeTerm = typing.Union[
    Module,
    Class,
    Function,
    UnboundMethod,
    Instance,
    BoundMethod
]
