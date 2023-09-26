class TypeInferenceClass:
    __slots__ = ('module_name', 'class_name')

    def __init__(self, module_name: str | None, name: str):
        self.module_name = module_name
        self.class_name = name

    def __eq__(self, other: object) -> bool:
        if isinstance(other, TypeInferenceClass):
            return self.module_name == other.module_name and self.class_name == other.class_name
        return False

    def __hash__(self) -> int:
        return hash((self.module_name, self.class_name))

    def __repr__(self) -> str:
        # Special representations for builtins.NoneType and builtins.ellipsis
        if self.module_name == 'builtins':
            if self.class_name == 'NoneType':
                return 'None'
            elif self.class_name == 'ellipsis':
                return '...'
        
        if self.module_name is not None:
            return f'{self.module_name}.{self.class_name}'
        else:
            return self.class_name
