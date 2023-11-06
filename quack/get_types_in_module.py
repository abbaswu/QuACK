from types import ModuleType


def get_types_in_module(module: ModuleType) -> set[type]:
    types_in_module: set[type] = set()
    for key, value in module.__dict__.items():
        if type in type(value).__mro__ and (
                value.__module__ == module.__name__ or value.__module__ == '_' + module.__name__ or value.__module__ == 'builtins'
        ):
            types_in_module.add(value)
    return types_in_module
