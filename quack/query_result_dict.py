import logging

from collections import defaultdict

from typing import Callable, TypeAlias


# Raw Result Defaultdict
FunctionLevelRawResultDefaultdict: TypeAlias = defaultdict[
    str,  # parameter_name_or_return
    list[
        str  # type_annotation_string
    ]
]

ClassLevelRawResultDefaultdict: TypeAlias = defaultdict[
    str,  # function_name
    FunctionLevelRawResultDefaultdict
]

ModuleLevelRawResultDefaultdict: TypeAlias = defaultdict[
    str,  # class_name_or_global
    ClassLevelRawResultDefaultdict
]

RawResultDefaultdict: TypeAlias = defaultdict[
    str,  # module_name
    ModuleLevelRawResultDefaultdict
]


def get_raw_result_defaultdict() -> RawResultDefaultdict:
    return RawResultDefaultdict(
        lambda: ModuleLevelRawResultDefaultdict(lambda: ClassLevelRawResultDefaultdict(lambda: FunctionLevelRawResultDefaultdict(list)))
    )
