from _typeshed import Incomplete

class TemplateSummaryDiffLabelsNewOld:
    openapi_types: Incomplete
    attribute_map: Incomplete
    discriminator: Incomplete
    def __init__(
        self, name: Incomplete | None = ..., color: Incomplete | None = ..., description: Incomplete | None = ...
    ) -> None: ...
    @property
    def name(self): ...
    @name.setter
    def name(self, name) -> None: ...
    @property
    def color(self): ...
    @color.setter
    def color(self, color) -> None: ...
    @property
    def description(self): ...
    @description.setter
    def description(self, description) -> None: ...
    def to_dict(self): ...
    def to_str(self): ...
    def __eq__(self, other): ...
    def __ne__(self, other): ...