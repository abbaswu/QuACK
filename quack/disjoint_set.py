"""
Adapted from:

- https://github.com/mrapacz/disjoint-set
- https://chat.openai.com/share/072d5aa1-c519-4a81-b5a4-914c0e5fbb41
"""
from collections import defaultdict
from typing import Dict, Generic, Iterator, DefaultDict, Tuple, Set, Callable, Optional
from typing import TypeVar

T = TypeVar("T")


class IdentityDict(Dict[T, T]):
    """A defaultdict implementation which places the requested key as its value in case it's missing."""
    def __missing__(self, key: T) -> T:
        self[key] = key
        return key


class DisjointSet(Generic[T]):
    """A disjoint set data structure."""

    def __init__(self, *args, **kwargs) -> None:
        self.element_to_parent_element: IdentityDict[T] = IdentityDict(*args, **kwargs)

    def __contains__(self, element: T) -> bool:
        return element in self.element_to_parent_element

    def __iter__(self) -> Iterator[Tuple[T, T]]:
        """Iterate over elements and the top elements of the sets containing the elements."""
        for element in self.element_to_parent_element.keys():
            yield element, self.find(element)

    def itersets(self) -> Iterator[Tuple[T, Set[T]]]:
        """
        Yield tuples of (<top element of a set of elements>, <the set of elements>).
        """
        top_element_to_set_of_elements: DefaultDict[T, Set[T]] = defaultdict(set)
        for element in self.element_to_parent_element:
            top_element_to_set_of_elements[self.find(element)].add(element)

        yield from top_element_to_set_of_elements.items()

    def find(self, element: T) -> T:
        """
        Finds the top element of the set containing the given element.
        Simultaneously performs path compression.
        """
        if self.element_to_parent_element[element] != element:
            self.element_to_parent_element[element] = self.find(self.element_to_parent_element[element])
        return self.element_to_parent_element[element]

    def union(
            self,
            first_element: T,
            second_element: T,
            callback: Optional[Callable[[T, T], None]] = None
    ) -> None:
        """
        Merge the sets containing first_element and second_element together if they are not already together.
        If a merge happens, `callback` is invoked with the top elements of the *target* set and the *acquirer* set as parameters.
        The terminology of *target* and *acquirer* comes from https://en.wikipedia.org/wiki/Takeover.
        """
        top_element_of_set_containing_first_element = self.find(first_element)
        top_element_of_set_containing_second_element = self.find(second_element)

        if top_element_of_set_containing_first_element != top_element_of_set_containing_second_element:
            if callback is not None:
                callback(top_element_of_set_containing_first_element, top_element_of_set_containing_second_element)
            self.element_to_parent_element[top_element_of_set_containing_first_element] = top_element_of_set_containing_second_element

    def connected(self, first_element: T, second_element: T) -> bool:
        """
        Return True if first_element and second_element belong to the same set (i.e. their containing sets have the same top element).
        """
        return self.find(first_element) == self.find(second_element)
