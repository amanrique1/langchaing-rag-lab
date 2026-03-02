from typing import Generic, TypeVar, Optional, Any

T = TypeVar("T")
E = TypeVar("E")

class Result(Generic[T, E]):
    """
    A generic Result type that encapsulates either a successful value or an error.
    """
    def __init__(self, is_success: bool, value: Optional[T], error: Optional[E]):
        self._is_success = is_success
        self._value = value
        self._error = error

    @property
    def is_success(self) -> bool:
        return self._is_success

    @property
    def is_failure(self) -> bool:
        return not self._is_success

    @property
    def value(self) -> T:
        if not self.is_success:
            raise ValueError(f"Cannot get value of a failed Result. Error was: {self._error}")
        return self._value  # type: ignore

    @property
    def error(self) -> E:
        if self.is_success:
            raise ValueError("Cannot get error of a successful Result.")
        return self._error  # type: ignore

    @classmethod
    def ok(cls, value: T) -> "Result[T, Any]":
        return cls(is_success=True, value=value, error=None)

    @classmethod
    def fail(cls, error: E) -> "Result[Any, E]":
        return cls(is_success=False, value=None, error=error)
