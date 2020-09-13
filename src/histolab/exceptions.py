class HistolabException(Exception):
    """Histolab custom exception main class"""

    def __init__(self, *args) -> None:
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        if self.message:
            return self.message
        return ""


class LevelError(HistolabException):
    """Raised when a requested level is not available"""


class FilterCompositionError(HistolabException):
    """Raised when a filter composition for the class is not available"""
