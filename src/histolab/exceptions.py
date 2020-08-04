class LevelError(Exception):
    """Raised when a requested level is not available"""

    def __init__(self, *args) -> None:
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        if self.message:
            return self.message
        else:
            return ""
